from __future__ import annotations

import csv
import gc
import importlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "wsp2_policy"
POLICY_SOURCE = PROJECT_ROOT / "experiments/robotwin" / POLICY_NAME
PHASE_CONFIG = {"clean": "demo_clean", "random": "demo_randomized"}
SUMMARY_PHASES = ("clean", "random")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.robotwin2.scheduler import (
    DynamicTaskScheduler,
    ProcessWorker,
    QueueEventSource,
)


@dataclass(frozen=True)
class EvaluationJob:
    task_name: str
    phase: str

    @property
    def job_id(self) -> str:
        return f"{self.task_name}:{self.phase}"


@dataclass(frozen=True)
class EvaluationJobResult:
    task_name: str
    phase: str
    gpu_id: int
    success: bool
    success_rate: float | None
    log_path: str
    error: str | None = None


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def _resolve_path(value: str, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _checkpoint_tag(checkpoint: Path) -> str:
    return checkpoint.name


def _ensure_policy_link(robotwin_root: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")
    target = policy_root / POLICY_NAME
    source = POLICY_SOURCE.resolve()
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source, target_is_directory=True)
    if target.is_symlink() and target.resolve() == source:
        return target
    if target.is_dir() and target.resolve() == source:
        return target
    raise RuntimeError(
        f"Policy path conflict: {target}; expected it to resolve to {source}"
    )


def _parse_rate(path: Path) -> float:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No success rate found in {path}")
    return values[-1]


def _run_render_preflight(*, gpu_id: int) -> None:
    """Run RoboTwin's own SAPIEN renderer probe before importing the policy."""

    try:
        module = importlib.import_module("script.test_render")
        probe = module.Sapien_TEST()
        if getattr(probe, "renderer", None) is None:
            raise RuntimeError("SAPIEN probe did not create a renderer")
        if getattr(probe, "scene", None) is None:
            raise RuntimeError("SAPIEN probe did not create a scene")
    except BaseException as exc:  # Sapien_TEST reports failure with SystemExit
        raise RuntimeError(
            f"RoboTwin SAPIEN render preflight failed on GPU {gpu_id}"
        ) from exc
    finally:
        if "probe" in locals():
            del probe
        gc.collect()


def _result_path(output: Path, job: EvaluationJob) -> Path:
    suffix = "clean" if job.phase == "clean" else "random"
    return output / job.task_name / job.phase / f"_result_{suffix}.txt"


def _curobo_planner_specs(
    robotwin_root: Path, phases: Iterable[str]
) -> list[tuple[str, list[float]]]:
    """Resolve the (curobo yml, robot pose) pairs `Robot.set_planner` will build.

    Mirrors script/eval_policy.main + envs/robot/robot.py so the prewarm goes
    through exactly the same cuRobo configs (and therefore the same Warp
    module hashes) as the evaluation workers.
    """

    task_config_dir = robotwin_root / "task_config"
    embodiment_types = yaml.safe_load(
        (task_config_dir / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    specs: dict[str, list[float]] = {}
    for phase in phases:
        task_args = yaml.safe_load(
            (task_config_dir / f"{PHASE_CONFIG[phase]}.yml").read_text(
                encoding="utf-8"
            )
        )
        embodiment = list(task_args["embodiment"])
        if len(embodiment) == 1:
            robot_files = [embodiment[0], embodiment[0]]
            yml_names = ["curobo_left.yml", "curobo_right.yml"]
        elif len(embodiment) == 3:
            robot_files = [embodiment[0], embodiment[1]]
            yml_names = ["curobo.yml", "curobo.yml"]
        else:
            raise ValueError(f"embodiment items should be 1 or 3: {embodiment}")
        for robot_file, yml_name in zip(robot_files, yml_names):
            robot_dir = robotwin_root / embodiment_types[robot_file]["file_path"]
            yml_path = (robot_dir / yml_name).resolve()
            if not yml_path.is_file():
                raise FileNotFoundError(f"cuRobo config not found: {yml_path}")
            robot_cfg = yaml.safe_load(
                (robot_dir / "config.yml").read_text(encoding="utf-8")
            )
            pose = [float(v) for v in robot_cfg["robot_pose"][0]]
            specs.setdefault(str(yml_path), pose)
    return list(specs.items())


def _prewarm_worker(
    robotwin_root: str,
    gpu_id: int,
    planner_specs: list[tuple[str, list[float]]],
) -> None:
    """Build every cuRobo planner once so Warp kernels land in the cache.

    Runs in a dedicated process before the GPU workers are spawned. Warp JIT
    compiles cuRobo kernels on the first `CuroboPlanner` construction; when
    eight workers hit a cold cache simultaneously, the shared precompiled
    header can be read while another process is still writing it and NVRTC
    fails with `CUDA kernel build failed with error code 6`. Warming the
    cache from a single process removes that race.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.chdir(robotwin_root)
    for path in (robotwin_root, os.path.join(robotwin_root, "policy")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import sapien.core as sapien  # noqa: PLC0415 - simulator import is heavy

    planner_module = importlib.import_module("envs.robot.planner")
    planner_cls = getattr(planner_module, "CuroboPlanner", None)
    if planner_cls is None:
        raise RuntimeError(
            "CuroboPlanner is unavailable in envs/robot/planner.py; "
            "expert_check cannot run without cuRobo."
        )
    for yml_path, pose in planner_specs:
        started = time.perf_counter()
        planner_cls(
            sapien.Pose(pose[:3], pose[3:]),
            [],
            [],
            yml_path=yml_path,
        )
        print(
            f"[warp-prewarm] built {os.path.basename(yml_path)} in "
            f"{time.perf_counter() - started:.1f}s"
        )


def _prewarm_warp_kernels(
    robotwin_root: Path,
    *,
    gpu_id: int,
    phases: Iterable[str],
    timeout_s: float,
) -> None:
    specs = _curobo_planner_specs(robotwin_root, phases)
    print(
        f"[warp-prewarm] gpu={gpu_id} cache="
        f"{os.environ.get('WARP_CACHE_PATH', '<warp default>')} "
        f"planners={[os.path.basename(p) for p, _ in specs]}",
        flush=True,
    )
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_prewarm_worker,
        args=(str(robotwin_root), int(gpu_id), specs),
        name="warp-prewarm",
    )
    process.start()
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError(
            f"Warp kernel prewarm exceeded {timeout_s:.0f}s on GPU {gpu_id}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"Warp kernel prewarm failed on GPU {gpu_id} "
            f"(exit code {process.exitcode}); refusing to start workers "
            "against a broken cuRobo/Warp runtime."
        )
    print(
        f"[warp-prewarm] done in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def _resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)  # type: ignore[return-value]



def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

@dataclass(frozen=True)
class NativeWorkerFactory:
    config: dict[str, Any]
    checkpoint: str
    robotwin_root: str
    output_dir: str

    def __call__(self, gpu_id: int) -> NativeRoboTwinTaskRunner:
        return NativeRoboTwinTaskRunner(
            self.config,
            checkpoint=Path(self.checkpoint),
            robotwin_root=Path(self.robotwin_root),
            output_dir=Path(self.output_dir),
            gpu_id=gpu_id,
        )


class NativeRoboTwinTaskRunner:
    """One long-lived GPU worker using RoboTwin's native eval_policy API."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        checkpoint: Path,
        robotwin_root: Path,
        output_dir: Path,
        gpu_id: int,
    ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.robotwin_root = robotwin_root
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.chdir(robotwin_root)
        for path in (
            PROJECT_ROOT,
            PROJECT_ROOT / "src",
            robotwin_root,
            robotwin_root / "policy",
            robotwin_root / "description/utils",
        ):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if bool(config["EVALUATION"].get("render_preflight", True)):
            _run_render_preflight(gpu_id=gpu_id)
        self.bridge = importlib.import_module("script.eval_policy")
        self.policy_module = importlib.import_module(POLICY_NAME)
        self.model: Any = None

    def __call__(self, job: EvaluationJob) -> EvaluationJobResult:
        log = self.output_dir / (
            f"eval_{job.task_name}_{PHASE_CONFIG[job.phase]}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        job_started = time.perf_counter()
        try:
            if self.model is not None:
                self.policy_module.prepare_model_for_evaluation_job(self.model)
            usr_args = self._usr_args(job)
            with log.open("w", encoding="utf-8") as stream:
                tee = _Tee(sys.__stdout__, stream)
                with redirect_stdout(tee), redirect_stderr(tee):
                    self.model = self.bridge.main(usr_args, model=self.model)
                    print(
                        f"[manager-timing] job_s="
                        f"{time.perf_counter() - job_started:.3f}"
                    )
            rate = _parse_rate(_result_path(self.output_dir, job))
            return EvaluationJobResult(
                job.task_name,
                job.phase,
                self.gpu_id,
                True,
                rate,
                str(log),
            )
        except BaseException as exc:  # noqa: BLE001 - task failures are results
            with log.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            return EvaluationJobResult(
                job.task_name,
                job.phase,
                self.gpu_id,
                False,
                None,
                str(log),
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            if self.model is not None:
                self.policy_module.reset_model(self.model)

    def _usr_args(self, job: EvaluationJob) -> dict[str, Any]:
        evaluation = self.config["EVALUATION"]
        args = {
            "task_name": job.task_name,
            "task_config": PHASE_CONFIG[job.phase],
            "ckpt_setting": str(self.checkpoint),
            "seed": int(self.config.get("seed", 0)),
            "policy_name": POLICY_NAME,
            "instruction_type": str(evaluation.get("instruction_type", "seen")),
            "eval_num_episodes": int(evaluation.get("eval_num_episodes", 100)),
            "max_seed_trials": int(evaluation.get("max_seed_trials", 1000)),
            "eval_output_dir": str(
                self.output_dir / job.task_name / job.phase
            ),
            "eval_video_log": bool(evaluation.get("save_video", False)),
            "clear_cache_freq": int(evaluation.get("clear_cache_freq", 1)),
            "sim_cfg_path": str(
                PROJECT_ROOT / "configs/eval/robotwin2_manager.yaml"
            ),
            "device": str(evaluation.get("device", "cuda")),
            "mixed_precision": str(self.config.get("mixed_precision", "bf16")),
            "mode": str(evaluation.get("mode", "auto")),
            "action_type": str(evaluation.get("action_type", "qpos")),
            "action_horizon": int(evaluation.get("action_horizon", 24)),
            "replan_steps": int(evaluation.get("replan_steps", 24)),
            "memory_reset_chunks": int(
                evaluation.get("memory_reset_chunks", 2)
            ),
            "skip_get_obs_within_replan": bool(
                evaluation.get("skip_get_obs_within_replan", True)
            ),
            "observation_interval": int(
                evaluation.get("observation_interval", 3)
            ),
            "vlm_history_num_frames": int(
                evaluation.get("vlm_history_num_frames", 8)
            ),
            "vlm_history_stride": int(
                evaluation.get(
                    "vlm_history_stride",
                    evaluation.get("action_horizon", 24),
                )
            ),
            "vlm_cot_prompt": str(evaluation["vlm_cot_prompt"]),
            "t5_prompt_template": str(evaluation["t5_prompt_template"]),
            "validate_checkpoint_artifacts": bool(
                evaluation.get("validate_checkpoint_artifacts", False)
            ),
            "log_inference": bool(evaluation.get("log_inference", True)),
        }
        diffusion_view_layout = evaluation.get("diffusion_view_layout")
        if diffusion_view_layout is not None:
            args["diffusion_view_layout"] = str(diffusion_view_layout)
        return args


def run_manager(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["EVALUATION"]
    multirun = config["MULTIRUN"]
    checkpoint = _resolve_path(str(config["ckpt"]))
    robotwin_root = _resolve_path(str(evaluation["robotwin_root"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not (robotwin_root / "script/eval_policy.py").is_file():
        raise FileNotFoundError(f"Invalid RoboTwin root: {robotwin_root}")
    _ensure_policy_link(robotwin_root)

    task_name = evaluation.get("task_name")
    if task_name:
        tasks = [str(task_name)]
    else:
        task_map = yaml.safe_load(
            (robotwin_root / "task_config/_eval_step_limit.yml").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(task_map, dict) or not task_map:
            raise ValueError("RoboTwin task catalog is empty")
        tasks = list(task_map)
    phase_name = str(multirun.get("eval_phases", "clean")).lower()
    if phase_name not in {"clean", "random", "both"}:
        raise ValueError(f"Unsupported eval_phases: {phase_name}")
    phases = ("clean", "random") if phase_name == "both" else (phase_name,)

    run_id = _resolve_path(str(evaluation["output_dir"])).name
    output = (
        PROJECT_ROOT
        / "evaluate_results/robotwin"
        / _checkpoint_tag(checkpoint)
        / run_id
    )
    output.mkdir(parents=True, exist_ok=True)
    jobs = [EvaluationJob(task, phase) for task in tasks for phase in phases]
    resume = bool(multirun.get("resume", True))
    pending = []
    resumed: list[EvaluationJobResult] = []
    for job in jobs:
        result_path = _result_path(output, job)
        if resume and result_path.is_file():
            try:
                resumed.append(
                    EvaluationJobResult(
                        job.task_name,
                        job.phase,
                        -1,
                        True,
                        _parse_rate(result_path),
                        "",
                    )
                )
                continue
            except ValueError:
                pass
        pending.append(job)

    gpu_ids = [int(value) for value in multirun.get("gpu_ids", [0])]
    workers_per_gpu = int(multirun.get("workers_per_gpu", 1))
    if not gpu_ids or workers_per_gpu < 1:
        raise ValueError("gpu_ids must be non-empty and workers_per_gpu positive")
    worker_count = min(len(pending), len(gpu_ids) * workers_per_gpu)
    results = list(resumed)
    _write_summary(output, results, tasks)
    _atomic_write_text(
        output / "manager_config.yaml",
        OmegaConf.to_yaml(OmegaConf.create(config), resolve=True),
    )

    def record_result(_sequence: int, result: EvaluationJobResult) -> None:
        results.append(result)
        # Keep a usable partial summary even if a later worker or the manager
        # is interrupted. The manager is the only writer for these files.
        _write_summary(output, results, tasks)

    if worker_count:
        if bool(evaluation.get("warp_prewarm", True)):
            _prewarm_warp_kernels(
                robotwin_root,
                gpu_id=gpu_ids[0],
                phases=phases,
                timeout_s=float(evaluation.get("warp_prewarm_timeout_s", 1800)),
            )
        context = multiprocessing.get_context("spawn")
        commands = context.Queue()
        events = context.Queue()
        factory = NativeWorkerFactory(
            _resolved_config(config),
            str(checkpoint),
            str(robotwin_root),
            str(output),
        )
        workers = [
            ProcessWorker(
                worker_id,
                device_id=gpu_ids[worker_id % len(gpu_ids)],
                context=context,
                commands=commands,
                events=events,
                runner_factory=factory,
            )
            for worker_id in range(worker_count)
        ]
        DynamicTaskScheduler(
            workers,
            commands,
            QueueEventSource(events),
            poll_interval_s=float(multirun.get("poll_interval_sec", 1.0)),
        ).run(
            pending,
            on_result=record_result,
        )
    return _write_summary(output, results, tasks)


def _write_summary(
    output: Path,
    results: list[EvaluationJobResult],
    tasks: list[str],
) -> dict[str, Any]:
    rates = {task: {phase: None for phase in SUMMARY_PHASES} for task in tasks}
    failures = []
    for result in results:
        if result.success:
            rates[result.task_name][result.phase] = result.success_rate
        else:
            failures.append(result.__dict__)
    overall = {
        phase: (
            sum(values) / len(values)
            if (
                values := [
                    row[phase]
                    for row in rates.values()
                    if row[phase] is not None
                ]
            )
            else None
        )
        for phase in SUMMARY_PHASES
    }
    summary = {
        "per_task": rates,
        "overall": overall,
        "completed_jobs": sum(result.success for result in results),
        "failed_jobs": len(failures),
    }
    _atomic_write_text(
        output / "summary.json",
        json.dumps(summary, indent=2),
    )
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        ["task_name", "clean_success_rate", "random_success_rate"]
    )
    for task, row in rates.items():
        writer.writerow([task, row["clean"], row["random"]])
    writer.writerow(["__overall__", overall["clean"], overall["random"]])
    _atomic_write_text(output / "summary.csv", stream.getvalue())
    _atomic_write_text(
        output / "failed_tasks.jsonl",
        "".join(json.dumps(item) + "\n" for item in failures),
    )
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../../configs/eval",
    config_name="robotwin2_manager",
)
def main(cfg: DictConfig) -> None:
    started = time.perf_counter()
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Resolved manager config must be a mapping")
    run_manager(config)
    print(f"[manager-timing] total_s={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
