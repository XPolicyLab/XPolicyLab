#!/usr/bin/env python3
"""Run the complete RoboDojo benchmark on one client process per GPU.

The default protocol is the official 42-task / 2100-episode evaluation:

* 12 Generalization tasks: 25 base + 25 random episodes per task.
* 30 other tasks: 50 episodes per task.

The supported launcher fixes ``TASKS_PER_GPU=1``: eight client GPUs mean eight
Isaac processes. Each process uses ``ROBODOJO_ENVS_PER_CLIENT`` vectorized
environments (default six) and consumes benchmark configs from a shared queue.
Requests are spread over ``NUM_SERVERS`` independent policy services started by
``scripts/serve_policy.sh``.
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as websocket_connect

from cogwam.eval.summarize import DIMENSIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_LAUNCHER = REPO_ROOT / "scripts/eval_robodojo.sh"
LOG_LOCK = threading.Lock()

GENERALIZATION = set(DIMENSIONS["Generalization"])
CANONICAL_TASKS = [task for tasks in DIMENSIONS.values() for task in tasks]
if len(CANONICAL_TASKS) != 42 or len(set(CANONICAL_TASKS)) != 42:
    raise RuntimeError("RoboDojo taxonomy must contain exactly 42 tasks")

ROBODOJO_REPOSITORY = "https://github.com/RoboDojo-Benchmark/RoboDojo.git"
ROBODOJO_COMMIT = "9b4cc885e8f530ed3ab14a30a312ae70242771c4"


def default_config_names() -> list[str]:
    configs: list[str] = []
    for dimension, tasks in DIMENSIONS.items():
        for task in tasks:
            configs.append(task)
            if dimension == "Generalization":
                configs.append(f"{task}_random")
    return configs


DEFAULT_CONFIGS = default_config_names()
if len(DEFAULT_CONFIGS) != 54:
    raise RuntimeError("RoboDojo full protocol must contain exactly 54 configs")


def log(message: str, *, worker_index: int | None = None) -> None:
    prefix = "[RoboDojo-fast]" if worker_index is None else f"[RoboDojo-fast][worker {worker_index}]"
    with LOG_LOCK:
        print(f"{prefix} {message}", flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def resolve_robodojo_root() -> Path:
    """Locate the simulator checkout. There is deliberately no default."""

    raw = os.getenv("ROBODOJO_ROOT", "").strip()
    if not raw:
        raise ValueError(
            "ROBODOJO_ROOT is not set. Clone the benchmark at the pinned commit and export it:\n"
            f"  git clone --recurse-submodules {ROBODOJO_REPOSITORY}\n"
            f"  cd RoboDojo && git checkout {ROBODOJO_COMMIT}\n"
            "  git submodule update --init --recursive\n"
            "  export ROBODOJO_ROOT=$PWD"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboDojo root does not exist: {root}")
    return root


def thread_cap() -> str:
    """Per-process CPU thread budget for the rollout subprocesses.

    torch takes half the visible cores per process, so eight concurrent clients
    on a 128-core node ask for ~512 threads on top of the simulators. That
    fan-out costs more than the work it parallelizes: the per-camera resizes
    measure 21.6 ms at 64 threads against 11.2 ms at 8, and mid-run the client
    pod burns 141 cores with every GPU at 0%. Output is bit-identical at any
    thread count, so this only moves wall time. Override with
    ROBODOJO_THREAD_CAP.
    """

    override = os.getenv("ROBODOJO_THREAD_CAP", "").strip()
    if override:
        return str(max(1, int(override)))
    gpus = len([x for x in os.getenv("CLIENT_CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()])
    slots = max(1, gpus * max(1, env_int("TASKS_PER_GPU", 1)))
    cores = os.cpu_count() or 8
    return str(max(1, cores // slots // 2))


_THREAD_CAP = thread_cap()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _first_int(environment: Mapping[str, str], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = environment.get(name)
        if value not in (None, ""):
            return int(value)
    return None


def parse_worker_info(environment: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Resolve worker topology and reject launcher/platform disagreement."""

    environment = os.environ if environment is None else environment
    explicit_index = _first_int(environment, ("CLIENT_WORKER_INDEX",))
    explicit_count = _first_int(environment, ("CLIENT_NUM_WORKERS",))

    platform_index = _first_int(environment, ("NODE_RANK", "MACHINE_RANK", "SLURM_PROCID"))
    platform_count = _first_int(environment, ("NNODES", "NUM_MACHINES", "SLURM_NNODES"))

    if explicit_index is not None and platform_index is not None and explicit_index != platform_index:
        raise ValueError(f"CLIENT_WORKER_INDEX disagrees with the platform rank: {explicit_index} != {platform_index}")
    if explicit_count is not None and platform_count is not None and explicit_count != platform_count:
        raise ValueError(
            "CLIENT_NUM_WORKERS disagrees with the platform worker count: "
            f"{explicit_count} != {platform_count}. Set the two launch configs to the same value."
        )

    index = explicit_index if explicit_index is not None else (platform_index or 0)
    count = explicit_count if explicit_count is not None else (platform_count or 1)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid worker topology index={index}, count={count}")
    return index, count


def canonical_task_name(config_name: str) -> str:
    return config_name.removesuffix("_random")


@dataclass(frozen=True)
class TaskSpec:
    config_id: int
    task_id: int
    dimension: str
    canonical_task: str
    config_name: str
    episodes: int


def _native_episode_counts(robodojo_root: Path) -> dict[str, int]:
    task_file = robodojo_root / "task/RoboDojo/config/_task.yml"
    payload = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
    common = int((payload.get("common") or {}).get("eval_nums", 50))
    overrides = payload.get("tasks") or {}
    return {
        config_name: int((overrides.get(config_name) or {}).get("eval_nums", common))
        for config_name in DEFAULT_CONFIGS
    }


def build_task_specs(
    robodojo_root: Path,
    *,
    task_names: list[str] | None = None,
    num_episodes: str = "native",
    max_configs: int = -1,
) -> list[TaskSpec]:
    selected = list(task_names or DEFAULT_CONFIGS)
    if len(selected) != len(set(selected)):
        raise ValueError("TASK_NAMES contains duplicate configs")
    unknown = sorted(set(selected) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError(f"unknown RoboDojo configs: {unknown}")
    if max_configs >= 0:
        selected = selected[:max_configs]
    if not selected:
        raise ValueError("no RoboDojo configs selected")

    native_counts = _native_episode_counts(robodojo_root)
    if num_episodes == "native":
        episode_counts = native_counts
    else:
        count = int(num_episodes)
        if count < 1:
            raise ValueError(f"NUM_EPISODES must be native or positive, got {num_episodes!r}")
        episode_counts = {name: count for name in DEFAULT_CONFIGS}

    dimensions = {task: dimension for dimension, tasks in DIMENSIONS.items() for task in tasks}
    selected_canonical = list(dict.fromkeys(canonical_task_name(name) for name in selected))
    task_ids = {name: index for index, name in enumerate(selected_canonical, start=1)}
    specs = [
        TaskSpec(
            config_id=config_id,
            task_id=task_ids[canonical_task_name(config_name)],
            dimension=dimensions[canonical_task_name(config_name)],
            canonical_task=canonical_task_name(config_name),
            config_name=config_name,
            episodes=episode_counts[config_name],
        )
        for config_id, config_name in enumerate(selected, start=1)
    ]

    if selected == DEFAULT_CONFIGS and num_episodes == "native":
        wrong = {
            spec.config_name: spec.episodes
            for spec in specs
            if spec.episodes != (25 if spec.canonical_task in GENERALIZATION else 50)
        }
        if wrong:
            raise ValueError(
                "official RoboDojo eval_nums contract is wrong; Generalization must be "
                f"25+25 and all other configs 50: {wrong}"
            )
        if len({spec.canonical_task for spec in specs}) != 42 or sum(spec.episodes for spec in specs) != 2100:
            raise RuntimeError("invalid default protocol; expected 42 tasks and 2100 episodes")
    return specs


def partition_specs(specs: list[TaskSpec], worker_count: int) -> list[list[TaskSpec]]:
    partitions: list[list[TaskSpec]] = [[] for _ in range(worker_count)]
    totals = [0] * worker_count
    for spec in sorted(specs, key=lambda item: (-item.episodes, item.config_id)):
        worker = min(
            range(worker_count),
            key=lambda index: (totals[index], len(partitions[index]), index),
        )
        partitions[worker].append(spec)
        totals[worker] += spec.episodes
    for partition in partitions:
        partition.sort(key=lambda item: item.config_id)
    return partitions


def parse_slots_per_gpu(
    gpu_ids: list[str],
    env_name: str = "TASKS_PER_GPU_LAYOUT",
) -> list[int]:
    """Return the number of concurrent Isaac processes assigned to each GPU."""

    raw_layout = os.getenv(env_name, "").strip()
    if not raw_layout and env_name != "TASKS_PER_GPU_LAYOUT":
        raw_layout = os.getenv("TASKS_PER_GPU_LAYOUT", "").strip()
    if raw_layout:
        values = [item.strip() for item in raw_layout.replace(" ", ",").split(",") if item.strip()]
        if len(values) != len(gpu_ids):
            raise ValueError(
                f"{env_name} must have one entry per client GPU: got {len(values)} entries for GPUs {gpu_ids}"
            )
        slots = [int(value) for value in values]
    else:
        slots = [env_int("TASKS_PER_GPU", 1)] * len(gpu_ids)
    if any(value < 1 for value in slots):
        raise ValueError(f"all task slots per GPU must be positive, got {slots}")
    return slots


def iter_slot_assignments(
    gpu_ids: list[str],
    slots_per_gpu: list[int],
) -> list[tuple[str, int]]:
    """Interleave GPUs so one card does not receive an initialization burst."""

    return [
        (gpu_id, local_slot)
        for local_slot in range(max(slots_per_gpu, default=0))
        for gpu_id, slot_count in zip(gpu_ids, slots_per_gpu, strict=True)
        if local_slot < slot_count
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def wait_for_manifest(path: Path, run_id: str, worker_count: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if payload.get("run_id") != run_id or int(payload.get("worker_count", -1)) != worker_count:
            raise ValueError(f"shared manifest does not match this launch: {path}")
        return payload
    raise TimeoutError(f"timed out waiting for rank-0 manifest: {path}")


def websocket_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with websocket_connect(
            f"ws://{host}:{port}",
            open_timeout=timeout,
            close_timeout=timeout,
            compression=None,
            max_size=None,
        ):
            return True
    except (OSError, TimeoutError, WebSocketException):
        return False


def wait_for_servers(host: str, base_port: int, count: int, timeout: int, worker_index: int) -> None:
    deadline = None if timeout < 0 else time.monotonic() + timeout
    while True:
        missing = [port for port in range(base_port, base_port + count) if not websocket_ready(host, port)]
        if not missing:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"policy servers unavailable: {host}:{missing}")
        log(f"waiting for policy servers; missing={missing}", worker_index=worker_index)
        time.sleep(3)


def ensure_proxy_ports_available(base_port: int, count: int) -> list[int]:
    ports: list[int] = []
    for port in range(base_port, base_port + 10000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            sock.close()
            continue
        sock.close()
        ports.append(port)
        if len(ports) == count:
            return ports
    raise OSError(f"cannot find {count} free proxy ports starting at {base_port}")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "warming-up"
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_snapshot(output_root: Path, manifest: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    statuses: dict[str, dict[str, Any]] = {}
    for path in (output_root / "task_status").glob("config_*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        statuses[str(status.get("config_name"))] = status

    successful = [status for status in statuses.values() if status.get("state") == "success"]
    failed = [status for status in statuses.values() if status.get("state") == "failed"]
    active = [status for status in statuses.values() if status.get("state") == "running"]
    completed_episodes = sum(
        min(
            int(status.get("actual_episodes", 0)),
            int(status.get("episodes", status.get("actual_episodes", 0))),
        )
        for status in successful
    )
    expected_episodes = int(manifest["expected_episodes"])
    elapsed = max(0.0, now - float(manifest["started_at_epoch"]))
    rate = completed_episodes / elapsed if completed_episodes > 0 and elapsed > 0 else 0.0
    eta = (expected_episodes - completed_episodes) / rate if rate > 0 else None

    expected_by_task: dict[str, set[str]] = {}
    for raw in manifest["specs"]:
        expected_by_task.setdefault(raw["canonical_task"], set()).add(raw["config_name"])
    successful_configs = {status["config_name"] for status in successful}
    completed_tasks = sum(configs <= successful_configs for configs in expected_by_task.values())
    return {
        "completed_configs": len(successful),
        "expected_configs": len(manifest["specs"]),
        "completed_tasks": completed_tasks,
        "expected_tasks": len(expected_by_task),
        "completed_episodes": completed_episodes,
        "expected_episodes": expected_episodes,
        "active": len(active),
        "failed": len(failed),
        "elapsed_seconds": elapsed,
        "episode_rate": rate,
        "eta_seconds": eta,
    }


def format_progress(snapshot: Mapping[str, Any]) -> str:
    rate = float(snapshot["episode_rate"])
    rate_text = "warming-up" if rate <= 0 else f"{rate:.2f} episode/s"
    return (
        f"configs={snapshot['completed_configs']}/{snapshot['expected_configs']} "
        f"tasks={snapshot['completed_tasks']}/{snapshot['expected_tasks']} "
        f"episodes={snapshot['completed_episodes']}/{snapshot['expected_episodes']} "
        f"active={snapshot['active']} failed={snapshot['failed']} "
        f"elapsed={format_duration(float(snapshot['elapsed_seconds']))} "
        f"throughput={rate_text} ETA={format_duration(snapshot['eta_seconds'])}"
    )


class ProgressReporter(threading.Thread):
    def __init__(self, output_root: Path, manifest: Mapping[str, Any], interval: int) -> None:
        super().__init__(name="robodojo-eta", daemon=True)
        self.output_root = output_root
        self.manifest = manifest
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            log("[ETA] " + format_progress(progress_snapshot(self.output_root, self.manifest)))

    def stop(self) -> None:
        self.stop_event.set()


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint: Path
    checkpoint_name: str
    host: str
    base_port: int
    num_servers: int
    worker_index: int
    worker_count: int
    shared_run_id: str
    output_root: Path
    seed: int
    action_chunk_size: int
    execute_horizon: int
    rtc_enabled: bool
    rtc_execution_horizon: int
    rtc_inference_delay: int
    rtc_max_guidance_weight: float
    rtc_prefix_attention_schedule: str
    rtc_debug_max_replans: int
    envs_per_process: int
    task_max_retries: int
    run_name: str


ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def terminate_all_processes() -> None:
    STOP_EVENT.set()
    with ACTIVE_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        terminate_process(process)


def _result_info(attempt_root: Path) -> tuple[int, Path | None, dict[str, Any] | None]:
    # A task attempt owns its directory exclusively, so search the complete
    # tree instead of coupling result discovery to one RoboDojo version's
    # native directory layout.
    candidates = list(attempt_root.rglob("_result.json"))
    ranked: list[tuple[int, int, Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            details = payload.get("details")
            episodes = len(details) if isinstance(details, dict) else 0
            ranked.append((episodes, path.stat().st_mtime_ns, path, payload))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    if not ranked:
        return 0, None, None
    episodes, _, path, payload = max(ranked, key=lambda item: (item[0], item[1]))
    return episodes, path, payload


def wait_for_result_info(attempt_root: Path, timeout: int) -> tuple[int, Path | None, dict[str, Any] | None]:
    """Wait for a completed process's result to become visible on shared storage."""

    deadline = time.monotonic() + max(0, timeout)
    while True:
        result = _result_info(attempt_root)
        if result[1] is not None:
            return result
        if time.monotonic() >= deadline:
            return result
        time.sleep(1)


def run_task(
    runtime: RuntimeConfig,
    spec: TaskSpec,
    *,
    gpu_id: str,
    local_slot: int,
    global_slot: int,
    proxy_port: int,
) -> dict[str, Any]:
    server_index = global_slot % runtime.num_servers
    server_port = runtime.base_port + server_index
    job_name = f"config_{spec.config_id:03d}_{sanitize(spec.config_name)}"
    job_root = runtime.output_root / "jobs" / job_name
    status_path = runtime.output_root / "task_status" / f"{job_name}.json"
    started_epoch = time.time()
    status: dict[str, Any] = {
        **asdict(spec),
        "state": "running",
        "worker_index": runtime.worker_index,
        "gpu_id": gpu_id,
        "local_slot": local_slot,
        "global_slot": global_slot,
        "server_index": server_index,
        "server_port": server_port,
        "proxy_port": proxy_port,
        "attempts": 0,
        "started_at_epoch": started_epoch,
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
    }
    atomic_write_json(status_path, status)

    last_error = ""
    for attempt in range(1, runtime.task_max_retries + 2):
        if STOP_EVENT.is_set():
            last_error = "launcher interrupted"
            break
        if attempt > 1:
            retry_backoff = env_float("TASK_RETRY_BACKOFF_SECONDS", 60.0)
            if retry_backoff < 0:
                raise ValueError("TASK_RETRY_BACKOFF_SECONDS must be non-negative")
            log(
                f"BACKOFF config={spec.config_id} {spec.config_name} for {retry_backoff:g}s before attempt={attempt}",
                worker_index=runtime.worker_index,
            )
            if STOP_EVENT.wait(retry_backoff):
                last_error = "launcher interrupted during retry backoff"
                break
        attempt_root = job_root / f"attempt_{attempt}"
        attempt_root.mkdir(parents=True, exist_ok=True)
        launcher_log = job_root / f"launcher_attempt_{attempt}.log"
        status.update(attempts=attempt, attempt_root=str(attempt_root), launcher_log=str(launcher_log))
        atomic_write_json(status_path, status)
        log(
            f"START config={spec.config_id} task={spec.task_id} {spec.dimension}/{spec.config_name} "
            f"episodes={spec.episodes} gpu={gpu_id} server={runtime.host}:{server_port} attempt={attempt}",
            worker_index=runtime.worker_index,
        )

        child_env = os.environ.copy()
        child_env.update(
            {
                "OMP_NUM_THREADS": _THREAD_CAP,
                "MKL_NUM_THREADS": _THREAD_CAP,
                "OPENBLAS_NUM_THREADS": _THREAD_CAP,
                "NUMEXPR_NUM_THREADS": _THREAD_CAP,
                "TORCH_NUM_THREADS": _THREAD_CAP,
                "COGWAM_POLICY_HOST": runtime.host,
                "COGWAM_POLICY_PORT": str(server_port),
                "COGWAM_CKPT_PATH": str(runtime.checkpoint),
                "ROBODOJO_CKPT_NAME": runtime.checkpoint_name,
                "ROBODOJO_TASK": spec.config_name,
                "ROBODOJO_TRIALS": str(spec.episodes),
                "ROBODOJO_EVAL_MODE": "fast",
                "ROBODOJO_ENVS_PER_CLIENT": str(runtime.envs_per_process),
                "ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE": str(runtime.action_chunk_size),
                "ROBODOJO_REPLAN_STEPS": str(runtime.execute_horizon),
                "ROBODOJO_RTC_ENABLED": str(runtime.rtc_enabled).lower(),
                "ROBODOJO_RTC_EXECUTION_HORIZON": str(runtime.rtc_execution_horizon),
                "ROBODOJO_RTC_INFERENCE_DELAY": str(runtime.rtc_inference_delay),
                "ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT": str(runtime.rtc_max_guidance_weight),
                "ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE": runtime.rtc_prefix_attention_schedule,
                "ROBODOJO_RTC_DEBUG_MAX_REPLANS": str(runtime.rtc_debug_max_replans),
                "ROBODOJO_POLICY_GPU": gpu_id,
                "ROBODOJO_ENV_GPU": gpu_id,
                "ROBODOJO_XPOLICY_PORT": str(proxy_port),
                "ROBODOJO_SEED": str(runtime.seed),
                "ROBODOJO_OUTPUT_RUN_ID": f"{runtime.shared_run_id}_{job_name}_a{attempt}",
                "ROBODOJO_NATIVE_RUN_ID": "run",
                "ROBODOJO_RUN_NAME": f"{runtime.run_name}_{spec.config_name}",
                "ROBODOJO_OUTPUT_ROOT": str(attempt_root),
                "ROBODOJO_SKIP_TABLE1_SUMMARY": "1",
            }
        )
        return_code = -1
        process: subprocess.Popen[Any] | None = None
        try:
            with launcher_log.open("a", encoding="utf-8") as stream:
                stream.write(f"\n===== attempt {attempt} =====\n")
                stream.flush()
                process = subprocess.Popen(
                    [
                        "bash",
                        str(FULL_LAUNCHER),
                        "RoboDojo",
                        spec.config_name,
                        runtime.checkpoint_name,
                        os.environ.get("ROBODOJO_ENV_CFG", "arx_x5"),
                        "joint",
                        str(runtime.seed),
                        gpu_id,
                        gpu_id,
                    ],
                    cwd=REPO_ROOT,
                    env=child_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with ACTIVE_LOCK:
                    ACTIVE_PROCESSES.add(process)
                return_code = process.wait()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                with ACTIVE_LOCK:
                    ACTIVE_PROCESSES.discard(process)
                terminate_process(process)

        actual, result_path, result_payload = wait_for_result_info(
            attempt_root,
            env_int("RESULT_DISCOVERY_TIMEOUT", 60),
        )
        if return_code == 0 and actual >= spec.episodes and result_path is not None:
            finished_epoch = time.time()
            status.update(
                state="success",
                return_code=return_code,
                actual_episodes=actual,
                result_path=str(result_path),
                result_success_rate=result_payload.get("success_rate") if result_payload else None,
                result_score=result_payload.get("score") if result_payload else None,
                # RoboDojo calls this eval_time, but it is an episode count.
                result_eval_time_episode_count=(result_payload or {}).get("eval_time"),
                finished_at_epoch=finished_epoch,
                finished_at=datetime.fromtimestamp(finished_epoch, timezone.utc).isoformat(),
                wall_time_seconds=finished_epoch - started_epoch,
            )
            atomic_write_json(status_path, status)
            log(
                f"DONE config={spec.config_id} {spec.config_name} episodes={actual}/{spec.episodes} "
                f"wall={format_duration(status['wall_time_seconds'])}",
                worker_index=runtime.worker_index,
            )
            return status

        if not last_error:
            last_error = f"exit={return_code}, episodes={actual}/{spec.episodes}"
        log(
            f"RETRY config={spec.config_id} {spec.config_name} attempt={attempt} failed: {last_error}",
            worker_index=runtime.worker_index,
        )

    finished_epoch = time.time()
    status.update(
        state="failed",
        error=last_error,
        finished_at_epoch=finished_epoch,
        finished_at=datetime.fromtimestamp(finished_epoch, timezone.utc).isoformat(),
        wall_time_seconds=finished_epoch - started_epoch,
    )
    atomic_write_json(status_path, status)
    log(f"FAILED config={spec.config_id} {spec.config_name}: {last_error}", worker_index=runtime.worker_index)
    return status


def wait_for_workers(output_root: Path, worker_count: int, timeout: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    status_dir = output_root / "worker_status"
    while time.monotonic() < deadline:
        statuses: list[dict[str, Any]] = []
        for worker_index in range(worker_count):
            path = status_dir / f"worker_{worker_index}.json"
            try:
                statuses.append(json.loads(path.read_text(encoding="utf-8")))
            except (FileNotFoundError, json.JSONDecodeError):
                break
        if len(statuses) == worker_count:
            return statuses
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for worker statuses in {status_dir}")


def _checkpoint_name(checkpoint: Path) -> str:
    explicit = os.getenv("ROBODOJO_CKPT_NAME")
    if explicit:
        return explicit
    if checkpoint.is_dir():
        # A released artifact directory is already named after the run.
        return checkpoint.name
    run_name = checkpoint.parents[1].name if checkpoint.parent.name == "checkpoints" else checkpoint.parent.name
    return f"{run_name}_{checkpoint.stem}"


def _model_root(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        return checkpoint
    return checkpoint.parents[1] if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _shared_run_id(worker_count: int) -> str:
    explicit = os.getenv("CLIENT_SHARED_RUN_ID")
    if explicit:
        run_id = sanitize(explicit)
        if not run_id:
            raise ValueError("CLIENT_SHARED_RUN_ID contains no usable characters")
        return run_id
    if worker_count > 1:
        raise ValueError("multi-worker rollout requires one identical CLIENT_SHARED_RUN_ID exported to every worker")
    return time.strftime("%Y%m%d_%H%M%S") + f"-{os.getpid()}"


def main() -> int:
    worker_index, worker_count = parse_worker_info()
    checkpoint = Path(os.environ["COGWAM_CKPT_PATH"]).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    host = os.getenv("COGWAM_POLICY_HOST")
    if not host:
        raise ValueError("set COGWAM_POLICY_HOST to the policy server address")
    robodojo_root = resolve_robodojo_root()
    for required in (FULL_LAUNCHER, robodojo_root / "task/RoboDojo/config/_task.yml"):
        if not required.is_file():
            raise FileNotFoundError(f"required rollout file is missing: {required}")

    shared_run_id = _shared_run_id(worker_count)
    run_name = os.getenv("ROBODOJO_FULL_RUN_NAME", "full_official_fast_h25")
    checkpoint_name = _checkpoint_name(checkpoint)
    model_root = _model_root(checkpoint)
    requested_output = os.getenv("OUTPUT_ROOT") or os.getenv("ROBODOJO_FULL_OUTPUT_ROOT")
    output_root = (
        Path(requested_output).expanduser().resolve()
        if requested_output
        else (model_root / "robodojo_eval_results" / f"{run_name}_{shared_run_id}").resolve()
    )

    requested_tasks = os.getenv("TASK_NAMES", "").replace(",", " ").split() or None
    num_episodes = os.getenv("NUM_EPISODES", os.getenv("EVAL_NUM", "native")).strip().lower()
    specs = build_task_specs(
        robodojo_root,
        task_names=requested_tasks,
        num_episodes=num_episodes,
        max_configs=env_int("MAX_TASKS", -1),
    )
    partitions = partition_specs(specs, worker_count)
    local_specs = partitions[worker_index]

    base_port = env_int("BASE_PORT", 7777)
    num_servers = env_int("NUM_SERVERS", 8)
    if num_servers < 1 or not 1 <= base_port <= 65535 or base_port + num_servers - 1 > 65535:
        raise ValueError(f"invalid policy server range: base={base_port}, count={num_servers}")
    gpu_ids = [
        item.strip()
        for item in os.getenv("CLIENT_CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0")).split(",")
        if item.strip()
    ]
    if not gpu_ids:
        raise ValueError("CLIENT_CUDA_VISIBLE_DEVICES must expose at least one GPU")
    num_clients = env_int("NUM_CLIENTS", len(gpu_ids))
    if num_clients != len(gpu_ids):
        raise ValueError(f"NUM_CLIENTS={num_clients}, but CLIENT_CUDA_VISIBLE_DEVICES has {len(gpu_ids)} entries")
    slots_per_gpu = parse_slots_per_gpu(gpu_ids)
    if slots_per_gpu != [1] * len(gpu_ids):
        raise ValueError(f"this rollout version requires exactly one Isaac process per GPU; got slots={slots_per_gpu}")
    phases = [("full", local_specs, slots_per_gpu)]

    action_chunk_size = env_int("ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE", 25)
    execute_horizon = env_int("N_ACTION_STEPS", env_int("ROBODOJO_REPLAN_STEPS", 10))
    if action_chunk_size < 1 or not 1 <= execute_horizon <= action_chunk_size:
        raise ValueError(
            f"N_ACTION_STEPS/ROBODOJO_REPLAN_STEPS must be in [1,{action_chunk_size}], got {execute_horizon}"
        )
    rtc_enabled = env_bool("ROBODOJO_RTC_ENABLED", False)
    rtc_overlap = action_chunk_size - execute_horizon
    rtc_execution_horizon = env_int("ROBODOJO_RTC_EXECUTION_HORIZON", max(rtc_overlap, 1))
    rtc_inference_delay = env_int("ROBODOJO_RTC_INFERENCE_DELAY", 1)
    rtc_max_guidance_weight = env_float("ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT", 20.0)
    rtc_prefix_attention_schedule = os.getenv("ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE", "linear").strip().lower()
    rtc_debug_max_replans = env_int("ROBODOJO_RTC_DEBUG_MAX_REPLANS", 8)
    if rtc_prefix_attention_schedule not in {"exp", "linear", "ones", "zeros"}:
        raise ValueError(
            "ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE must be exp, linear, ones, or zeros; "
            f"got {rtc_prefix_attention_schedule!r}"
        )
    if rtc_max_guidance_weight <= 0:
        raise ValueError("ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT must be positive")
    if rtc_debug_max_replans < 0:
        raise ValueError("ROBODOJO_RTC_DEBUG_MAX_REPLANS must be non-negative")
    if rtc_enabled and (
        rtc_overlap < 1
        or not 1 <= rtc_execution_horizon <= rtc_overlap
        or not 0 <= rtc_inference_delay <= rtc_execution_horizon
    ):
        raise ValueError(
            f"RTC with H{action_chunk_size}/replan{execute_horizon} requires "
            f"execution_horizon in [1,{rtc_overlap}] and inference_delay in "
            f"[0,execution_horizon]; got horizon={rtc_execution_horizon}, delay={rtc_inference_delay}"
        )
    envs_per_process = env_int("ROBODOJO_ENVS_PER_CLIENT", 1)
    if envs_per_process < 1:
        raise ValueError("ROBODOJO_ENVS_PER_CLIENT must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    canonical_tasks = list(dict.fromkeys(spec.canonical_task for spec in specs))
    manifest: dict[str, Any] = {
        "run_id": shared_run_id,
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint_name,
        "worker_count": worker_count,
        "host": host,
        "base_port": base_port,
        "num_servers": num_servers,
        "phases": [
            {
                "name": name,
                "configs": [spec.config_name for spec in phase_specs],
                "tasks_per_gpu": phase_slots,
                "concurrent_slots": sum(phase_slots),
            }
            for name, phase_specs, phase_slots in phases
        ],
        "gpus_per_worker": len(gpu_ids),
        "envs_per_process": envs_per_process,
        "action_chunk_size": action_chunk_size,
        "execute_horizon": execute_horizon,
        "rtc": {
            "enabled": rtc_enabled,
            "overlap": rtc_overlap,
            "execution_horizon": rtc_execution_horizon,
            "inference_delay": rtc_inference_delay,
            "max_guidance_weight": rtc_max_guidance_weight,
            "prefix_attention_schedule": rtc_prefix_attention_schedule,
            "debug_max_replans": rtc_debug_max_replans,
        },
        "save_video": False,
        "num_episodes": num_episodes,
        "expected_tasks": len(canonical_tasks),
        "expected_configs": len(specs),
        "expected_episodes": sum(spec.episodes for spec in specs),
        "started_at_epoch": time.time(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "specs": [asdict(spec) for spec in specs],
        "partitions": [[spec.config_name for spec in partition] for partition in partitions],
    }
    if worker_index == 0:
        if manifest_path.exists() and not env_bool("ROBODOJO_RESUME", False):
            raise FileExistsError(
                f"run manifest already exists: {manifest_path}; "
                "choose a new CLIENT_SHARED_RUN_ID or set ROBODOJO_RESUME=1"
            )
        if manifest_path.exists():
            manifest = wait_for_manifest(manifest_path, shared_run_id, worker_count, 1)
        else:
            atomic_write_json(manifest_path, manifest)
    else:
        manifest = wait_for_manifest(
            manifest_path,
            shared_run_id,
            worker_count,
            env_int("CLIENT_MANIFEST_MAX_WAIT", 1800),
        )

    runtime = RuntimeConfig(
        checkpoint=checkpoint,
        checkpoint_name=checkpoint_name,
        host=host,
        base_port=base_port,
        num_servers=num_servers,
        worker_index=worker_index,
        worker_count=worker_count,
        shared_run_id=shared_run_id,
        output_root=output_root,
        seed=env_int("ROBODOJO_SEED", env_int("SEED", 0)),
        action_chunk_size=action_chunk_size,
        execute_horizon=execute_horizon,
        rtc_enabled=rtc_enabled,
        rtc_execution_horizon=rtc_execution_horizon,
        rtc_inference_delay=rtc_inference_delay,
        rtc_max_guidance_weight=rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=rtc_prefix_attention_schedule,
        rtc_debug_max_replans=rtc_debug_max_replans,
        envs_per_process=envs_per_process,
        task_max_retries=env_int("TASK_MAX_RETRIES", 1),
        run_name=run_name,
    )

    if worker_index == 0:
        log(
            f"protocol={len(canonical_tasks)} tasks / {len(specs)} configs / "
            f"{sum(spec.episodes for spec in specs)} episodes"
        )
        if specs == build_task_specs(robodojo_root):
            log("Generalization=12 x (25 base + 25 random); other dimensions=30 x 50")
        log(
            f"topology={worker_count} workers x {len(gpu_ids)} GPU; "
            f"policy_servers={num_servers}; H{action_chunk_size}/replan{execute_horizon}; "
            f"envs/process={envs_per_process}; videos=disabled"
        )
        log(
            "rtc="
            + (
                f"enabled overlap={rtc_overlap} horizon={rtc_execution_horizon} "
                f"delay={rtc_inference_delay} schedule={rtc_prefix_attention_schedule} "
                f"weight={rtc_max_guidance_weight:g} debug={rtc_debug_max_replans}"
                if rtc_enabled
                else f"disabled (the {rtc_overlap}-step chunk tail is discarded)"
            )
        )
        for phase_index, (phase_name, phase_specs, phase_slots) in enumerate(phases, start=1):
            log(
                f"phase={phase_index}/{len(phases)} name={phase_name} "
                f"configs={len(phase_specs)} tasks/GPU={phase_slots} concurrent_slots={sum(phase_slots)}"
            )
        log(f"output={output_root}")
        log("[ETA] " + format_progress(progress_snapshot(output_root, manifest)))

    wait_for_servers(host, base_port, num_servers, env_int("SERVER_MAX_WAIT", 1800), worker_index)
    statuses: list[dict[str, Any]] = []
    statuses_lock = threading.Lock()

    reporter = ProgressReporter(output_root, manifest, env_int("PROGRESS_INTERVAL", 60)) if worker_index == 0 else None
    if reporter is not None:
        reporter.start()

    def run_phase(
        phase_index: int,
        phase_name: str,
        phase_specs: list[TaskSpec],
        phase_slots: list[int],
    ) -> None:
        phase_slot_count = sum(phase_slots)
        proxy_ports = ensure_proxy_ports_available(env_int("PROXY_BASE_PORT", 20000), phase_slot_count)
        task_queue: queue.Queue[TaskSpec] = queue.Queue()
        for spec in phase_specs:
            task_queue.put(spec)

        def slot_worker(gpu_id: str, local_slot: int, slot_index: int) -> None:
            global_slot = worker_index * phase_slot_count + slot_index
            while not STOP_EVENT.is_set():
                try:
                    spec = task_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    status = run_task(
                        runtime,
                        spec,
                        gpu_id=gpu_id,
                        local_slot=local_slot,
                        global_slot=global_slot,
                        proxy_port=proxy_ports[slot_index],
                    )
                    with statuses_lock:
                        statuses.append(status)
                finally:
                    task_queue.task_done()

        log(
            f"PHASE START {phase_index}/{len(phases)} name={phase_name} "
            f"configs={len(phase_specs)} slots={phase_slot_count}",
            worker_index=worker_index,
        )
        threads: list[threading.Thread] = []
        assignments = iter_slot_assignments(gpu_ids, phase_slots)
        start_stagger = env_float("SLOT_START_STAGGER_SECONDS", 15.0)
        if start_stagger < 0:
            raise ValueError("SLOT_START_STAGGER_SECONDS must be non-negative")
        for slot_index, (gpu_id, local_slot) in enumerate(assignments):
            thread = threading.Thread(
                target=slot_worker,
                args=(gpu_id, local_slot, slot_index),
                name=f"robodojo-{phase_name}-gpu{gpu_id}-slot{local_slot}",
            )
            thread.start()
            threads.append(thread)
            if start_stagger and slot_index + 1 < len(assignments):
                if STOP_EVENT.wait(start_stagger):
                    break
        for thread in threads:
            thread.join()
        log(f"PHASE DONE {phase_index}/{len(phases)} name={phase_name}", worker_index=worker_index)

    try:
        for phase_index, (phase_name, phase_specs, phase_slots) in enumerate(phases, start=1):
            if STOP_EVENT.is_set():
                break
            run_phase(phase_index, phase_name, phase_specs, phase_slots)
    finally:
        terminate_all_processes()

    worker_success = len(statuses) == len(local_specs) and all(status.get("state") == "success" for status in statuses)
    worker_status = {
        "worker_index": worker_index,
        "success": worker_success,
        "expected_configs": len(local_specs),
        "completed_configs": sum(status.get("state") == "success" for status in statuses),
        "failed_configs": [status["config_name"] for status in statuses if status.get("state") != "success"],
        "configs": [spec.config_name for spec in local_specs],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_root / "worker_status" / f"worker_{worker_index}.json", worker_status)
    log(f"worker complete success={worker_success}", worker_index=worker_index)

    if worker_index != 0:
        return 0 if worker_success else 1

    try:
        worker_statuses = wait_for_workers(output_root, worker_count, env_int("CLIENT_WORKER_SYNC_MAX_WAIT", 14400))
    finally:
        if reporter is not None:
            reporter.stop()
            reporter.join(timeout=5)

    final_progress = progress_snapshot(output_root, manifest)
    log("[ETA] " + format_progress(final_progress))
    summary_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "cogwam.eval.summarize",
            "--eval-root",
            str(output_root),
            "--checkpoint",
            str(checkpoint),
            "--ckpt-name",
            checkpoint_name,
            "--seed",
            str(runtime.seed),
            "--output-dir",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    all_success = (
        all(status.get("success") for status in worker_statuses)
        and int(final_progress["completed_configs"]) == len(specs)
        and summary_status == 0
    )
    readme = "\n".join(
        [
            "RoboDojo fast rollout",
            f"checkpoint={checkpoint}",
            f"run_id={shared_run_id}",
            f"tasks={final_progress['completed_tasks']}/{final_progress['expected_tasks']}",
            f"configs={final_progress['completed_configs']}/{final_progress['expected_configs']}",
            f"episodes={final_progress['completed_episodes']}/{final_progress['expected_episodes']}",
            f"wall_time_seconds={final_progress['elapsed_seconds']:.3f}",
            "eta_uses_wall_time_and_completed_episode_count=true",
            "robodojo_eval_time_is_episode_count=true",
            f"table_markdown={output_root / f'table1_seed{runtime.seed}.md'}",
        ]
    )
    (output_root / "README_RESULTS.txt").write_text(readme + "\n", encoding="utf-8")
    if all_success:
        log(f"SUCCESS: all {len(canonical_tasks)} tasks and {sum(spec.episodes for spec in specs)} episodes completed")
        return 0
    log("ERROR: rollout is incomplete; inspect task_status/, worker_status/, and jobs/*/launcher_attempt_*.log")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_all_processes()
        raise SystemExit(130) from None
    except Exception as exc:
        terminate_all_processes()
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
