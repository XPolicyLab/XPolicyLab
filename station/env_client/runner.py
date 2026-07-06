"""In-process env trial execution for dispatch orchestration."""

from __future__ import annotations

import glob
import inspect
import json
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from station.env_client.api import (
    EnvClientBaselineConfig,
    dispatch_trial_to_deploy_cfg,
)
from station.eval_env_type import (
    DEFAULT_EVAL_ENV_TYPE,
    is_real_world,
    normalize_eval_env_type,
)
from station.schemas import DispatchPayload
from station.trial.config import build_trial_run_config

EnvTrialRunner = Callable[..., dict[str, Any]]
DebugTrialRunner = EnvTrialRunner
TrialRunnerFn = Callable[[DispatchPayload, dict[str, Any], str], dict[str, Any]]
StopCheckFactory = Callable[[dict[str, Any]], Callable[[], bool]]


def _never_stop() -> bool:
    return False


class TrialRunnerError(RuntimeError):
    def __init__(self, message: str, *, error: dict[str, Any] | None = None):
        super().__init__(message)
        self.error = error


def _ensure_pipeline_paths(root_dir: str) -> None:
    for path in (f"{root_dir}/src", f"{root_dir}/XPolicyLab", root_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _cleanup_env(env: Any) -> None:
    close = getattr(env.model_client, "close", None)
    if callable(close):
        close()
    cleanup = getattr(env, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _run_trial_loop(
    env: Any,
    *,
    stop_check: Callable[[], bool],
    eval_batch: bool,
    max_episodes: int | None = None,
) -> int:
    episodes = 0
    total_steps = 0
    while not stop_check():
        if max_episodes is not None and episodes >= max_episodes:
            break
        env.reset()
        if eval_batch:
            env.eval_one_episode_batch()
        else:
            env.eval_one_episode()
        total_steps += env.episode_step
        # Reset the robot/policy before finish webhook and trial video export.
        env.reset()
        env.finish_episode()
        episodes += 1
    return total_steps


def _completed_trial_result(
    deploy_cfg: Mapping[str, Any],
    *,
    steps: int,
    default_eval_env_type: str,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "trial_id": deploy_cfg.get("trial_id"),
        "steps": steps,
        "eval_env_type": deploy_cfg.get("eval_env_type", default_eval_env_type),
        "policy_name": deploy_cfg.get("policy_name"),
    }


def _hdf5_path_from_env(env: Any) -> str | None:
    """Best-effort absolute path of the HDF5 just recorded by the robot collector.

    The collector writes {save_dir}/{task_name}/{type}/{episode_index}.hdf5 and
    increments episode_index after each write, so the last recording is at
    episode_index - 1. Returns None for non-recording envs (e.g. debug/sim).
    """
    collector = getattr(getattr(env, "robot", None), "collector", None)
    if collector is None:
        return None
    episode_index = getattr(collector, "episode_index", 0)
    cfg = getattr(collector, "collect_cfg", None)
    if not isinstance(cfg, Mapping) or episode_index <= 0:
        return None
    try:
        path = os.path.abspath(
            os.path.join(
                cfg["save_dir"],
                cfg["task_name"],
                cfg["type"],
                f"{episode_index - 1}.hdf5",
            )
        )
    except (KeyError, TypeError):
        return None
    return path if os.path.isfile(path) else None


def baseline_to_reset_deploy_cfg(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        baseline.model_dump()
        if isinstance(baseline, EnvClientBaselineConfig)
        else dict(baseline)
    )

    task_name = payload.get("task_name") or "trial"
    payload.setdefault("evaluation_id", "idle-reset")
    payload.setdefault("trial_id", f"{task_name}-reset")
    payload.setdefault("action_case_id", f"{task_name}_case_1")
    # EnvClientBaselineConfig has no repeat_index field, but TestEnv/RealEnv
    # read deploy_cfg["repeat_index"] unconditionally.
    payload.setdefault("repeat_index", None)
    if (
        payload.get("protocol", "ws") == "ws"
        and not payload.get("policy_server_url")
    ):
        host = payload.get("host") or "localhost"
        port = payload.get("port")
        if port is not None:
            payload["policy_server_url"] = f"ws://{host}:{int(port)}"
    return payload


def _sync_host_port_from_policy_url(
    deploy_cfg: dict[str, Any],
    policy_server_url: str,
) -> None:
    parsed = urlparse(policy_server_url)
    if parsed.hostname:
        deploy_cfg["host"] = parsed.hostname
    if parsed.port:
        deploy_cfg["port"] = parsed.port


def _overlay_dispatch_for_reset(
    deploy_cfg: dict[str, Any],
    dispatch: DispatchPayload,
    *,
    evaluation_id: str,
) -> dict[str, Any]:
    from station.dispatch.planner import build_trial_runs

    trial_runs = build_trial_runs(dispatch, evaluation_id=evaluation_id)
    if not trial_runs:
        deploy_cfg["policy_server_url"] = dispatch.policy_server_url
        return deploy_cfg

    config = build_trial_run_config(
        dispatch,
        trial_runs[0],
        evaluation_id=evaluation_id,
        eval_env_type=deploy_cfg.get("eval_env_type"),
    )
    overlay = {
        "policy_server_url": config.policy_server_url,
        "policy_name": config.policy_name,
        "task_name": config.task_name,
        "env_cfg_type": config.env_cfg_type,
        "eval_env_type": config.eval_env_type,
        "trial_id": f"{config.trial_id}-reset",
        "action_case_id": config.action_case_id,
        "evaluation_id": evaluation_id,
    }
    action_type = config.case_meta.get("action_type")
    if action_type in ("joint", "ee"):
        overlay["action_type"] = action_type
    deploy_cfg.update(overlay)
    _sync_host_port_from_policy_url(deploy_cfg, config.policy_server_url)
    return deploy_cfg


def reset_idle_env(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
    *,
    dispatch: DispatchPayload | None = None,
    evaluation_id: str | None = None,
) -> None:
    """Reset policy + robot state while no trial is executing."""

    if isinstance(baseline, Mapping) and not isinstance(baseline, EnvClientBaselineConfig):
        baseline = EnvClientBaselineConfig.model_validate(baseline)

    deploy_cfg = baseline_to_reset_deploy_cfg(baseline)
    if dispatch is not None and evaluation_id:
        deploy_cfg = _overlay_dispatch_for_reset(
            deploy_cfg,
            dispatch,
            evaluation_id=evaluation_id,
        )
    deploy_cfg = _prepare_real_deploy_cfg(deploy_cfg)
    eval_env_type = _baseline_eval_env_type(baseline)

    if is_real_world(eval_env_type):
        if not baseline.root_dir:
            message = "root_dir is required for real_world eval_env_type reset"
            raise TrialRunnerError(
                message,
                error={"code": "missing_root_dir", "message": message},
            )
        _ensure_pipeline_paths(str(baseline.root_dir))
        from task_env.real_env_client import RealEnv

        env = RealEnv(deploy_cfg, setup_cameras=False)
    else:
        from debug_env_client import TestEnv

        env = TestEnv(deploy_cfg)
    try:
        env.reset()
    finally:
        _cleanup_env(env)


def _wire_env_stop_check(env: Any, stop_check: Callable[[], bool]) -> None:
    set_stop_check = getattr(env, "set_stop_check", None)
    if callable(set_stop_check):
        set_stop_check(stop_check)


def _run_env_trial(
    deploy_cfg: dict[str, Any],
    *,
    stop_check: Callable[[], bool],
    default_eval_env_type: str,
    env_factory: Callable[[dict[str, Any]], Any],
    max_episodes: int | None,
) -> dict[str, Any]:
    env = env_factory(deploy_cfg)
    _wire_env_stop_check(env, stop_check)
    hdf5_path: str | None = None
    try:
        total_steps = _run_trial_loop(
            env,
            stop_check=stop_check,
            eval_batch=deploy_cfg["eval_batch"],
            max_episodes=max_episodes,
        )
        hdf5_path = _hdf5_path_from_env(env)
    finally:
        _cleanup_env(env)
    result = _completed_trial_result(
        deploy_cfg,
        steps=total_steps,
        default_eval_env_type=default_eval_env_type,
    )
    if hdf5_path:
        result["hdf5_path"] = hdf5_path
    return result


def run_debug_trial(
    deploy_cfg: dict[str, Any],
    *,
    stop_check: Callable[[], bool] = _never_stop,
) -> dict[str, Any]:
    from debug_env_client import TestEnv

    return _run_env_trial(
        deploy_cfg,
        stop_check=stop_check,
        default_eval_env_type="debug",
        env_factory=TestEnv,
        max_episodes=deploy_cfg["eval_episode_num"],
    )


def run_real_trial(
    deploy_cfg: dict[str, Any],
    *,
    stop_check: Callable[[], bool] = _never_stop,
) -> dict[str, Any]:
    root_dir = deploy_cfg.get("root_dir")
    if not root_dir:
        return {
            "status": "failed",
            "error": {
                "code": "missing_root_dir",
                "message": "root_dir is required for real_world eval_env_type",
            },
        }

    _ensure_pipeline_paths(str(root_dir))
    from task_env.real_env_client import RealEnv

    # Real-robot rollouts have no batch inference path; never dispatch
    # eval_one_episode_batch against a physical robot.
    deploy_cfg = {**deploy_cfg, "eval_batch": False}
    return _run_env_trial(
        deploy_cfg,
        stop_check=stop_check,
        default_eval_env_type="real_world",
        env_factory=RealEnv,
        max_episodes=1,
    )


def _repo_root_dir() -> str:
    # runner.py -> env_client -> station -> XPolicyLab -> <repo root>
    return str(Path(__file__).resolve().parents[4])


def _sim_root_dir(deploy_cfg: Mapping[str, Any]) -> str:
    return (
        deploy_cfg.get("root_dir")
        or os.environ.get("XPOLICYLAB_SIM_ROOT")
        or _repo_root_dir()
    )


def _sim_conda_env(deploy_cfg: Mapping[str, Any]) -> str | None:
    """Conda env with Isaac Sim for the eval subprocess.

    The daemon usually runs in the eval-station env, which cannot import Isaac
    Sim. Mirror run_sim_env_client.sh by activating the simulator conda env
    when one is configured; otherwise inherit the daemon environment.
    """
    env = deploy_cfg.get("eval_env_conda_env") or os.environ.get(
        "XPOLICYLAB_SIM_CONDA_ENV"
    )
    return str(env) if env else None


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def _load_sim_result(root_dir: str, run_id: str) -> dict[str, Any] | None:
    """Locate the ``_result.json`` written by the sim eval client for run_id.

    eval_env.py writes ``{cwd}/eval_result/.../{run_id}/_result.json``; the
    subprocess runs with cwd=root_dir.
    """
    pattern = os.path.join(str(root_dir), "eval_result", "**", run_id, "_result.json")
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        return None
    try:
        with open(matches[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_sim_trial(
    deploy_cfg: dict[str, Any],
    *,
    stop_check: Callable[[], bool] = _never_stop,
) -> dict[str, Any]:
    """Run a simulator trial by shelling out to ``scripts/eval_policy.sh``.

    The Isaac Sim rollout runs as a subprocess (mirroring the local
    ``run_sim_env_client.sh`` path); there is no in-process sim env factory.
    ``device_id`` / ``seed`` fall back to the daemon environment when the
    dispatch payload does not carry them. ``eval_episode_num`` is ignored:
    the episode count is decided by eval_policy.sh / EVAL_NUM / the task
    config, exactly as in the shell path. Exit code 0 alone is not trusted;
    the trial only completes when ``_result.json`` reports ``eval_time >= 1``.
    """
    root_dir = _sim_root_dir(deploy_cfg)
    eval_policy = os.path.join(str(root_dir), "scripts", "eval_policy.sh")
    if not os.path.isfile(eval_policy):
        message = f"scripts/eval_policy.sh not found under root_dir={root_dir}"
        return {
            "status": "failed",
            "error": {"code": "missing_eval_policy_script", "message": message},
        }

    missing = [
        key
        for key in ("task_name", "env_cfg_type", "policy_name", "port")
        if not deploy_cfg.get(key)
    ]
    if missing:
        message = (
            "sim eval_env_type is missing required deploy fields: "
            f"{', '.join(missing)}"
        )
        return {
            "status": "failed",
            "error": {
                "code": "missing_sim_deploy_cfg",
                "message": message,
                "missing": missing,
            },
        }

    host = deploy_cfg.get("host") or "localhost"
    device_id = str(
        deploy_cfg.get("device_id")
        or os.environ.get("XPOLICYLAB_SIM_DEVICE_ID")
        or (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
    )
    seed = str(deploy_cfg.get("seed", os.environ.get("XPOLICYLAB_SIM_SEED", "0")))
    eval_batch = "true" if deploy_cfg.get("eval_batch") else "false"
    additional_info = deploy_cfg.get("additional_info")
    if not additional_info:
        action_type = deploy_cfg.get("action_type")
        additional_info = f"action_type={action_type}" if action_type else ""

    cmd = [
        "bash",
        eval_policy,
        "--root_dir", str(root_dir),
        "--bench_name", str(deploy_cfg.get("bench_name") or ""),
        "--task_name", str(deploy_cfg["task_name"]),
        "--env_cfg_type", str(deploy_cfg["env_cfg_type"]),
        "--policy_name", str(deploy_cfg["policy_name"]),
        "--device_id", device_id,
        "--host", str(host),
        "--port", str(deploy_cfg["port"]),
        "--eval_batch", eval_batch,
        "--additional_info", str(additional_info),
        "--seed", seed,
    ]
    conda_env = _sim_conda_env(deploy_cfg)
    if conda_env:
        cmd = [
            "bash",
            "-c",
            'source "$(conda info --base)/etc/profile.d/conda.sh" '
            f"&& conda activate {shlex.quote(conda_env)} "
            f"&& exec {shlex.join(cmd)}",
        ]

    # Pin the run id so the _result.json written by eval_env.py can be found
    # afterwards. eval_policy.sh keeps a pre-set ROBODOJO_RUN_ID as-is.
    run_id = os.environ.get("ROBODOJO_RUN_ID") or "{}_{}".format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        os.getpid(),
    )
    env = {**os.environ, "ROBODOJO_RUN_ID": run_id}

    # New session so stop/cleanup can kill the whole process group (bash ->
    # python -> Isaac Sim), not just the bash wrapper.
    proc = subprocess.Popen(cmd, cwd=str(root_dir), env=env, start_new_session=True)
    try:
        while True:
            try:
                returncode = proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                if stop_check():
                    _signal_process_group(proc, signal.SIGTERM)
                    try:
                        returncode = proc.wait(timeout=15.0)
                    except subprocess.TimeoutExpired:
                        _signal_process_group(proc, signal.SIGKILL)
                        returncode = proc.wait()
                    break
    finally:
        if proc.poll() is None:
            _signal_process_group(proc, signal.SIGKILL)
            proc.wait()

    if returncode != 0:
        message = f"scripts/eval_policy.sh exited with code {returncode}"
        return {
            "status": "failed",
            "error": {
                "code": "sim_eval_failed",
                "message": message,
                "returncode": returncode,
            },
        }

    # A clean exit code alone is not smoke success (see repo guidance);
    # require a _result.json with at least one evaluated episode.
    result_json = _load_sim_result(str(root_dir), run_id)
    eval_time = (result_json or {}).get("eval_time", 0)
    if not isinstance(eval_time, (int, float)) or eval_time < 1:
        message = (
            "sim eval exited 0 but no valid _result.json was found "
            f"(run_id={run_id}, eval_time={eval_time!r})"
        )
        return {
            "status": "failed",
            "error": {
                "code": "missing_sim_result",
                "message": message,
                "run_id": run_id,
            },
        }

    result = _completed_trial_result(
        deploy_cfg,
        steps=0,
        default_eval_env_type="sim",
    )
    result["sim_result"] = result_json
    return result


def _baseline_eval_env_type(baseline: EnvClientBaselineConfig | Mapping[str, Any]) -> str:
    if isinstance(baseline, Mapping):
        raw = baseline.get("eval_env_type", baseline.get("eval_env", DEFAULT_EVAL_ENV_TYPE))
        return normalize_eval_env_type(str(raw))
    return normalize_eval_env_type(baseline.eval_env_type)


def _deploy_eval_env_type(deploy_cfg: Mapping[str, Any]) -> str:
    raw = deploy_cfg.get("eval_env_type", deploy_cfg.get("eval_env", DEFAULT_EVAL_ENV_TYPE))
    return normalize_eval_env_type(str(raw))


def _prepare_real_deploy_cfg(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    deploy_cfg["eval_env_type"] = _deploy_eval_env_type(deploy_cfg)
    if is_real_world(deploy_cfg["eval_env_type"]):
        _apply_validated_action_type(deploy_cfg)
    _validate_real_deploy_cfg(deploy_cfg)
    return deploy_cfg


def _apply_validated_action_type(deploy_cfg: dict[str, Any]) -> None:
    from task_env.real_env_client import validate_deploy_cfg

    try:
        deploy_cfg["action_type"] = validate_deploy_cfg(deploy_cfg)
    except ValueError as exc:
        raise TrialRunnerError(
            str(exc),
            error={
                "code": "invalid_deploy_cfg",
                "message": str(exc),
                "field": "action_type",
            },
        ) from exc


def _validate_real_deploy_cfg(deploy_cfg: Mapping[str, Any]) -> None:
    if not is_real_world(_deploy_eval_env_type(deploy_cfg)):
        return

    missing: list[str] = []
    if deploy_cfg.get("action_type") not in ("joint", "ee"):
        missing.append("action_type")
    for key in ("env_cfg_type", "task_name", "policy_server_url"):
        if not deploy_cfg.get(key):
            missing.append(key)
    if not missing:
        return

    raise TrialRunnerError(
        "real_world eval_env_type reset is missing required deploy fields: "
        f"{', '.join(missing)}. Provide them via dispatch payload "
        "or env client startup args (ACTION_TYPE, ENV_CFG_TYPE, etc.).",
        error={
            "code": "missing_reset_deploy_cfg",
            "message": f"reset deploy_cfg missing: {', '.join(missing)}",
            "missing": missing,
        },
    )


def _call_env_trial_runner(
    env_trial_runner: EnvTrialRunner,
    deploy_cfg: dict[str, Any],
    stop_check: Callable[[], bool],
) -> dict[str, Any]:
    if "stop_check" in inspect.signature(env_trial_runner).parameters:
        return env_trial_runner(deploy_cfg, stop_check=stop_check)
    return env_trial_runner(deploy_cfg)


def _default_env_trial_runner(eval_env_type: str) -> EnvTrialRunner:
    if is_real_world(eval_env_type):
        return run_real_trial
    if eval_env_type == "debug":
        return run_debug_trial
    if eval_env_type == "sim":
        return run_sim_trial
    raise TrialRunnerError(
        f"unsupported eval_env_type for env trial runner: {eval_env_type}",
        error={
            "code": "unsupported_eval_env_type",
            "message": f"unsupported eval_env_type for env trial runner: {eval_env_type}",
            "eval_env_type": eval_env_type,
        },
    )


def make_dispatch_trial_runner(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
    *,
    run_trial: EnvTrialRunner | None = None,
    eval_episode_num: int | None = 1,
    stop_check_factory: StopCheckFactory | None = None,
) -> TrialRunnerFn:
    eval_env_type = _baseline_eval_env_type(baseline)
    if run_trial is None:
        run_trial = _default_env_trial_runner(eval_env_type)
    episode_override = None if is_real_world(eval_env_type) else eval_episode_num

    def runner(
        dispatch: DispatchPayload,
        trial_run: dict[str, Any],
        evaluation_id: str,
    ) -> dict[str, Any]:
        deploy_cfg = dispatch_trial_to_deploy_cfg(
            dispatch,
            trial_run,
            baseline,
            evaluation_id=evaluation_id,
            eval_episode_num=episode_override,
        )
        deploy_cfg = _prepare_real_deploy_cfg(deploy_cfg)
        stop_check = (
            stop_check_factory(deploy_cfg) if stop_check_factory else _never_stop
        )
        result = _call_env_trial_runner(run_trial, deploy_cfg, stop_check)
        if result.get("status") == "failed":
            raw_error = result.get("error")
            error = raw_error if isinstance(raw_error, dict) else {}
            raise TrialRunnerError(
                str(error.get("message", "env trial failed")),
                error=error or None,
            )
        summary = {
            "trial_id": result.get("trial_id"),
            "steps": result.get("steps"),
            "eval_env_type": result.get("eval_env_type"),
            "policy_name": result.get("policy_name"),
            "hdf5_path": result.get("hdf5_path"),
            "actions": [],
        }
        if result.get("sim_result") is not None:
            summary["sim_result"] = result["sim_result"]
        return summary

    return runner
