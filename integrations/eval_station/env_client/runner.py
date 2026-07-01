"""In-process env trial execution for dispatch orchestration."""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

from eval_station.env_client.api import (
    EnvClientBaselineConfig,
    dispatch_trial_to_deploy_cfg,
)
from eval_station.schemas import DispatchPayload
from eval_station.trial.config import build_trial_run_config

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
    default_eval_env: str,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "trial_id": deploy_cfg.get("trial_id"),
        "steps": steps,
        "eval_env": deploy_cfg.get("eval_env", default_eval_env),
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
    if (
        payload.get("protocol", "robodojo_ws") == "robodojo_ws"
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
    from eval_station.dispatch.planner import build_trial_runs

    trial_runs = build_trial_runs(dispatch, evaluation_id=evaluation_id)
    if not trial_runs:
        deploy_cfg["policy_server_url"] = dispatch.policy_server_url
        return deploy_cfg

    config = build_trial_run_config(
        dispatch,
        trial_runs[0],
        evaluation_id=evaluation_id,
        eval_env=deploy_cfg.get("eval_env"),
    )
    overlay = {
        "policy_server_url": config.policy_server_url,
        "policy_name": config.policy_name,
        "task_name": config.task_name,
        "env_cfg_type": config.env_cfg_type,
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
    eval_env = _baseline_eval_env(baseline)

    if eval_env == "real":
        if not baseline.root_dir:
            message = "root_dir is required for real eval_env reset"
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
    default_eval_env: str,
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
        default_eval_env=default_eval_env,
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
        default_eval_env="debug",
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
                "message": "root_dir is required for real eval_env",
            },
        }

    _ensure_pipeline_paths(str(root_dir))
    from task_env.real_env_client import RealEnv

    return _run_env_trial(
        deploy_cfg,
        stop_check=stop_check,
        default_eval_env="real",
        env_factory=RealEnv,
        max_episodes=1,
    )


def _baseline_eval_env(baseline: EnvClientBaselineConfig | Mapping[str, Any]) -> str:
    if isinstance(baseline, Mapping):
        return str(baseline.get("eval_env", "debug"))
    return baseline.eval_env


def _prepare_real_deploy_cfg(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    if deploy_cfg.get("eval_env") == "real":
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
    if deploy_cfg.get("eval_env") != "real":
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
        "real eval_env reset is missing required deploy fields: "
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


def make_dispatch_trial_runner(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
    *,
    run_trial: EnvTrialRunner | None = None,
    eval_episode_num: int | None = 1,
    stop_check_factory: StopCheckFactory | None = None,
) -> TrialRunnerFn:
    eval_env = _baseline_eval_env(baseline)
    if run_trial is None:
        run_trial = run_real_trial if eval_env == "real" else run_debug_trial
    episode_override = None if eval_env == "real" else eval_episode_num

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
        return {
            "trial_id": result.get("trial_id"),
            "steps": result.get("steps"),
            "eval_env": result.get("eval_env"),
            "policy_name": result.get("policy_name"),
            "hdf5_path": result.get("hdf5_path"),
            "actions": [],
        }

    return runner
