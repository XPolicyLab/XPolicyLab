"""In-process env trial execution for dispatch orchestration."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

from robodojo.env_client.api import (
    EnvClientBaselineConfig,
    dispatch_trial_to_deploy_cfg,
)
from robodojo.schemas import DispatchPayload

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
        env.finish_episode()
        episodes += 1
        total_steps += env.episode_step
    env.reset()
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


def baseline_to_reset_deploy_cfg(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(baseline, EnvClientBaselineConfig):
        payload = baseline.model_dump()
    else:
        payload = dict(baseline)

    payload.setdefault("evaluation_id", "idle-reset")
    payload.setdefault("trial_id", f"{payload.get('task_name', 'trial')}-reset")
    payload.setdefault("action_case_id", f"{payload.get('task_name', 'trial')}_case_1")
    if payload.get("protocol", "robodojo_ws") == "robodojo_ws" and not payload.get(
        "policy_server_url"
    ):
        host = payload.get("host", "localhost")
        port = int(payload["port"])
        payload["policy_server_url"] = f"ws://{host}:{port}"
    return payload


def reset_idle_env(baseline: EnvClientBaselineConfig | Mapping[str, Any]) -> None:
    """Reset policy + robot state while no trial is executing."""

    if isinstance(baseline, Mapping) and not isinstance(baseline, EnvClientBaselineConfig):
        baseline = EnvClientBaselineConfig.model_validate(baseline)

    deploy_cfg = baseline_to_reset_deploy_cfg(baseline)
    eval_env = _baseline_eval_env(baseline)

    if eval_env == "real":
        if not baseline.root_dir:
            raise TrialRunnerError(
                "root_dir is required for real eval_env reset",
                error={"code": "missing_root_dir", "message": "root_dir is required for real eval_env reset"},
            )
        _ensure_pipeline_paths(str(baseline.root_dir))
        from task_env.real_env_client import RealEnv

        def env_factory(cfg: dict[str, Any]) -> Any:
            return RealEnv(cfg, setup_cameras=False)
    else:
        from debug_env_client import TestEnv

        env_factory = TestEnv

    env = env_factory(deploy_cfg)
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
    try:
        total_steps = _run_trial_loop(
            env,
            stop_check=stop_check,
            eval_batch=deploy_cfg["eval_batch"],
            max_episodes=max_episodes,
        )
    finally:
        _cleanup_env(env)
    return _completed_trial_result(
        deploy_cfg,
        steps=total_steps,
        default_eval_env=default_eval_env,
    )


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
        max_episodes=None,
    )


def _baseline_eval_env(baseline: EnvClientBaselineConfig | Mapping[str, Any]) -> str:
    if isinstance(baseline, Mapping):
        return str(baseline.get("eval_env", "debug"))
    return baseline.eval_env


# deploy_cfg keys that may be auto-filled from policy server HELLO meta when
# neither the dispatch payload nor the daemon baseline provided them.
_META_FILL_KEYS = (
    "policy_name",
    "action_type",
    "env_cfg_type",
    "task_name",
    "dataset_name",
)

_META_PLACEHOLDER_VALUES = {None, "", "auto"}


def _missing_meta_keys(deploy_cfg: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key in _META_FILL_KEYS
        if deploy_cfg.get(key) in _META_PLACEHOLDER_VALUES
    ]


def _policy_server_ws_url(deploy_cfg: Mapping[str, Any]) -> str | None:
    url = deploy_cfg.get("policy_server_url")
    if url:
        return str(url)
    host, port = deploy_cfg.get("host"), deploy_cfg.get("port")
    if host and port:
        return f"ws://{host}:{int(port)}"
    return None


def _fill_deploy_cfg_from_meta(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    if deploy_cfg.get("eval_env") != "real":
        return deploy_cfg
    missing = _missing_meta_keys(deploy_cfg)
    if not missing:
        return deploy_cfg

    url = _policy_server_ws_url(deploy_cfg)
    if url is None:
        return deploy_cfg

    from robodojo.env_client.ws_adapter import fetch_policy_meta

    try:
        meta = fetch_policy_meta(url, max_connect_attempts=3)
    except Exception as exc:
        print(
            f"[env_client] policy meta fetch failed ({url}): {exc}",
            file=sys.stderr,
        )
        return deploy_cfg

    filled = {key: meta[key] for key in missing if meta.get(key) is not None}
    if filled:
        print(f"[env_client] deploy_cfg auto-filled from policy meta: {filled}")
        deploy_cfg.update(filled)
    return deploy_cfg


def _validate_real_deploy_cfg(deploy_cfg: Mapping[str, Any]) -> None:
    if deploy_cfg.get("eval_env") != "real":
        return
    if deploy_cfg.get("action_type") not in ("joint", "ee"):
        raise TrialRunnerError(
            "action_type is required for real eval_env but was not provided by "
            "the dispatch payload, policy server meta, or daemon startup args "
            "(upgrade the policy server to advertise meta, or set ACTION_TYPE).",
            error={
                "code": "missing_action_type",
                "message": "action_type unresolved for real eval_env",
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
        deploy_cfg = _fill_deploy_cfg_from_meta(deploy_cfg)
        _validate_real_deploy_cfg(deploy_cfg)
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
            "actions": [],
        }

    return runner
