"""HTTP API schemas for the env client daemon."""

from __future__ import annotations

from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from station.eval_env_type import DEFAULT_EVAL_ENV_TYPE
from station.schemas import DispatchPayload
from station.trial.config import build_trial_run_config, normalize_policy_name

DEBUG_ENV_CLIENT_DEPLOY_CFG_KEYS = (
    "bench_name",
    "task_name",
    "env_cfg_type",
    "policy_name",
    "protocol",
    "host",
    "port",
    "policy_server_url",
    "evaluation_id",
    "action_case_id",
    "trial_id",
    "repeat_index",
    "eval_episode_num",
    "eval_batch",
    "action_type",
    "base_cfg",
)

_DEPLOY_CFG_CASE_META_KEYS = (
    "bench_name",
    "env_cfg_type",
    "task_name",
    "policy_name",
    "eval_batch",
    "eval_episode_num",
    "protocol",
    "action_type",
)


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvClientBaselineConfig(_StrictSchema):
    """Startup deploy_cfg from daemon launch (``debug_env_client`` CLI parity).

    Most fields are optional: per-trial values come from the dispatch payload;
    startup values only act as fallbacks.
    """

    bench_name: str | None = None
    task_name: str | None = None
    env_cfg_type: str | None = None
    policy_name: str | None = None
    protocol: Literal["legacy_tcp", "ws"] = "ws"
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    eval_batch: bool = False
    eval_episode_num: int = Field(default=10, ge=1)
    eval_env_type: str = DEFAULT_EVAL_ENV_TYPE
    root_dir: str | None = None
    action_type: Literal["joint", "ee"] | None = None
    base_cfg: str | None = None


class TrialRunRequest(_StrictSchema):
    evaluation_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    trial_index: int | None = Field(default=None, ge=1)
    action_case_id: str = Field(min_length=1)
    policy_server_url: str = Field(min_length=1)
    case_meta: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)


class TrialRunResponse(_StrictSchema):
    status: Literal["completed", "failed"]
    trial_id: str | None = None
    steps: int | None = Field(default=None, ge=0)
    eval_env_type: str | None = None
    policy_name: str | None = None
    error: dict[str, Any] | None = None


class HealthResponse(_StrictSchema):
    ok: bool = True
    policy_name: str | None = None
    eval_env_type: str
    deploy_yml: str | None = None
    last_trial_id: str | None = None


def _baseline_deploy_cfg(
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(baseline, EnvClientBaselineConfig):
        cfg = baseline.model_dump()
    else:
        cfg = dict(baseline)
    # Unset baseline fields must not shadow dispatch-provided values.
    for key in (
        "bench_name",
        "task_name",
        "env_cfg_type",
        "policy_name",
        "host",
        "port",
        "action_type",
        "base_cfg",
    ):
        if cfg.get(key) is None:
            cfg.pop(key, None)
    return cfg


def _normalize_case_meta_action_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped.lower()
    normalized = str(value).strip().lower()
    return normalized or None


def trial_request_to_deploy_cfg(
    request: TrialRunRequest,
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Merge daemon startup config with a per-trial HTTP request."""
    deploy_cfg = _baseline_deploy_cfg(baseline)
    deploy_cfg.update(
        {
            "evaluation_id": request.evaluation_id,
            "trial_id": request.trial_id,
            "action_case_id": request.action_case_id,
            "policy_server_url": request.policy_server_url,
        }
    )

    parsed = urlparse(request.policy_server_url)
    if parsed.hostname is not None:
        deploy_cfg["host"] = parsed.hostname
    if parsed.port is not None:
        deploy_cfg["port"] = parsed.port

    case_meta = request.case_meta
    for key in _DEPLOY_CFG_CASE_META_KEYS:
        value = case_meta.get(key)
        if key == "action_type":
            value = _normalize_case_meta_action_type(value)
        if value is not None and value != "":
            deploy_cfg[key] = value

    # Older control planes still send the pre-rename ``dataset_name`` key.
    if not deploy_cfg.get("bench_name") and case_meta.get("dataset_name"):
        deploy_cfg["bench_name"] = case_meta["dataset_name"]

    if deploy_cfg.get("policy_name"):
        deploy_cfg["policy_name"] = normalize_policy_name(str(deploy_cfg["policy_name"]))

    deploy_cfg["repeat_index"] = case_meta.get("repeat_index")
    deploy_cfg.update(
        {
            key: value
            for key, value in request.overrides.items()
            if value is not None and key in DEBUG_ENV_CLIENT_DEPLOY_CFG_KEYS
        }
    )
    return deploy_cfg


def debug_env_client_deploy_cfg_view(deploy_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deploy_cfg[key] for key in DEBUG_ENV_CLIENT_DEPLOY_CFG_KEYS if key in deploy_cfg}


def baseline_deploy_cfg_view(deploy_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Debug-env client fields plus baseline-only keys such as ``eval_env_type`` and ``root_dir``."""
    view = debug_env_client_deploy_cfg_view(deploy_cfg)
    for key in ("eval_env_type", "root_dir"):
        if key in deploy_cfg:
            view[key] = deploy_cfg[key]
    return view


def _request_from_trial_run_config(
    config: Any,
    trial_run: dict[str, Any],
    *,
    eval_episode_num: int | None,
) -> TrialRunRequest:
    case_meta = {
        **config.case_meta,
        "env_cfg_type": config.env_cfg_type,
        "task_name": config.task_name,
        "policy_name": config.policy_name,
    }
    if config.eval_batch:
        case_meta["eval_batch"] = True
    if config.repeat_index is not None:
        case_meta["repeat_index"] = config.repeat_index
    action_type = _normalize_case_meta_action_type(config.case_meta.get("action_type"))
    if action_type in ("joint", "ee"):
        case_meta["action_type"] = action_type

    trial_index = trial_run.get("trial_index")
    overrides: dict[str, Any] = {}
    if eval_episode_num is not None:
        overrides["eval_episode_num"] = eval_episode_num
    return TrialRunRequest(
        evaluation_id=config.evaluation_id,
        trial_id=config.trial_id,
        trial_index=int(trial_index) if trial_index is not None else None,
        action_case_id=config.action_case_id,
        policy_server_url=config.policy_server_url,
        case_meta=case_meta,
        overrides=overrides,
    )


def dispatch_trial_to_request(
    dispatch: DispatchPayload,
    trial_run: dict[str, Any],
    *,
    evaluation_id: str,
    eval_episode_num: int | None = 1,
    eval_env_type: str | None = None,
) -> TrialRunRequest:
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id=evaluation_id,
        eval_env_type=eval_env_type,
    )
    return _request_from_trial_run_config(
        config, trial_run, eval_episode_num=eval_episode_num
    )


def _baseline_eval_env_type(baseline_cfg: Mapping[str, Any]) -> str | None:
    """Resolve the startup eval env type, honoring the legacy ``eval_env`` key."""
    return baseline_cfg.get("eval_env_type") or baseline_cfg.get("eval_env")


def dispatch_trial_to_deploy_cfg(
    dispatch: DispatchPayload,
    trial_run: dict[str, Any],
    baseline: EnvClientBaselineConfig | Mapping[str, Any],
    *,
    evaluation_id: str,
    eval_episode_num: int | None = 1,
) -> dict[str, Any]:
    baseline_cfg = _baseline_deploy_cfg(baseline)
    # Build the trial config once; reuse it for both the request and the
    # resolved eval_env_type stamped onto deploy_cfg.
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id=evaluation_id,
        eval_env_type=_baseline_eval_env_type(baseline_cfg),
    )
    request = _request_from_trial_run_config(
        config, trial_run, eval_episode_num=eval_episode_num
    )
    deploy_cfg = trial_request_to_deploy_cfg(request, baseline)
    deploy_cfg["eval_env_type"] = config.eval_env_type
    return deploy_cfg
