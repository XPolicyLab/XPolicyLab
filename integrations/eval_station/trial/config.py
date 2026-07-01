"""Dispatch trial metadata helpers for env client deploy_cfg mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval_station.schemas import DispatchPayload, EvaluationTrialPayload


@dataclass(frozen=True)
class TrialRunConfig:
    policy_server_url: str
    evaluation_id: str
    trial_id: str
    action_case_id: str
    policy_name: str
    task_name: str
    env_cfg_type: str
    eval_env: str
    eval_batch: bool
    case_meta: dict[str, Any]
    instruction: str
    repeat_index: int | None = None


def normalize_policy_name(name: str) -> str:
    return name.replace("-", "_")


def _normalize_dispatch_action_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped.lower()
    normalized = str(value).strip().lower()
    return normalized or None


def _dispatch_extra(dispatch: DispatchPayload) -> dict[str, Any]:
    return getattr(dispatch, "__pydantic_extra__", None) or {}


def _first_non_empty_str(*values: object, default: str) -> str:
    for value in values:
        if value:
            return str(value)
    return default


def _find_dispatch_trial(
    dispatch: DispatchPayload, trial_run: dict[str, Any]
) -> EvaluationTrialPayload | None:
    trial_index = trial_run.get("trial_index")
    if trial_index is None:
        return None
    for trial in dispatch.evaluation_plan.trials:
        if trial.trial_index == trial_index:
            return trial
    return None


def _resolve_instruction(
    dispatch: DispatchPayload,
    trial_run: dict[str, Any],
    case_meta: dict[str, Any],
) -> str:
    dispatch_trial = _find_dispatch_trial(dispatch, trial_run)
    return _first_non_empty_str(
        case_meta.get("instruction"),
        case_meta.get("language_instruction"),
        trial_run.get("instruction"),
        dispatch_trial.instruction if dispatch_trial is not None else None,
        default="",
    )


def build_trial_run_config(
    dispatch: DispatchPayload,
    trial_run: dict[str, Any],
    *,
    evaluation_id: str,
    eval_env: str | None = None,
) -> TrialRunConfig:
    case_meta = dict(trial_run.get("case_meta") or {})
    task = dispatch.evaluation_plan.task
    dispatch_extra = _dispatch_extra(dispatch)
    resolved_eval_env = _first_non_empty_str(
        eval_env,
        case_meta.get("eval_env"),
        dispatch_extra.get("eval_env"),
        default="debug",
    )
    # Debug keeps legacy hard defaults; real envs require dispatch or startup args.
    is_debug = resolved_eval_env == "debug"
    env_cfg_type = _first_non_empty_str(
        case_meta.get("env_cfg_type"),
        trial_run.get("env_cfg_type"),
        task.env_cfg_type if task is not None else "",
        default="arx_x5" if is_debug else "",
    )
    task_name = _first_non_empty_str(
        case_meta.get("task_name"),
        task.name if task is not None else "",
        dispatch.task_id,
        task.id if task is not None else "",
        default="debug_task" if is_debug else "",
    )
    policy_name = normalize_policy_name(
        _first_non_empty_str(
            case_meta.get("policy_name"),
            dispatch.model_name,
            default="demo_policy" if is_debug else "",
        )
    )
    eval_batch = bool(
        case_meta.get("eval_batch", dispatch_extra.get("eval_batch", False))
    )
    instruction = _resolve_instruction(dispatch, trial_run, case_meta)
    if instruction:
        case_meta["instruction"] = instruction
    action_type = _normalize_dispatch_action_type(
        case_meta.get("action_type") or dispatch_extra.get("action_type")
    )
    if action_type in ("joint", "ee"):
        case_meta["action_type"] = action_type
    repeat_index = case_meta.get("repeat_index", trial_run.get("repeat_index"))

    return TrialRunConfig(
        policy_server_url=dispatch.policy_server_url,
        evaluation_id=evaluation_id,
        trial_id=str(trial_run["trial_id"]),
        action_case_id=str(trial_run["action_case_id"]),
        policy_name=policy_name,
        task_name=task_name,
        env_cfg_type=str(env_cfg_type),
        eval_env=str(resolved_eval_env),
        eval_batch=eval_batch,
        case_meta=case_meta,
        instruction=instruction,
        repeat_index=int(repeat_index) if repeat_index is not None else None,
    )
