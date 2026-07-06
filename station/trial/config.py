"""Dispatch trial metadata helpers for env client deploy_cfg mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from station.eval_env_type import (
    DEFAULT_EVAL_ENV_TYPE,
    is_debug,
    normalize_eval_env_type,
)
from station.schemas import DispatchPayload, EvaluationTrialPayload


@dataclass(frozen=True)
class TrialRunConfig:
    policy_server_url: str
    evaluation_id: str
    trial_id: str
    action_case_id: str
    policy_name: str
    task_name: str
    env_cfg_type: str
    eval_env_type: str
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


def _payload_extra(payload: object) -> dict[str, Any]:
    return getattr(payload, "__pydantic_extra__", None) or {}


def _first_non_empty_str(*values: object, default: str) -> str:
    for value in values:
        if value:
            return str(value)
    return default


def _resolve_eval_env_type(*values: object) -> str:
    for value in values:
        if value:
            return normalize_eval_env_type(str(value))
    return DEFAULT_EVAL_ENV_TYPE


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
    eval_env_type: str | None = None,
) -> TrialRunConfig:
    case_meta = dict(trial_run.get("case_meta") or {})
    task = dispatch.evaluation_plan.task
    task_extra = _payload_extra(task)
    dispatch_extra = _dispatch_extra(dispatch)
    resolved_eval_env_type = _resolve_eval_env_type(
        eval_env_type,
        case_meta.get("eval_env_type"),
        case_meta.get("eval_env"),
        dispatch_extra.get("eval_env_type"),
        dispatch_extra.get("eval_env"),
    )
    debug_mode = is_debug(resolved_eval_env_type)
    env_cfg_type = _first_non_empty_str(
        case_meta.get("env_cfg_type"),
        trial_run.get("env_cfg_type"),
        task.env_cfg_type if task is not None else "",
        default="arx_x5" if debug_mode else "",
    )
    task_name = _first_non_empty_str(
        case_meta.get("task_name"),
        task.name if task is not None else "",
        dispatch.task_id,
        task.id if task is not None else "",
        default="debug_task" if debug_mode else "",
    )
    policy_name = normalize_policy_name(
        _first_non_empty_str(
            case_meta.get("policy_name"),
            dispatch.model_name,
            default="demo_policy" if debug_mode else "",
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
    bench_name = _first_non_empty_str(
        case_meta.get("bench_name"),
        task_extra.get("bench_name"),
        dispatch_extra.get("bench_name"),
        case_meta.get("dataset_name"),
        task_extra.get("dataset_name"),
        dispatch_extra.get("dataset_name"),
        default="",
    )
    if bench_name:
        case_meta["bench_name"] = bench_name
    repeat_index = case_meta.get("repeat_index", trial_run.get("repeat_index"))

    return TrialRunConfig(
        policy_server_url=dispatch.policy_server_url,
        evaluation_id=evaluation_id,
        trial_id=str(trial_run["trial_id"]),
        action_case_id=str(trial_run["action_case_id"]),
        policy_name=policy_name,
        task_name=task_name,
        env_cfg_type=str(env_cfg_type),
        eval_env_type=resolved_eval_env_type,
        eval_batch=eval_batch,
        case_meta=case_meta,
        instruction=instruction,
        repeat_index=int(repeat_index) if repeat_index is not None else None,
    )
