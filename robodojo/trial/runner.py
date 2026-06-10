"""Run one RoboDojo trial through env client + policy server."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from robodojo.env_client import RoboDojoModelClient
from robodojo.schemas import DispatchPayload, EvaluationTrialPayload
from robodojo.trial.env import DebugTrialEnv, TrialEnv
from robodojo.trial.sim import SimEnvConfig, create_sim_trial_env


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
    seed: int | None = None
    device_id: str | None = None
    root_dir: str | None = None
    sim_env_factory: str | None = None
    repeat_index: int | None = None
    episode_step_limit: int = 5


class ActionRecorder:
    def __init__(self, client: RoboDojoModelClient):
        self._client = client
        self.actions: list[Any] = []
        self.steps = 0

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        result = self._client.call(func_name=func_name, obs=obs, **kwargs)
        if func_name == "get_action":
            self._record_single_env_actions(result)
        elif func_name == "get_action_batch":
            self._record_batch_actions(result)
        return result

    def _record_single_env_actions(self, result: Any) -> None:
        self.steps += 1
        if isinstance(result, list):
            self.actions.extend(result)
        elif result is not None:
            self.actions.append(result)

    def _record_batch_actions(self, result: Any) -> None:
        self.steps += 1
        if not isinstance(result, list):
            return
        for env_actions in result:
            if isinstance(env_actions, list):
                self.actions.extend(env_actions)
            elif env_actions is not None:
                self.actions.append(env_actions)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ActionRecorder:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def normalize_policy_name(name: str) -> str:
    return name.replace("-", "_")


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
    root_dir: str | None = None,
    sim_env_factory: str | None = None,
    episode_step_limit: int = 5,
) -> TrialRunConfig:
    case_meta = dict(trial_run.get("case_meta") or {})
    task = dispatch.evaluation_plan.task
    dispatch_extra = _dispatch_extra(dispatch)
    env_cfg_type = _first_non_empty_str(
        case_meta.get("env_cfg_type"),
        trial_run.get("env_cfg_type"),
        task.env_cfg_type if task is not None else "",
        default="arx_x5",
    )
    task_name = _first_non_empty_str(
        case_meta.get("task_name"),
        dispatch.task_id,
        task.id if task is not None else "",
        default="debug_task",
    )
    policy_name = normalize_policy_name(
        _first_non_empty_str(
            case_meta.get("policy_name"),
            dispatch.model_name,
            default="demo_policy",
        )
    )
    resolved_eval_env = _first_non_empty_str(
        eval_env,
        case_meta.get("eval_env"),
        dispatch_extra.get("eval_env"),
        default="debug",
    )
    eval_batch = bool(
        case_meta.get("eval_batch", dispatch_extra.get("eval_batch", False))
    )
    seed = case_meta.get("seed", dispatch_extra.get("seed"))
    device_id = case_meta.get("device_id", dispatch_extra.get("device_id"))
    instruction = _resolve_instruction(dispatch, trial_run, case_meta)
    if instruction:
        case_meta["instruction"] = instruction
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
        seed=int(seed) if seed is not None else None,
        device_id=str(device_id) if device_id is not None else None,
        root_dir=root_dir or dispatch_extra.get("root_dir"),
        sim_env_factory=sim_env_factory or dispatch_extra.get("sim_env_factory"),
        repeat_index=int(repeat_index) if repeat_index is not None else None,
        episode_step_limit=episode_step_limit,
    )


def create_trial_env(config: TrialRunConfig) -> TrialEnv:
    additional_info = {
        "trial_id": config.trial_id,
        "action_case_id": config.action_case_id,
    }
    if config.instruction:
        additional_info["instruction"] = config.instruction
    if config.eval_env == "debug":
        return DebugTrialEnv(
            env_cfg_type=config.env_cfg_type,
            instruction=config.instruction,
            episode_step_limit=config.episode_step_limit,
            additional_info=additional_info,
        )
    if config.eval_env == "sim":
        return create_sim_trial_env(
            SimEnvConfig(
                task_name=config.task_name,
                env_cfg_type=config.env_cfg_type,
                seed=config.seed,
                device_id=config.device_id,
                additional_info=str(config.case_meta.get("additional_info", "")),
                case_meta=config.case_meta,
                root_dir=config.root_dir,
                sim_env_factory=config.sim_env_factory,
            )
        )
    raise ValueError(
        f"Unsupported eval_env {config.eval_env!r}; expected 'debug' or 'sim'"
    )


def _load_deploy_module(policy_name: str):
    module_name = f"XPolicyLab.policy.{policy_name}.deploy"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import policy deploy module: {module_name}"
        ) from exc


def run_eval_episode(
    task_env: TrialEnv,
    model_client: Any,
    *,
    policy_name: str,
    eval_batch: bool,
) -> None:
    deploy_module = _load_deploy_module(policy_name)
    eval_fn_name = "eval_one_episode_batch" if eval_batch else "eval_one_episode"
    eval_fn = getattr(deploy_module, eval_fn_name, None)
    if eval_fn is None:
        raise AttributeError(
            f"policy {policy_name}.deploy is missing {eval_fn_name}"
        )
    eval_fn(TASK_ENV=task_env, model_client=model_client)


def run_policy_trial(
    policy_server_url: str,
    evaluation_id: str,
    trial_run: dict[str, Any],
    *,
    dispatch: DispatchPayload | None = None,
    eval_env: str | None = None,
    root_dir: str | None = None,
    sim_env_factory: str | None = None,
    episode_step_limit: int = 5,
) -> dict[str, Any]:
    if dispatch is None:
        dispatch = DispatchPayload.model_validate(
            {
                "policy_server_url": policy_server_url,
                "evaluation_plan": {"trials": []},
            }
        )

    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id=evaluation_id,
        eval_env=eval_env,
        root_dir=root_dir,
        sim_env_factory=sim_env_factory,
        episode_step_limit=episode_step_limit,
    )
    task_env = create_trial_env(config)

    with RoboDojoModelClient(
        url=config.policy_server_url,
        evaluation_id=config.evaluation_id,
        trial_id=config.trial_id,
        action_case_id=config.action_case_id,
        repeat_index=config.repeat_index,
    ) as raw_client:
        with ActionRecorder(raw_client) as model_client:
            model_client.call(func_name="prepare_case", obs=config.case_meta)
            task_env.reset()
            model_client.call(func_name="reset")
            run_eval_episode(
                task_env,
                model_client,
                policy_name=config.policy_name,
                eval_batch=config.eval_batch,
            )
            model_client.call(
                func_name="trial_end",
                obs={
                    "result": "success",
                    "steps": model_client.steps,
                },
            )
            return {
                "trial_id": config.trial_id,
                "actions": list(model_client.actions),
                "steps": model_client.steps,
                "eval_env": config.eval_env,
            }
