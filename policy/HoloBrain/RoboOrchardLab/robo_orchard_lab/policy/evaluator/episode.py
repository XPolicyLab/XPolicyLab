# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Generator, NoReturn

import torch

from robo_orchard_lab.envs.base import EnvStepReturn, finalize_env_episode
from robo_orchard_lab.policy.evaluator.contracts import (
    EpisodeResult,
    EvaluationRequest,
    EvaluationStatus,
    PreparedEnvStart,
    TerminalReason,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
    MetricUpdateTiming,
)
from robo_orchard_lab.utils.state import State

if TYPE_CHECKING:
    from robo_orchard_lab.policy.evaluator.base import PolicyEvaluator

__all__ = [
    "evaluate_episode",
]

_StepRet = (
    EnvStepReturn
    | tuple[
        Any,
        Any,
        bool | torch.Tensor | None,
        bool | torch.Tensor | None,
        dict[str, Any],
    ]
)


def _resolve_step_ret_terminal_flags(
    step_ret: _StepRet,
) -> tuple[bool, bool]:
    def normalize_terminal_flag(flag: bool | torch.Tensor | None) -> bool:
        if isinstance(flag, bool):
            return flag
        if flag is None:
            return False
        if isinstance(flag, torch.Tensor):
            return bool(torch.any(flag).item())
        raise ValueError(
            "The terminal flag must be a boolean, torch.Tensor, or None."
        )

    if isinstance(step_ret, tuple):
        return (
            normalize_terminal_flag(step_ret[2]),
            normalize_terminal_flag(step_ret[3]),
        )
    if isinstance(step_ret, EnvStepReturn):
        return (
            normalize_terminal_flag(step_ret.terminated),
            normalize_terminal_flag(step_ret.truncated),
        )
    raise NotImplementedError(
        "The `step_ret` must be either `EnvStepReturn` or tuple."
    )


def evaluate_episode(
    evaluator: PolicyEvaluator,
    request: EvaluationRequest,
) -> EpisodeResult:
    """Run one evaluator episode to completion and return its result payload.

    This helper owns the shared local episode loop used by the public
    evaluator facade. It computes metrics only after terminal metric updates
    have completed, and wraps compute failures in the public execution error
    taxonomy.
    """

    episode_iter = _run_episode_loop(evaluator, request)
    while True:
        try:
            next(episode_iter)
        except StopIteration as stop_iteration:
            terminal_reason, episode_steps = stop_iteration.value
            break

    try:
        metrics = evaluator.compute_metrics()
    except Exception as exception:
        _raise_execution_error(
            message="Failed to compute policy evaluation metrics.",
            terminal_reason=TerminalReason.ERROR,
            episode_steps=episode_steps,
            exception=exception,
        )

    return EpisodeResult(
        status=EvaluationStatus.SUCCEEDED,
        terminal_reason=terminal_reason,
        episode_steps=episode_steps,
        metrics=metrics,
    )


def _run_episode_loop(
    evaluator: PolicyEvaluator,
    request: EvaluationRequest,
) -> Generator[int, None, tuple[TerminalReason, int]]:
    from robo_orchard_lab.policy.evaluator.base import (
        PolicyEvaluationError,
        PolicyEvaluationExecutionError,
    )

    if not evaluator._is_ready():
        raise PolicyEvaluationError(
            "PolicyEvaluator is not ready. Please call setup() first "
            "or reconfigure the policy, environment, and metrics."
        )
    evaluator_metrics = evaluator._evaluator_metrics

    episode_steps = 0
    last_action = None
    last_step_ret: _StepRet | None = None
    episode_terminal_condition_triggered = False

    try:
        if request.max_steps <= 0:
            _raise_execution_error(
                message="Policy evaluation produced no rollout steps.",
                terminal_reason=TerminalReason.EMPTY_ROLLOUT,
                episode_steps=0,
                exception=ValueError("Episode evaluation produced no steps."),
            )
        if request.rollout_steps <= 0:
            _raise_execution_error(
                message="Policy evaluation received an invalid rollout size.",
                terminal_reason=TerminalReason.ERROR,
                episode_steps=0,
                exception=ValueError(
                    "The rollout step count must be greater than zero."
                ),
            )

        init_obs, env_step_callback = _prepare_episode_start(
            evaluator,
            request,
            evaluator_metrics=evaluator_metrics,
        )

        for i in range(0, request.max_steps, request.rollout_steps):
            rollout_ret = evaluator.env.rollout(
                init_obs=init_obs,
                max_steps=min(
                    request.rollout_steps,
                    request.max_steps - i,
                ),
                policy=evaluator.policy,
                env_step_callback=env_step_callback,
                terminal_condition=request.rollout_stop_condition,
                keep_last_results=1,
            )
            if not rollout_ret.actions or not rollout_ret.step_results:
                _raise_execution_error(
                    message="Policy evaluation produced an empty rollout.",
                    terminal_reason=TerminalReason.EMPTY_ROLLOUT,
                    episode_steps=episode_steps,
                    exception=ValueError(
                        "Environment rollout returned no steps."
                    ),
                )
            if rollout_ret.rollout_actual_steps <= 0:
                _raise_execution_error(
                    message="Policy evaluation returned an invalid rollout.",
                    terminal_reason=TerminalReason.ERROR,
                    episode_steps=episode_steps,
                    exception=ValueError(
                        "Environment rollout returned a non-positive "
                        "step count."
                    ),
                )

            episode_steps += rollout_ret.rollout_actual_steps
            if isinstance(rollout_ret.step_results[-1], EnvStepReturn):
                init_obs = rollout_ret.step_results[-1].observations
            else:
                init_obs = rollout_ret.step_results[-1][0]
            last_action = rollout_ret.actions[-1]
            last_step_ret = rollout_ret.step_results[-1]
            yield rollout_ret.rollout_actual_steps
            if rollout_ret.terminal_condition_triggered:
                episode_terminal_condition_triggered = True
                break

        if last_action is None or last_step_ret is None:
            _raise_execution_error(
                message="Policy evaluation produced no rollout steps.",
                terminal_reason=TerminalReason.EMPTY_ROLLOUT,
                episode_steps=episode_steps,
                exception=ValueError("Episode evaluation produced no steps."),
            )

        terminal_reason = _resolve_terminal_reason(
            step_ret=last_step_ret,
            terminal_condition_triggered=episode_terminal_condition_triggered,
            episode_steps=episode_steps,
            max_steps=request.max_steps,
        )
        evaluator_metrics.update(
            timing=MetricUpdateTiming.TERMINAL,
            action=last_action,
            step_ret=last_step_ret,
        )
        return (
            terminal_reason,
            episode_steps,
        )
    except PolicyEvaluationExecutionError:
        raise
    except Exception as exception:
        _raise_execution_error(
            message="Policy evaluation failed during episode execution.",
            terminal_reason=TerminalReason.ERROR,
            episode_steps=episode_steps,
            exception=exception,
        )
    finally:
        finalize_env_episode(evaluator.env)


def _prepare_episode_start(
    evaluator: PolicyEvaluator,
    request: EvaluationRequest,
    *,
    evaluator_metrics: EvaluatorMetrics,
) -> tuple[Any, Callable[[Any, Any], None] | None]:
    """Reset env/policy once and build the optional step metric callback."""

    env_reset_input = request.env_reset_input
    if isinstance(env_reset_input, PreparedEnvStart):
        env_reset_ret = env_reset_input.observations, env_reset_input.info
    elif (
        env_reset_input is None
        or isinstance(env_reset_input, dict)
        or isinstance(env_reset_input, State)
    ):
        env_reset_ret = evaluator.reset_env(env_reset_input=env_reset_input)
    else:
        raise TypeError(
            "env_reset_input must be dict, State, PreparedEnvStart, or None. "
            f"Got {type(env_reset_input).__name__}."
        )

    evaluator.reset_policy(**(request.policy_reset_input or {}))
    if not evaluator_metrics.requires_step_callback:
        return env_reset_ret[0], None

    def step_metric_callback(action: Any, step_ret: Any) -> None:
        evaluator_metrics.update(
            timing=MetricUpdateTiming.STEP,
            action=action,
            step_ret=step_ret,
        )

    return env_reset_ret[0], step_metric_callback


def _resolve_terminal_reason(
    step_ret: _StepRet,
    terminal_condition_triggered: bool,
    episode_steps: int,
    max_steps: int,
) -> TerminalReason:
    terminated, truncated = _resolve_step_ret_terminal_flags(step_ret)

    if truncated:
        return TerminalReason.TRUNCATED
    if terminated:
        return TerminalReason.TERMINATED
    if episode_steps >= max_steps:
        return TerminalReason.MAX_STEPS_REACHED
    if terminal_condition_triggered:
        return TerminalReason.TERMINATED
    return TerminalReason.ERROR


def _raise_execution_error(
    *,
    message: str,
    terminal_reason: TerminalReason,
    episode_steps: int,
    exception: Exception,
) -> NoReturn:
    from robo_orchard_lab.policy.evaluator.base import (
        PolicyEvaluationExecutionError,
    )

    result = EpisodeResult(
        status=EvaluationStatus.FAILED,
        terminal_reason=terminal_reason,
        episode_steps=episode_steps,
        metrics={},
    )
    raise PolicyEvaluationExecutionError(
        message,
        result=result,
        cause_type=type(exception).__name__,
        cause_message=str(exception),
    ) from exception
