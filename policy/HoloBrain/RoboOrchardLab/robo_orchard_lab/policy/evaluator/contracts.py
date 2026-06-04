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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias

from typing_extensions import Self

from robo_orchard_lab.envs.base import EnvStepReturn
from robo_orchard_lab.utils.state import State

RollOutStopCondition = Callable[
    [EnvStepReturn | tuple[Any, Any, bool, bool, dict[str, Any]]], bool
]


def _default_rollout_stop_condition() -> RollOutStopCondition:
    from robo_orchard_lab.policy.evaluator.base import (
        evaluate_rollout_stop_condition,
    )

    return evaluate_rollout_stop_condition


class EvaluationStatus(str, Enum):
    """Episode execution status carried in evaluator result payloads."""

    SUCCEEDED = "SUCCEEDED"
    """The episode loop and metric computation completed successfully."""

    FAILED = "FAILED"
    """The evaluator wrapped an execution failure in a result payload."""


class TerminalReason(str, Enum):
    """Caller-visible reason why an evaluator episode stopped."""

    TERMINATED = "TERMINATED"
    """The environment reported ordinary task termination."""

    TRUNCATED = "TRUNCATED"
    """The environment reported truncation before ordinary termination."""

    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    """The evaluator consumed the requested step budget."""

    EMPTY_ROLLOUT = "EMPTY_ROLLOUT"
    """The rollout produced no usable environment step."""

    ERROR = "ERROR"
    """The episode stopped because an execution error was raised."""


@dataclass(slots=True)
class PreparedEnvStart:
    """Already-reset environment start for one evaluator episode.

    Use this payload when a caller has already reset the environment and
    wants evaluation to begin from the returned observation without issuing a
    second env reset. Policy reset still uses the ordinary per-episode policy
    reset input.
    """

    observations: Any
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.info, dict):
            raise TypeError(
                "PreparedEnvStart.info must be a dict. "
                f"Got {type(self.info).__name__}."
            )
        self.info = dict(self.info)

    @classmethod
    def from_reset_return(
        cls,
        reset_ret: tuple[Any, dict[str, Any]],
    ) -> Self:
        """Wrap an ``env.reset(...)`` return value as a prepared start."""

        observations, info = reset_ret
        return cls(observations=observations, info=info)


EnvResetInput: TypeAlias = dict[str, Any] | State | None
EnvStartInput: TypeAlias = EnvResetInput | PreparedEnvStart
PolicyResetInput: TypeAlias = dict[str, Any] | None


@dataclass(slots=True)
class EvaluationRequest:
    """Canonical request for one policy-evaluator episode.

    ``env_reset_input`` accepts reset-triggering inputs for ordinary episode
    starts, or :class:`PreparedEnvStart` when the caller has already reset the
    environment and is providing the initial observation directly.
    """

    max_steps: int
    rollout_steps: int = 5
    env_reset_input: EnvStartInput = None
    policy_reset_input: PolicyResetInput = None
    rollout_stop_condition: RollOutStopCondition = field(
        default_factory=_default_rollout_stop_condition
    )

    def __post_init__(self) -> None:
        if isinstance(self.env_reset_input, dict):
            self.env_reset_input = dict(self.env_reset_input)
        elif self.env_reset_input is not None and not isinstance(
            self.env_reset_input,
            (State, PreparedEnvStart),
        ):
            raise TypeError(
                "env_reset_input must be dict, State, PreparedEnvStart, "
                "or None. "
                f"Got {type(self.env_reset_input).__name__}."
            )

        if isinstance(self.policy_reset_input, dict):
            self.policy_reset_input = dict(self.policy_reset_input)
        elif self.policy_reset_input is not None:
            raise TypeError(
                "policy_reset_input must be dict or None. "
                f"Got {type(self.policy_reset_input).__name__}."
            )


@dataclass(slots=True)
class EpisodeResult:
    """Result envelope for one evaluator episode attempt.

    This payload is used both for successful direct helper returns and for
    public ``PolicyEvaluationExecutionError.result`` values. ``metrics`` is
    filled only after successful terminal metric update and compute; failure
    results keep it empty because the metric surface may have been rolled
    back by the evaluator facade.
    """

    status: EvaluationStatus
    """Whether the episode completed successfully or failed."""

    terminal_reason: TerminalReason
    """Why rollout stopped, or why the failure result was produced."""

    episode_steps: int
    """Number of environment steps completed before stop or failure."""

    metrics: dict[str, Any]
    """Computed evaluator metrics for successful episodes."""
