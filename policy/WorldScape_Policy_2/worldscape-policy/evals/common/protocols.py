from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch

from worldscape_policy.types import (
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)


@dataclass(frozen=True)
class RolloutInput:
    observation: ObservationBatch
    prompts: PromptBatch


RawObservationT_contra = TypeVar("RawObservationT_contra", contravariant=True)
AdaptedObservationT_co = TypeVar("AdaptedObservationT_co", covariant=True)
AdaptedActionT_co = TypeVar("AdaptedActionT_co", covariant=True)


@runtime_checkable
class ObservationAdapter(
    Protocol[RawObservationT_contra, AdaptedObservationT_co]
):
    """Convert a backend-native observation to a policy-facing value."""

    def observation(
        self, value: RawObservationT_contra
    ) -> AdaptedObservationT_co: ...


@runtime_checkable
class ActionAdapter(Protocol[AdaptedActionT_co]):
    """Convert a policy output to a backend-native action."""

    def action(self, output: WorldActionOutput) -> AdaptedActionT_co: ...


class RuntimeProtocol(Protocol):
    """The transactional subset of PolicyRuntime used by rollout backends."""

    @property
    def has_pending_prediction(self) -> bool: ...

    def reset(self, mode: str) -> None: ...

    def predict(
        self,
        *,
        observation: ObservationBatch,
        prompts: PromptBatch,
        generator: torch.Generator,
    ) -> WorldActionOutput: ...

    def commit(self, output: WorldActionOutput | None = None) -> None: ...

    def discard(self) -> None: ...


class ObservationSource(Protocol):
    def read(self, step_index: int) -> RolloutInput:
        """Read and adapt the next environment observation."""


class ActionExecutor(Protocol):
    def execute(
        self,
        output: WorldActionOutput,
        *,
        timeout_s: float | None,
    ) -> None:
        """Execute an action or raise, including TimeoutError on deadline."""


@runtime_checkable
class EvaluationEnvironment(Protocol):
    """Backend-neutral environment contract used by the evaluation runner.

    ``reset`` and ``step`` deliberately return backend-native values.  The
    selected observation/action adapter owns conversion to public policy
    schemas, while ``success`` keeps benchmark-specific success logic in the
    environment implementation.
    """

    def reset(self, task: Any, *, seed: int | None = None) -> Any: ...

    def step(self, action: Any) -> Any: ...

    def success(self) -> bool: ...

    def close(self) -> None: ...
