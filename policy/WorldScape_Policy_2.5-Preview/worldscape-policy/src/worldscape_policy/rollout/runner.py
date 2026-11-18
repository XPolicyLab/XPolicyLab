from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from evals.common.protocols import (
    ActionExecutor,
    ObservationSource,
    RuntimeProtocol,
)
from evals.common.schemas import (
    EpisodeLatencyMetrics,
    EpisodeRecord,
    StepLatencyMetrics,
    StepRecord,
)
from worldscape_policy.types import InteractionMode, WorldActionOutput


@dataclass(frozen=True)
class RolloutConfig:
    mode: InteractionMode | str
    max_steps: int
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    execution_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if (
            self.execution_timeout_s is not None
            and self.execution_timeout_s <= 0
        ):
            raise ValueError("execution_timeout_s must be positive")


@dataclass(frozen=True)
class RolloutResult:
    outputs: tuple[WorldActionOutput, ...]
    record: EpisodeRecord
    error: Exception | None = field(default=None, repr=False, compare=False)

    def raise_for_error(self) -> None:
        if self.error is not None:
            raise self.error

    def jsonl_records(self) -> tuple[dict[str, Any], ...]:
        """Return step records followed by an episode summary record."""

        return tuple(step.to_dict() for step in self.record.steps) + (
            self.record.summary(),
        )


class RolloutRunner:
    """Run a PolicyRuntime transaction against environment adapters."""

    def __init__(
        self,
        runtime: RuntimeProtocol,
        observation_source: ObservationSource,
        action_executor: ActionExecutor,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._runtime = runtime
        self._observation_source = observation_source
        self._action_executor = action_executor
        self._clock = clock

    def run(
        self,
        config: RolloutConfig,
        *,
        generator: torch.Generator,
    ) -> RolloutResult:
        mode = InteractionMode.parse(config.mode)
        self._runtime.reset(mode.value)
        episode_started = self._clock()
        outputs: list[WorldActionOutput] = []
        records: list[StepRecord] = []
        error: Exception | None = None

        for step_index in range(config.max_steps):
            step_started = self._clock()
            observation_finished = step_started
            prediction_finished = step_started
            execution_finished = step_started
            phase = "observation"
            try:
                rollout_input = self._observation_source.read(step_index)
                observation_finished = self._clock()
                phase = "prediction"
                output = self._runtime.predict(
                    observation=rollout_input.observation,
                    prompts=rollout_input.prompts,
                    generator=generator,
                )
                prediction_finished = self._clock()
                phase = "execution"
                self._action_executor.execute(
                    output,
                    timeout_s=config.execution_timeout_s,
                )
                execution_finished = self._clock()
                phase = "commit"
                self._runtime.commit(output)
                step_finished = self._clock()
            except Exception as exc:
                error = exc
                failure_finished = self._clock()
                if phase == "observation":
                    observation_finished = failure_finished
                    prediction_finished = failure_finished
                    execution_finished = failure_finished
                elif phase == "prediction":
                    prediction_finished = failure_finished
                    execution_finished = failure_finished
                elif phase == "execution":
                    execution_finished = failure_finished
                if self._runtime.has_pending_prediction:
                    self._runtime.discard()
                records.append(
                    StepRecord(
                        episode_id=config.episode_id,
                        step_index=step_index,
                        status=(
                            "timed_out"
                            if isinstance(exc, TimeoutError)
                            else "failed"
                        ),
                        latency=_latency(
                            step_started,
                            observation_finished,
                            prediction_finished,
                            execution_finished,
                            failure_finished,
                        ),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                break

            outputs.append(output)
            records.append(
                StepRecord(
                    episode_id=config.episode_id,
                    step_index=step_index,
                    status="completed",
                    latency=_latency(
                        step_started,
                        observation_finished,
                        prediction_finished,
                        execution_finished,
                        step_finished,
                    ),
                )
            )

        episode_finished = self._clock()
        status = "completed"
        if isinstance(error, TimeoutError):
            status = "timed_out"
        elif error is not None:
            status = "failed"
        record = EpisodeRecord(
            episode_id=config.episode_id,
            mode=mode.value,
            status=status,
            requested_steps=config.max_steps,
            completed_steps=len(outputs),
            latency=EpisodeLatencyMetrics.from_steps(
                (item.latency for item in records),
                episode_ms=_milliseconds(episode_finished - episode_started),
            ),
            steps=tuple(records),
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        return RolloutResult(tuple(outputs), record, error)


def _latency(
    step_started: float,
    observation_finished: float,
    prediction_finished: float,
    execution_finished: float,
    step_finished: float,
) -> StepLatencyMetrics:
    return StepLatencyMetrics(
        observation_ms=_milliseconds(observation_finished - step_started),
        prediction_ms=_milliseconds(prediction_finished - observation_finished),
        execution_ms=_milliseconds(execution_finished - prediction_finished),
        total_ms=_milliseconds(step_finished - step_started),
    )


def _milliseconds(seconds: float) -> float:
    return max(0.0, float(seconds) * 1000.0)
