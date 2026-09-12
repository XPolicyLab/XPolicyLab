from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from evals.common.environment import normalize_environment_step
from evals.common.protocols import (
    EvaluationEnvironment,
    RuntimeProtocol,
)
from evals.common.schemas import (
    EpisodeLatencyMetrics,
    EpisodeRecord,
    StepLatencyMetrics,
    StepRecord,
)
from evals.common.simulator import SimulatorAdapterProtocol
from evals.common.suite import EpisodeSpec, TaskSuite
from worldscape_policy.types import InteractionMode, WorldActionOutput


@dataclass(frozen=True)
class EvaluationConfig:
    mode: InteractionMode | str
    max_steps: int
    execution_timeout_s: float | None = None
    device: torch.device | str = "cpu"
    control_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.execution_timeout_s is not None and self.execution_timeout_s <= 0:
            raise ValueError("execution_timeout_s must be positive")
        if self.control_frequency_hz is not None and self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")


@dataclass(frozen=True)
class EvaluationEpisodeResult:
    spec: EpisodeSpec
    outputs: tuple[WorldActionOutput, ...]
    record: EpisodeRecord
    frames: tuple[np.ndarray, ...] = ()
    error: Exception | None = None


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EvaluationEpisodeResult, ...]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(item.record.success is True for item in self.episodes) / len(
            self.episodes
        )


class EvaluationRunner:
    """Evaluate task suites with transactional policy-state updates."""

    def __init__(
        self,
        runtime: RuntimeProtocol,
        environment: EvaluationEnvironment,
        adapter: SimulatorAdapterProtocol,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.runtime = runtime
        self.environment = environment
        self.adapter = adapter
        self.clock = clock

    def run(
        self,
        suite: TaskSuite,
        config: EvaluationConfig,
        *,
        generator: torch.Generator,
    ) -> EvaluationResult:
        episodes = []
        try:
            for spec in suite.episodes():
                episodes.append(
                    self._run_episode(
                        spec,
                        config,
                        generator,
                        suite_id=suite.suite_id,
                        suite_metadata=suite.metadata,
                    )
                )
        finally:
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            self.environment.close()
        return EvaluationResult(tuple(episodes))

    def _run_episode(
        self,
        spec: EpisodeSpec,
        config: EvaluationConfig,
        generator: torch.Generator,
        *,
        suite_id: str,
        suite_metadata: Mapping[str, Any],
    ) -> EvaluationEpisodeResult:
        mode = InteractionMode.parse(config.mode)
        self.runtime.reset(mode.value)
        started = self.clock()
        records: list[StepRecord] = []
        outputs: list[WorldActionOutput] = []
        frames: list[np.ndarray] = []
        error: Exception | None = None
        native_observation = None
        try:
            native_observation = self.environment.reset(spec.task, seed=spec.seed)
            for step_index in range(config.max_steps):
                step_started = self.clock()
                observation = self.adapter.observation(
                    native_observation, device=config.device
                )
                prompts = self.adapter.prompt(spec.task.instruction, mode=mode)
                observation_finished = self.clock()
                output = self.runtime.predict(
                    observation=observation,
                    prompts=prompts,
                    generator=generator,
                )
                prediction_finished = self.clock()
                step_value = self.environment.step(self.adapter.action(output))
                execution_finished = self.clock()
                step = normalize_environment_step(step_value)
                # The candidate state becomes authoritative only after the
                # environment accepted and completed the action.
                self.runtime.commit(output)
                step_finished = self.clock()
                outputs.append(output.result_snapshot())
                frames.extend(step.frames)
                records.append(
                    StepRecord(
                        episode_id=spec.episode_id,
                        step_index=step_index,
                        status="completed",
                        latency=_step_latency(
                            step_started,
                            observation_finished,
                            prediction_finished,
                            execution_finished,
                            step_finished,
                        ),
                    )
                )
                native_observation = step.observation
                if step.done or self.environment.success():
                    break
        except Exception as exc:  # noqa: BLE001 - episode failures are artifacts
            error = exc
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            records.append(
                StepRecord(
                    episode_id=spec.episode_id,
                    step_index=len(outputs),
                    status="timed_out" if isinstance(exc, TimeoutError) else "failed",
                    latency=StepLatencyMetrics(0.0, 0.0, 0.0, 0.0),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        finished = self.clock()
        metrics = _environment_metrics(self.environment)
        success = False if error is not None else bool(self.environment.success())
        metrics.setdefault("success", float(success))
        status = (
            "timed_out"
            if isinstance(error, TimeoutError)
            else "failed"
            if error is not None
            else "completed"
        )
        record = EpisodeRecord(
            episode_id=spec.episode_id,
            mode=mode.value,
            status=status,
            requested_steps=config.max_steps,
            completed_steps=len(outputs),
            latency=EpisodeLatencyMetrics.from_steps(
                (item.latency for item in records),
                episode_ms=max(0.0, (finished - started) * 1000.0),
            ),
            steps=tuple(records),
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            task_id=spec.task.task_id,
            success=success,
            seed=spec.seed,
            suite_id=suite_id,
            horizon=config.max_steps,
            control_frequency_hz=config.control_frequency_hz,
            metrics=metrics,
            suite_metadata=dict(suite_metadata),
            task_metadata=dict(spec.task.metadata),
        )
        return EvaluationEpisodeResult(
            spec=spec,
            outputs=tuple(outputs),
            record=record,
            frames=tuple(frames),
            error=error,
        )


def _step_latency(
    started: float,
    observation_finished: float,
    prediction_finished: float,
    execution_finished: float,
    finished: float,
) -> StepLatencyMetrics:
    def milliseconds(value: float) -> float:
        return max(0.0, value * 1000.0)

    return StepLatencyMetrics(
        observation_ms=milliseconds(observation_finished - started),
        prediction_ms=milliseconds(prediction_finished - observation_finished),
        execution_ms=milliseconds(execution_finished - prediction_finished),
        total_ms=milliseconds(finished - started),
    )


def _environment_metrics(environment: Any) -> dict[str, float]:
    metric_fn = getattr(environment, "metrics", None)
    if not callable(metric_fn):
        return {}
    values = metric_fn()
    if not isinstance(values, Mapping):
        raise TypeError("Evaluation environment metrics() must return a mapping")
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, (bool, int, float, np.number)):
            metrics[str(name)] = float(value)
    return metrics
