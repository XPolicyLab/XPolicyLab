from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepLatencyMetrics:
    """Wall-clock latency for one rollout step."""

    observation_ms: float
    prediction_ms: float
    execution_ms: float
    total_ms: float


@dataclass(frozen=True)
class LatencySummary:
    count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> LatencySummary:
        samples = sorted(float(value) for value in values)
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total = sum(samples)
        return cls(
            count=len(samples),
            total_ms=total,
            mean_ms=total / len(samples),
            p50_ms=_percentile(samples, 0.50),
            p95_ms=_percentile(samples, 0.95),
            max_ms=samples[-1],
        )


@dataclass(frozen=True)
class EpisodeLatencyMetrics:
    observation: LatencySummary
    prediction: LatencySummary
    execution: LatencySummary
    step: LatencySummary
    episode_ms: float

    @classmethod
    def from_steps(
        cls,
        steps: Iterable[StepLatencyMetrics],
        *,
        episode_ms: float,
    ) -> EpisodeLatencyMetrics:
        values = list(steps)
        return cls(
            observation=LatencySummary.from_values(
                item.observation_ms for item in values
            ),
            prediction=LatencySummary.from_values(
                item.prediction_ms for item in values
            ),
            execution=LatencySummary.from_values(
                item.execution_ms for item in values
            ),
            step=LatencySummary.from_values(item.total_ms for item in values),
            episode_ms=float(episode_ms),
        )


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    step_index: int
    status: str
    latency: StepLatencyMetrics
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "rollout_step",
            "schema_version": 1,
            **asdict(self),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    mode: str
    status: str
    requested_steps: int
    completed_steps: int
    latency: EpisodeLatencyMetrics
    steps: tuple[StepRecord, ...] = field(default_factory=tuple)
    error_type: str | None = None
    error_message: str | None = None
    task_id: str | None = None
    success: bool | None = None
    seed: int | None = None
    suite_id: str | None = None
    horizon: int | None = None
    control_frequency_hz: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    suite_metadata: Mapping[str, Any] = field(default_factory=dict)
    task_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "rollout_episode",
            "schema_version": 1,
            "episode_id": self.episode_id,
            "mode": self.mode,
            "status": self.status,
            "requested_steps": self.requested_steps,
            "completed_steps": self.completed_steps,
            "latency": asdict(self.latency),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "task_id": self.task_id,
            "success": self.success,
            "seed": self.seed,
            "suite_id": self.suite_id,
            "horizon": self.horizon,
            "control_frequency_hz": self.control_frequency_hz,
            "metrics": dict(self.metrics),
            "suite_metadata": dict(self.suite_metadata),
            "task_metadata": dict(self.task_metadata),
        }
        if include_steps:
            record["steps"] = [step.to_dict() for step in self.steps]
        return record

    def to_json(self, *, include_steps: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_steps=include_steps),
            sort_keys=True,
        )

    def summary(self) -> dict[str, Any]:
        """Return an episode-level record suitable for summary JSONL."""

        return self.to_dict(include_steps=False)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
