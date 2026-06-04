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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import torch

from robo_orchard_lab.envs.base import EnvBaseCfg
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator.contracts import (
    EnvResetInput,
    PolicyResetInput,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)

__all__ = [
    "BenchmarkAttemptRequest",
    "BenchmarkAttemptError",
    "BenchmarkDriver",
    "BenchmarkEpisode",
    "BenchmarkEpisodeRecord",
    "BenchmarkEvaluateFailedEvent",
    "BenchmarkEvaluateSucceededEvent",
    "BenchmarkEvaluator",
    "BenchmarkPrepareFailedEvent",
    "BenchmarkPrepareJob",
    "BenchmarkPrepareSucceededEvent",
    "BenchmarkResult",
    "BenchmarkTerminalEvent",
]


@dataclass(slots=True)
class BenchmarkEpisode:
    """Identify one logical benchmark episode independently of attempts.

    Domain drivers use this payload for stable logging, retry, result
    records, and aggregation grouping. Concrete reset inputs are allocated
    later in :meth:`BenchmarkDriver.make_attempt_request`, when the backend
    has actually reserved a worker for the attempt.
    """

    episode_key: str
    """Stable logical id for logs, retry bookkeeping, and result records."""

    group_key: str
    """Aggregation key, for example a task name in a multi-task benchmark."""

    episode_id: int
    """Zero-based logical episode index within its benchmark/group scope."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Domain-owned annotations. The generic backend never interprets them."""


@dataclass(slots=True)
class BenchmarkPrepareJob:
    """Describe a ready logical attempt before it reserves a worker.

    Drivers return these jobs from :meth:`BenchmarkDriver.get_ready_jobs`
    after applying domain readiness rules. The backend converts each job into
    a :class:`BenchmarkAttemptRequest` only when a worker slot is acquired, so
    domain state such as offset assignment is not advanced prematurely.
    """

    episode: BenchmarkEpisode
    """Logical episode this attempt belongs to."""

    attempt_index: int
    """Zero-based retry/attempt index for the logical episode."""

    max_steps: int
    """Rollout step cap used by evaluate_episode after prepare succeeds."""

    policy_reset_input: PolicyResetInput = None
    """Optional policy reset payload forwarded at rollout time."""

    metric_reset_input: dict[str, Any] = field(default_factory=dict)
    """Keyword payload for per-attempt worker metric reset."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Domain-owned attempt annotations. The backend never interprets them."""


@dataclass(slots=True)
class BenchmarkAttemptRequest:
    """Fully specify one benchmark attempt for the backend pipeline.

    The env reset input must be a reset-triggering input. The backend executes
    it during prepare, wraps the returned observation and reset info in a
    prepared-start payload, and then evaluates the same attempt without a
    second env reset.
    """

    episode: BenchmarkEpisode
    """Logical episode this concrete attempt belongs to."""

    attempt_index: int
    """Zero-based retry/attempt index copied from the prepare job."""

    env_cfg: EnvBaseCfg
    """Environment config for setup or reconfigure on the reserved worker."""

    env_reset_input: EnvResetInput
    """Reset-triggering input consumed during prepare/reset."""

    max_steps: int
    """Rollout step cap used by evaluate_episode."""

    policy_reset_input: PolicyResetInput = None
    """Optional policy reset payload forwarded at rollout time."""

    metric_reset_input: dict[str, Any] = field(default_factory=dict)
    """Keyword payload for per-attempt worker metric reset."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Domain-owned concrete attempt annotations."""


@dataclass(slots=True)
class BenchmarkPrepareSucceededEvent:
    """Notify a driver that prepare/reset completed for an attempt."""

    request: BenchmarkAttemptRequest
    """Concrete attempt that completed prepare/reset."""

    reset_info: dict[str, Any] = field(default_factory=dict)
    """Env reset info returned by the worker, normalized to a dict."""

    worker_id: int | None = None
    """Backend worker id for diagnostics; assignment is not stable."""


@dataclass(slots=True)
class BenchmarkPrepareFailedEvent:
    """Notify a driver that setup, metric reset, or env reset failed."""

    request: BenchmarkAttemptRequest
    """Concrete attempt that failed before rollout submission."""

    reset_info: dict[str, Any] = field(default_factory=dict)
    """Best-effort reset info from exceptions that expose reset_info."""

    error_type: str | None = None
    """Exception class name, stored as data so results stay serializable."""

    error_message: str | None = None
    """Exception message suitable for result records and logs."""

    worker_id: int | None = None
    """Backend worker id for diagnostics and tests."""


@dataclass(slots=True)
class BenchmarkEvaluateSucceededEvent:
    """Notify a driver that rollout and worker metric capture completed.

    This event reports infrastructure success only. Domain/task success is
    still encoded in ``episode_metrics`` or in the driver-owned aggregate
    metric policy.
    """

    request: BenchmarkAttemptRequest
    """Concrete attempt that completed rollout and worker metric capture."""

    episode_metrics: dict[str, Any]
    """Per-episode metrics returned by the worker evaluator."""

    worker_metrics: EvaluatorMetrics
    """Serialized worker metric snapshot for driver-side aggregation."""

    reset_info: dict[str, Any] = field(default_factory=dict)
    """Env reset info captured during prepare and carried through rollout."""

    worker_id: int | None = None
    """Backend worker id for diagnostics and tests."""


@dataclass(slots=True)
class BenchmarkEvaluateFailedEvent:
    """Notify a driver that rollout or worker metric capture failed."""

    request: BenchmarkAttemptRequest
    """Concrete attempt that failed after prepare succeeded."""

    reset_info: dict[str, Any] = field(default_factory=dict)
    """Env reset info captured during prepare and carried through failure."""

    error_type: str | None = None
    """Exception class name, stored as data so results stay serializable."""

    error_message: str | None = None
    """Exception message suitable for result records and logs."""

    worker_id: int | None = None
    """Backend worker id for diagnostics and tests."""


BenchmarkTerminalEvent: TypeAlias = (
    BenchmarkPrepareFailedEvent
    | BenchmarkEvaluateSucceededEvent
    | BenchmarkEvaluateFailedEvent
)


class BenchmarkAttemptError(RuntimeError):
    """Raised when fail-fast benchmark policy aborts on an attempt failure.

    The error carries the serializable terminal failure event rather than the
    original exception object because remote worker failures may cross process
    boundaries. The event is an infrastructure failure; it does not represent
    domain/task success or failure.
    """

    terminal_event: BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent
    """Failure event that caused fail-fast benchmark termination."""

    def __init__(
        self,
        terminal_event: (
            BenchmarkPrepareFailedEvent | BenchmarkEvaluateFailedEvent
        ),
    ) -> None:
        self.terminal_event = terminal_event
        request = terminal_event.request
        episode = request.episode
        error_type = terminal_event.error_type or type(terminal_event).__name__
        error_message = terminal_event.error_message or "attempt failed"
        super().__init__(
            "Benchmark attempt failed "
            f"for episode={episode.episode_key!r} "
            f"group={episode.group_key!r} "
            f"attempt={request.attempt_index}: "
            f"{error_type}: {error_message}"
        )


@dataclass(slots=True)
class BenchmarkEpisodeRecord:
    """Represent the final domain outcome for one logical episode."""

    episode: BenchmarkEpisode
    """Logical episode represented by this final record."""

    succeeded: bool
    """Domain-level outcome after retries are exhausted or succeeded."""

    attempts: int
    """Number of concrete attempts consumed for this logical episode."""

    episode_metrics: dict[str, Any] = field(default_factory=dict)
    """Final per-episode metrics chosen by the driver."""

    error_type: str | None = None
    """Final failure exception class name, when the episode failed."""

    error_message: str | None = None
    """Final failure message, when the episode failed."""

    attempt_errors: list[dict[str, Any]] = field(default_factory=list)
    """Per-attempt failure summaries retained by the driver."""


@dataclass(slots=True)
class BenchmarkResult:
    """Return aggregate benchmark metrics with per-episode records."""

    metrics: dict[str, Any]
    """Driver-owned aggregate metrics for the full benchmark run."""

    episodes: list[BenchmarkEpisodeRecord]
    """Final record for each logical episode."""

    metadata: dict[str, Any]
    """Benchmark-level annotations such as backend or domain settings."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for compatibility wrappers."""

        return {
            "metrics": self.metrics,
            "episodes": [
                {
                    "episode": {
                        "episode_key": record.episode.episode_key,
                        "group_key": record.episode.group_key,
                        "episode_id": record.episode.episode_id,
                        "metadata": record.episode.metadata,
                    },
                    "succeeded": record.succeeded,
                    "attempts": record.attempts,
                    "episode_metrics": record.episode_metrics,
                    "error_type": record.error_type,
                    "error_message": record.error_message,
                    "attempt_errors": record.attempt_errors,
                }
                for record in self.episodes
            ],
            "metadata": self.metadata,
        }


class BenchmarkEvaluator(Protocol):
    """Common callable interface for benchmark-level policy evaluation.

    Use this protocol when code needs to accept any concrete benchmark
    evaluator without depending on a specific domain such as RoboTwin or a
    specific backend such as :class:`RemoteBenchmarkBackend`. Concrete
    evaluators own their configuration, backend choice, and domain policy.
    """

    def evaluate(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        *,
        device: str | torch.device | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark for one policy and return aggregate results."""
        ...


class BenchmarkDriver(Protocol):
    """Domain policy interface consumed by :class:`RemoteBenchmarkBackend`.

    All callbacks are invoked serially from the backend scheduler loop. Driver
    implementations can therefore own domain readiness, retry, offset
    advancement, and metric aggregation without adding their own locks.
    """

    def has_unfinished_work(self) -> bool:
        """Return whether the benchmark still has logical work to finish."""
        ...

    def get_ready_jobs(
        self,
        max_jobs: int | None = None,
    ) -> Sequence[BenchmarkPrepareJob]:
        """Return currently ready jobs, honoring ``max_jobs`` when provided."""
        ...

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        """Allocate concrete attempt inputs after a worker is reserved."""
        ...

    def on_attempt_prepared(
        self,
        event: BenchmarkPrepareSucceededEvent,
    ) -> None:
        """Consume prepare/reset success before evaluate is submitted."""
        ...

    def on_terminal_event(
        self,
        event: BenchmarkTerminalEvent,
    ) -> None:
        """Consume a terminal prepare/evaluate event."""
        ...

    def result(self) -> BenchmarkResult:
        """Return the final benchmark result after all work is finished."""
        ...
