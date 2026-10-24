# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned RoboRSI interaction messages and their JSON Schema contracts."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from roborsi.evaluation.protocol import InteractionEnvelope, MessageKind

MessageT = TypeVar("MessageT", bound="RoboRSIMessage")


def _case_relative_ref(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("artifact ref must be a case-relative path")
    return value


class RoboRSIMessage(BaseModel):
    """Strict JSON boundary shared by the Agent and Benchmark services."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def _json_boundary(self) -> RoboRSIMessage:
        self.as_dict()
        return self

    def as_dict(self) -> dict[str, Any]:
        """Return detached, finite JSON; reject a mutated nested payload."""
        document = self.model_dump(mode="python")
        try:
            return json.loads(json.dumps(document, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RoboRSI message must contain only finite JSON data"
            ) from exc

    @classmethod
    def from_dict(cls: type[MessageT], value: dict[str, Any]) -> MessageT:
        """Validate a detached JSON object using the wire representation."""
        if not isinstance(value, dict):
            raise TypeError("RoboRSI message must be a JSON object")
        return cls.model_validate_json(json.dumps(value, allow_nan=False))


class ArtifactInfo(RoboRSIMessage):
    """Immutable public metadata; ref is an archive name, never a host path."""

    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    kind: Literal["image", "native_result", "log", "file"]
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0, le=256 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_index: int | None = Field(default=None, ge=0)

    @field_validator("ref")
    @classmethod
    def _safe_ref(cls, value: str) -> str:
        value = _case_relative_ref(value)
        if value.split("/")[0] not in {"media", "native", "trace"}:
            raise ValueError("artifact must use media, native or trace archive names")
        return value


class Task(RoboRSIMessage):
    """One requested native benchmark task and exact nonnegative seed."""

    benchmark: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    instruction: str = Field(min_length=1)
    max_steps: int = Field(gt=0)
    native: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class MediaRef(RoboRSIMessage):
    """Case-relative reference to a separately transferred media artifact."""

    ref: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("ref")
    @classmethod
    def _safe_ref(cls, value: str) -> str:
        return _case_relative_ref(value)


class Observation(RoboRSIMessage):
    """One observation, with large camera payloads referenced separately."""

    benchmark: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    data: dict[str, Any] = Field(default_factory=dict)
    media: tuple[MediaRef, ...] = ()


class StartResult(RoboRSIMessage):
    """Actual task, capability set and first observation returned by Adapter."""

    task: Task
    observation: Observation
    capabilities: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_initial_observation(self) -> StartResult:
        if (
            self.task.benchmark != self.observation.benchmark
            or self.observation.step_index != 0
        ):
            raise ValueError("start result needs a matching step-zero observation")
        return self


class Action(RoboRSIMessage):
    """Native action payload chosen for one observation step."""

    benchmark: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    data: dict[str, Any]


class BenchmarkQuery(RoboRSIMessage):
    """Request a benchmark-owned operation that does not advance the scene."""

    benchmark: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    query_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class QueryResult(RoboRSIMessage):
    """Reply to one query at the unchanged native observation step."""

    benchmark: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    query_id: str = Field(min_length=1)
    data: dict[str, Any]


class EpisodeStop(RoboRSIMessage):
    """Request native termination at the current observation, without a step.

    The benchmark decides the outcome and score. Policies submit any final
    physical action separately; operational failures use cancellation instead.
    """

    benchmark: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    reason: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class StepResult(RoboRSIMessage):
    """Native feedback after an action or an explicit zero-step termination."""

    observation: Observation
    terminated: bool
    truncated: bool
    reward: float | None = None
    termination_reason: str | None = None
    native: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_termination(self) -> StepResult:
        if (self.terminated or self.truncated) != (self.termination_reason is not None):
            raise ValueError("termination_reason must be set exactly when episode ends")
        if self.termination_reason == "":
            raise ValueError("termination_reason must be non-empty")
        return self


class NativeResult(RoboRSIMessage):
    """Benchmark-owned score before Runner adds provenance and error status."""

    task: Task
    episode_id: str = Field(min_length=1)
    success: bool
    score: float
    details: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("source")
    @classmethod
    def _safe_source(cls, value: str | None) -> str | None:
        return _case_relative_ref(value) if value is not None else None

    @model_validator(mode="after")
    def _check_source(self) -> NativeResult:
        if (self.source is None) != (self.source_sha256 is None):
            raise ValueError("native source and digest must appear together")
        return self


class ServiceError(RoboRSIMessage):
    """Machine-readable failure independent of a benchmark's native error."""

    code: Literal[
        "invalid_request",
        "unsupported_version",
        "agent_error",
        "adapter_error",
        "infrastructure_error",
        "timeout",
        "cancelled",
    ]
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class Provenance(RoboRSIMessage):
    """Recorded immutable input versions; unknown values remain explicit."""

    benchmark_version: str | None = None
    benchmark_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    docker_image_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    assets_version: str | None = None
    adapter_version: str | None = None
    agent_version: str | None = None

    @field_validator(
        "benchmark_version",
        "benchmark_commit",
        "docker_image_digest",
        "assets_version",
        "adapter_version",
        "agent_version",
    )
    @classmethod
    def _nonempty_version(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("version source must be non-empty or null")
        return value


class BenchmarkRuntimeInfo(RoboRSIMessage):
    """Versions and scoring rules read by the Adapter inside its runtime."""

    benchmark: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    benchmark_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_version: str = Field(min_length=1)
    evaluation: dict[str, Any]
    platform: dict[str, str]


class AgentRuntimeRecord(RoboRSIMessage):
    """A saved Agent environment checked by its owning launcher."""

    agent_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    python: dict[str, Any]
    installed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_after_close: bool = False


class ContainerAgentRuntimeRecord(RoboRSIMessage):
    """Agent image and launch settings checked by the owning Docker launcher."""

    agent_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    docker_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    factory: str = Field(min_length=1)
    configuration: dict[str, Any]
    verified_after_close: bool = False


class BenchmarkRuntimeRecord(RoboRSIMessage):
    """Adapter information, checked assets and available deployment evidence."""

    info: BenchmarkRuntimeInfo | None = None
    docker_image_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    assets_version: str | None = Field(default=None, min_length=1)
    assets: tuple[dict[str, Any], ...]
    configuration: dict[str, Any]
    verified_after_close: bool = False


class RuntimeConditions(RoboRSIMessage):
    """Runner settings that must agree when comparing two scored cases."""

    timeout_s: float = Field(gt=0)
    case_timeout_s: float = Field(gt=0)
    evaluation: dict[str, Any]


class RuntimeVerification(RoboRSIMessage):
    """Observed versions; missing or unchecked owners leave explicit nulls."""

    conditions: RuntimeConditions
    agent: AgentRuntimeRecord | ContainerAgentRuntimeRecord | None = None
    benchmark: BenchmarkRuntimeRecord | None = None
    observed: Provenance

    @staticmethod
    def observed_versions(
        agent: AgentRuntimeRecord | ContainerAgentRuntimeRecord | None,
        benchmark: BenchmarkRuntimeRecord | None,
    ) -> Provenance:
        values: dict[str, Any] = {}
        if agent is not None and agent.verified_after_close:
            values["agent_version"] = agent.agent_version
        if benchmark is not None and benchmark.verified_after_close:
            values.update(
                docker_image_digest=benchmark.docker_image_digest,
                assets_version=benchmark.assets_version,
            )
            if benchmark.info is not None:
                for field in (
                    "benchmark_version",
                    "benchmark_commit",
                    "adapter_version",
                ):
                    values[field] = getattr(benchmark.info, field)
        return Provenance(**values)

    @model_validator(mode="after")
    def _check_observed(self) -> RuntimeVerification:
        if self.observed != self.observed_versions(self.agent, self.benchmark):
            raise ValueError("observed versions differ from runtime checks")
        return self

    def verify_requested(self, task: Task, provenance: Provenance) -> None:
        """Refuse an observed mismatch while preserving explicitly unknown data."""
        for field, actual in self.observed.as_dict().items():
            if field != "schema_version" and actual is not None:
                if actual != getattr(provenance, field):
                    raise ValueError(f"runtime {field} differs from requested version")
        if self.benchmark is not None and self.benchmark.info is not None:
            info = self.benchmark.info
            if info.benchmark != task.benchmark:
                raise ValueError("runtime Benchmark differs from requested task")
            if info.evaluation != self.conditions.evaluation:
                raise ValueError("runtime scoring rules differ from requested rules")


class EvaluationResult(RoboRSIMessage):
    """Per-case result with distinct task and operational failure states."""

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "success"}}},
                    "then": {
                        "properties": {
                            "success": {"const": True},
                            "score": {"type": "number"},
                            "error": {"type": "null"},
                        }
                    },
                },
                {
                    "if": {"properties": {"status": {"const": "task_failure"}}},
                    "then": {
                        "properties": {
                            "success": {"const": False},
                            "score": {"type": "number"},
                            "error": {"type": "null"},
                        }
                    },
                },
                {
                    "if": {
                        "properties": {
                            "status": {
                                "enum": [
                                    "agent_error",
                                    "infrastructure_error",
                                    "cancelled",
                                ]
                            }
                        }
                    },
                    "then": {
                        "properties": {
                            "success": {"type": "null"},
                            "score": {"type": "null"},
                            "error": {"type": "object"},
                        },
                        "required": ["error"],
                    },
                },
            ]
        }
    )

    task: Task
    episode_id: str = Field(min_length=1)
    status: Literal[
        "success", "task_failure", "agent_error", "infrastructure_error", "cancelled"
    ]
    success: bool | None
    score: float | None
    native_score_source: str | None = None
    native_score_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    trace_manifest: str | None = None
    provenance: Provenance
    runtime: RuntimeVerification | None = None
    benchmark_artifacts: tuple[ArtifactInfo, ...] = ()
    error: ServiceError | None = None

    @field_validator("native_score_source", "trace_manifest")
    @classmethod
    def _safe_artifact_ref(cls, value: str | None) -> str | None:
        return _case_relative_ref(value) if value is not None else None

    @model_validator(mode="after")
    def _check_outcome(self) -> EvaluationResult:
        if self.status in {"success", "task_failure"}:
            if self.success is not (self.status == "success") or self.score is None:
                raise ValueError("completed result needs native success and score")
            if self.error is not None:
                raise ValueError("completed result cannot contain an error")
            if self.runtime is not None:
                self.runtime.verify_requested(self.task, self.provenance)
        elif self.success is not None or self.score is not None or self.error is None:
            raise ValueError("operational error needs error details and unknown score")
        if self.error is not None and self.status == "agent_error":
            if self.error.code not in {"agent_error", "timeout"}:
                raise ValueError("agent_error status needs an agent or timeout error")
        if self.error is not None and self.status == "cancelled":
            if self.error.code != "cancelled":
                raise ValueError("cancelled status needs a cancelled error")
        if (self.native_score_source is None) != (self.native_score_sha256 is None):
            raise ValueError("native score reference and digest must appear together")
        return self


InteractionMessage: TypeAlias = (
    Task
    | StartResult
    | Observation
    | Action
    | EpisodeStop
    | BenchmarkQuery
    | QueryResult
    | StepResult
    | NativeResult
    | EvaluationResult
    | ServiceError
)
_MESSAGE_TYPES: dict[str, type[RoboRSIMessage]] = {
    "task": Task,
    "start_result": StartResult,
    "observation": Observation,
    "action": Action,
    "episode_stop": EpisodeStop,
    "benchmark_query": BenchmarkQuery,
    "query_result": QueryResult,
    "step_result": StepResult,
    "native_result": NativeResult,
    "evaluation_result": EvaluationResult,
    "error": ServiceError,
}


def message_schema(kind: MessageKind) -> dict[str, Any]:
    """Return a detached JSON Schema for a version 1 message payload."""
    model = _MESSAGE_TYPES.get(kind)
    if model is None:
        raise ValueError(f"unknown message kind: {kind}")
    return model.model_json_schema()


def to_envelope(
    message: InteractionMessage,
    *,
    benchmark: str,
    episode_id: str,
    request_id: str | None = None,
) -> InteractionEnvelope:
    """Serialize a typed payload and verify its declared episode identity."""
    kind = next(
        (name for name, model in _MESSAGE_TYPES.items() if type(message) is model),
        None,
    )
    if kind is None:
        raise TypeError("unsupported RoboRSI message type")
    _check_identity(message, benchmark, episode_id)
    return InteractionEnvelope(
        kind, benchmark, episode_id, message.as_dict(), request_id=request_id
    )


def from_envelope(envelope: InteractionEnvelope) -> InteractionMessage:
    """Validate a wire envelope into its concrete message type."""
    model = _MESSAGE_TYPES[envelope.kind]
    message = model.from_dict(envelope.as_dict()["payload"])
    _check_identity(message, envelope.benchmark, envelope.episode_id)
    return message


def _check_identity(
    message: InteractionMessage, benchmark: str, episode_id: str
) -> None:
    if isinstance(message, Task):
        actual_benchmark, actual_episode = message.benchmark, episode_id
    elif isinstance(message, StartResult):
        actual_benchmark = message.task.benchmark
        actual_episode = message.observation.episode_id
    elif isinstance(message, StepResult):
        actual_benchmark = message.observation.benchmark
        actual_episode = message.observation.episode_id
    elif isinstance(message, NativeResult):
        actual_benchmark, actual_episode = message.task.benchmark, message.episode_id
    elif isinstance(message, EvaluationResult):
        actual_benchmark, actual_episode = message.task.benchmark, message.episode_id
    elif isinstance(message, ServiceError):
        actual_benchmark, actual_episode = benchmark, episode_id
    else:
        actual_benchmark, actual_episode = message.benchmark, message.episode_id
    if (actual_benchmark, actual_episode) != (benchmark, episode_id):
        raise ValueError("message identity does not match envelope")
