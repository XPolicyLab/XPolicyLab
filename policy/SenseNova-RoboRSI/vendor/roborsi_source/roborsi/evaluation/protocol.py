# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Versioned task identity and reproducibility fields for embodied evaluation."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any, Literal

RESULT_SCHEMA_VERSION = 1
INTERACTION_SCHEMA_VERSION = 1
MessageKind = Literal[
    "task",
    "start_result",
    "observation",
    "action",
    "episode_stop",
    "benchmark_query",
    "query_result",
    "step_result",
    "native_result",
    "evaluation_result",
    "error",
]
_KINDS = frozenset(
    (
        "task",
        "start_result",
        "observation",
        "action",
        "episode_stop",
        "benchmark_query",
        "query_result",
        "step_result",
        "native_result",
        "evaluation_result",
        "error",
    )
)


def interaction_envelope_schema() -> dict[str, Any]:
    """Return the version 1 wire envelope JSON Schema."""
    resource = files("roborsi.schemas").joinpath("roborsi_interaction_envelope.v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class InteractionEnvelope:
    """JSON boundary shared by an agent and a benchmark adapter.

    ``payload`` belongs to the benchmark or agent. The envelope validates
    transport shape and version; the receiving adapter validates native data.
    """

    kind: MessageKind
    benchmark: str
    episode_id: str
    payload: dict[str, Any]
    schema_version: int = 1
    request_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != INTERACTION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported interaction schema_version")
        if self.kind not in _KINDS:
            raise ValueError("unsupported interaction kind")
        if not isinstance(self.benchmark, str) or not self.benchmark:
            raise ValueError("benchmark must be non-empty text")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be non-empty text")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a JSON object")
        if self.request_id is not None and (
            not isinstance(self.request_id, str) or not self.request_id.strip()
        ):
            raise ValueError("request_id must be non-empty text")
        try:
            encoded = json.dumps(self.payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must contain only finite JSON data") from exc
        decoded = json.loads(encoded)
        if decoded != self.payload:
            raise ValueError("payload must use JSON object keys and values")
        object.__setattr__(self, "payload", decoded)

    def as_dict(self) -> dict[str, Any]:
        document = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "benchmark": self.benchmark,
            "episode_id": self.episode_id,
            "payload": copy.deepcopy(self.payload),
        }
        if self.request_id is not None:
            document["request_id"] = self.request_id
        return document

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InteractionEnvelope:
        """Reject malformed or unknown wire fields at the service boundary."""
        required = {"schema_version", "kind", "benchmark", "episode_id", "payload"}
        if (
            not isinstance(value, dict)
            or not required <= set(value)
            or set(value) - required - {"request_id"}
        ):
            raise ValueError("interaction envelope fields do not match schema")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Identify one native benchmark task and seed without changing its semantics."""

    benchmark: str
    task_id: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.benchmark, str)
            or not self.benchmark
            or not isinstance(self.task_id, str)
            or not self.task_id
        ):
            raise ValueError("benchmark and task_id must be non-empty")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Pinned input versions; ``None`` explicitly means the source is unknown."""

    benchmark_version: str | None = None
    benchmark_commit: str | None = None
    docker_image_digest: str | None = None
    assets_version: str | None = None
    adapter_version: str | None = None
    agent_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty text or None")

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)
