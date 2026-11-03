# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned runtime events shared by persistence and live consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RuntimeEvent:
    """One observed fact, ordered by the writer that owns its run."""

    schema_version: int
    event_id: str
    seq: int
    type: str
    timestamp_utc: str
    elapsed_s: float
    run_id: str
    agent_id: str | None = None
    parent_agent_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    attempt: int | None = None
    tool_call_id: str | None = None
    episode_id: str | None = None
    step_idx: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible public envelope."""
        return asdict(self)


class RuntimeEventSink(Protocol):
    """A synchronous consumer of a run's ordered event stream."""

    def emit(self, event: RuntimeEvent) -> None:
        """Consume an event without taking ownership of its run."""
