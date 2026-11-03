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

"""Runner-owned RoboRSI interaction records using the shared runtime event format."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    InteractionMessage,
    MediaRef,
    NativeResult,
    Observation,
    Provenance,
    RoboRSIMessage,
    ServiceError,
    StartResult,
    StepResult,
    Task,
    from_envelope,
    to_envelope,
)
from roborsi.evaluation.protocol import InteractionEnvelope
from roborsi.evaluation.result import write_json_atomic
from roborsi.runtime.events import RuntimeEvent

TRACE_MANIFEST = "runner_trace/manifest.json"


class CaseTraceManifest(RoboRSIMessage):
    """Identity and checksum for a complete or interrupted case recording."""

    format: Literal["roborsi_case_trace"] = "roborsi_case_trace"
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    task: Task
    provenance: Provenance
    status: Literal[
        "running",
        "incomplete",
        "success",
        "task_failure",
        "agent_error",
        "infrastructure_error",
        "cancelled",
    ]
    complete: bool
    last_seq: int = Field(ge=0)
    events: MediaRef | None = None
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _completion(self) -> CaseTraceManifest:
        if self.complete and (
            self.events is None
            or self.errors
            or self.status in {"running", "incomplete"}
        ):
            raise ValueError(
                "complete trace needs an event file and no recording errors"
            )
        if self.events is not None and self.events.mime_type != "application/x-ndjson":
            raise ValueError("trace event file must be JSONL")
        return self


class CaseTrace:
    """Append accepted protocol messages without copying their camera bytes.

    A case owns one recorder. Recording errors stop further evaluation, while
    the caller still owns service cleanup and publication of an error result.
    """

    def __init__(
        self,
        directory: Path,
        run_id: str,
        case_id: str,
        episode_id: str,
        task: Task,
        provenance: Provenance,
    ) -> None:
        self.directory = directory
        self._trace_dir = directory / "runner_trace"
        self._trace_dir.mkdir(exist_ok=False)
        self._events = self._trace_dir / "events.jsonl"
        self._events.touch(mode=0o600, exist_ok=False)
        self._started = time.monotonic()
        self._step: int | None = None
        self._closed = False
        self._seq = 0
        self._errors: list[str] = []
        self._manifest = CaseTraceManifest(
            run_id=run_id,
            case_id=case_id,
            episode_id=episode_id,
            task=task,
            provenance=provenance,
            status="running",
            complete=False,
            last_seq=0,
        )
        self.emit(
            "run_start", {"task": task.as_dict(), "provenance": provenance.as_dict()}
        )
        self._write_manifest("running", complete=False)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Durably append a JSON event; retain write errors for the manifest."""
        if self._closed:
            raise RuntimeError("case trace is closed")
        event = RuntimeEvent(
            schema_version=1,
            event_id=uuid.uuid4().hex,
            seq=self._seq + 1,
            type=event_type,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.monotonic() - self._started,
            run_id=self._manifest.run_id,
            episode_id=self._manifest.episode_id,
            turn_id=f"step-{self._step}" if self._step is not None else None,
            step_idx=self._step,
            payload=payload,
        )
        try:
            data = (
                json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            ).encode()
            with self._events.open("ab") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, ValueError, TypeError) as exc:
            self._errors.append(f"{type(exc).__name__}: {exc}")
            raise
        self._seq += 1

    def record(self, message: InteractionMessage, *, stage: str) -> None:
        """Save one validated message accepted by the Runner."""
        envelope = to_envelope(
            message,
            benchmark=self._manifest.task.benchmark,
            episode_id=self._manifest.episode_id,
        )
        self.emit("roborsi_message", {"stage": stage, "message": envelope.as_dict()})
        observation = (
            message.observation
            if isinstance(message, (StartResult, StepResult))
            else message
            if isinstance(message, Observation)
            else None
        )
        if observation is not None:
            for media in observation.media:
                self.emit(
                    "artifact_written",
                    {
                        "artifact_ref": media.ref,
                        "media_type": media.mime_type,
                        "sha256": media.sha256,
                        "size_bytes": media.size_bytes,
                        "step_idx": observation.step_index,
                    },
                )

    def begin_step(self, observation: Observation) -> None:
        """Associate the input observation and following action with one turn."""
        self._step = observation.step_index
        self.emit("turn_start", {"observation": observation.as_dict()})

    def end_step(self, status: str = "completed") -> None:
        if self._step is not None:
            self.emit("turn_end", {"status": status})
            self._step = None

    def _write_manifest(self, status: str, *, complete: bool) -> None:
        data = self._events.read_bytes()
        self._manifest = self._manifest.model_copy(
            update={
                "status": status,
                "complete": complete,
                "last_seq": self._seq,
                "events": MediaRef(
                    ref="runner_trace/events.jsonl",
                    mime_type="application/x-ndjson",
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                ),
                "errors": tuple(self._errors),
            }
        )
        write_json_atomic(self._trace_dir / "manifest.json", self._manifest.as_dict())

    def finish(self, status: str, error: ServiceError | None = None) -> str:
        """Finalize after service cleanup, including an error's stage and reason."""
        if self._closed:
            return TRACE_MANIFEST
        self.end_step(status)
        if error is not None:
            self.record(error, stage=error.details.get("stage", "case.run"))
        self.emit("run_end", {"status": status})
        self._write_manifest(
            "incomplete" if self._errors else status, complete=not self._errors
        )
        self._closed = True
        return TRACE_MANIFEST


def read_case_trace(
    directory: Path,
    reference: str,
    *,
    task: Task,
    provenance: Provenance,
    episode_id: str,
) -> tuple[CaseTraceManifest, list[dict[str, Any]]]:
    """Check file bytes, event identity and every recorded observation's media."""
    root = directory.resolve()

    def read_file(name: str) -> bytes:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"trace file unavailable or outside case: {name}")
        return path.read_bytes()

    manifest = CaseTraceManifest.model_validate_json(read_file(reference))
    if (manifest.task, manifest.provenance, manifest.episode_id) != (
        task,
        provenance,
        episode_id,
    ):
        raise ValueError("trace identity differs from result")
    if manifest.case_id != directory.name or manifest.run_id != directory.parent.name:
        raise ValueError("trace run/case differs from its directory")
    if manifest.events is None:
        raise ValueError("trace event file is missing")
    if manifest.events.ref != (Path(reference).parent / "events.jsonl").as_posix():
        raise ValueError("trace event file must belong to its manifest directory")
    data = read_file(manifest.events.ref)
    if manifest.complete and not data.endswith(b"\n"):
        raise ValueError("complete trace has an unfinished final event")
    if (
        len(data) != manifest.events.size_bytes
        or hashlib.sha256(data).hexdigest() != manifest.events.sha256
    ):
        raise ValueError("trace event file checksum mismatch")
    events = []
    channel: CaseMediaChannel | None = None

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard trace JSON constant: {value}")

    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n") and not manifest.complete:
            break
        event = json.loads(line, parse_constant=reject_constant)
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("type"), str)
            or not isinstance(event.get("payload"), dict)
            or event.get("schema_version") != 1
            or event.get("seq") != len(events) + 1
            or event.get("run_id") != manifest.run_id
            or event.get("episode_id") != episode_id
        ):
            raise ValueError("trace event version, identity or order differs")
        if event["type"] == "heart_message":
            raise ValueError("unsupported trace event type: heart_message")
        if event.get("type") == "roborsi_message":
            wire = event["payload"].get("message")
            if not isinstance(wire, dict):
                raise ValueError("trace event needs a protocol message")
            envelope = InteractionEnvelope.from_dict(wire)
            if (envelope.benchmark, envelope.episode_id) != (
                task.benchmark,
                episode_id,
            ):
                raise ValueError("trace message identity differs from result")
            message = from_envelope(envelope)
            actual_task = (
                message.task
                if isinstance(message, (StartResult, NativeResult))
                else message
                if isinstance(message, Task)
                else None
            )
            if actual_task is not None and actual_task != task:
                raise ValueError("trace message task differs from result")
            observation = (
                message.observation
                if isinstance(message, (StartResult, StepResult))
                else message
                if isinstance(message, Observation)
                else None
            )
            if observation is not None and observation.media:
                if channel is None:
                    channel = CaseMediaChannel(root / "media")
                channel.verify_observation(observation)
        events.append(event)
    if len(events) != manifest.last_seq:
        raise ValueError("trace event count differs from manifest")
    if (
        not events
        or events[0]["type"] != "run_start"
        or events[0]["payload"]
        != {"task": task.as_dict(), "provenance": provenance.as_dict()}
    ):
        raise ValueError("trace must start with the claimed task and versions")
    if manifest.complete and (not events or events[-1]["type"] != "run_end"):
        raise ValueError("complete trace needs a terminal event")
    if manifest.complete and events[-1]["payload"].get("status") != manifest.status:
        raise ValueError("trace terminal event differs from manifest status")
    return manifest, events
