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

"""Run-owned traces independent of the agent's replaceable working history."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, ImageUrl

from roborsi.evaluation.result import write_json_atomic
from roborsi.runtime.events import RuntimeEvent, RuntimeEventSink
from roborsi.utils.logging import get_logger

logger = get_logger("runtime.trace")
_SECRET_KEYS = frozenset(
    {"api_key", "authorization", "password", "secret", "access_token"}
)


@dataclass(frozen=True)
class TraceConfig:
    """Choose full content, metadata only, or disabled trace collection."""

    mode: Literal["full", "metadata", "off"] = "full"
    capture_video: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"full", "metadata", "off"}:
            raise ValueError("trace mode must be full, metadata, or off")
        if not isinstance(self.capture_video, bool):
            raise TypeError("capture_video must be a boolean")


@dataclass(frozen=True)
class TraceScope:
    """Task-local identity inherited by model calls and explicitly scoped tools."""

    agent_id: str | None = None
    parent_agent_id: str | None = None
    turn_id: str | None = None
    tool_call_id: str | None = None
    purpose: str = "agent"


_recorder: ContextVar[TraceRecorder | None] = ContextVar("roborsi_trace", default=None)
_scope: ContextVar[TraceScope] = ContextVar("roborsi_trace_scope", default=TraceScope())


def current_trace() -> TraceRecorder | None:
    """Return the recorder active in this asynchronous task or copied thread context."""
    return _recorder.get()


def current_trace_scope() -> TraceScope:
    """Return the task-local agent, turn and purpose identifiers."""
    return _scope.get()


@contextmanager
def trace_scope(**values: Any) -> Iterator[TraceScope]:
    """Temporarily override identifiers or request purpose in the current task."""
    scope = replace(_scope.get(), **values)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def exception_status(error: BaseException) -> str:
    """Classify terminal outcomes without confusing cancellation with success."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    from roborsi.planner.base import PlannerFailure

    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, PlannerFailure) and error.kind == "cancelled":
        return "cancelled"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, UsageLimitExceeded):
        return "limit"
    return "error"


def reported_usage(response: Any) -> dict[str, Any] | None:
    """Keep reported counters; SDK class defaults are unknown, not reported zero."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    # RequestUsage deliberately only puts explicitly supplied fields in __dict__.
    # The provider may also assign counters as streamed usage arrives.
    supplied = vars(usage)
    names = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    if not any(name in supplied for name in names):
        return None
    return {name: supplied.get(name) for name in names}


class TraceRecorder:
    """Persist and fan out one run's events without interrupting resource cleanup.

    Args:
        output_dir: Existing run directory; all trace references are relative to it.
        config: Collection mode and media preference.
        run_id: Optional stable run identifier.
        sinks: Live event consumers. Their failures mark the trace incomplete.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: TraceConfig | None = None,
        *,
        run_id: str | None = None,
        sinks: Sequence[RuntimeEventSink] = (),
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.config = config or TraceConfig()
        self.run_id = run_id or uuid.uuid4().hex
        self.trace_dir = self.output_dir / "trace"
        self._started = time.monotonic()
        self._lock = threading.RLock()
        self._seq = 0
        self._closed = False
        self._sinks = tuple(sinks)
        self._errors: list[str] = []
        self._media: dict[str, dict[str, Any]] = {}
        self._open_contexts: set[str] = set()
        self._agent_stops: dict[str, tuple[str, str]] = {}
        self._terminal_status = "completed"
        self._terminal_reason: str | None = None
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "mode": self.config.mode,
            "capture_video": self.config.capture_video,
            "status": "running",
            "complete": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.config.mode != "off":
            # Refuse to mix two run identities, including when a previous run
            # was interrupted. The owner can keep/reuse its live recorder.
            if (self.trace_dir / "events.jsonl").exists():
                raise FileExistsError(f"trace already exists: {self.trace_dir}")
            try:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self._failed(error)
            self.emit("run_start", {"mode": self.config.mode})
            self._write_manifest()

    @property
    def enabled(self) -> bool:
        """Whether this recorder collects any events."""
        return self.config.mode != "off"

    @property
    def terminal_status(self) -> str:
        """Return the last root invocation's observed or explicitly requested outcome."""
        return self._terminal_status

    def mark_agent_stop(self, sdk_run_id: str, *, status: str, reason: str) -> None:
        """Declare why the owning planner intentionally leaves an SDK iterator."""
        with self._lock:
            self._agent_stops[sdk_run_id] = (status, reason)

    def mark_run_stop(self, *, status: str, reason: str) -> None:
        """Record a session stop between agent invocations, such as terminal EOF."""
        with self._lock:
            self._terminal_status, self._terminal_reason = status, reason

    def agent_outcome(
        self, sdk_run_id: str, parent_agent_id: str | None, status: str
    ) -> tuple[str, str | None]:
        """Resolve iterator cleanup against an explicit stop, preserving real failures."""
        with self._lock:
            declared = self._agent_stops.pop(sdk_run_id, None)
            reason = None
            # SDK iterator teardown raises CancelledError even for a deliberate
            # finish/limit boundary. Only that known boundary overrides it.
            if declared is not None and status in {"completed", "cancelled"}:
                status, reason = declared
            if parent_agent_id is None:
                self._terminal_status, self._terminal_reason = status, reason
            return status, reason

    @contextmanager
    def activate(self) -> Iterator[TraceRecorder]:
        """Make this run available to nested agents and model wrappers."""
        token = _recorder.set(self if self.enabled else None)
        scope_token = _scope.set(TraceScope())
        try:
            yield self
        finally:
            _scope.reset(scope_token)
            _recorder.reset(token)

    def _failed(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        self._errors.append(message)
        logger.warning("trace recording failed: %s", message)

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._manifest.update(
                last_seq=self._seq,
                media=list(self._media.values()),
                errors=list(self._errors),
            )
            if self._errors:
                self._manifest.update(status="incomplete", complete=False)
            try:
                write_json_atomic(self.trace_dir / "manifest.json", self._manifest)
            except OSError as error:
                self._failed(error)

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        **identifiers: Any,
    ) -> RuntimeEvent | None:
        """Append one canonical event and notify sinks under the same sequence lock."""
        if not self.enabled or self._closed:
            return None
        scope = _scope.get()
        ids = {
            key: getattr(scope, key)
            for key in ("agent_id", "parent_agent_id", "turn_id", "tool_call_id")
        }
        ids.update(identifiers)
        with self._lock:
            self._seq += 1
            event = RuntimeEvent(
                schema_version=1,
                event_id=uuid.uuid4().hex,
                seq=self._seq,
                type=event_type,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_s=round(time.monotonic() - self._started, 6),
                run_id=self.run_id,
                payload=dict(payload or {}),
                **ids,
            )
            if event.turn_id is not None:
                if event_type == "context_start":
                    self._open_contexts.add(event.turn_id)
                elif event_type == "context_end":
                    self._open_contexts.discard(event.turn_id)
            try:
                with (self.trace_dir / "events.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError) as error:
                self._failed(error)
            # Persistence is visible before live callbacks. A callback may
            # immediately read the manifest/index from a different thread.
            if event_type in {"artifact_written", "artifact_failed"}:
                ref = str(event.payload.get("artifact_ref", event.event_id))
                self._media[ref] = {
                    **event.payload,
                    "status": "ready" if event_type == "artifact_written" else "failed",
                }
                self._write_manifest()
            elif event_type in {"run_start", "run_end"}:
                self._write_manifest()
            error_count = len(self._errors)
            for sink in self._sinks:
                try:
                    sink.emit(event)
                except Exception as error:  # noqa: BLE001 - observability cannot own action cleanup
                    self._failed(error)
            if len(self._errors) != error_count:
                self._write_manifest()
            if event_type.endswith(("_start", "_end", "_error", "_failed")):
                log = (
                    logger.info
                    if event_type.startswith(("run_", "agent_", "turn_"))
                    else logger.debug
                )
                log(
                    "runtime event %s run=%s agent=%s turn=%s status=%s",
                    event_type,
                    self.run_id,
                    event.agent_id,
                    event.turn_id,
                    event.payload.get("status"),
                )
            return event

    def finish_context(self, status: str, **identifiers: Any) -> None:
        """Close a context phase interrupted before the model wrapper was reached."""
        with self._lock:
            if identifiers.get("turn_id") in self._open_contexts:
                self.emit("context_end", {"status": status}, **identifiers)

    def _binary(
        self, data: bytes, media_type: str = "application/octet-stream"
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        suffix = mimetypes.guess_extension(media_type) or ""
        path = self.trace_dir / "content" / f"{digest}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = {
            "artifact_ref": path.relative_to(self.output_dir).as_posix(),
            "media_type": media_type,
            "sha256": digest,
            "size_bytes": len(data),
        }
        if not path.exists():
            path.write_bytes(data)
            self.emit("artifact_written", reference)
        return reference

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, BinaryContent):
            return self._binary(value.data, value.media_type)
        if isinstance(value, ImageUrl) and value.url.startswith("data:"):
            header, encoded = value.url.split(",", 1)
            if ";base64" in header:
                return self._binary(base64.b64decode(encoded), header[5:].split(";")[0])
        if isinstance(value, bytes):
            return self._binary(value)
        if isinstance(value, Mapping):
            return {
                str(key): "[redacted]"
                if str(key).lower() in _SECRET_KEYS
                else self._serialize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._serialize(item) for item in value]
        if isinstance(value, Enum):
            return self._serialize(value.value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: self._serialize(getattr(value, field.name))
                for field in fields(value)
                if not field.name.startswith("_")
            }
        if hasattr(value, "model_dump"):
            return self._serialize(value.model_dump(mode="python"))
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Do not serialize arbitrary object reprs that can embed clients or credentials.
        return {"type": type(value).__name__}

    def snapshot(self, category: str, name: str, value: Any) -> str | None:
        """Write full normalized content and return a run-relative JSON reference."""
        if self.config.mode != "full" or self._closed:
            return None
        if (
            category not in {"requests", "responses", "messages", "tools", "chunks"}
            or not name.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("invalid trace snapshot name")
        with self._lock:
            try:
                path = self.trace_dir / category / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                normalized = self._serialize(value)
                write_json_atomic(
                    path,
                    normalized
                    if isinstance(normalized, dict)
                    else {"content": normalized},
                )
                return path.relative_to(self.output_dir).as_posix()
            except (OSError, TypeError, ValueError) as error:
                self._failed(error)
                return None

    def receive_message(
        self, message: Any, *, seen: set[str], **identifiers: Any
    ) -> None:
        """Persist newly observed raw history while containing all recording errors."""
        reference = None
        if self.config.mode == "full":
            with self._lock:
                try:
                    normalized = self._serialize(message)
                    digest = hashlib.sha256(
                        json.dumps(normalized, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    if digest in seen:
                        return
                    seen.add(digest)
                    reference = self.snapshot("messages", digest, normalized)
                except (OSError, TypeError, ValueError) as error:
                    self._failed(error)
        self.emit(
            "message_received",
            {"source": "input", "message_ref": reference},
            **identifiers,
        )

    def record_artifact(
        self, path: str | Path, *, media_type: str | None = None, **metadata: Any
    ) -> None:
        """Publish a completed media artifact already inside the run directory."""
        resolved = Path(path).resolve()
        try:
            reference = resolved.relative_to(self.output_dir).as_posix()
        except ValueError:
            raise ValueError("artifact must be inside the run directory") from None
        if not resolved.is_file():
            self.emit(
                "artifact_failed",
                {"artifact_ref": reference, "error": "file_missing", **metadata},
            )
            return
        self.emit(
            "artifact_written",
            {
                "artifact_ref": reference,
                "media_type": media_type
                or mimetypes.guess_type(resolved.name)[0]
                or "application/octet-stream",
                **metadata,
            },
        )

    def close(
        self, status: str = "completed", error: BaseException | str | None = None
    ) -> None:
        """Finalize after tool/video cleanup; repeated calls have no effect."""
        if self._closed:
            return
        with self._lock:
            self._manifest.update(
                status=status,
                complete=True,
                ended_at=datetime.now(timezone.utc).isoformat(),
            )
            self.emit(
                "run_end",
                {
                    "status": status,
                    "error_type": type(error).__name__ if error is not None else None,
                    "reason": self._terminal_reason,
                },
                agent_id=None,
                parent_agent_id=None,
                turn_id=None,
                tool_call_id=None,
            )
            self._closed = True


class RuntimeTraceCapability(AbstractCapability):
    """Capture each invocation before history processing, including child agents."""

    def __init__(self) -> None:
        super().__init__()
        self._agent_id: str | None = None
        self._parent_id: str | None = None
        self._turn_id: str | None = None
        self._seen: set[str] = set()

    async def for_run(self, ctx: Any) -> RuntimeTraceCapability:
        return RuntimeTraceCapability()

    def _ids(self) -> dict[str, str | None]:
        return {
            "agent_id": self._agent_id,
            "parent_agent_id": self._parent_id,
            "turn_id": self._turn_id,
        }

    def _end_turn(
        self, recorder: TraceRecorder, status: str, reason: str | None = None
    ) -> None:
        if self._turn_id is not None:
            recorder.finish_context(status, **self._ids())
            recorder.emit(
                "turn_end", {"status": status, "reason": reason}, **self._ids()
            )
            self._turn_id = None

    async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
        recorder = current_trace()
        if recorder is None:
            return await handler()
        self._agent_id = uuid.uuid4().hex
        self._parent_id = current_trace_scope().agent_id
        status = "completed"
        error_type = None
        with trace_scope(
            agent_id=self._agent_id,
            parent_agent_id=self._parent_id,
            turn_id=None,
            tool_call_id=None,
        ):
            recorder.emit(
                "agent_start",
                {
                    "name": getattr(ctx.agent, "name", None) or "root",
                    "model": ctx.model.model_name,
                    "purpose": current_trace_scope().purpose,
                },
            )
            try:
                return await handler()
            except BaseException as error:
                status = exception_status(error)
                error_type = type(error).__name__
                raise
            finally:
                status, reason = recorder.agent_outcome(
                    ctx.run_id, self._parent_id, status
                )
                if error_type == "CancelledError" and status != "cancelled":
                    error_type = None
                self._end_turn(recorder, status, reason)
                recorder.emit(
                    "agent_end",
                    {"status": status, "reason": reason, "error_type": error_type},
                    **self._ids(),
                )

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        recorder = current_trace()
        if recorder is None:
            return request_context
        self._end_turn(recorder, "completed")
        self._turn_id = uuid.uuid4().hex
        _scope.set(replace(current_trace_scope(), **self._ids(), tool_call_id=None))
        recorder.emit("turn_start", {"run_step": ctx.run_step}, **self._ids())
        # Content-addressed identities retain new user/tool input while avoiding
        # repeated historical snapshots after SDK history replacement.
        for message in request_context.messages:
            recorder.receive_message(message, seen=self._seen, **self._ids())
        recorder.emit(
            "context_start",
            {"message_count": len(request_context.messages)},
            **self._ids(),
        )
        return request_context

    async def wrap_tool_execute(
        self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any
    ) -> Any:
        recorder = current_trace()
        if recorder is None:
            return await handler(args)
        with trace_scope(**self._ids(), tool_call_id=call.tool_call_id):
            name = uuid.uuid4().hex
            arguments_ref = recorder.snapshot("tools", f"{name}-arguments", args)
            recorder.emit(
                "tool_start", {"name": call.tool_name, "arguments_ref": arguments_ref}
            )
            status = "completed"
            result_ref = None
            error_type = None
            try:
                result = await handler(args)
                result_ref = recorder.snapshot("tools", f"{name}-result", result)
                recorder.emit(
                    "message_received", {"source": "tool", "message_ref": result_ref}
                )
                return result
            except BaseException as error:
                status = exception_status(error)
                error_type = type(error).__name__
                raise
            finally:
                recorder.emit(
                    "tool_end",
                    {
                        "name": call.tool_name,
                        "status": status,
                        "result_ref": result_ref,
                        "error_type": error_type,
                    },
                )
