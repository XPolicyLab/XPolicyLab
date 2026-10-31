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

"""Bounded request retries and sanitized error records for LLM providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import BinaryContent, CachePoint, ImageUrl, ModelRequest
from pydantic_ai.models.wrapper import WrapperModel

from roborsi.utils.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings, RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

logger = get_logger("llm.retry")
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}
_SAFE_ERROR_VALUE = re.compile(r"[A-Za-z0-9_.:/ -]{1,128}\Z")
_SAFE_MESSAGE = re.compile(r"[A-Za-z0-9_.:/(),;= -]{1,300}\Z")
_DIAGNOSTIC_PREFIXES = (
    "bad request",
    "context",
    "gateway",
    "image",
    "internal",
    "invalid",
    "maximum",
    "model",
    "overload",
    "rate limit",
    "server",
    "timeout",
    "token",
    "too many",
    "upstream",
)
_SENSITIVE_BODY_KEY = re.compile(
    r"(?i)(?:authorization|api[_-]?key|password|secret|prompt(?:_text)?|"
    r"input(?:_text|_image|_body)?|messages|image(?:_url|_data|_bytes)?|"
    r"content(?:_text|_data)?|request_body)\Z"
)


def _redact_response_value(value: Any) -> Any:
    """Retain server diagnostics while removing echoed requests and secrets."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[redacted]"
                if _SENSITIVE_BODY_KEY.fullmatch(str(key))
                else _redact_response_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_response_value(item) for item in value]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", value)
        value = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted-key]", value)
        value = re.sub(r"data:image/[^\s]+", "[redacted-image]", value)
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _response_body_for_log(error: Exception) -> dict[str, Any]:
    """Describe every API failure, including an explicitly absent response."""
    body = getattr(error, "body", None)
    return {
        "available": body is not None,
        "body": _redact_response_value(body) if body is not None else None,
        "exception_message": (
            f"HTTP {error.status_code}"
            if isinstance(error, ModelHTTPError)
            else _redact_response_value(str(error))
        ),
    }


def _http_error_details(error: Exception | None) -> dict[str, Any] | None:
    """Keep provider diagnostics without writing raw response bodies."""
    if not isinstance(error, ModelHTTPError):
        return None
    body = error.body
    serialized = json.dumps(body, sort_keys=True, default=str)
    nested = body.get("error", body) if isinstance(body, dict) else body
    details: dict[str, Any] = {
        "body_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "body_type": type(body).__name__,
    }
    if isinstance(nested, dict):
        for field in ("code", "type", "param"):
            value = nested.get(field)
            if isinstance(value, (str, int)) and _SAFE_ERROR_VALUE.fullmatch(
                str(value)
            ):
                details[field] = str(value)
        message = nested.get("message", nested.get("msg"))
    else:
        message = nested
    if isinstance(message, str):
        normalized = message.strip()
        request_id = re.search(
            r"(?i)request[ _-]?id\s*[:=]\s*([A-Za-z0-9_-]{6,128})", normalized
        )
        if request_id:
            details["server_request_id"] = request_id.group(1)
        if (
            _SAFE_MESSAGE.fullmatch(normalized)
            and normalized.lower().startswith(_DIAGNOSTIC_PREFIXES)
            and not re.search(
                r"(?i)sk-[a-z0-9]|bearer|api.?key|password|secret|authorization|data:image|base64|prompt",
                normalized,
            )
        ):
            details["message"] = normalized
        else:
            details["message_redacted"] = True
    headers = error.headers or {}
    for name in (
        "x-request-id",
        "request-id",
        "x-ms-request-id",
        "x-correlation-id",
        "retry-after",
    ):
        value = next(
            (value for key, value in headers.items() if key.lower() == name), None
        )
        if isinstance(value, str) and _SAFE_ERROR_VALUE.fullmatch(value):
            details[name.replace("-", "_")] = value
    return details


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry transient provider failures; ``max_retries`` excludes the first call."""

    max_retries: int = 3
    initial_delay_s: float = 0.5
    max_delay_s: float = 4.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay_s < 0 or self.max_delay_s < self.initial_delay_s:
            raise ValueError("retry delays must be non-negative and ordered")

    def delay(self, attempt: int) -> float:
        """Return capped exponential backoff after the given failed attempt."""
        return min(self.initial_delay_s * 2 ** (attempt - 1), self.max_delay_s)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        return (
            exc.status_code in _RETRYABLE_HTTP_STATUSES or 500 <= exc.status_code < 600
        )
    return isinstance(exc, (ModelAPIError, ConnectionError, TimeoutError))


def _request_shape(
    messages: list[ModelMessage], parameters: ModelRequestParameters
) -> dict[str, int]:
    """Measure a request without retaining any text, image, or tool arguments."""
    shape = {
        "message_count": len(messages),
        "text_chars": 0,
        "image_count": 0,
        "image_bytes": 0,
        "other_binary_bytes": 0,
        "cache_points": 0,
        "tool_count": len(parameters.function_tools) + len(parameters.output_tools),
        "tool_schema_chars": 0,
    }

    def measure(value: Any) -> None:
        if isinstance(value, str):
            shape["text_chars"] += len(value)
        elif isinstance(value, BinaryContent):
            if value.media_type.startswith("image/"):
                shape["image_count"] += 1
                shape["image_bytes"] += len(value.data)
            else:
                shape["other_binary_bytes"] += len(value.data)
        elif isinstance(value, ImageUrl):
            shape["image_count"] += 1
        elif isinstance(value, CachePoint):
            shape["cache_points"] += 1
        elif isinstance(value, dict):
            for item in value.values():
                measure(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                measure(item)

    for message in messages:
        if isinstance(message, ModelRequest):
            measure(message.instructions)
        for part in message.parts:
            if isinstance(part, CachePoint):
                shape["cache_points"] += 1
            else:
                measure(getattr(part, "content", None))
                measure(getattr(part, "args", None))
    for tool in (*parameters.function_tools, *parameters.output_tools):
        shape["tool_schema_chars"] += len(tool.description or "") + len(
            json.dumps(tool.parameters_json_schema, ensure_ascii=False)
        )
    for instruction in parameters.instruction_parts or ():
        measure(getattr(instruction, "content", None))
    return shape


class RetryLoggingModel(WrapperModel):
    """Retry a failed model request without replaying completed robot tool calls.

    Error files keep redacted response bodies; request logs omit them. A stream is
    retried only before it opens; replaying a partially consumed stream could
    duplicate output or downstream actions.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        policy: RetryPolicy | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.policy = policy or RetryPolicy()
        self.log_path = Path(log_path) if log_path is not None else None
        self.request_log_path = (
            self.log_path.with_name("llm_requests.jsonl")
            if self.log_path is not None
            else None
        )
        # Both SDKs otherwise retry internally before an error reaches this
        # layer, making the configured attempt limit and logs misleading.
        provider = wrapped.provider
        if provider is not None and provider.name in {"openai", "anthropic"}:
            provider.client.max_retries = 0

    def _write_record(self, path: Path, record: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            if os.geteuid() == 0:
                for parent in path.parents:
                    owner = parent.stat()
                    if owner.st_uid != 0:
                        try:
                            os.fchown(fd, owner.st_uid, owner.st_gid)
                        except OSError as ownership_error:
                            logger.warning(
                                "could not assign LLM log ownership: %s",
                                ownership_error,
                            )
                        break
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as log_error:
            logger.warning("could not write LLM request log: %s", log_error)

    def _record_attempt(
        self,
        *,
        request_id: str,
        attempt: int,
        shape: dict[str, int],
        elapsed_s: float,
        response: ModelResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.request_log_path is None:
            return
        usage = response.usage if response is not None else None
        self._write_record(
            self.request_log_path,
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "attempt": attempt,
                "provider": self.system,
                "model": self.model_name,
                "elapsed_s": round(elapsed_s, 3),
                "outcome": "error" if error is not None else "success",
                "error_type": type(error).__name__ if error is not None else None,
                "status_code": (
                    error.status_code if isinstance(error, ModelHTTPError) else None
                ),
                "server_error": _http_error_details(error),
                "request_shape": shape,
                "usage": (
                    {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cache_read_tokens,
                        "cache_write_tokens": usage.cache_write_tokens,
                    }
                    if usage is not None
                    else None
                ),
            },
        )

    def _failed(
        self,
        exc: Exception,
        attempt: int,
        *,
        can_retry: bool,
        request_id: str,
        shape: dict[str, int],
        elapsed_s: float,
    ) -> float:
        delay = self.policy.delay(attempt) if can_retry else 0.0
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "provider": self.system,
            "model": self.model_name,
            "error_type": type(exc).__name__,
            "status_code": exc.status_code if isinstance(exc, ModelHTTPError) else None,
            "attempt": attempt,
            "max_attempts": self.policy.max_retries + 1,
            "will_retry": can_retry,
            "retry_delay_s": delay,
            "request_id": request_id,
            "request_shape": shape,
            "server_error": _http_error_details(exc),
            "elapsed_s": round(elapsed_s, 3),
        }
        detailed_record = {**record, "response_body": _response_body_for_log(exc)}
        log = logger.warning if can_retry else logger.error
        log("LLM request failed: %s", json.dumps(detailed_record, ensure_ascii=False))
        if self.log_path is not None:
            self._write_record(self.log_path, detailed_record)
        return delay

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Retry only the provider request that failed."""
        request_id = uuid.uuid4().hex
        shape = _request_shape(messages, model_request_parameters)
        shape["max_output_tokens"] = int((model_settings or {}).get("max_tokens") or 0)
        trace = _RequestTrace(
            self, request_id, messages, model_settings, model_request_parameters
        )
        try:
            for attempt in range(1, self.policy.max_retries + 2):
                started = time.monotonic()
                trace.attempt_start(attempt)
                try:
                    response = await self.wrapped.request(
                        messages, model_settings, model_request_parameters
                    )
                except BaseException as exc:
                    elapsed_s = time.monotonic() - started
                    self._record_attempt(
                        request_id=request_id,
                        attempt=attempt,
                        shape=shape,
                        elapsed_s=elapsed_s,
                        error=exc,
                    )
                    trace.attempt_end(attempt, elapsed_s, error=exc)
                    if not isinstance(exc, Exception):
                        raise
                    can_retry = _retryable(exc) and attempt <= self.policy.max_retries
                    delay = self._failed(
                        exc,
                        attempt,
                        can_retry=can_retry,
                        request_id=request_id,
                        shape=shape,
                        elapsed_s=elapsed_s,
                    )
                    if not can_retry:
                        raise
                    trace.retry(attempt, delay)
                    await asyncio.sleep(delay)
                else:
                    elapsed_s = time.monotonic() - started
                    self._record_attempt(
                        request_id=request_id,
                        attempt=attempt,
                        shape=shape,
                        elapsed_s=elapsed_s,
                        response=response,
                    )
                    trace.attempt_end(attempt, elapsed_s, response=response)
                    trace.end()
                    return response
        except BaseException as error:
            trace.end(error)
            raise
        raise AssertionError("unreachable retry state")

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Retry stream setup; never replay a stream after it has opened."""
        request_id = uuid.uuid4().hex
        shape = _request_shape(messages, model_request_parameters)
        shape["max_output_tokens"] = int((model_settings or {}).get("max_tokens") or 0)
        trace = _RequestTrace(
            self, request_id, messages, model_settings, model_request_parameters
        )
        try:
            for attempt in range(1, self.policy.max_retries + 2):
                opened = False
                stream = None
                started = time.monotonic()
                trace.attempt_start(attempt)
                try:
                    async with self.wrapped.request_stream(
                        messages, model_settings, model_request_parameters, run_context
                    ) as stream:
                        opened = True
                        yield (
                            _TracedStream(stream, trace, attempt)
                            if trace.recorder is not None
                            else stream
                        )
                except BaseException as exc:
                    elapsed_s = time.monotonic() - started
                    # get() includes partial provider-reported usage even when
                    # consumption or stream cleanup failed. Never replay it.
                    response = _stream_response(stream)
                    self._record_attempt(
                        request_id=request_id,
                        attempt=attempt,
                        shape=shape,
                        elapsed_s=elapsed_s,
                        response=response,
                        error=exc,
                    )
                    trace.attempt_end(attempt, elapsed_s, response=response, error=exc)
                    if not isinstance(exc, Exception):
                        raise
                    can_retry = (
                        not opened
                        and _retryable(exc)
                        and attempt <= self.policy.max_retries
                    )
                    delay = self._failed(
                        exc,
                        attempt,
                        can_retry=can_retry,
                        request_id=request_id,
                        shape=shape,
                        elapsed_s=elapsed_s,
                    )
                    if not can_retry:
                        raise
                    trace.retry(attempt, delay)
                    await asyncio.sleep(delay)
                else:
                    response = _stream_response(stream)
                    elapsed_s = time.monotonic() - started
                    self._record_attempt(
                        request_id=request_id,
                        attempt=attempt,
                        shape=shape,
                        elapsed_s=elapsed_s,
                        response=response,
                    )
                    trace.attempt_end(attempt, elapsed_s, response=response)
                    trace.end()
                    return
        except BaseException as error:
            trace.end(error)
            raise
        raise AssertionError("unreachable retry state")


def _stream_response(stream: StreamedResponse | None) -> ModelResponse | None:
    if stream is not None:
        try:
            return stream.get()
        except Exception:  # noqa: BLE001 - recording cannot replace the provider error
            pass
    return None


class _TracedStream:
    """Observe a requested stream without enabling streaming on nonstreaming agents."""

    def __init__(
        self, wrapped: StreamedResponse, trace: _RequestTrace, attempt: int
    ) -> None:
        self._wrapped = wrapped
        self._trace = trace
        self._attempt = attempt

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for event in self._wrapped:
            reference = self._trace.recorder.snapshot("chunks", uuid.uuid4().hex, event)
            self._trace._emit(
                "model_response_chunk", {"chunk_ref": reference}, attempt=self._attempt
            )
            yield event


class _RequestTrace:
    """Correlate attempt records without adding mutable state to shared models."""

    def __init__(
        self,
        model: RetryLoggingModel,
        request_id: str,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        parameters: ModelRequestParameters,
    ) -> None:
        from roborsi.runtime.trace import current_trace, current_trace_scope

        self.recorder = current_trace()
        self.request_id = request_id
        self.scope = current_trace_scope()
        self.model = model.model_name
        self.provider = model.system
        self.response_ref: str | None = None
        if self.recorder is not None:
            if self.scope.turn_id is not None:
                self._emit(
                    "context_end",
                    {"message_count": len(messages), "status": "completed"},
                )
            reference = self.recorder.snapshot(
                "requests",
                request_id,
                {
                    "model": self.model,
                    "provider": self.provider,
                    "messages": messages,
                    "model_settings": settings,
                    "model_request_parameters": parameters,
                },
            )
            self._emit("model_request_start", {"request_ref": reference})

    def _emit(
        self, kind: str, payload: dict[str, Any], *, attempt: int | None = None
    ) -> None:
        if self.recorder is not None:
            self.recorder.emit(
                kind,
                {
                    "model": self.model,
                    "provider": self.provider,
                    "purpose": self.scope.purpose,
                    **payload,
                },
                request_id=self.request_id,
                attempt=attempt,
                agent_id=self.scope.agent_id,
                parent_agent_id=self.scope.parent_agent_id,
                turn_id=self.scope.turn_id,
                tool_call_id=self.scope.tool_call_id,
            )

    def attempt_start(self, attempt: int) -> None:
        self._emit("model_attempt_start", {}, attempt=attempt)

    def attempt_end(
        self,
        attempt: int,
        elapsed_s: float,
        *,
        response: ModelResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.recorder is None:
            return
        from roborsi.runtime.trace import exception_status, reported_usage

        if response is not None:
            self.response_ref = self.recorder.snapshot(
                "responses", f"{self.request_id}-{attempt}", response
            )
            self._emit(
                "message_received",
                {"source": "model", "message_ref": self.response_ref},
                attempt=attempt,
            )
        self._emit(
            "model_attempt_end",
            {
                "status": exception_status(error) if error is not None else "completed",
                "elapsed_s": elapsed_s,
                "error_type": type(error).__name__ if error is not None else None,
                "usage": reported_usage(response),
                "response_ref": self.response_ref,
            },
            attempt=attempt,
        )

    def retry(self, attempt: int, delay: float) -> None:
        self._emit("model_retry", {"retry_delay_s": delay}, attempt=attempt)

    def end(self, error: BaseException | None = None) -> None:
        from roborsi.runtime.trace import exception_status

        self._emit(
            "model_request_end",
            {
                "status": exception_status(error) if error is not None else "completed",
                "response_ref": self.response_ref,
                "error_type": type(error).__name__ if error is not None else None,
            },
        )
