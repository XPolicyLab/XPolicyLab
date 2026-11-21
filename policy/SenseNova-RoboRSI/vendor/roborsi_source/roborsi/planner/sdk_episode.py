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

"""Episode boundaries shared by SDK tool transports."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from pydantic_ai.messages import BinaryContent

from roborsi.planner.base import PlannerFailure
from roborsi.tools.toolkit import ToolResult


def prompt_parts(
    system_prompt: str, user_message: str | list[str | BinaryContent]
) -> str | list[str | BinaryContent]:
    """Keep the established text prompt and preserve multimodal input blocks."""
    if isinstance(user_message, str):
        return f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
    for part in user_message:
        if isinstance(part, BinaryContent) and not part.media_type.startswith("image/"):
            raise ValueError("SDK planners support image BinaryContent only")
        if not isinstance(part, (str, BinaryContent)):
            raise TypeError(
                "SDK user messages must contain text or image BinaryContent"
            )
    return ([system_prompt] if system_prompt else []) + list(user_message)


def codex_input(prompt: str | list[str | BinaryContent]) -> Any:
    """Encode the installed Codex SDK's text and image input dataclasses."""
    if isinstance(prompt, str):
        return prompt
    import openai_codex

    return [
        openai_codex.TextInput(part)
        if isinstance(part, str)
        else openai_codex.ImageInput(
            f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}"
        )
        for part in prompt
    ]


def claude_input(prompt: str | list[str | BinaryContent]) -> Any:
    """Encode the installed Claude SDK's streaming user envelope."""
    if isinstance(prompt, str):
        return prompt

    async def messages() -> AsyncIterator[dict[str, Any]]:
        content = [
            {"type": "text", "text": part}
            if isinstance(part, str)
            else {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": base64.b64encode(part.data).decode("ascii"),
                },
            }
            for part in prompt
        ]
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

    return messages()


class SdkEpisode:
    """Observe actual tool results and stop further dispatch after a boundary.

    MCP implementations may convert handler exceptions to ordinary model feedback.
    This wrapper retains fatal error categories before that conversion happens.
    Cancellation never acquires the dispatch lock: deferred handlers hold it while
    waiting for the next environment transition.
    """

    def __init__(self, toolkit: Any, cancel_event: threading.Event | None) -> None:
        self.toolkit = toolkit
        self.cancel_event = cancel_event
        self.stopped = threading.Event()
        self.finish_result: dict[str, Any] | None = None
        self.failure: PlannerFailure | None = None
        self.feedback: list[str] = []
        self.dispatches = 0
        self.acknowledged = threading.Event()
        self.terminal_name: str | None = None
        self.terminal_token: str | None = None
        self._terminal_at: float | None = None
        self._dispatch_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_cancelled = False
        self._closing = threading.Event()
        self._client_closing = threading.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.toolkit, name)

    def stop(self, message: str, *, kind: str, reason: str) -> None:
        """Retain the first episode boundary, including its failure category."""
        with self._state_lock:
            if not self.stopped.is_set():
                self.failure = PlannerFailure(message, kind=kind, reason=reason)
                self.stopped.set()

    def poll_cancel(self) -> None:
        """Observe caller cancellation without waiting for tool dispatch."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            if self.finish_result is not None and not self.acknowledged.is_set():
                with self._state_lock:
                    if self.failure is None:
                        self.failure = PlannerFailure(
                            "SDK episode cancelled", kind="cancelled", reason="closed"
                        )
            else:
                self.stop("SDK episode cancelled", kind="cancelled", reason="closed")

    def execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch once and keep terminal results ahead of transport errors."""
        with self._dispatch_lock:
            self.poll_cancel()
            if self.stopped.is_set() or self._closing.is_set():
                return ToolResult(name, {"error": "SDK episode has ended"})
            self.dispatches += 1
            try:
                result = self.toolkit.execute_tool(name, args)
            except PlannerFailure as exc:
                if not (self._closing.is_set() and exc.kind == "cancelled"):
                    self.stop(str(exc), kind=exc.kind, reason=exc.reason)
                result = ToolResult(name, {"error": str(exc)})
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.stop(message, kind="infrastructure", reason="tool_error")
                result = ToolResult(name, {"error": message})
            if getattr(result, "is_finish", False):
                with self._state_lock:
                    if not self.stopped.is_set():
                        self.finish_result = dict(result.result)
                        self.terminal_name = name
                        self.terminal_token = uuid.uuid4().hex
                        self._terminal_at = time.monotonic()
                        self.stopped.set()
                if self.finish_result is not None:
                    result = copy.copy(result)
                    result.content_blocks = [
                        *result.content_blocks,
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "_finish": True,
                                    "_roborsi_terminal_ack": self.terminal_token,
                                }
                            ),
                        },
                    ]
            if self.stopped.is_set():
                self.feedback.extend(
                    block.get("text", "")
                    for block in getattr(result, "content_blocks", [])
                    if block.get("type") == "text"
                )
            return result

    def acknowledge(self, name: str | None, content: Any) -> None:
        """Observe delivery of the terminal payload, never an older tool result."""
        if (
            self.finish_result is not None
            and (name is None or name == self.terminal_name)
            and _has_terminal_payload(content, self.terminal_token)
        ):
            self.acknowledged.set()

    def shutdown_failure(self, message: str, *, reason: str) -> None:
        """Keep terminal evidence while reporting missing delivery or cleanup."""
        with self._state_lock:
            if self.failure is None:
                self.failure = PlannerFailure(
                    message, kind="infrastructure", reason=reason
                )
            self.stopped.set()

    def should_shutdown(self, *, timeout_expired: bool = False) -> bool:
        """Wait for SDK acknowledgment with a bounded terminal feedback deadline."""
        self.poll_cancel()
        if self.finish_result is not None and self.acknowledged.is_set():
            return True
        if timeout_expired:
            self.shutdown_failure("SDK episode timed out", reason="timeout")
        if self.finish_result is not None and not self.acknowledged.is_set():
            if (
                self._terminal_at is not None
                and time.monotonic() - self._terminal_at >= 5
            ):
                self.shutdown_failure(
                    "SDK did not acknowledge terminal tool feedback",
                    reason="terminal_feedback_timeout",
                )
            return self.failure is not None
        return self.stopped.is_set()

    def finalize(self) -> None:
        """Reject an ended SDK stream that omitted its terminal feedback event."""
        if self.finish_result is not None and not self.acknowledged.is_set():
            self.shutdown_failure(
                "SDK stream ended before terminal tool feedback was acknowledged",
                reason="terminal_feedback_missing",
            )

    def cancel_pending(self) -> None:
        """Release deferred handlers before waiting for active toolkit work."""
        with self._pending_lock:
            if not self._pending_cancelled:
                cancel = getattr(self.toolkit, "cancel_pending", None)
                if cancel is not None:
                    cancel()
                self._pending_cancelled = True

    def cancel_active_and_wait(self) -> None:
        """Close the deferred bridge before toolkit cancellation takes its lock."""
        try:
            self.cancel_pending()
            self.toolkit.cancel_active_and_wait()
        except Exception as exc:
            self.shutdown_failure(
                f"SDK toolkit cleanup failed: {type(exc).__name__}: {exc}",
                reason="cleanup_failed",
            )
            raise

    async def run(
        self, coroutine: Coroutine[Any, Any, Any], timeout_s: float | None
    ) -> Any:
        """Watch independent of SDK cancellation, waking deferred handlers first."""
        task = asyncio.create_task(coroutine)
        started = time.monotonic()
        error: BaseException | None = None
        try:
            while not task.done():
                if self.should_shutdown(
                    timeout_expired=bool(
                        timeout_s and time.monotonic() - started >= timeout_s
                    )
                ):
                    return None
                await asyncio.sleep(0.02)
            return await task
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._closing.set()
            try:
                try:
                    self.cancel_pending()
                finally:
                    try:
                        if not task.done():
                            if not self._client_closing.is_set():
                                task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task
                    finally:
                        await asyncio.to_thread(self.cancel_active_and_wait)
            except Exception as exc:
                self.shutdown_failure(
                    f"SDK episode cleanup failed: {type(exc).__name__}: {exc}",
                    reason="cleanup_failed",
                )
                if error is None:
                    raise
            finally:
                self.finalize()

    def client_context(self, client: Any) -> Any:
        """Record synchronous SDK context cleanup failures."""
        episode = self

        class Context:
            def __enter__(self):
                return client.__enter__()

            def __exit__(self, *args):
                try:
                    return client.__exit__(*args)
                except Exception as exc:
                    episode.shutdown_failure(
                        f"SDK client cleanup failed: {type(exc).__name__}: {exc}",
                        reason="cleanup_failed",
                    )
                    raise

        return Context()

    def async_client_context(self, client: Any) -> Any:
        """Record asynchronous SDK context cleanup failures."""
        episode = self

        class Context:
            async def __aenter__(self):
                return await client.__aenter__()

            async def __aexit__(self, *args):
                episode._client_closing.set()
                try:
                    return await client.__aexit__(*args)
                except Exception as exc:
                    episode.shutdown_failure(
                        f"SDK client cleanup failed: {type(exc).__name__}: {exc}",
                        reason="cleanup_failed",
                    )
                    raise
                finally:
                    episode._client_closing.clear()

        return Context()


def _has_terminal_payload(content: Any, token: str | None) -> bool:
    """Find this dispatch's acknowledgment, including truncated native feedback."""
    if isinstance(content, str):
        try:
            return _has_terminal_payload(json.loads(content), token)
        except (ValueError, TypeError):
            return False
    if isinstance(content, list):
        return any(_has_terminal_payload(block, token) for block in content)
    if isinstance(content, dict):
        if token is not None and content.get("_roborsi_terminal_ack") == token:
            return True
        return any(
            _has_terminal_payload(content.get(key), token)
            for key in ("content", "result", "text")
        )
    return False
