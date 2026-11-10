# coding=utf-8
"""Vision-language supervision over Chat Completions or Responses APIs.

Protocol reference:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
JSON mode is used for provider compatibility; decisions are validated locally.
"""

from __future__ import annotations

import json
import hashlib
import math
from concurrent.futures import Executor, Future
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urlparse

from .llm_auth import CredentialError, StaticCredential
from .llm_images import MAX_IMAGE_BYTES, MAX_REQUEST_BYTES, bounded_image_data_url
from .trace import TraceWriter

from .state import (
    Decision,
    PolicySelection,
    SupervisorRequest,
)

DEFAULT_SELECTION_PROMPT = (
    "Choose one policy_id from available_policies using the task, current camera "
    "images, robot state and declared capabilities. The selected policy will remain "
    "fixed for the whole episode. Return only a JSON object with exactly policy_id "
    "and reason as nonempty strings. Do not invent policies or supply model paths."
)
DEFAULT_REASONING_EFFORT = None
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def validate_api_settings(
    wire_api, headers, stream, reasoning_effort=DEFAULT_REASONING_EFFORT
):
    if not isinstance(wire_api, str) or wire_api not in {
        "chat_completions",
        "responses",
    }:
        raise ValueError("supervisor.wire_api must be chat_completions or responses")
    if type(stream) is not bool or stream and wire_api != "responses":
        raise ValueError("supervisor.stream is supported only for Responses")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in REASONING_EFFORTS
    ):
        raise ValueError(
            "supervisor.reasoning_effort must be null or one of "
            + ", ".join(REASONING_EFFORTS)
        )
    # Only non-secret integration metadata may enter public config snapshots.
    allowed = {"editor-version", "copilot-integration-id", "user-agent"}
    if (
        not isinstance(headers, dict)
        or any(
            not isinstance(key, str)
            or key.lower() not in allowed
            or not isinstance(value, str)
            or not value.strip()
            or any(ord(char) < 32 or ord(char) > 126 for char in value)
            for key, value in headers.items()
        )
        or len({key.lower() for key in headers}) != len(headers)
    ):
        raise ValueError("Only non-secret integration headers are allowed")


class SupervisorError(RuntimeError):
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


class SupervisorEpisodeError(RuntimeError):
    """A review failed; the action policy itself has not failed."""


class SupervisorExecutor(Executor):
    """Daemon workers keep an unresponsive remote call from pinning shutdown.

    Runtime limits outstanding requests and drains them between jobs. Workers
    only return decisions and never mutate the policy or episode state.
    """

    def __init__(self, workers):
        self._queue = queue.Queue()
        self._closed = False
        self._threads = [
            threading.Thread(target=self._work, daemon=True, name="orimapi-supervisor")
            for _ in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def _work(self):
        while (item := self._queue.get()) is not None:
            future, function, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)

    def submit(self, function, /, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Supervisor executor is closed")
        future = Future()
        self._queue.put((future, function, args, kwargs))
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        if not self._closed:
            self._closed = True
            if cancel_futures:
                while True:
                    try:
                        self._queue.get_nowait()[0].cancel()
                    except queue.Empty:
                        break
            for _ in self._threads:
                self._queue.put(None)
        if wait:
            for thread in self._threads:
                thread.join()


class APISupervisor:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        credential=None,
        wire_api: str = "chat_completions",
        headers: dict | None = None,
        stream: bool = False,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        system_prompt: str,
        initial_prompt: str | None = None,
        review_prompt: str | None = None,
        selection_prompt: str = DEFAULT_SELECTION_PROMPT,
        timeout_s: float = 30.0,
        max_response_bytes: int = 1_048_576,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        transport: Callable[[dict], dict] | None = None,
        trace: TraceWriter | None = None,
    ):
        address = urlparse(endpoint)
        if address.scheme not in {"https", "http"} or not address.hostname:
            raise ValueError("A full HTTP(S) supervisor endpoint is required")
        if address.username or address.password:
            raise ValueError(
                "Put API credentials in the configured environment variable"
            )
        initial_prompt = system_prompt if initial_prompt is None else initial_prompt
        review_prompt = system_prompt if review_prompt is None else review_prompt
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                model,
                system_prompt,
                initial_prompt,
                review_prompt,
                selection_prompt,
            )
        ):
            raise ValueError("Model and system prompt are required")
        if (api_key is None) == (credential is None):
            raise ValueError("Supply exactly one API key or credential provider")
        headers = {} if headers is None else dict(headers)
        validate_api_settings(wire_api, headers, stream, reasoning_effort)
        if (
            type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("A positive API timeout is required")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("A positive response byte limit is required")
        for name, limit in (
            ("image", max_image_bytes),
            ("request", max_request_bytes),
        ):
            if type(limit) is not int or limit < 1:
                raise ValueError(f"A positive {name} byte limit is required")
        self.endpoint, self.model = endpoint, model
        self._credential = (
            credential if credential is not None else StaticCredential(api_key)
        )
        self.system_prompt = system_prompt
        self.initial_prompt, self.review_prompt = initial_prompt, review_prompt
        self._trace = trace
        self.wire_api, self.headers, self.stream = wire_api, headers, stream
        self.reasoning_effort = reasoning_effort
        self.selection_prompt = selection_prompt
        self.timeout_s, self.max_response_bytes = timeout_s, max_response_bytes
        self.max_image_bytes, self.max_request_bytes = (
            max_image_bytes,
            max_request_bytes,
        )
        self._transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        deadline = time.monotonic() + self.timeout_s
        try:
            for attempt in range(2):
                token = self._credential.get(deadline)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SupervisorError("Supervisor request deadline exceeded")
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload, allow_nan=False).encode(),
                    headers={
                        **self.headers,
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=remaining) as response:
                        if self.stream:
                            result = self._read_stream(response, deadline)
                        else:
                            raw = response.read(self.max_response_bytes + 1)
                            if len(raw) > self.max_response_bytes:
                                raise SupervisorError(
                                    "Supervisor response exceeded the byte limit"
                                )
                            try:
                                result = json.loads(raw)
                            except (ValueError, UnicodeError):
                                raise SupervisorError(
                                    "Supervisor API returned invalid JSON"
                                ) from None
                        if time.monotonic() > deadline:
                            raise SupervisorError(
                                "Supervisor request deadline exceeded"
                            )
                        return result
                except urllib.error.HTTPError as error:
                    code = error.code
                    error.close()
                    if code == 401 and attempt == 0 and self._credential.refreshable:
                        self._credential.invalidate(token, deadline=deadline)
                        continue
                    raise SupervisorError(
                        f"Supervisor API returned HTTP {code}",
                        retryable=code in {408, 409, 429} or 500 <= code < 600,
                    ) from None
        except CredentialError as error:
            raise SupervisorError(str(error), retryable=error.retryable) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SupervisorError(
                "Supervisor API transport failed or timed out"
            ) from None

    def _read_stream(self, response, deadline):
        used, data = 0, []
        while True:
            if time.monotonic() >= deadline:
                raise SupervisorError("Supervisor request deadline exceeded")
            line = response.readline(self.max_response_bytes - used + 1)
            used += len(line)
            if used > self.max_response_bytes:
                raise SupervisorError("Supervisor response exceeded the byte limit")
            if not line:
                raise SupervisorError(
                    "Supervisor stream ended without a completed response"
                )
            if line.rstrip(b"\r\n") == b"":
                if not data:
                    continue
                try:
                    event = json.loads(b"\n".join(data))
                    kind = event["type"]
                    if kind == "response.completed":
                        return event["response"]
                    if kind in {"error", "response.failed", "response.incomplete"}:
                        raise SupervisorError(
                            "Supervisor stream failed or was incomplete"
                        )
                except (KeyError, TypeError, ValueError, UnicodeError):
                    raise SupervisorError(
                        "Supervisor returned an invalid stream event"
                    ) from None
                data.clear()
            elif line.startswith(b"data:"):
                data.append(line[5:].strip())

    def select_policy(self, request: SupervisorRequest) -> PolicySelection:
        try:
            selection = PolicySelection.from_dict(
                self._query(
                    request,
                    self.selection_prompt,
                    prompt_phase="selection",
                    available_policies=request.candidates,
                )
            )
            if selection.policy_id not in request.candidates:
                raise ValueError("Unknown policy ID")
            return selection
        except (TypeError, ValueError):
            raise SupervisorError(
                "Supervisor returned an invalid policy selection"
            ) from None

    def decide(self, request: SupervisorRequest) -> Decision:
        phase = "initial" if request.memory.get("active_subgoal") is None else "review"
        prompt = self.initial_prompt if phase == "initial" else self.review_prompt
        try:
            return Decision.from_dict(self._query(request, prompt, prompt_phase=phase))
        except (TypeError, ValueError):
            raise SupervisorError("Supervisor returned an invalid decision") from None

    def _query(
        self,
        request: SupervisorRequest,
        system_prompt: str,
        *,
        prompt_phase: str,
        **extra,
    ) -> dict:
        observation = request.observation
        context = {
            "task_goal": observation.instruction,
            "robot_state": observation.robot_state,
            "observation_step": observation.step,
            "plan_version": request.plan_version,
            "memory": request.memory,
            "active_policy": request.policy,
            **extra,
        }
        if self._trace is not None:
            self._trace.write(
                "supervisor_prompt",
                request_id=request.request_id,
                episode_id=request.episode_id,
                env_id=observation.env_id,
                observation_step=observation.step,
                plan_version=request.plan_version,
                prompt_phase=prompt_phase,
                reasoning_effort=self.reasoning_effort,
                prompt_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
            )
        return self.query_json(context, observation.images, system_prompt)

    def query_json(self, context, images, system_prompt):
        """Use the configured authenticated transport for bounded JSON tasks."""
        responses = self.wire_api == "responses"
        text_type = "input_text" if responses else "text"
        content = [{"type": text_type, "text": json.dumps(context, allow_nan=False)}]
        if responses:
            # Some compatible providers validate JSON mode against input only.
            content.append({"type": "input_text", "text": "Return only a JSON object."})
        for camera, image in images.items():
            try:
                url = bounded_image_data_url(image, max_bytes=self.max_image_bytes)
            except (ValueError, OSError) as error:
                raise SupervisorError(
                    f"Supervisor image preparation failed for {camera}: {error}",
                    retryable=False,
                ) from error
            content.extend(
                [
                    {"type": text_type, "text": f"Camera: {camera}"},
                    (
                        {"type": "input_image", "image_url": url}
                        if responses
                        else {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    ),
                ]
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        if responses:
            payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_object"}},
                "stream": self.stream,
                "store": False,
            }
        if self.reasoning_effort is not None:
            if responses:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            else:
                payload["reasoning_effort"] = self.reasoning_effort
        # Use exactly the serialization used by _post, including Base64 and text.
        # Check before calling even an injected transport or fetching credentials.
        request_bytes = len(json.dumps(payload, allow_nan=False).encode())
        if request_bytes > self.max_request_bytes:
            raise SupervisorError(
                f"Supervisor request is {request_bytes} bytes; limit is "
                f"{self.max_request_bytes}. Reduce image views or text context.",
                retryable=False,
            )
        try:
            result = self._transport(payload)
            if responses:
                if result.get("status") != "completed" or result.get("error"):
                    raise SupervisorError(
                        "Supervisor response failed or was incomplete"
                    )
                texts = []
                for item in result["output"]:
                    if item["type"] == "reasoning":
                        continue
                    if (
                        item["type"] != "message"
                        or item.get("role") != "assistant"
                        or item.get("status", "completed") != "completed"
                    ):
                        raise ValueError("Unexpected response output")
                    for part in item["content"]:
                        if part["type"] != "output_text" or not isinstance(
                            part["text"], str
                        ):
                            raise ValueError("Expected a completed text decision")
                        texts.append(part["text"])
                return json.loads("".join(texts))
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice.get("finish_reason") in {
                "length",
                "content_filter",
            }:
                raise SupervisorError(
                    "Supervisor declined or did not finish its decision"
                )
            return json.loads(message["content"])
        except SupervisorError:
            raise
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            raise SupervisorError("Supervisor returned an invalid decision") from None
