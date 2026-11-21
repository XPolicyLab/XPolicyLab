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

"""Benchmark HTTP API with private episode files and downloadable artifacts."""

from __future__ import annotations

import hmac
import json
import re
import shutil
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, unquote, urlsplit

from roborsi.evaluation.adapter_service import AdapterFacade, BenchmarkAdapter
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    ServiceError,
    StartResult,
    StepResult,
    Task,
    from_envelope,
    to_envelope,
)
from roborsi.evaluation.protocol import InteractionEnvelope
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.service_protocol import RoboRSIRequestError
from roborsi.utils.logging import get_logger

logger = get_logger("benchmark_http")
MAX_REQUEST_BYTES = 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TERMINAL = {"completed", "failed", "cancelled", "closed"}


class BenchmarkProvider(Protocol):
    """Benchmark-owned discovery and fresh native Adapter construction."""

    name: str

    def tasks(self) -> list[Task]: ...
    def runtime_info(self) -> BenchmarkRuntimeInfo | None: ...
    def assets(self) -> list[dict[str, Any]]: ...
    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> BenchmarkAdapter: ...


class ApiError(Exception):
    def __init__(
        self, status: int, message: str, code: str = "invalid_request"
    ) -> None:
        self.status = status
        self.error = ServiceError(code=code, message=message)
        super().__init__(message)


def _message(envelope: InteractionEnvelope, model):
    message = from_envelope(envelope)
    if not isinstance(message, model):
        raise ValueError("unexpected interaction message")
    return message


def _envelope(document: dict, kind: str) -> InteractionEnvelope:
    envelope = InteractionEnvelope.from_dict(document)
    if envelope.kind != kind or not envelope.request_id:
        raise ApiError(400, "request needs the expected message kind and request_id")
    if not _ID.fullmatch(envelope.episode_id):
        raise ApiError(
            400, "episode_id must be 1-128 letters, digits, dots, dashes or underscores"
        )
    return envelope


class _Episode:
    def __init__(
        self, envelope: InteractionEnvelope, directory: Path, mode: str
    ) -> None:
        self.episode_id = envelope.episode_id
        self.start_request = envelope.as_dict()
        self.task = _message(envelope, Task)
        self.directory = directory
        (directory / "trace/events").mkdir(parents=True)
        self.mode = mode
        self.state = "starting"
        self.step_index = 0
        self.error: ServiceError | None = None
        self.observation: Observation | None = None
        self.result: dict | None = None
        self.facade: AdapterFacade | None = None
        self.artifacts = ArtifactPublisher(directory, self.episode_id)
        self.cancelled = threading.Event()
        self.operation = threading.Lock()
        self.lock = threading.RLock()
        self.events = 0
        self.test_request: dict | None = None
        self.replies: dict[str, tuple[str, dict]] = {}
        self.active_request: str | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "schema_version": 1,
                "episode_id": self.episode_id,
                "mode": self.mode,
                "state": self.state,
                "step_index": self.step_index,
                "error": self.error.as_dict() if self.error else None,
                "observation": self.observation.as_dict() if self.observation else None,
            }

    def log(self, method: str, message: dict) -> None:
        with self.lock:
            ref = f"trace/events/{self.events:08d}.json"
            write_json_atomic(
                self.directory / ref, {"method": method, "message": message}
            )
            self.artifacts.publish(ref, "log", "application/json")
            self.events += 1

    def publish_observation(self, observation: Observation) -> None:
        CaseMediaChannel(self.directory / "media").verify_observation(observation)
        with self.lock:
            for ref in observation.media:
                self.artifacts.publish(
                    ref.ref,
                    "image",
                    ref.mime_type,
                    step_index=observation.step_index,
                    sha256=ref.sha256,
                    size_bytes=ref.size_bytes,
                )
            self.observation = observation
            self.step_index = observation.step_index


class BenchmarkApplication:
    """Serve one active native episode while retaining completed HTTP results."""

    def __init__(self, provider: BenchmarkProvider, root: Path) -> None:
        self.provider, self.root = provider, root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes: dict[str, _Episode] = {}
        self._lock = threading.RLock()

    def _get(self, episode_id: str) -> _Episode:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise ApiError(404, "episode not found")
        return episode

    def _create(self, envelope: InteractionEnvelope, mode: str) -> _Episode:
        task = _message(envelope, Task)
        if task.benchmark != self.provider.name:
            raise ApiError(400, "task belongs to another Benchmark")
        with self._lock:
            existing = self._episodes.get(envelope.episode_id)
            if existing:
                if (
                    existing.start_request != envelope.as_dict()
                    or existing.mode != mode
                ):
                    raise ApiError(
                        409, "episode_id already belongs to a different request"
                    )
                return existing
            if any(
                e.state not in _TERMINAL or e.operation.locked()
                for e in self._episodes.values()
            ):
                raise ApiError(409, "another episode is still running")
            directory = self.root / uuid.uuid4().hex
            directory.mkdir()
            episode = _Episode(envelope, directory, mode)
            self._episodes[envelope.episode_id] = episode
            return episode

    def _execute(self, episode: _Episode, method: str, *args: Any) -> dict:
        if method != "finish":
            args = (json.dumps(json.loads(args[0]), sort_keys=True),)
        request_id = (
            args[0] if method == "finish" else json.loads(args[0])["request_id"]
        )
        fingerprint = json.dumps([method, *args], sort_keys=True)
        with episode.lock:
            acquired = episode.operation.acquire(blocking=False)
            if acquired:
                episode.active_request = fingerprint
            elif episode.active_request != fingerprint:
                raise ApiError(409, "an episode operation is still running")
        if not acquired:
            # A lost response may be retried while the first call still owns the
            # native environment. Wait for its result without executing twice.
            if not episode.operation.acquire(timeout=30.0):
                raise ApiError(409, "the original request is still running")
            with episode.lock:
                episode.active_request = fingerprint
        try:
            with episode.lock:
                if episode.cancelled.is_set():
                    raise ApiError(409, "episode was cancelled")
                previous = episode.replies.get(request_id)
                if previous is not None:
                    if previous[0] != fingerprint:
                        raise ApiError(
                            409, "request_id was reused with a different request"
                        )
                    return previous[1]
                if episode.state in _TERMINAL:
                    raise ApiError(409, "episode is no longer executable")
            if episode.facade is None:
                episode.facade = AdapterFacade(
                    self.provider.create_adapter(episode.directory, episode.artifacts)
                )
            response = json.loads(getattr(episode.facade, method)(*args))
            envelope = InteractionEnvelope.from_dict(response)
            if method == "start":
                episode.publish_observation(_message(envelope, StartResult).observation)
            elif method in {"step", "stop"}:
                episode.publish_observation(_message(envelope, StepResult).observation)
            elif method == "finish":
                native = _message(envelope, NativeResult)
                if native.source:
                    with episode.lock:
                        episode.artifacts.publish(
                            native.source,
                            "native_result",
                            "application/json",
                            sha256=native.source_sha256,
                        )
            episode.log(method, response)
            with episode.lock:
                if episode.cancelled.is_set():
                    raise ApiError(409, "episode was cancelled")
                if method == "finish":
                    episode.result = response
                    episode.state = "completed"
                elif episode.state != "completed":
                    episode.state = "running"
                episode.replies[request_id] = (fingerprint, response)
            return response
        except ApiError:
            raise
        except RoboRSIRequestError as exc:
            raise ApiError(409, str(exc)) from exc
        except Exception as exc:
            error = ServiceError(
                code="adapter_error",
                message="Benchmark execution or artifact publication failed",
                details={"operation": method, "exception": type(exc).__name__},
            )
            with episode.lock:
                episode.error = error
                episode.state = "failed"
            try:
                episode.log("error", error.as_dict())
            except Exception:
                logger.warning("could not save Benchmark error record", exc_info=True)
            if episode.facade is not None:
                episode.facade.close()
            raise ApiError(422, error.message, "adapter_error") from exc
        finally:
            try:
                if episode.cancelled.is_set():
                    self._finish_cancel(episode)
            finally:
                with episode.lock:
                    episode.active_request = None
                    episode.operation.release()

    def _finish_cancel(self, episode: _Episode) -> None:
        try:
            if episode.facade is not None:
                episode.facade.close()
        except Exception as exc:
            with episode.lock:
                episode.error = ServiceError(
                    code="infrastructure_error",
                    message="Benchmark cleanup failed",
                    details={"operation": "close", "exception": type(exc).__name__},
                )
                episode.state = "failed"
                episode.result = None
        else:
            with episode.lock:
                episode.result = None
                episode.state = "cancelled"

    def _cancel(self, episode: _Episode) -> dict:
        with episode.lock:
            if episode.state in _TERMINAL:
                return episode.snapshot()
            episode.cancelled.set()
            episode.state = "cancelling"
        if episode.operation.acquire(blocking=False):

            def cleanup():
                try:
                    self._finish_cancel(episode)
                finally:
                    episode.operation.release()

            threading.Thread(target=cleanup, daemon=True).start()
        return episode.snapshot()

    def _run_test(self, episode: _Episode, document: dict) -> None:
        try:
            self._execute(episode, "start", json.dumps(episode.start_request))
            for index in range(episode.task.max_steps):
                actions = document["actions"]
                if index >= len(actions) and not document.get("repeat_last", False):
                    raise ApiError(
                        422,
                        "action sequence ended before the task finished",
                        "adapter_error",
                    )
                action = Action(
                    benchmark=episode.task.benchmark,
                    episode_id=episode.episode_id,
                    step_index=index,
                    data=actions[min(index, len(actions) - 1)],
                )
                wire = to_envelope(
                    action,
                    benchmark=action.benchmark,
                    episode_id=action.episode_id,
                    request_id=f"test-step-{index}",
                ).as_dict()
                reply = self._execute(episode, "step", json.dumps(wire))
                step = _message(InteractionEnvelope.from_dict(reply), StepResult)
                if step.terminated or step.truncated:
                    self._execute(episode, "finish", "test-finish")
                    return
            raise ApiError(
                422, "Benchmark did not finish within max_steps", "adapter_error"
            )
        except Exception as exc:
            if episode.cancelled.is_set():
                self._cancel(episode)
                return
            with episode.lock:
                if episode.error is None:
                    episode.error = (
                        exc.error
                        if isinstance(exc, ApiError)
                        else ServiceError(
                            code="adapter_error",
                            message="complete test failed",
                        )
                    )
                episode.state = "failed"
            if episode.facade is not None:
                episode.facade.close()

    def handle(
        self, method: str, path: str, body: dict | None
    ) -> tuple[int, dict | bytes, dict]:
        parts = [unquote(p) for p in urlsplit(path).path.split("/") if p]
        if parts[:1] != ["v1"]:
            raise ApiError(404, "unknown API path")
        if method == "GET" and parts == ["v1", "health"]:
            with self._lock:
                busy = any(e.state not in _TERMINAL for e in self._episodes.values())
            return (
                200,
                {
                    "status": "ok",
                    "benchmark_api_versions": [1],
                    "busy": busy,
                    **(
                        {"episode_stop_versions": [1]}
                        if getattr(self.provider, "supports_episode_stop", False)
                        is True
                        else {}
                    ),
                    **(
                        {"benchmark_query_versions": [1]}
                        if getattr(self.provider, "supports_benchmark_queries", False)
                        is True
                        else {}
                    ),
                },
                {},
            )
        if method == "GET" and parts == ["v1", "runtime"]:
            info = self.provider.runtime_info()
            return (
                200,
                {
                    "runtime": info.as_dict() if info else None,
                    "assets": self.provider.assets(),
                },
                {},
            )
        if method == "GET" and parts == ["v1", "benchmark"]:
            info = self.provider.runtime_info()
            return (
                200,
                {
                    "schema_version": 1,
                    "benchmark": self.provider.name,
                    "runtime": info.as_dict() if info else None,
                    "assets": self.provider.assets(),
                    "max_concurrent": 1,
                },
                {},
            )
        if method == "GET" and parts == ["v1", "tasks"]:
            return (
                200,
                {"tasks": [task.as_dict() for task in self.provider.tasks()]},
                {},
            )
        if method == "POST" and parts == ["v1", "episodes"]:
            envelope = _envelope(body, "task")
            episode = self._create(envelope, "interactive")
            return 201, self._execute(episode, "start", json.dumps(body)), {}
        if method == "POST" and parts == ["v1", "tests"]:
            required = {"request_id", "episode_id", "task", "actions"}
            if (
                not isinstance(body, dict)
                or not required <= body.keys()
                or body.keys() - required - {"repeat_last"}
            ):
                raise ApiError(400, "complete test fields do not match the API")
            task = Task.from_dict(body["task"])
            actions = body["actions"]
            if (
                not isinstance(actions, list)
                or not 0 < len(actions) <= task.max_steps
                or any(not isinstance(a, dict) for a in actions)
                or type(body.get("repeat_last", False)) is not bool
            ):
                raise ApiError(
                    400, "actions must be a nonempty bounded list of objects"
                )
            envelope = _envelope(
                to_envelope(
                    task,
                    benchmark=task.benchmark,
                    episode_id=body["episode_id"],
                    request_id=body["request_id"],
                ).as_dict(),
                "task",
            )
            episode = self._create(envelope, "test")
            with episode.lock:
                if episode.test_request is not None and episode.test_request != body:
                    raise ApiError(
                        409, "request_id already belongs to another complete test"
                    )
                if episode.test_request is None:
                    episode.test_request = body
                    threading.Thread(
                        target=self._run_test, args=(episode, body), daemon=True
                    ).start()
            return 202, episode.snapshot(), {}
        if len(parts) < 3 or parts[1] != "episodes":
            raise ApiError(404, "unknown API path")
        episode = self._get(parts[2])
        tail = parts[3:]
        if method == "GET" and not tail:
            return 200, episode.snapshot(), {}
        if method == "GET" and tail == ["result"]:
            with episode.lock:
                return (
                    (200, episode.result, {})
                    if episode.result is not None
                    else (202, episode.snapshot(), {})
                )
        if method == "GET" and tail == ["artifacts"]:
            with episode.lock:
                query_string = urlsplit(path).query
                query = (
                    parse_qs(query_string, strict_parsing=True) if query_string else {}
                )
                if set(query) - {"step_index", "kind"} or any(
                    len(v) != 1 for v in query.values()
                ):
                    raise ApiError(400, "unsupported artifact filter")
                artifacts = episode.artifacts.list()
                if "step_index" in query:
                    artifacts = [
                        a
                        for a in artifacts
                        if a.step_index == int(query["step_index"][0])
                    ]
                if "kind" in query:
                    artifacts = [a for a in artifacts if a.kind == query["kind"][0]]
                return 200, {"artifacts": [a.as_dict() for a in artifacts]}, {}
        if method == "GET" and len(tail) == 2 and tail[0] == "artifacts":
            try:
                with episode.lock:
                    info, data = episode.artifacts.get(tail[1])
            except KeyError as exc:
                raise ApiError(404, "artifact not found in this episode") from exc
            return (
                200,
                data,
                {"Content-Type": info.mime_type, "X-Content-SHA256": info.sha256},
            )
        if method == "POST" and tail == ["cancel"]:
            return 202, self._cancel(episode), {}
        if method == "POST" and tail == ["close"]:
            if episode.mode == "test" and episode.state not in _TERMINAL:
                raise ApiError(409, "cancel the complete test before closing")
            if not episode.operation.acquire(blocking=False):
                raise ApiError(409, "cancel the active operation before closing")
            try:
                if episode.facade is not None:
                    episode.facade.close()
                with episode.lock:
                    if episode.state not in _TERMINAL:
                        episode.state = "closed"
                return 200, episode.snapshot(), {}
            finally:
                episode.operation.release()
        if method == "DELETE" and not tail:
            with self._lock:
                if episode.state not in _TERMINAL or not episode.operation.acquire(
                    blocking=False
                ):
                    raise ApiError(
                        409, "stop the episode before deleting its artifacts"
                    )
                try:
                    shutil.rmtree(episode.directory)
                    del self._episodes[episode.episode_id]
                finally:
                    episode.operation.release()
            return 200, {"deleted": True, "episode_id": episode.episode_id}, {}
        if method == "POST" and tail in (["actions"], ["stop"], ["query"], ["finish"]):
            if episode.mode != "interactive":
                raise ApiError(409, "complete tests own their action sequence")
            if tail in (["actions"], ["stop"], ["query"]):
                operation, kind = {
                    "actions": ("step", "action"),
                    "stop": ("stop", "episode_stop"),
                    "query": ("query", "benchmark_query"),
                }[tail[0]]
                envelope = _envelope(body, kind)
                if envelope.episode_id != episode.episode_id:
                    raise ApiError(409, "request belongs to another episode")
                return (
                    200,
                    self._execute(episode, operation, json.dumps(body)),
                    {},
                )
            if (
                not isinstance(body, dict)
                or set(body) != {"request_id"}
                or not isinstance(body["request_id"], str)
                or not body["request_id"].strip()
            ):
                raise ApiError(400, "finish requires request_id")
            return 200, self._execute(episode, "finish", body["request_id"]), {}
        raise ApiError(404, "unknown API path")

    def close(self) -> None:
        for episode in list(self._episodes.values()):
            self._cancel(episode)


class BenchmarkHttpServer(ThreadingHTTPServer):
    """HTTP server with bounded JSON requests and raw artifact downloads."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        application: BenchmarkApplication,
        *,
        token: str | None = None,
    ) -> None:
        self.application, self.token = application, token
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _handle(self) -> None:
        try:
            if self.server.token and not hmac.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {self.server.token}"
            ):
                raise ApiError(401, "authentication required")
            length = int(self.headers.get("Content-Length", "0"))
            if (
                self.headers.get("Transfer-Encoding")
                or not 0 <= length <= MAX_REQUEST_BYTES
            ):
                raise ApiError(413, "request body exceeds supported size")
            self.connection.settimeout(30.0)
            body = None
            if length:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ApiError(400, "incomplete request body")
                body = json.loads(
                    raw,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ValueError("nonfinite JSON")
                    ),
                )
            status, document, headers = self.server.application.handle(
                self.command, self.path, body
            )
        except ApiError as exc:
            status, document, headers = exc.status, {"error": exc.error.as_dict()}, {}
        except (ValueError, TypeError, KeyError) as exc:
            status, document, headers = (
                400,
                {
                    "error": ServiceError(
                        code="invalid_request",
                        message="request or artifact validation failed",
                        details={"exception": type(exc).__name__},
                    ).as_dict()
                },
                {},
            )
        except Exception:
            logger.exception("Benchmark HTTP request failed")
            status, document, headers = (
                500,
                {
                    "error": ServiceError(
                        code="infrastructure_error",
                        message="Benchmark service failed",
                    ).as_dict()
                },
                {},
            )
        data = (
            document
            if isinstance(document, bytes)
            else json.dumps(document, allow_nan=False).encode()
        )
        self.send_response(status)
        self.send_header(
            "Content-Type", headers.pop("Content-Type", "application/json")
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = do_POST = do_DELETE = _handle
