"""Common session lifecycle and serial request scheduling for policy servers.

Connection threads never enter a policy handler. They validate framing, enqueue
one synchronous request, and send its response. A single dispatcher owns all
calls into the shared model runtime and its profiler.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """The peer sent a malformed or unsupported protocol message."""


class SessionError(RuntimeError):
    """A request failed without making the shared runtime unsafe."""


class SessionCapacityError(SessionError):
    """The server has no free concurrent session slot."""


class FatalServerError(RuntimeError):
    """A request made the shared model runtime unsafe to continue using."""


class PolicySession(Protocol):
    def handle(self, command: str, payload: object) -> object: ...

    def close(self) -> None: ...


class SessionFactory(Protocol):
    def create(self, client_id: str, metadata: dict[str, object]) -> PolicySession: ...


@dataclass(frozen=True)
class RegisteredSession:
    session_id: str
    client_id: str
    metadata: dict[str, object]
    policy: PolicySession
    opened_at: float


class SessionRegistry:
    """Tracks one policy session per live client connection."""

    def __init__(self, factory: SessionFactory, max_sessions: int) -> None:
        if int(max_sessions) <= 0:
            raise ValueError("max_sessions must be positive")
        self._factory = factory
        self.max_sessions = int(max_sessions)
        self._lock = threading.Lock()
        self._by_session: dict[str, RegisteredSession] = {}
        self._by_client: dict[str, str] = {}
        self._peak_sessions = 0
        self._sessions_created = 0
        self._sessions_closed = 0

    def open(self, client_id: str, metadata: dict[str, object]) -> RegisteredSession:
        client_id = str(client_id).strip()
        if not client_id:
            raise ProtocolError("client_id must be non-empty")
        with self._lock:
            if client_id in self._by_client:
                raise ProtocolError(f"duplicate client_id: {client_id}")
            if len(self._by_session) >= self.max_sessions:
                raise SessionCapacityError(
                    f"server session capacity reached: {self.max_sessions}"
                )
            # Reserve the client id while factory construction runs. Session
            # construction is intentionally serialized and expected to be cheap.
            self._by_client[client_id] = ""
            try:
                policy = self._factory.create(client_id, dict(metadata))
                session_id = uuid.uuid4().hex
                session = RegisteredSession(
                    session_id=session_id,
                    client_id=client_id,
                    metadata=dict(metadata),
                    policy=policy,
                    opened_at=time.time(),
                )
                self._by_session[session_id] = session
                self._by_client[client_id] = session_id
                self._sessions_created += 1
                self._peak_sessions = max(self._peak_sessions, len(self._by_session))
                return session
            except BaseException:
                self._by_client.pop(client_id, None)
                raise

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._by_session.pop(session_id, None)
            if session is None:
                return
            self._by_client.pop(session.client_id, None)
            self._sessions_closed += 1
        session.policy.close()

    def close_all(self) -> None:
        with self._lock:
            session_ids = list(self._by_session)
        for session_id in session_ids:
            self.close(session_id)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_sessions": len(self._by_session),
                "peak_sessions": self._peak_sessions,
                "sessions_created": self._sessions_created,
                "sessions_closed": self._sessions_closed,
                "max_clients": self.max_sessions,
            }


@dataclass
class _ScheduledRequest:
    sequence: int
    session: RegisteredSession
    request_id: str
    command: str
    payload: object
    enqueued_perf: float = field(default_factory=time.perf_counter)
    done: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    cancelled: bool = False
    result: object = None
    error: BaseException | None = None
    error_traceback: str = ""
    queue_wait_s: float = 0.0
    handler_s: float = 0.0


class SerialRequestScheduler:
    """Runs all policy calls on one FIFO dispatcher thread."""

    _STOP = object()

    def __init__(self, max_pending: int, request_timeout_sec: float = 300.0) -> None:
        if int(max_pending) <= 0:
            raise ValueError("max_pending must be positive")
        if float(request_timeout_sec) <= 0:
            raise ValueError("request_timeout_sec must be positive")
        self.request_timeout_sec = float(request_timeout_sec)
        self._queue: queue.Queue[_ScheduledRequest | object] = queue.Queue(
            maxsize=int(max_pending)
        )
        self._lock = threading.Lock()
        self._sequence = 0
        self._requests_completed = 0
        self._queue_wait_total_s = 0.0
        self._handler_total_s = 0.0
        self._queue_wait_samples: deque[float] = deque(maxlen=10000)
        self._handler_samples: deque[float] = deque(maxlen=10000)
        self._peak_queue_depth = 0
        self._last_request_started_at: float | None = None
        self._last_request_completed_at: float | None = None
        self._inflight: _ScheduledRequest | None = None
        self._fatal_error: str | None = None
        self.stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="policy-request-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        session: RegisteredSession,
        request_id: str,
        command: str,
        payload: object,
    ) -> _ScheduledRequest:
        if self.stop_event.is_set():
            raise FatalServerError(self.fatal_error or "policy dispatcher is stopping")
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        request = _ScheduledRequest(
            sequence=sequence,
            session=session,
            request_id=str(request_id),
            command=str(command),
            payload=payload,
        )
        try:
            self._queue.put(request, timeout=self.request_timeout_sec)
        except queue.Full as exc:
            fatal = FatalServerError("policy request queue remained full")
            self.fail_fatally(fatal)
            raise fatal from exc
        with self._lock:
            self._peak_queue_depth = max(self._peak_queue_depth, self._queue.qsize())
        if not request.done.wait(timeout=self.request_timeout_sec):
            with self._lock:
                if not request.started:
                    request.cancelled = True
            fatal = FatalServerError(
                f"policy request timed out after {self.request_timeout_sec:.1f}s: "
                f"client={session.client_id} command={command} request_id={request_id}"
            )
            self.fail_fatally(fatal)
            raise fatal
        if request.error is not None:
            raise request.error
        return request

    @property
    def fatal_error(self) -> str | None:
        with self._lock:
            return self._fatal_error

    def fail_fatally(self, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = message
        self.stop_event.set()

    def _dispatch_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                self._queue.task_done()
                return
            request = item
            assert isinstance(request, _ScheduledRequest)
            if request.cancelled:
                request.error = SessionError("request cancelled before dispatch")
                request.done.set()
                self._queue.task_done()
                continue
            started_perf = time.perf_counter()
            with self._lock:
                request.started = True
                request.queue_wait_s = started_perf - request.enqueued_perf
                self._inflight = request
                self._last_request_started_at = time.time()
            try:
                request.result = request.session.policy.handle(
                    request.command, request.payload
                )
            except SessionError as exc:
                request.error = exc
                request.error_traceback = traceback.format_exc()
            except BaseException as exc:
                fatal = FatalServerError(
                    "unclassified policy handler failure: "
                    f"client={request.session.client_id} command={request.command}: {exc}"
                )
                fatal.__cause__ = exc
                request.error = fatal
                request.error_traceback = traceback.format_exc()
                self.fail_fatally(fatal)
            finally:
                request.handler_s = time.perf_counter() - started_perf
                with self._lock:
                    self._requests_completed += 1
                    self._queue_wait_total_s += request.queue_wait_s
                    self._handler_total_s += request.handler_s
                    self._queue_wait_samples.append(request.queue_wait_s)
                    self._handler_samples.append(request.handler_s)
                    self._last_request_completed_at = time.time()
                    self._inflight = None
                request.done.set()
                self._queue.task_done()
            if self.stop_event.is_set():
                self._fail_pending_requests()
                return

    def _fail_pending_requests(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if isinstance(item, _ScheduledRequest):
                    item.error = FatalServerError(
                        self.fatal_error or "policy dispatcher stopped"
                    )
                    item.done.set()
            finally:
                self._queue.task_done()

    def snapshot(self) -> dict[str, object]:
        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
            return float(ordered[index])

        with self._lock:
            inflight = self._inflight
            queue_wait_samples = list(self._queue_wait_samples)
            handler_samples = list(self._handler_samples)
            return {
                "queue_depth": self._queue.qsize(),
                "peak_queue_depth": self._peak_queue_depth,
                "requests_completed": self._requests_completed,
                "queue_wait_total_s": self._queue_wait_total_s,
                "queue_wait_p50_s": percentile(queue_wait_samples, 0.50),
                "queue_wait_p95_s": percentile(queue_wait_samples, 0.95),
                "queue_wait_max_s": max(queue_wait_samples, default=0.0),
                "handler_total_s": self._handler_total_s,
                "handler_p50_s": percentile(handler_samples, 0.50),
                "handler_p95_s": percentile(handler_samples, 0.95),
                "handler_max_s": max(handler_samples, default=0.0),
                "last_request_started_at": self._last_request_started_at,
                "last_request_completed_at": self._last_request_completed_at,
                "inflight_request_id": None if inflight is None else inflight.request_id,
                "inflight_command": None if inflight is None else inflight.command,
                "inflight_client_id": (
                    None if inflight is None else inflight.session.client_id
                ),
                "fatal_error": self._fatal_error,
            }

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout_sec: float = 10.0) -> bool:
        self.stop_event.set()
        if not self._thread.is_alive():
            return True
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            # The dispatcher will free a slot; avoid losing the stop request.
            try:
                self._queue.put(self._STOP, timeout=max(float(timeout_sec), 0.1))
            except queue.Full:
                return False
        self._thread.join(timeout=max(float(timeout_sec), 0.0))
        return not self._thread.is_alive()


class _HealthReporter:
    def __init__(
        self,
        path: Path,
        snapshot: Callable[[], dict[str, object]],
        interval_sec: float,
    ) -> None:
        self.path = path
        self.snapshot = snapshot
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="policy-health-reporter",
            daemon=True,
        )

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread.start()

    def _write(self) -> None:
        payload = self.snapshot()
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._write()
            except OSError:
                # Server lifecycle must not depend on diagnostic filesystem IO.
                pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self.interval_sec, 1.0))
        try:
            self._write()
        except OSError:
            pass


class MultiClientSessionServer:
    """Concurrent connection/session server with serial policy execution."""

    def __init__(
        self,
        *,
        listener: socket.socket,
        recv_message: Callable[[socket.socket], object],
        send_message: Callable[[socket.socket, object], None],
        registry: SessionRegistry,
        scheduler: SerialRequestScheduler,
        server_id: str,
        persistent: bool,
        health_path: Path | None = None,
        health_interval_sec: float = 5.0,
        server_metadata: dict[str, object] | None = None,
    ) -> None:
        self.listener = listener
        self.recv_message = recv_message
        self.send_message = send_message
        self.registry = registry
        self.scheduler = scheduler
        self.server_id = str(server_id)
        self.persistent = bool(persistent)
        self.server_metadata = dict(server_metadata or {})
        self.stop_event = threading.Event()
        self._connections_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._threads: set[threading.Thread] = set()
        self._ready = False
        self._started_at = time.time()
        self._health = (
            _HealthReporter(Path(health_path), self.health_snapshot, health_interval_sec)
            if health_path is not None
            else None
        )

    def health_snapshot(self) -> dict[str, object]:
        return {
            "pid": os.getpid(),
            "server_id": self.server_id,
            "ready": self._ready and not self.stop_event.is_set(),
            "protocol_version": PROTOCOL_VERSION,
            "started_at": self._started_at,
            "updated_at": time.time(),
            **self.server_metadata,
            **self.registry.snapshot(),
            **self.scheduler.snapshot(),
        }

    def request_stop(self) -> None:
        self.stop_event.set()
        try:
            self.listener.close()
        except OSError:
            pass

    @staticmethod
    def _require_dict(message: object, label: str) -> dict[str, object]:
        if not isinstance(message, dict):
            raise ProtocolError(f"{label} must be a dict, got {type(message)!r}")
        return message

    def _send_error(
        self,
        connection: socket.socket,
        *,
        request_id: object = None,
        error: BaseException,
        fatal: bool = False,
        error_traceback: str = "",
    ) -> None:
        self.send_message(
            connection,
            {
                "type": "response",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "fatal": bool(fatal),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": error_traceback,
            },
        )

    def _serve_connection(self, connection: socket.socket) -> None:
        session: RegisteredSession | None = None
        with self._connections_lock:
            self._connections.add(connection)
        try:
            hello = self._require_dict(self.recv_message(connection), "hello")
            if hello.get("type") != "hello":
                raise ProtocolError("first message must have type='hello'")
            if int(hello.get("protocol_version", -1)) != PROTOCOL_VERSION:
                raise ProtocolError(
                    f"unsupported protocol_version={hello.get('protocol_version')}; "
                    f"server={PROTOCOL_VERSION}"
                )
            client_id = str(hello.get("client_id", "")).strip()
            metadata_value = hello.get("metadata", {})
            if not isinstance(metadata_value, dict):
                raise ProtocolError("hello.metadata must be a dict")
            metadata = dict(metadata_value)
            session = self.registry.open(client_id, metadata)
            self.send_message(
                connection,
                {
                    "type": "hello_ack",
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": True,
                    "server_id": self.server_id,
                    "session_id": session.session_id,
                    "max_clients": self.registry.max_sessions,
                },
            )
            while not self.stop_event.is_set() and not self.scheduler.stop_event.is_set():
                request = self._require_dict(self.recv_message(connection), "request")
                if request.get("type") != "request":
                    raise ProtocolError("request message must have type='request'")
                if int(request.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise ProtocolError("request protocol version changed within session")
                request_id = str(request.get("request_id", "")).strip()
                command = str(request.get("command", "")).strip()
                if not request_id or not command:
                    raise ProtocolError("request_id and command must be non-empty")
                if command == "health":
                    self.send_message(
                        connection,
                        {
                            "type": "response",
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "ok": True,
                            "result": self.health_snapshot(),
                            "queue_wait_s": 0.0,
                            "handler_s": 0.0,
                            "server_call_s": 0.0,
                        },
                    )
                    continue
                try:
                    scheduled = self.scheduler.submit(
                        session,
                        request_id,
                        command,
                        request.get("payload"),
                    )
                except FatalServerError as exc:
                    self._send_error(
                        connection,
                        request_id=request_id,
                        error=exc,
                        fatal=True,
                        error_traceback=traceback.format_exc(),
                    )
                    self.request_stop()
                    break
                except SessionError as exc:
                    self._send_error(
                        connection,
                        request_id=request_id,
                        error=exc,
                        error_traceback=traceback.format_exc(),
                    )
                    break
                self.send_message(
                    connection,
                    {
                        "type": "response",
                        "protocol_version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "ok": True,
                        "result": scheduled.result,
                        "queue_wait_s": scheduled.queue_wait_s,
                        "handler_s": scheduled.handler_s,
                        # Compatibility with the original RoboTwin client.
                        "server_call_s": scheduled.handler_s,
                    },
                )
        except (BrokenPipeError, ConnectionError, EOFError, socket.timeout):
            pass
        except (ProtocolError, SessionCapacityError) as exc:
            try:
                self._send_error(connection, error=exc)
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        except OSError:
            if not self.stop_event.is_set():
                raise
        finally:
            if session is not None:
                try:
                    self.registry.close(session.session_id)
                except BaseException as exc:
                    self.scheduler.fail_fatally(exc)
                    self.request_stop()
            with self._connections_lock:
                self._connections.discard(connection)
                self._threads.discard(threading.current_thread())
            try:
                connection.close()
            except OSError:
                pass

    def serve(self) -> None:
        self.listener.settimeout(1.0)
        self._ready = True
        if self._health is not None:
            self._health.start()
        accepted = 0
        try:
            while not self.stop_event.is_set() and not self.scheduler.stop_event.is_set():
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                accepted += 1
                thread = threading.Thread(
                    target=self._serve_connection,
                    args=(connection,),
                    name=f"policy-connection-{accepted}",
                    daemon=True,
                )
                with self._connections_lock:
                    self._threads.add(thread)
                thread.start()
                if not self.persistent:
                    thread.join()
                    break
        finally:
            self._ready = False
            self.stop_event.set()
            try:
                self.listener.close()
            except OSError:
                pass
            with self._connections_lock:
                connections = list(self._connections)
                threads = list(self._threads)
            for connection in connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except (AttributeError, OSError):
                    # multiprocessing.connection.Connection does not expose
                    # socket.shutdown(); close() below is its transport-level
                    # interruption primitive.
                    pass
                try:
                    connection.close()
                except OSError:
                    pass
            scheduler_stopped = self.scheduler.stop()
            if not scheduler_stopped:
                self.scheduler.fail_fatally(
                    FatalServerError("policy dispatcher did not stop within grace period")
                )
            for thread in threads:
                thread.join(timeout=10.0)
            if scheduler_stopped:
                self.registry.close_all()
            if self._health is not None:
                self._health.stop()
