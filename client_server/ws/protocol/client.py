"""Eval Client over WebSocket (msgpack frames)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

import websockets

from client_server.ws.protocol.codec import decode_envelope, encode_frame
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import REQUEST_RESPONSE_PAIRS, MessageType
from client_server.ws.protocol.schemas import Frame

logger = logging.getLogger(__name__)

_RED = "\033[31m"
_GREEN = "\033[32m"
_BLUE = "\033[34m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _status(level: str, color: str, message: str) -> None:
    print(f"{color}{_BOLD}[{level}]{_RESET} {message}", flush=True)


class WebSocketConnection(Protocol):
    async def send(self, message: Any, text: bool | None = None) -> None: ...

    async def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[bytes | str]: ...


@dataclass
class PolicyEvalClientConfig:
    url: str
    evaluation_id: str
    connect_timeout_s: float = 30.0
    request_timeout_s: float = 120.0
    max_connect_attempts: int = 10
    connect_retry_delay_s: float = 5.0
    ws_ping_interval_s: float = 20.0
    ws_ping_timeout_s: float = 20.0
    proxy: str | Literal[True] | None = None


@dataclass
class PolicyEvalClient:
    config: PolicyEvalClientConfig
    _ws: WebSocketConnection | None = field(default=None, init=False)
    _recv_task: asyncio.Task[None] | None = field(default=None, init=False)
    _pending: dict[str, asyncio.Future[Frame]] = field(default_factory=dict, init=False)
    _closed: bool = field(default=False, init=False)

    async def _sleep_with_countdown(self, seconds: float, label: str) -> None:
        remaining = max(0, int(seconds))
        if remaining <= 0:
            return
        for left in range(remaining, 0, -1):
            print(
                f"\r{_YELLOW}{_BOLD}[RECONNECT]{_RESET} {label}; retry in {left:2d}s",
                end="",
                flush=True,
            )
            await asyncio.sleep(1)
        print("\r" + " " * 96 + "\r", end="", flush=True)

    async def _reset_connection_state(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def connect(
        self,
        *,
        handshake: bool = True,
        evaluation_plan: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> Frame | None:
        if self._ws is not None:
            return None
        self._closed = False
        connect_kwargs: dict[str, Any] = {
            "max_size": None,
            "ping_interval": self.config.ws_ping_interval_s,
            "ping_timeout": self.config.ws_ping_timeout_s,
        }
        # `proxy` is only supported by the websockets>=14 asyncio client; passing it
        # to older releases (e.g. the legacy client shipped with Isaac Sim's
        # websockets 12) raises TypeError inside loop.create_connection.
        if self.config.proxy is not None:
            connect_kwargs["proxy"] = self.config.proxy
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_connect_attempts + 1):
            _status(
                "CONNECTING",
                _BLUE,
                f"websocket policy server {self.config.url} "
                f"(attempt {attempt}/{self.config.max_connect_attempts}, timeout={self.config.connect_timeout_s:g}s)",
            )
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.config.url,
                        **connect_kwargs,
                    ),
                    timeout=self.config.connect_timeout_s,
                )
                self._recv_task = asyncio.create_task(self._recv_loop())
                logger.info("websocket connected: %s", self.config.url)
                _status("CONNECTED", _GREEN, f"websocket policy server connected: {self.config.url}")
                if handshake:
                    return await self.hello(evaluation_plan=evaluation_plan)
                return None
            except Exception as exc:
                last_err = exc
                logger.warning("connect attempt %s failed: %s", attempt, exc)
                await self._reset_connection_state()
                _status(
                    "CONNECT-FAILED",
                    _YELLOW,
                    f"attempt {attempt}/{self.config.max_connect_attempts} failed: {exc}",
                )
                if attempt < self.config.max_connect_attempts:
                    await self._sleep_with_countdown(
                        self.config.connect_retry_delay_s,
                        f"reconnecting to {self.config.url}",
                    )
        _status(
            "ERROR",
            _RED,
            f"failed to connect to {self.config.url} after {self.config.max_connect_attempts} attempts",
        )
        raise ConnectionError(
            f"failed to connect after {self.config.max_connect_attempts} attempts: {last_err}"
        ) from last_err

    async def close(self) -> None:
        self._closed = True
        self._fail_pending(WsError(ErrorCode.INTERNAL, "client closed"))
        await self._reset_connection_state()

    def _fail_pending(self, exc: BaseException) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        close_exc: BaseException | None = None
        try:
            async for raw in self._ws:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    frame = decode_envelope(bytes(raw))
                except WsError as exc:
                    logger.error("invalid frame: %s", exc)
                    continue
                self._dispatch_incoming(frame)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("recv loop ended: %s", exc)
            close_exc = exc
        finally:
            if not self._closed:
                self._fail_pending(
                    close_exc or WsError(ErrorCode.INTERNAL, "connection closed")
                )
                self._ws = None

    def _dispatch_incoming(self, frame: Frame) -> None:
        if frame.message_type == MessageType.ERROR:
            err = frame.payload
            try:
                code = ErrorCode(err.get("code", ErrorCode.INTERNAL.value))
            except ValueError:
                code = ErrorCode.INTERNAL
            exc = WsError(
                code,
                str(err.get("message", "policy server error")),
                details=err.get("details"),
            )
            req_id = frame.request_id
            if req_id in self._pending:
                self._pending.pop(req_id).set_exception(exc)
            return

        req_id = frame.request_id
        fut = self._pending.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(frame)

    async def request(
        self,
        msg_type: MessageType,
        payload: dict[str, Any],
        *,
        action_case_id: str | None = None,
        trial_id: str | None = None,
        repeat_index: int | None = None,
        step: int = 0,
        timeout_s: float | None = None,
        _reconnect_attempted: bool = False,
    ) -> Frame:
        expected = REQUEST_RESPONSE_PAIRS.get(msg_type)
        if expected is None and msg_type != MessageType.CLOSE:
            raise ValueError(f"no response pairing for {msg_type}")
        if self._ws is None:
            if self._closed:
                raise RuntimeError("not connected; client is closed")
            if msg_type == MessageType.HELLO:
                raise RuntimeError("not connected; call connect() first")
            _status(
                "RECONNECT",
                _YELLOW,
                f"connection to {self.config.url} is closed; reconnecting before {msg_type.value}",
            )
            await self.connect(handshake=True)

        request_id = str(uuid4())
        frame = Frame(
            message_type=msg_type,
            request_id=request_id,
            evaluation_id=self.config.evaluation_id,
            action_case_id=action_case_id,
            trial_id=trial_id,
            repeat_index=repeat_index,
            step=step,
            payload=payload,
        )
        if expected is None:
            await self._ws.send(encode_frame(frame))
            return frame

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Frame] = loop.create_future()
        self._pending[request_id] = fut
        try:
            await self._ws.send(encode_frame(frame))
        except Exception:
            self._pending.pop(request_id, None)
            if not _reconnect_attempted and not self._closed and msg_type != MessageType.HELLO:
                _status("RECONNECT", _YELLOW, f"send failed for {msg_type.value}; reconnecting to {self.config.url}")
                await self._reset_connection_state()
                await self.connect(handshake=True)
                return await self.request(
                    msg_type,
                    payload,
                    action_case_id=action_case_id,
                    trial_id=trial_id,
                    repeat_index=repeat_index,
                    step=step,
                    timeout_s=timeout_s,
                    _reconnect_attempted=True,
                )
            raise
        timeout = timeout_s if timeout_s is not None else self.config.request_timeout_s
        try:
            response = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise WsError(ErrorCode.TIMEOUT, f"timeout waiting for {expected}") from exc
        except Exception:
            self._pending.pop(request_id, None)
            if self._ws is None and not _reconnect_attempted and not self._closed and msg_type != MessageType.HELLO:
                _status(
                    "RECONNECT",
                    _YELLOW,
                    f"connection dropped while waiting for {msg_type.value}; reconnecting to {self.config.url}",
                )
                await self.connect(handshake=True)
                return await self.request(
                    msg_type,
                    payload,
                    action_case_id=action_case_id,
                    trial_id=trial_id,
                    repeat_index=repeat_index,
                    step=step,
                    timeout_s=timeout_s,
                    _reconnect_attempted=True,
                )
            raise
        if expected is not None and response.message_type != expected:
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"expected {expected.value}, got {response.message_type.value}",
            )
        return response

    async def hello(
        self,
        *,
        evaluation_plan: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> Frame:
        payload: dict[str, Any] = {}
        if evaluation_plan is not None:
            payload["evaluation_plan"] = dict(evaluation_plan)
        return await self.request(MessageType.HELLO, payload)

    async def prepare_case(
        self,
        action_case_id: str,
        case_meta: dict[str, Any] | None = None,
    ) -> Frame:
        body = {"action_case_id": action_case_id, **(case_meta or {})}
        return await self.request(
            MessageType.PREPARE_CASE,
            body,
            action_case_id=action_case_id,
        )

    async def reset(
        self,
        *,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Frame:
        body = {"trial_id": trial_id, **(payload or {})}
        return await self.request(
            MessageType.RESET,
            body,
            action_case_id=action_case_id,
            trial_id=trial_id,
            repeat_index=repeat_index,
        )

    async def infer(
        self,
        observation: dict[str, Any],
        *,
        trial_id: str | None = None,
        action_case_id: str | None = None,
        step: int = 0,
    ) -> Frame:
        return await self.request(
            MessageType.INFER,
            {"observation": observation},
            trial_id=trial_id,
            action_case_id=action_case_id,
            step=step,
        )

    async def trial_end(
        self,
        *,
        trial_id: str,
        result: dict[str, Any] | None = None,
        action_case_id: str | None = None,
    ) -> Frame:
        body = {"trial_id": trial_id, **(result or {})}
        return await self.request(
            MessageType.TRIAL_END,
            body,
            trial_id=trial_id,
            action_case_id=action_case_id,
        )

    async def send_close(self, reason: str = "") -> Frame:
        return await self.request(MessageType.CLOSE, {"reason": reason})
