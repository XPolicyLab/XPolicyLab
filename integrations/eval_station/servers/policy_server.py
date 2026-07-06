"""WebSocket policy server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml
from websockets.asyncio.server import Server, ServerConnection, serve

from eval_station.protocol.codec import decode_envelope, encode_frame
from eval_station.protocol.exceptions import ErrorCode, WsError
from eval_station.protocol.messages import MessageType
from eval_station.protocol.schemas import Frame

logger = logging.getLogger(__name__)


def _ok_payload(result: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True}
    if result is not None:
        payload["result"] = result
    return payload


@dataclass
class PolicyServerConfig:
    host: str = "0.0.0.0"
    port: int = 19000
    ws_ping_interval_s: float = 20.0
    ws_ping_timeout_s: float = 20.0


@dataclass
class PolicyServer:
    model: Any
    config: PolicyServerConfig = field(default_factory=PolicyServerConfig)
    _server: Server | None = field(default=None, init=False)
    _model_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle_connection,
            self.config.host,
            self.config.port,
            max_size=None,
            ping_interval=self.config.ws_ping_interval_s,
            ping_timeout=self.config.ws_ping_timeout_s,
        )
        logger.info("websocket policy server listening on %s", self.url)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    @property
    def url(self) -> str:
        port = self.config.port
        if self._server is not None and self._server.sockets:
            port = self._server.sockets[0].getsockname()[1]
        return f"ws://{self.config.host}:{port}"

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        send_lock = asyncio.Lock()
        pending: set[asyncio.Task[None]] = set()

        async def respond(frame: Frame) -> None:
            try:
                response = await self.process_frame(frame)
                if response is None:
                    await websocket.close()
                    return
                async with send_lock:
                    await websocket.send(encode_frame(response))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to send policy server response")

        try:
            async for raw in websocket:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    frame = decode_envelope(bytes(raw))
                except WsError as exc:
                    logger.error("invalid request frame: %s", exc)
                    continue
                task = asyncio.create_task(respond(frame))
                pending.add(task)
                task.add_done_callback(pending.discard)
        finally:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def process_frame(self, frame: Frame) -> Frame | None:
        try:
            return await self._dispatch_frame(frame)
        except WsError as exc:
            return self._error_reply(frame, exc.code, exc.message, exc.details)
        except Exception as exc:
            logger.exception("policy server request failed")
            return self._error_reply(frame, ErrorCode.INTERNAL, str(exc))

    def _reply(
        self,
        frame: Frame,
        message_type: MessageType,
        payload: dict[str, Any] | None = None,
    ) -> Frame:
        return Frame(
            message_type=message_type,
            request_id=frame.request_id,
            evaluation_id=frame.evaluation_id,
            action_case_id=frame.action_case_id,
            trial_id=frame.trial_id,
            repeat_index=frame.repeat_index,
            step=frame.step,
            payload=payload or {},
        )

    def _error_reply(
        self,
        frame: Frame,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Frame:
        return self._reply(
            frame,
            MessageType.ERROR,
            {
                "code": code.value,
                "message": message,
                "details": details or {},
            },
        )

    async def _invoke_method(self, method, *args: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        result = await asyncio.to_thread(method, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _call_model_method(self, method, *args: Any) -> Any:
        async with self._model_lock:
            return await self._invoke_method(method, *args)

    async def _dispatch_frame(self, frame: Frame) -> Frame | None:
        if frame.message_type == MessageType.HELLO:
            return self._reply(
                frame,
                MessageType.HELLO_ACK,
                {
                    "ok": True,
                    "server": "demo_policy_server",
                },
            )
        if frame.message_type == MessageType.PREPARE_CASE:
            return await self._handle_prepare_case(frame)
        if frame.message_type == MessageType.RESET:
            return await self._handle_reset(frame)
        if frame.message_type == MessageType.INFER:
            return await self._handle_infer(frame)
        if frame.message_type == MessageType.TRIAL_END:
            return await self._handle_trial_end(frame)
        if frame.message_type == MessageType.HEARTBEAT:
            return self._reply(frame, MessageType.HEARTBEAT_ACK, {"ok": True})
        if frame.message_type == MessageType.CLOSE:
            return None
        raise WsError(
            ErrorCode.UNKNOWN_MESSAGE_TYPE,
            f"unsupported request type: {frame.message_type.value}",
        )

    async def _handle_prepare_case(self, frame: Frame) -> Frame:
        case_meta = dict(frame.payload)
        if frame.action_case_id is not None:
            case_meta.setdefault("action_case_id", frame.action_case_id)

        hook = getattr(self.model, "prepare_case", None)
        hook_result = (
            await self._call_model_method(hook, case_meta) if callable(hook) else None
        )
        return self._reply(
            frame, MessageType.PREPARE_CASE_ACK, _ok_payload(hook_result)
        )

    async def _handle_reset(self, frame: Frame) -> Frame:
        method = getattr(self.model, "reset", None)
        if not callable(method):
            raise WsError(ErrorCode.RESET_FAILED, "model.reset is not callable")
        try:
            result = await self._call_model_method(method)
        except Exception as exc:
            raise WsError(ErrorCode.RESET_FAILED, str(exc)) from exc
        return self._reply(frame, MessageType.RESET_RESULT, _ok_payload(result))

    async def _handle_infer(self, frame: Frame) -> Frame:
        observation = frame.payload.get("observation")
        if observation is None:
            raise WsError(ErrorCode.INVALID_FRAME, "infer payload missing observation")

        start = time.perf_counter()
        try:
            async with self._model_lock:
                update_obs = getattr(self.model, "update_obs", None)
                get_action = getattr(self.model, "get_action", None)
                if callable(update_obs) and callable(get_action):
                    if inspect.iscoroutinefunction(
                        update_obs
                    ) or inspect.iscoroutinefunction(get_action):
                        update_result = update_obs(observation)
                        if inspect.isawaitable(update_result):
                            await update_result
                        result = get_action()
                        if inspect.isawaitable(result):
                            result = await result
                    else:
                        def run_legacy_infer() -> Any:
                            update_result = update_obs(observation)
                            if inspect.isawaitable(update_result):
                                raise TypeError(
                                    "update_obs returned an awaitable but is not async"
                                )
                            result = get_action()
                            if inspect.isawaitable(result):
                                raise TypeError(
                                    "get_action returned an awaitable but is not async"
                                )
                            return result

                        result = await asyncio.to_thread(run_legacy_infer)
                else:
                    infer = getattr(self.model, "infer", None)
                    if not callable(infer):
                        raise AttributeError(
                            "model must implement update_obs()/get_action() or infer(observation)"
                        )
                    result = await self._invoke_method(infer, observation)
        except Exception as exc:
            raise WsError(ErrorCode.INFER_FAILED, str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(result, Mapping) and "actions" in result:
            payload: dict[str, Any] = dict(result)
            payload.setdefault("latency_ms", latency_ms)
        else:
            payload = {"actions": result, "latency_ms": latency_ms}

        return self._reply(frame, MessageType.INFER_RESULT, payload)

    async def _handle_trial_end(self, frame: Frame) -> Frame:
        hook = getattr(self.model, "on_trial_end", None)
        hook_result = (
            await self._call_model_method(hook, dict(frame.payload))
            if callable(hook)
            else None
        )
        return self._reply(frame, MessageType.TRIAL_END_ACK, _ok_payload(hook_result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", "--config_path", dest="config_path", required=True
    )
    parser.add_argument("--protocol", choices=("ws",), default="ws")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--relay-url", dest="relay_url")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {args.config_path}")
    deploy_cfg = data
    if args.host is not None:
        deploy_cfg["host"] = args.host
    if args.port is not None:
        deploy_cfg["port"] = args.port
    if args.relay_url is not None:
        deploy_cfg["relay_url"] = args.relay_url

    policy_name = deploy_cfg.get("policy_name")
    if not policy_name:
        raise ValueError("policy_name must be specified in config")
    module = importlib.import_module(f"XPolicyLab.policy.{policy_name}.model")
    model_class = getattr(module, "Model")
    model = model_class(deploy_cfg)

    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=deploy_cfg.get("host", "0.0.0.0"),
            port=int(deploy_cfg.get("port", 19000)),
        ),
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
