"""Standalone RoboDojo WebSocket client; no server imports or GPU dependencies."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import time
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.client import connect

DEFAULT_URL = "wss://salvhn155nb8sv6j2dvpj.apigateway-cn-beijing.volceapi.com/"
PROTOCOL = "robodojo-policy-v1"
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
DEFAULT_CONFIG = Path(__file__).with_name("client_config.json")


def load_client_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the adjacent config, or an explicitly selected JSON file."""
    settings = {"url": DEFAULT_URL, "timeout": 120}
    path = DEFAULT_CONFIG if config_path is None else Path(config_path).expanduser()
    if config_path is None and not path.exists():
        return settings  # Retain support for copying just client.py.
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - set(settings):
        raise ValueError("Client config must be a JSON object with only url and timeout fields")
    settings.update(value)
    return settings


def pack(message: dict[str, Any]) -> bytes:
    return msgpack.packb(message, default=msgpack_numpy.encode, use_bin_type=True)


def _numeric_arrays(value: dict) -> Any:
    if b"nd" in value:
        # Object arrays in msgpack-numpy may invoke pickle. Only numeric arrays
        # belong to this protocol, including for local saved request samples.
        if value.get(b"kind") in (b"O", b"V") or np.dtype(value[b"type"]).kind not in "buif":
            raise ValueError("Only numeric NumPy arrays are accepted")
    return msgpack_numpy.decode(value)


def unpack(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError("Expected binary MsgPack, not a text/JSON WebSocket frame")
    value = msgpack.unpackb(data, object_hook=_numeric_arrays, raw=False, strict_map_key=True)
    if not isinstance(value, dict):
        raise TypeError("Expected a MsgPack mapping")
    return value


class RoboDojoClient:
    """One connection, sequential requests, and no automatic prediction retries."""

    def __init__(self, url: str | None = None, timeout: float | None = None, *,
                 config_path: str | Path | None = None) -> None:
        settings = load_client_config(config_path)
        url = settings["url"] if url is None else url
        timeout = settings["timeout"] if timeout is None else timeout
        if not isinstance(url, str) or not url.startswith(("ws://", "wss://")):
            raise ValueError("Use ws:// or wss://; HTTPS URLs must use wss://")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a positive number of seconds")
        self.url, self.timeout = url, timeout
        self.socket = None
        self.metadata: dict[str, Any] = {}
        self.ready = False
        self.next_index = 0

    def __enter__(self) -> RoboDojoClient:
        options: dict[str, Any] = {}
        parameters = inspect.signature(connect).parameters
        if "proxy" in parameters:
            options["proxy"] = None
        if "ping_interval" in parameters:
            options["ping_interval"] = None
        self.socket = connect(
            self.url, open_timeout=min(30, self.timeout), close_timeout=5,
            max_size=MAX_MESSAGE_BYTES, compression=None, **options,
        )
        try:
            self.metadata = unpack(self.socket.recv(timeout=self.timeout))
            if self.metadata.get("protocol") != PROTOCOL or not self.metadata.get("session_id"):
                raise ValueError("Unexpected server handshake")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.ready = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _call(self, operation: str, **fields: Any) -> dict[str, Any]:
        if self.socket is None:
            raise RuntimeError("Open the client with a with statement")
        started = time.perf_counter()
        try:
            self.socket.send(pack({
                "operation": operation, "session_id": self.metadata["session_id"], **fields,
            }))
            response = unpack(self.socket.recv(timeout=self.timeout))
            if (response.get("session_id") != self.metadata["session_id"]
                    or response.get("operation") != operation):
                raise RuntimeError("Response does not match the current request/session")
        except BaseException:
            # A late response may still arrive after a timeout; don't reuse the
            # connection and accidentally consume it as another request's reply.
            self.close()
            raise
        if response.get("ok") is not True:
            self.ready = False
            raise RuntimeError(f"Server rejected request; reset required: {response.get('error')}")
        response["client_roundtrip_ms"] = (time.perf_counter() - started) * 1000
        return response

    def reset(self, instruction: str, *, task: str | None = None, seed: int = 0,
              source_color_order: str = "rgb") -> dict[str, Any]:
        self.ready = False
        response = self._call("reset", instruction=instruction, task=task, seed=seed,
                              source_color_order=source_color_order)
        self.next_index, self.ready = 0, True
        return response

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Call reset before predict")
        response = self._call("predict", request_index=self.next_index, images=images, state=state)
        result = response.get("result", {})
        action = result.get("action")
        if (result.get("request_index") != self.next_index
                or not isinstance(action, np.ndarray) or action.dtype != np.float32
                or action.shape != (24, 14) or not np.isfinite(action).all()):
            self.close()
            raise RuntimeError("Invalid action response; reconnect and reset before retrying")
        self.next_index += 1
        return response

    def predict_current(self, images: dict[str, np.ndarray], state: np.ndarray, *,
                        instruction: str, task: str | None = None, seed: int = 0,
                        source_color_order: str = "rgb") -> dict[str, Any]:
        """Start at any task time using one current frame per camera.

        Reset only server memory, then predict; never reset the environment.
        Earlier predictions may have been ignored or partially executed, and
        another controller may have acted in between. No history is retained
        across these calls. Use predict() for a fully executed continuation.
        Calls on this client must be sequential, including reset + predict.
        """
        if not isinstance(images, dict) or set(images) != {"head", "left_wrist", "right_wrist"}:
            raise ValueError("images must contain head, left_wrist and right_wrist")
        for name, value in images.items():
            if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
                    or value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3
                    or min(value.shape[1:3]) < 1):
                raise ValueError(f"images.{name} must be uint8 [1,H,W,3]")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must contain 14 finite raw values")
        started = time.perf_counter()
        self.reset(instruction, task=task, seed=seed, source_color_order=source_color_order)
        response = self.predict(images, state)
        response["client_current_request_ms"] = (time.perf_counter() - started) * 1000
        return response
