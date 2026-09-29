# Copyright 2025 starVLA community. All rights reserved.
# SPDX-License-Identifier: MIT
#
# The WebSocket policy server and client in this module are ported from the
# starVLA research tree (deployment/model_server/tools/websocket_policy_server.py
# and websocket_policy_client.py), implemented by Jinhui YE (HKUST) in 2025 and
# released under the MIT License.
#
# Those upstream files derive from openpi
# (https://github.com/Physical-Intelligence/openpi), Copyright Physical
# Intelligence, licensed under the Apache License 2.0.
#
# The numpy array codec (pack_array/unpack_array) is adapted from msgpack-numpy
# (https://github.com/lebedov/msgpack-numpy), Copyright (c) 2013-2022 Lev E.
# Givon, licensed under the BSD 3-Clause License.
#
# ---------------------------------------------------------------------------
# MODIFICATION NOTICE (Apache License 2.0, Section 4(b), for the openpi-derived
# portions; recorded for all upstreams above)
#
# This file was changed by The CogWAM Authors in 2025.
# Modifications Copyright (c) 2025 The CogWAM Authors.
#
# Changes relative to the originals:
#   - MERGED three upstream modules into this single module:
#     msgpack_numpy.py, websocket_policy_server.py and
#     websocket_policy_client.py. The array codec is folded in verbatim; the
#     server's handshake-metadata push, ping/infer/predict_action routing,
#     flat-dict-as-payload fallback, response envelope and idle watchdog are
#     byte-identical to upstream.
#   - The server now imports the msgpack helpers from this same module rather
#     than a sibling module.
#   - REMOVED the typing_extensions @override decorator on the client (no base
#     Policy class exists in this repository, so it was a no-op dependency).
#   - REMOVED the server module's __main__ NotImplementedError stub.
#   - KEPT the client's inspect-based ping_interval/ping_timeout filtering for
#     older pinned `websockets` versions.
#
# The upstream paths and the SHA-256 of the exact revisions this file was
# ported from are recorded in UPSTREAM_SOURCES.json at the repository root.
# ---------------------------------------------------------------------------

"""Wire protocol for the CogWAM policy server: msgpack-numpy over WebSocket.

msgpack is used instead of pickle for (de)serializing observations and action
chunks over the network because:

- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); ~4x faster than pickle for
    serializing large arrays using the below strategy

The array codec is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library
directly is that it falls back to pickle for object arrays.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import websockets.asyncio.server
import websockets.frames
import websockets.sync.client

# ---------------------------------------------------------------------------
# msgpack <-> numpy
# ---------------------------------------------------------------------------


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class WebsocketPolicyServer:
    """Serves a policy over WebSocket.

    On connect the server PUSHES ``metadata`` as the first frame, before the
    client sends anything: the client treats that handshake payload as a hard
    ABI and validates its own configuration against it.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                ret = self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknown request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class WebsocketClientPolicy:
    """Client side of :class:`WebsocketPolicyServer`.

    The server metadata frame is consumed during construction, so
    ``get_server_metadata()`` is available before the first inference call.
    """

    def __init__(self, host: str = "127.0.0.1", port: int | None = 10093, api_key: str | None = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300):
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = dict(
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=None,
                    ping_timeout=60,
                )
                # The RoboDojo simulator image pins an older ``websockets`` whose
                # sync client does not expose the keepalive knobs. Drop only the
                # options it cannot accept instead of failing to connect.
                named = {
                    name
                    for name, parameter in inspect.signature(websockets.sync.client.connect).parameters.items()
                    if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                }
                for option in ("ping_interval", "ping_timeout"):
                    if option not in named:
                        connect_kwargs.pop(option, None)
                conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def predict_action(self, query_info: dict) -> dict[str, Any]:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)


__all__ = [
    "Packer",
    "Unpacker",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "pack_array",
    "packb",
    "unpack_array",
    "unpackb",
]
