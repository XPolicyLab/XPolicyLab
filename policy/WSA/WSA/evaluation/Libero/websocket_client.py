from __future__ import annotations

import logging
import time
from typing import Optional

import websockets.sync.client

try:
    from .msgpack_numpy import Packer, unpackb
except ImportError:
    from msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Simple websocket client for the LIBERO evaluation policy server."""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for LIBERO policy server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except OSError as exc:
                logging.info("Still waiting for LIBERO policy server (%s)...", exc)
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        pack_start = time.perf_counter()
        data = self._packer.pack(obs)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0

        start_time = time.perf_counter()
        self._ws.send(data)
        response = self._ws.recv()
        round_trip_ms = (time.perf_counter() - start_time) * 1000.0
        if isinstance(response, str):
            raise RuntimeError(f"Error in LIBERO policy server:\n{response}")

        unpack_start = time.perf_counter()
        result = unpackb(response)
        unpack_ms = (time.perf_counter() - unpack_start) * 1000.0
        if isinstance(result, dict):
            client_timing = result.setdefault("client_timing", {})
            client_timing["pack_ms"] = float(pack_ms)
            client_timing["round_trip_ms"] = float(round_trip_ms)
            client_timing["unpack_ms"] = float(unpack_ms)
            client_timing["total_client_ms"] = float(pack_ms + round_trip_ms + unpack_ms)
            client_timing["payload_bytes"] = int(len(data))
        return result

    def reset(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
