from __future__ import annotations

import argparse
import logging
import socket
import struct
import threading
from io import BytesIO
from typing import Any, Dict

import torch

from deploy_policy import get_model, reset_model


LOGGER = logging.getLogger("motus.robotwin.server")


class TorchSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        buffer = BytesIO(data)
        return torch.load(buffer, map_location="cpu", weights_only=False)


class MotusPolicyServer:
    def __init__(self, host: str, port: int, model_args: Dict[str, Any]):
        self.host = host
        self.port = int(port)
        self.model = get_model(model_args)
        self._lock = threading.Lock()

    def handle_request(self, request: Dict[str, Any]) -> Any:
        endpoint = request.get("endpoint")

        if endpoint == "ping":
            return {"status": "ok"}

        if endpoint == "reset":
            with self._lock:
                reset_model(self.model)
            return {"status": "ok"}

        if endpoint != "inference":
            raise ValueError(f"unsupported endpoint: {endpoint}")

        observation = request.get("observation")
        instruction = request.get("instruction")
        if observation is None:
            raise ValueError("missing observation")
        if instruction is None:
            raise ValueError("missing instruction")

        with self._lock:
            self.model.set_instruction(str(instruction))
            self.model.update_obs(observation)
            return self.model.get_action()

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            LOGGER.info("Motus server listening on %s:%s", self.host, self.port)

            while True:
                conn, addr = server_socket.accept()
                LOGGER.info("accepted client from %s:%s", addr[0], addr[1])
                thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            while True:
                try:
                    payload = self._recv_message(conn)
                    if payload is None:
                        return

                    request = TorchSerializer.from_bytes(payload)
                    response = self.handle_request(request)
                    self._send_message(conn, TorchSerializer.to_bytes(response))
                except Exception as exc:
                    LOGGER.exception("request handling failed")
                    self._send_message(conn, TorchSerializer.to_bytes({"error": str(exc)}))
                    return

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = conn.recv(size - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)

    def _recv_message(self, conn: socket.socket) -> bytes | None:
        header = self._recv_exact(conn, 8)
        if header is None:
            return None
        (size,) = struct.unpack("!Q", header)
        return self._recv_exact(conn, size)

    @staticmethod
    def _send_message(conn: socket.socket, payload: bytes) -> None:
        conn.sendall(struct.pack("!Q", len(payload)))
        conn.sendall(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motus RoboTwin policy server")
    parser.add_argument("--checkpoint_path", required=True, help="Motus checkpoint directory")
    parser.add_argument("--wan_path", required=True, help="WAN model directory")
    parser.add_argument("--vlm_path", required=True, help="VLM model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    model_args = {
        "ckpt_setting": args.checkpoint_path,
        "wan_path": args.wan_path,
        "vlm_path": args.vlm_path,
        "log_dir": args.log_dir,
        "task_name": args.task_name,
    }
    server = MotusPolicyServer(host=args.host, port=args.port, model_args=model_args)
    server.serve_forever()


if __name__ == "__main__":
    main()