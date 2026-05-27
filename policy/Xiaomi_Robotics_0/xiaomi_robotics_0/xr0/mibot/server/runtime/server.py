# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import pickle
import socket
import struct
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from mibot.utils.io import ACTION_EPS, denormalize_action


class Server(mp.Process):
    def __init__(self, host: str, port: int, model, mean, std, action_mask, device: str) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.action_mask = action_mask.to(device)

    @staticmethod
    def _recv_all(conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _recv(self, conn):
        head = self._recv_all(conn, 4)
        if not head:
            return None
        size = struct.unpack(">I", head)[0]
        body = self._recv_all(conn, size)
        return None if body is None else pickle.loads(body)

    @staticmethod
    def _send(conn, payload):
        payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        conn.sendall(struct.pack(">I", len(payload)) + payload)

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, _ = server.accept()
                try:
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            request = self._recv(conn)
                            if request is None:
                                break

                            tic = time.time()
                            batch = {
                                key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                                for key, value in request.items()
                            }
                            mask = self.action_mask.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)

                            if "action" in batch:
                                batch["action"] = ((batch["action"] - self.mean) / (self.std + ACTION_EPS)) * mask
                            else:
                                batch["action"] = torch.zeros((batch["input_ids"].shape[0], *self.mean.shape), device=self.device, dtype=torch.bfloat16)

                            batch["action_mask"] = mask if "action_mask" not in batch else batch["action_mask"].to(self.device) * mask

                            action = self.model.generate(batch)
                            action = denormalize_action(action * mask, self.mean, self.std) * mask
                            self._send(conn, action.cpu())

                            pbar.update(1)
                            pbar.set_postfix({"avg_time": f"{(time.time() - tic) * 1000:.2f}ms"})
                except Exception as error:
                    print(f"Error handling connection: {error}")
                finally:
                    conn.close()


if __name__ == "__main__":
    raise SystemExit("Use `python mibot/server/deploy.py --model <dir>` to start the inference server.")
