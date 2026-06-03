# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import pickle
import socket
import struct

import numpy as np
import torch
from transformers import AutoProcessor

from mibot.utils.io import compose_state, recover_action, resize_image, split_action


class Client:
    def __init__(self, host: str = "localhost", port: int = 50000) -> None:
        self.socket = socket.create_connection((host, port))
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _recv_all(sock, length):
        data = b""
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                raise ConnectionError("connection closed while receiving response")
            data += packet
        return data

    def _send(self, payload):
        payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.socket.sendall(struct.pack(">I", len(payload)) + payload)

    def _recv(self):
        size = struct.unpack(">I", self._recv_all(self.socket, 4))[0]
        return pickle.loads(self._recv_all(self.socket, size))

    @staticmethod
    def _messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
                    {"type": "image", "image": ego_obs},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left_wrist_obs},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": right_wrist_obs},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction}"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<bot></bot>"}]},
        ]

    def __call__(self, robot_state, ego_obs, left_wrist_obs, right_wrist_obs, instruction):
        ego_obs, left_wrist_obs, right_wrist_obs = [
            resize_image(image, factor=32, max_pixels=90000) for image in (ego_obs, left_wrist_obs, right_wrist_obs)
        ]

        payload = self.processor.apply_chat_template(
            [self._messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        payload["state"] = torch.from_numpy(
            compose_state(
                left_gripper=np.asarray(robot_state["left_gripper_pos"], dtype=np.float32),
                left_joint=np.asarray(robot_state["left_arm_joint"], dtype=np.float32),
                right_gripper=np.asarray(robot_state["right_gripper_pos"], dtype=np.float32),
                right_joint=np.asarray(robot_state["right_arm_joint"], dtype=np.float32),
            )
        )[None]

        self._send(payload)
        action = self._recv()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (1, 30, 32):
            action = action[0]
        if action.shape != (30, 32):
            raise ValueError(f"expected server output shape (30, 32), got {action.shape}")

        return {
            "raw_action": action,
            "action_components": split_action(action),
            "action_targets": recover_action(action, robot_state),
        }

    def close(self) -> None:
        self.socket.close()
