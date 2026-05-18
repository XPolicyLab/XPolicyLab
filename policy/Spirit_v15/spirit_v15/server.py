from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SpiritVLAPolicy
from robochallenge.runner.executor import _post_process_action
from robochallenge.runner.task_info import TASK_INFO, TASKS_USE_LESS_CHUNK_SIZE, TASTS_APPLY_GRIPPER_BINARIZATION


LOGGER = logging.getLogger("spirit.robotwin.server")
TASK_NAME_ALIASES = {
    "stack_bowls_three": "stack_bowls",
    "stack_bowls_two": "stack_bowls",
}


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


class SpiritRobotWinPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        task_name: str | None = None,
        used_chunk_size: int = 60,
        raw_embodiment_stats_json_path: str | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SpiritVLAPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.default_task_name = None
        if task_name:
            self.default_task_name = self._resolve_task_name(task_name)
        self.used_chunk_size = int(used_chunk_size)
        self.raw_embodiment_stats = None
        if raw_embodiment_stats_json_path:
            with open(raw_embodiment_stats_json_path, "r", encoding="utf-8") as file:
                self.raw_embodiment_stats = json.load(file)

    def reset(self) -> None:
        return None

    def infer(self, observation: Dict[str, Any], instruction: str | None = None, task_name: str | None = None) -> Dict[str, Any]:
        resolved_task_name = self._resolve_task_name(task_name)
        batch = self._prepare_batch(observation, resolved_task_name)

        used_chunk_size = self.used_chunk_size
        if resolved_task_name in TASKS_USE_LESS_CHUNK_SIZE:
            used_chunk_size = 40

        binarization_threshold = TASTS_APPLY_GRIPPER_BINARIZATION.get(resolved_task_name)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad(),
        ):
            action_tensor = self.policy.select_action(batch).cpu()

        actions = _post_process_action(
            action_tensor.squeeze(0).numpy(),
            batch["observation.state.before_norm"].numpy(),
            TASK_INFO[resolved_task_name]["robot_type"],
            used_chunk_size,
            self.raw_embodiment_stats,
            binarization_threshold,
        )
        return {
            "actions": actions,
            "action_type": self._resolve_action_type(resolved_task_name),
            "task_name": resolved_task_name,
            "instruction": instruction,
        }

    def _resolve_task_name(self, task_name: str | None) -> str:
        for candidate in (task_name, self.default_task_name):
            if not candidate:
                continue
            if candidate in TASK_INFO:
                return candidate
            alias = TASK_NAME_ALIASES.get(candidate)
            if alias in TASK_INFO:
                return alias
        available = ", ".join(sorted(TASK_INFO.keys()))
        raise KeyError(f"unsupported Spirit task name: {task_name!r}; available tasks: {available}")

    def _prepare_batch(self, observation: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        robot_type = TASK_INFO[task_name]["robot_type"]
        item: Dict[str, Any] = {
            "task": [TASK_INFO[task_name]["task"]],
            "normalized_in_getitem": torch.tensor([False]),
            "batch_source": "rb",
            "robot_type": [robot_type],
        }

        state_tensor = self._extract_internal_state(observation, robot_type)
        item["observation.state.before_norm"] = state_tensor.clone()
        item["observation.state"] = state_tensor.unsqueeze(0).to(self.device)

        semantic_images = {
            "high": observation["observation"]["head_camera"]["rgb"],
            "left_hand": observation["observation"]["left_camera"]["rgb"],
            "right_hand": observation["observation"]["right_camera"]["rgb"],
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            image = semantic_images[TASK_INFO[task_name][key]]
            item[key] = self._image_to_tensor(image).unsqueeze(0).to(self.device)
        return item

    def _extract_internal_state(self, observation: Dict[str, Any], robot_type: str) -> torch.Tensor:
        endpose = observation.get("endpose") or {}
        if robot_type == "aloha":
            if all(key in endpose for key in ("left_endpose", "left_gripper", "right_endpose", "right_gripper")):
                return self._dual_ee_to_internal_state(
                    left_endpose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    left_gripper=float(endpose["left_gripper"]),
                    right_endpose=np.asarray(endpose["right_endpose"], dtype=np.float32),
                    right_gripper=float(endpose["right_gripper"]),
                )

        if robot_type in {"ARX5", "Franka", "UR5"}:
            if "left_endpose" in endpose and "left_gripper" in endpose:
                return self._single_ee_to_internal_state(
                    ee_pose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    gripper=float(endpose["left_gripper"]),
                    robot_type=robot_type,
                )

        if "joint_action" in observation and "vector" in observation["joint_action"]:
            return self._robotwin_joint_state_to_internal(
                np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                robot_type,
            )
        if "action" in observation:
            return self._robotwin_joint_state_to_internal(np.asarray(observation["action"], dtype=np.float32), robot_type)
        raise KeyError("missing usable state in endpose, joint_action.vector, or action")

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected RGB image with shape [H, W, 3], got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0) if np.issubdtype(array.dtype, np.floating) else array
            if array.max() <= 1.0:
                array = (array * 255.0).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        resized = Image.fromarray(array, mode="RGB").resize((320, 240), Image.BILINEAR)
        return torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0

    @staticmethod
    def _robotwin_joint_state_to_internal(state: np.ndarray, robot_type: str) -> torch.Tensor:
        from scipy.spatial.transform import Rotation

        state_tensor = torch.zeros(14, dtype=torch.float32)
        if robot_type == "ARX5":
            if state.shape[0] != 7:
                raise ValueError(f"expected 7-dim ARX5 state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_euler("xyz", state[3:6], degrees=False).as_rotvec())
            state_tensor[6] = torch.tensor(state[6], dtype=torch.float32)
            return state_tensor
        if robot_type == "UR5":
            if state.shape[0] not in {7, 14}:
                raise ValueError(f"expected 7-dim or 14-dim UR5 state, got {state.shape}")
            state_tensor[:7] = torch.from_numpy(state[:7])
            return state_tensor
        if robot_type == "Franka":
            if state.shape[0] not in {8, 16}:
                raise ValueError(f"expected 8-dim or 16-dim Franka state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_quat(state[3:7]).as_rotvec())
            state_tensor[6] = torch.tensor(state[7], dtype=torch.float32)
            return state_tensor
        if robot_type == "aloha":
            raise ValueError(
                f"aloha requires endpose fields in observation; received only joint state with shape {state.shape}"
            )
        raise ValueError(f"unsupported robot type: {robot_type}")

    @staticmethod
    def _single_ee_to_internal_state(ee_pose: np.ndarray, gripper: float, robot_type: str) -> torch.Tensor:
        from scipy.spatial.transform import Rotation

        if ee_pose.shape[0] != 7:
            raise ValueError(f"expected 7-dim end-effector pose, got {ee_pose.shape}")

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(ee_pose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(ee_pose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _dual_ee_to_internal_state(
        left_endpose: np.ndarray,
        left_gripper: float,
        right_endpose: np.ndarray,
        right_gripper: float,
    ) -> torch.Tensor:
        from scipy.spatial.transform import Rotation

        if left_endpose.shape[0] != 7 or right_endpose.shape[0] != 7:
            raise ValueError(
                f"expected 7-dim dual-arm endpose, got left={left_endpose.shape}, right={right_endpose.shape}"
            )

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(left_endpose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(left_endpose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(left_gripper, dtype=torch.float32)
        state_tensor[7:10] = torch.from_numpy(right_endpose[:3])
        state_tensor[10:13] = torch.tensor(Rotation.from_quat(right_endpose[3:]).as_rotvec())
        state_tensor[13] = torch.tensor(right_gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _resolve_action_type(task_name: str) -> str:
        action_type = TASK_INFO[task_name].get("action_type")
        if action_type == "leftjoint":
            return "qpos"
        return "ee"


class SpiritPolicyServer:
    def __init__(self, host: str, port: int, model: SpiritRobotWinPolicy):
        self.host = host
        self.port = int(port)
        self.model = model
        self._lock = threading.Lock()

    def handle_request(self, request: Dict[str, Any]) -> Any:
        endpoint = request.get("endpoint")

        if endpoint == "ping":
            return {"status": "ok"}

        if endpoint == "reset":
            with self._lock:
                self.model.reset()
            return {"status": "ok"}

        if endpoint != "inference":
            raise ValueError(f"unsupported endpoint: {endpoint}")

        observation = request.get("observation")
        if observation is None:
            raise ValueError("missing observation")

        with self._lock:
            return self.model.infer(
                observation=observation,
                instruction=request.get("instruction"),
                task_name=request.get("task_name"),
            )

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            LOGGER.info("Spirit server listening on %s:%s", self.host, self.port)

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
    parser = argparse.ArgumentParser(description="Spirit RoboTwin policy server")
    parser.add_argument("--checkpoint_path", required=True, help="Spirit checkpoint directory")
    parser.add_argument("--task_name", default=None, help="Default Spirit task key or RobotWin alias")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--used_chunk_size", type=int, default=60)
    parser.add_argument("--raw_embodiment_stats_json_path", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    model = SpiritRobotWinPolicy(
        checkpoint_path=args.checkpoint_path,
        task_name=args.task_name,
        used_chunk_size=args.used_chunk_size,
        raw_embodiment_stats_json_path=args.raw_embodiment_stats_json_path,
    )
    server = SpiritPolicyServer(host=args.host, port=args.port, model=model)
    server.serve_forever()


if __name__ == "__main__":
    main()