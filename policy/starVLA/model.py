from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_CUR_DIR = Path(__file__).resolve().parent


def _optional_path(value: str | None, *base_dirs: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return base_dirs[0] / path


def _decode_image(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed image bytes.")
        image = decoded

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], camera_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb", "colors"):
                if image_key in camera_obs:
                    return _decode_image(camera_obs[image_key])
        else:
            return _decode_image(camera_obs)
    raise KeyError(f"Missing camera from candidates: {camera_names}")


def _xpolicy_to_starvla_joint_order(vector: np.ndarray, robot_action_dim_info: dict[str, Any]) -> np.ndarray:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) != 2 or arm_dims != [6, 6] or ee_dims != [1, 1]:
        return vector
    left_arm = vector[..., 0:6]
    left_ee = vector[..., 6:7]
    right_arm = vector[..., 7:13]
    right_ee = vector[..., 13:14]
    return np.concatenate([left_arm, right_arm, left_ee, right_ee], axis=-1)


def _starvla_to_xpolicy_joint_order(vector: np.ndarray, robot_action_dim_info: dict[str, Any]) -> np.ndarray:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) != 2 or arm_dims != [6, 6] or ee_dims != [1, 1]:
        return vector
    left_arm = vector[..., 0:6]
    right_arm = vector[..., 6:12]
    left_ee = vector[..., 12:13]
    right_ee = vector[..., 13:14]
    return np.concatenate([left_arm, left_ee, right_arm, right_ee], axis=-1)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("starVLA currently supports action_type='joint' first.")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("starVLA requires env_cfg_type.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        starvla_root = _optional_path(
            self.model_cfg.get("starvla_root"),
            _CUR_DIR,
        ) or (_CUR_DIR / "source_starvla")
        if str(starvla_root) not in sys.path:
            sys.path.insert(0, str(starvla_root))

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(
            self.model_cfg.get("starvla_server_host", "127.0.0.1"),
            int(self.model_cfg.get("starvla_server_port", 5694)),
        )
        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = int(server_meta["action_chunk_size"])
        self.unnorm_key = self.model_cfg.get("unnorm_key", "new_embodiment")
        self.use_ddim = bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        self.image_size = tuple(self.model_cfg.get("image_size", [224, 224]))
        self.input_color_order = self.model_cfg.get("input_color_order", "bgr").lower()

        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[starVLA] connected to StarVLA server, action_dim={self.action_dim}, "
            f"chunk={self.action_chunk_size}, metadata={server_meta}"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = [
            _extract_camera(observation, ["cam_head", "head_camera"]),
            _extract_camera(observation, ["cam_left_wrist", "left_camera"]),
            _extract_camera(observation, ["cam_right_wrist", "right_camera"]),
        ]
        if self.input_color_order == "bgr":
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = [
            cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
            for image in images
        ]

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.model_cfg.get("task_name", "")

        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

        return {
            "lang": str(instruction),
            "image": images,
            "state": _xpolicy_to_starvla_joint_order(state, self.robot_action_dim_info),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self._convert_obs(obs)

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        if env_idx not in self.obs_by_env:
            raise AssertionError("update_obs must be called before get_action.")

        vla_input = {
            "examples": [self.obs_by_env[env_idx]],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "unnorm_key": self.unnorm_key,
        }
        response = self.client.predict_action(vla_input)
        return np.asarray(response["data"]["actions"][0], dtype=np.float32)

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or step % self.action_chunk_size == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk

        action_idx = min(step % self.action_chunk_size, len(chunk) - 1)
        self.step_by_env[env_idx] = step + 1
        action = _starvla_to_xpolicy_joint_order(
            np.asarray(chunk[action_idx], dtype=np.float32),
            self.robot_action_dim_info,
        )
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        return action

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [
            [
                unpack_robot_state(
                    self._next_action_vector(int(env_idx)),
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            ]
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.obs_by_env.clear()
        self.action_chunks_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]
