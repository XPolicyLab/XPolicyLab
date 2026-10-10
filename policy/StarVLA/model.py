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
from .runtime_config import (
    parse_bool,
    resolve_checkpoint_framework,
    resolve_include_state,
    validate_server_runtime_contract,
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
    image = np.asarray(image)

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
        execute_horizon = self.model_cfg.get("execute_horizon", self.action_chunk_size)
        if execute_horizon in (None, "", "null", "None"):
            execute_horizon = self.action_chunk_size
        self.execute_horizon = int(execute_horizon)
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}.")
        if self.execute_horizon > self.action_chunk_size:
            raise ValueError(
                f"execute_horizon={self.execute_horizon} exceeds action_chunk_size={self.action_chunk_size}."
            )
        self.unnorm_key = self.model_cfg.get("unnorm_key", "arx_x5")
        if self.unnorm_key in (None, "", "null", "None", "auto"):
            self.unnorm_key = None
        self.use_ddim = bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        self.image_size = tuple(self.model_cfg.get("image_size", [224, 224]))
        self.include_state = resolve_include_state(
            self.model_cfg.get("include_state", "auto"),
            self.model_cfg.get("checkpoint_path"),
        )
        self.require_runtime_contract = parse_bool(
            self.model_cfg.get("require_runtime_contract", True)
        )
        if self.require_runtime_contract:
            expected_framework = resolve_checkpoint_framework(
                self.model_cfg.get("checkpoint_path")
            )
            expected_pi_v3_forward = self.model_cfg.get("required_pi_v3_forward")
            if expected_pi_v3_forward in (None, "", "auto", "none", "null", "None"):
                expected_pi_v3_forward = None
            else:
                expected_pi_v3_forward = str(expected_pi_v3_forward)
            validate_server_runtime_contract(
                server_meta,
                include_state=self.include_state,
                action_dim=self.action_dim,
                unnorm_key=self.unnorm_key,
                expected_framework=expected_framework,
                expected_pi_v3_forward=expected_pi_v3_forward,
            )

        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[starVLA] connected to StarVLA server, action_dim={self.action_dim}, "
            f"chunk={self.action_chunk_size}, execute_horizon={self.execute_horizon}, "
            f"include_state={self.include_state}, action_order=xpolicy, metadata={server_meta}"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = [
            _extract_camera(observation, ["cam_head", "head_camera"]),
            _extract_camera(observation, ["cam_left_wrist", "left_camera"]),
            _extract_camera(observation, ["cam_right_wrist", "right_camera"]),
        ]
        images = [
            cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
            for image in images
        ]

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.model_cfg.get("task_name", "")

        converted_obs = {
            "lang": str(instruction),
            "image": images,
        }
        if self.include_state:
            state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
            if state.ndim == 1:
                state = state[None, :]
            if state.ndim != 2 or state.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected state shape (T, {self.action_dim}), got {state.shape}."
                )
            converted_obs["state"] = state

        return converted_obs

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
        }
        if self.unnorm_key is not None:
            vla_input["unnorm_key"] = self.unnorm_key
        response = self.client.predict_action(vla_input)
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        return np.asarray(response["data"]["actions"][0], dtype=np.float32)

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or step % self.execute_horizon == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk

        action_idx = min(step % self.execute_horizon, len(chunk) - 1)
        self.step_by_env[env_idx] = step + 1
        action = np.asarray(chunk[action_idx], dtype=np.float32)
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
