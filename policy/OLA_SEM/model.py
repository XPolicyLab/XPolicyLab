from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


XPL_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = Path(__file__).resolve().parent
VENDORED_ROOT = POLICY_DIR / "ola_sem"
UPSTREAM_POLICY_DIR = VENDORED_ROOT / "inference" / "robotwin" / "Motus"


def _instruction(obs: dict[str, Any], fallback: str = "follow the instruction") -> str:
    value = obs.get("instruction", obs.get("instructions", fallback))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    text = str(value).strip()
    return text or fallback


def _rgb_image(obs: dict[str, Any], camera: str) -> np.ndarray:
    try:
        image = np.asarray(obs["vision"][camera]["color"])
    except KeyError as exc:
        raise KeyError(f"Missing RGB camera observation: vision.{camera}.color") from exc
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"vision.{camera}.color must be HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def compose_three_view_rgb(obs: dict[str, Any]) -> np.ndarray:
    """Reproduce OLA-SEM's 320x360 three-view mosaic from standard RGB arrays."""
    head = cv2.resize(_rgb_image(obs, "cam_head"), (320, 240))
    left = cv2.resize(_rgb_image(obs, "cam_left_wrist"), (160, 120))
    right = cv2.resize(_rgb_image(obs, "cam_right_wrist"), (160, 120))
    return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)


def pack_joint_state(obs: dict[str, Any], dim_info: dict[str, list[int]]) -> np.ndarray:
    arm_dims = list(dim_info["arm_dim"])
    ee_dims = list(dim_info["ee_dim"])
    if len(arm_dims) != 2 or len(ee_dims) != 2:
        raise ValueError(f"OLA-SEM requires a dual-arm robot, got {dim_info}")
    state = obs.get("state")
    if not isinstance(state, dict):
        raise KeyError("Observation is missing state")
    parts: list[np.ndarray] = []
    for side, arm_dim, ee_dim in zip(("left", "right"), arm_dims, ee_dims):
        arm_key = f"{side}_arm_joint_state"
        ee_key = f"{side}_ee_joint_state"
        arm = np.asarray(state[arm_key], dtype=np.float32).reshape(-1)
        ee = np.asarray(state[ee_key], dtype=np.float32).reshape(-1)
        if arm.size != arm_dim or ee.size != ee_dim:
            raise ValueError(
                f"State shape mismatch for {arm_key}/{ee_key}: "
                f"got {arm.size}/{ee.size}, expected {arm_dim}/{ee_dim}"
            )
        parts.extend((arm, ee))
    return np.concatenate(parts, axis=0)


def unpack_joint_actions(
    actions: Any, dim_info: dict[str, list[int]]
) -> list[dict[str, np.ndarray]]:
    packed = np.asarray(actions, dtype=np.float32)
    if packed.ndim == 1:
        packed = packed[None, :]
    arm_dims = list(dim_info["arm_dim"])
    ee_dims = list(dim_info["ee_dim"])
    expected = sum(arm_dims) + sum(ee_dims)
    if packed.ndim != 2 or packed.shape[1] != expected:
        raise ValueError(f"Expected action chunk [T, {expected}], got {packed.shape}")
    result: list[dict[str, np.ndarray]] = []
    for row in packed:
        offset = 0
        item: dict[str, np.ndarray] = {}
        for side, arm_dim, ee_dim in zip(("left", "right"), arm_dims, ee_dims):
            item[f"{side}_arm_joint_state"] = row[offset : offset + arm_dim].copy()
            offset += arm_dim
            item[f"{side}_ee_joint_state"] = row[offset : offset + ee_dim].copy()
            offset += ee_dim
        result.append(item)
    return result


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg["action_type"])
        self.env_cfg_type = str(self.model_cfg["env_cfg_type"])
        if self.action_type != "joint":
            raise ValueError("OLA-SEM supports only action_type=joint")
        if self.env_cfg_type != "aloha_agilex":
            raise ValueError("OLA-SEM supports only env_cfg_type=aloha_agilex")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        if action_dim != 14:
            raise ValueError(f"OLA-SEM checkpoint requires 14-D qpos, got {action_dim}")

        checkpoint_root = resolve_checkpoint_root(
            self.model_cfg,
            POLICY_DIR / "checkpoints",
            policy_dir=POLICY_DIR,
        )
        checkpoint_model_dir = (
            checkpoint_root / "pytorch_model"
            if (checkpoint_root / "pytorch_model").is_dir()
            else checkpoint_root
        )
        self.checkpoint_root = checkpoint_root
        self.checkpoint_model_dir = checkpoint_model_dir
        self._validate_history_metadata()

        self.wan_path = self._required_path("wan_path", "OLA_SEM_WAN_PATH")
        self.vlm_path = self._required_path("vlm_path", "OLA_SEM_VLM_PATH")
        self.inference_mode = str(self.model_cfg.get("inference_mode", "history_flow"))
        self.num_inference_timesteps = int(
            self.model_cfg.get("num_inference_timesteps", 4)
        )
        self.history_action_noise_std = float(
            self.model_cfg.get("history_action_noise_std", 0.02)
        )
        self.future_video_denoise_fraction = float(
            self.model_cfg.get("future_video_denoise_fraction", 1.0)
        )
        if self.inference_mode != "history_flow":
            raise ValueError("This adapter requires inference_mode=history_flow")

        self.policy = self._load_policy()
        self._has_observation = False
        self._last_obs: dict[str, Any] | None = None
        print(
            "[OLA_SEM] initialized "
            f"checkpoint={self.checkpoint_model_dir} mode={self.inference_mode}"
        )

    def _required_path(self, config_key: str, env_key: str) -> Path:
        import os

        value = self.model_cfg.get(config_key) or os.environ.get(env_key)
        if not value:
            raise ValueError(f"Set deploy.yml {config_key} or {env_key}")
        path = Path(str(value)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{config_key} not found: {path}")
        return path

    def _validate_history_metadata(self) -> None:
        import json

        metadata_path = self.checkpoint_root / "config.json"
        if not metadata_path.is_file() and self.checkpoint_root.name == "pytorch_model":
            metadata_path = self.checkpoint_root.parent / "config.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        common = metadata.get("common", {})
        flow = metadata.get("flow_source", {})
        expected_chunk = int(common.get("action_chunk_size", 0))
        if expected_chunk == 0:
            expected_chunk = int(common.get("num_video_frames", 0)) * int(
                common.get("video_action_freq_ratio", 0)
            )
        expected = {
            "mode": "history",
            "video_mode": "gaussian",
            "history_length": expected_chunk,
        }
        actual = {
            "mode": flow.get("mode"),
            "video_mode": flow.get("video_mode", flow.get("mode")),
            "history_length": int(flow.get("history_length", -1)),
        }
        action_noise_std = float(flow.get("action_noise_std", -1.0))
        if expected_chunk != 16 or actual != expected or action_noise_std != 0.02:
            raise ValueError(
                f"Incompatible history-flow metadata in {metadata_path}: "
                f"expected {expected} with action_noise_std=0.02, "
                f"got {actual} with action_noise_std={action_noise_std}"
            )

    def _load_policy(self):
        import sys

        upstream_dir = str(UPSTREAM_POLICY_DIR)
        if upstream_dir not in sys.path:
            sys.path.insert(0, upstream_dir)
        from XPolicyLab.policy.OLA_SEM.ola_sem.inference.robotwin.Motus.deploy_policy import (
            MotusPolicy,
        )

        return MotusPolicy(
            checkpoint_path=str(self.checkpoint_model_dir),
            config_path=str(UPSTREAM_POLICY_DIR / "utils" / "robotwin.yml"),
            wan_path=str(self.wan_path),
            vlm_path=str(self.vlm_path),
            inference_mode=self.inference_mode,
            num_inference_timesteps=self.num_inference_timesteps,
            history_action_noise_std=self.history_action_noise_std,
            future_video_denoise_fraction=self.future_video_denoise_fraction,
            device=str(self.model_cfg.get("device", "cuda")),
        )

    def _native_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": compose_three_view_rgb(obs),
            "joint_action": {
                "vector": pack_joint_state(obs, self.robot_action_dim_info)
            },
        }

    def update_obs(self, obs: dict[str, Any]):
        native = self._native_observation(obs)
        self.policy.set_instruction(_instruction(obs))
        if self._has_observation:
            self.policy.record_executed_qpos(native)
        self.policy.update_obs(native)
        self._has_observation = True
        self._last_obs = obs

    def update_obs_batch(self, obs_list: list[dict[str, Any]]):
        if len(obs_list) != 1:
            raise NotImplementedError("OLA-SEM supports only single-environment inference")
        self.update_obs(obs_list[0])

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if not self._has_observation:
            raise RuntimeError("Call update_obs before get_action")
        actions = self.policy.get_action()
        result = unpack_joint_actions(actions, self.robot_action_dim_info)
        if len(result) != 16:
            raise ValueError(f"OLA-SEM must return a 16-step action chunk, got {len(result)}")
        return result

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise NotImplementedError("OLA-SEM supports only single-environment inference")
        return [self.get_action()]

    def reset(self):
        self.policy.obs_cache.clear()
        self.policy.action_cache.clear()
        self.policy.current_state = None
        self.policy.current_state_norm = None
        self.policy.is_first_step = True
        self.policy.prev_action = None
        self.policy.real_qpos_history.clear()
        self._has_observation = False
        self._last_obs = None
