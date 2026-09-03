from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .runtime_config import checkpoint_run_dir, load_and_validate_profile, resolve_checkpoint_file


_POLICY_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _POLICY_DIR / "source_starvla"


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an image array with ndim=3, got {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            for key in ("color", "rgb", "colors"):
                if key in value:
                    return _ensure_hwc_uint8(value[key])
        else:
            return _ensure_hwc_uint8(value)
    raise KeyError(f"Missing RGB camera; tried {names}.")


def _normalize_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
    scale = np.maximum(high - low, 1e-8)
    normalized = np.where(mask, 2.0 * (values - low) / scale - 1.0, values)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _unnormalize_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
    clipped = np.clip(values, -1.0, 1.0)
    return np.where(mask, 0.5 * (clipped + 1.0) * (high - low) + low, clipped).astype(np.float32)


class Model(ModelTemplate):
    """Evaluation-only RoboDojo adapter for the released Qwen3.5 MMDiT profile."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        checkpoint_root = resolve_checkpoint_root(
            self.model_cfg,
            _POLICY_DIR / "checkpoints",
            policy_dir=_POLICY_DIR,
        )
        self.checkpoint_path = resolve_checkpoint_file(checkpoint_root)
        _, statistics = load_and_validate_profile(self.checkpoint_path, self.model_cfg)

        self.action_type = self.model_cfg["action_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.model_cfg["env_cfg_type"])
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        if self.action_dim != 14:
            raise ValueError(f"This profile requires a 14D dual-arm robot, got {self.action_dim}.")

        self.execute_horizon = int(self.model_cfg["execute_horizon"])
        self.image_size = tuple(int(value) for value in self.model_cfg["image_size"])
        embodiment = statistics["new_embodiment"]
        self.state_stats = embodiment["state"]
        self.action_stats = embodiment["action"]

        base_vlm = self.model_cfg.get("base_vlm")
        if base_vlm in (None, "", "null", "None"):
            bundled_base_vlm = checkpoint_run_dir(self.checkpoint_path) / "base_vlm"
            if bundled_base_vlm.is_dir():
                base_vlm = bundled_base_vlm
        if base_vlm in (None, "", "null", "None"):
            raise FileNotFoundError(
                "Qwen3.5 tokenizer/config assets were not found. Keep base_vlm/ beside "
                "pytorch_model.pt or set STARVLA_BASE_VLM."
            )
        os.environ["STARVLA_BASE_VLM"] = str(base_vlm)
        if str(_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(_SOURCE_ROOT))

        from starVLA.model.framework.base_framework import baseframework

        if not torch.cuda.is_available():
            raise RuntimeError("Meituan_Robotics_0 evaluation requires a CUDA GPU.")
        self.device = torch.device("cuda")
        self.model = baseframework.from_pretrained(str(self.checkpoint_path)).eval().to(self.device)
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list = [0]
        print(
            "[Meituan_Robotics_0] loaded "
            f"{self.checkpoint_path}; RGB=640x480, action=abs/q99, horizon=50, "
            f"execute_horizon={self.execute_horizon}, Euler steps=10"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = [
            _extract_camera(observation, ("cam_head", "head_camera")),
            _extract_camera(observation, ("cam_left_wrist", "left_camera")),
            _extract_camera(observation, ("cam_right_wrist", "right_camera")),
        ]
        images = [
            cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            if (image.shape[1], image.shape[0]) != self.image_size
            else image
            for image in images
        ]
        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if not instruction:
            instruction = self.model_cfg.get("task_name", "")

        raw_state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        state = _normalize_q99(raw_state, self.state_stats)
        if state.ndim == 1:
            state = state[None, :]
        return {"image": images, "lang": str(instruction), "state": state}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self._convert_obs(obs)

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        missing = [int(index) for index in env_idx_list if int(index) not in self.obs_by_env]
        if missing:
            raise AssertionError(f"update_obs_batch must be called first for envs {missing}.")
        examples = [self.obs_by_env[int(index)] for index in env_idx_list]
        prediction = self.model.predict_action(examples=examples)["normalized_actions"]
        actions = _unnormalize_q99(np.asarray(prediction, dtype=np.float32), self.action_stats)
        actions = actions[:, : self.execute_horizon]
        return [
            unpack_robot_state(
                env_actions,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for env_actions in actions
        ]

    def reset(self):
        self.obs_by_env.clear()
        self._latest_env_idx_list = [0]
