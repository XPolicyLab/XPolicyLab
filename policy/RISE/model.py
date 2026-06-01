from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = POLICY_DIR / "RISE"
OFFLINE_DIR = UPSTREAM_DIR / "policy_and_value" / "policy_offline_and_value"
SRC_DIR = OFFLINE_DIR / "src"


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.expected_action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        self._obs_by_env: dict[int, dict] = {}
        self.policy = None

        if self.action_type != "joint":
            raise ValueError("RISE upstream policy is joint-action based. Use action_type=joint for real inference.")

        self.policy = self._load_policy(model_cfg)
        print("[RISE] Model initialized from checkpoint.")

    def _load_policy(self, model_cfg):
        checkpoint_path = model_cfg.get("checkpoint_path")
        if not checkpoint_path or checkpoint_path == "null":
            raise FileNotFoundError(
                "RISE checkpoint_path is required for real inference. "
                "Set RISE_CHECKPOINT_PATH or deploy.yml checkpoint_path."
            )

        checkpoint_dir = _resolve_policy_path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"RISE checkpoint does not exist: {checkpoint_path}")

        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from openpi_value.policies import policy_config
        from openpi_value.training import config as training_config

        config_name = model_cfg.get("config_name") or "Policy_offline_release"
        train_cfg = training_config.get_config(config_name)
        train_cfg = _override_train_config(train_cfg, model_cfg)
        default_prompt = _none_if_null(model_cfg.get("default_prompt"))
        device = "cuda:0" if str(model_cfg.get("gpu_id", "0")) != "cpu" else "cpu"

        return policy_config.create_trained_policy(
            train_cfg,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=device,
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._obs_by_env[env_idx] = encode_obs(obs, self.action_type, self.robot_action_dim_info)

    def get_action(self):
        return self.get_action_batch([0])[0]

    def get_action_batch(self, env_idx_list):
        if self.policy is None:
            raise RuntimeError("RISE policy is not loaded.")

        action_batch = []
        for env_idx in env_idx_list:
            env_idx = int(env_idx)
            if env_idx not in self._obs_by_env:
                raise KeyError(f"No observation has been provided for env_idx={env_idx}.")

            result = self.policy.infer(_copy_obs_for_policy(self._obs_by_env[env_idx]))
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.shape[-1] < self.expected_action_dim:
                raise ValueError(
                    f"RISE returned action dim {actions.shape[-1]}, "
                    f"but XPolicyLab expects {self.expected_action_dim}."
                )

            actions = actions[..., : self.expected_action_dim]
            action_batch.append(
                unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type="obs")
            )

        return action_batch

    def reset(self):
        self._obs_by_env.clear()


def encode_obs(observation, action_type, robot_action_dim_info):
    return {
        "images": {
            "top_head": _standardize_rgb(observation["vision"]["cam_head"]["color"]),
            "hand_left": _standardize_rgb(observation["vision"]["cam_left_wrist"]["color"]),
            "hand_right": _standardize_rgb(observation["vision"]["cam_right_wrist"]["color"]),
        },
        "state": pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32),
        "prompt": _extract_prompt(observation),
    }


def _copy_obs_for_policy(observation):
    return {
        "images": dict(observation["images"]),
        "state": observation["state"],
        "prompt": observation["prompt"],
    }


def _standardize_rgb(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}.")

    # XPolicyLab env/debug/sim observations are already RGB-ordered.
    # Match training data resolution; upstream ResizeImages applies resize_with_pad to 224x224.
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized image shape (240, 320, 3), got {image.shape}.")
    return image.astype(np.uint8, copy=False)


def _extract_prompt(observation) -> str:
    instruction = observation.get("instruction", observation.get("instructions", ""))
    if isinstance(instruction, (list, tuple)):
        return str(instruction[0]) if instruction else ""
    return str(instruction)


def _resolve_policy_path(value) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (POLICY_DIR / path).resolve()


def _none_if_null(value):
    if value in (None, "null", "None", ""):
        return None
    return value


def _override_train_config(train_cfg, model_cfg):
    action_dim = _none_if_null(model_cfg.get("model_action_dim"))
    if action_dim:
        train_cfg = dataclasses.replace(
            train_cfg,
            model=dataclasses.replace(train_cfg.model, action_dim=int(action_dim)),
        )

    seed = model_cfg.get("seed")
    if seed is not None:
        train_cfg = dataclasses.replace(train_cfg, seed=int(seed))

    asset_id = _none_if_null(model_cfg.get("asset_id"))
    if asset_id:
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(
                train_cfg.data,
                assets=dataclasses.replace(train_cfg.data.assets, asset_id=str(asset_id)),
            ),
        )

    return train_cfg
