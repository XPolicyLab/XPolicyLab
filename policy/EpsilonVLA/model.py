#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import os
import sys
from typing import Any

import cv2
import numpy as np

from epsilonvla.policies import policy_config as _policy_config
from epsilonvla.shared import normalize as _normalize
from epsilonvla.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots, ckpt_name_is_path
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _is_weight_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "params").exists() or (path / "assets").exists())


def _cfg_or_env(model_cfg: dict[str, Any], key: str, *env_names: str) -> str:
    # Wrapper exports EPSILONVLA_* from platform train_config_name / repo_id.
    for env_name in env_names:
        env = str(os.environ.get(env_name) or "").strip()
        if env:
            return env
    return str(model_cfg.get(key) or "").strip()


def _explicit_ckpt_requested(model_cfg: dict[str, Any]) -> bool:
    if any(str(model_cfg.get(key) or "").strip() for key in ("model_path", "checkpoint_path")):
        return True
    return ckpt_name_is_path(model_cfg.get("ckpt_name"))


def _resolve_epsilonvla_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for EpsilonVLA.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    # Platform/custom eval passes an absolute path to the ensured step dir.
    # Trust it when the directory itself is loadable; do not re-pick by
    # deploy.yml checkpoint_num (often a different step than the user filled).
    if _is_weight_dir(checkpoint_root) and _explicit_ckpt_requested(model_cfg):
        return checkpoint_root

    candidate_dirs = []
    if _is_weight_dir(checkpoint_root):
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if _is_weight_dir(child)
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = _cfg_or_env(model_cfg, "checkpoint_num", "EPSILONVLA_CHECKPOINT_NUM")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


def _resolve_repo_id(model_cfg: dict[str, Any], model_root: Path) -> str | None:
    configured = _cfg_or_env(model_cfg, "repo_id", "EPSILONVLA_REPO_ID")
    assets = model_root / "assets"
    if configured and (assets / configured).exists():
        return configured
    if assets.is_dir():
        subdirs = sorted(path.name for path in assets.iterdir() if path.is_dir())
        if len(subdirs) == 1:
            return subdirs[0]
    return configured or None


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.policy_batch_size = max(1, int(model_cfg.get("policy_batch_size", 5)))
        execute_steps = model_cfg.get("execute_steps")
        self.execute_steps = None if execute_steps is None else int(execute_steps)
        if self.execute_steps is not None and self.execute_steps < 1:
            raise ValueError(f"execute_steps must be positive, got {self.execute_steps}.")
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = (
            _cfg_or_env(model_cfg, "train_config_name", "EPSILONVLA_TRAIN_CONFIG_NAME")
            or "epsilonvla_robodojo_posttrain_abs_joint_infer"
        )
        model_root = _resolve_epsilonvla_model_root(model_cfg)
        repo_id = _resolve_repo_id(model_cfg, model_root)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for start in range(0, len(env_idx_list), self.policy_batch_size):
            stop = min(start + self.policy_batch_size, len(env_idx_list))
            batch_observations = [
                slice_stacked_obs(self.observation_window, batch_index)
                for batch_index in range(start, stop)
            ]
            if not hasattr(self.policy, "infer_batch"):
                batch_actions = [
                    self.policy.infer(observation, **kwargs)["actions"]
                    for observation in batch_observations
                ]
            elif len(batch_observations) == 1:
                batch_actions = [
                    self.policy.infer(batch_observations[0], **kwargs)["actions"]
                ]
            else:
                batch_actions = self.policy.infer_batch(batch_observations, **kwargs)["actions"]

            for actions in batch_actions:
                if self.execute_steps is not None:
                    actions = actions[: self.execute_steps]
                if self.robot_action_dim_info is None:
                    action_list.append(actions)
                else:
                    action_list.append(
                        unpack_robot_state(
                            actions,
                            self.action_type,
                            self.robot_action_dim_info,
                            source_type="obs",
                        )
                    )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_compressed_image(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)

    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_compressed_image(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)
