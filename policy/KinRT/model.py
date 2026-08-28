import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_SUPPORTED_ENV_CFG_TYPE = "arx_x5"
_SUPPORTED_ACTION_TYPE = "joint"


def _extract_step_number(value: Any) -> int | None:
    final_component = str(value).rstrip("/").split("/")[-1]
    digits = "".join(character for character in final_component if character.isdigit())
    return int(digits) if digits else None


def _resolve_model_root(model_cfg: dict[str, Any]) -> Path:
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name, model_path, or checkpoint_path is required for KinRT.")

    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_steps: list[Path] = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_steps.append(checkpoint_root)
    candidate_steps.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_steps:
        return checkpoint_root

    desired_step = _extract_step_number(model_cfg.get("checkpoint_num"))
    if desired_step is not None:
        for candidate in candidate_steps:
            if _extract_step_number(candidate.name) == desired_step:
                return candidate

    numeric_steps = [candidate for candidate in candidate_steps if _extract_step_number(candidate.name) is not None]
    if numeric_steps:
        return max(numeric_steps, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_steps[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.action_type = model_cfg.get("action_type", _SUPPORTED_ACTION_TYPE)
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        if self.action_type != _SUPPORTED_ACTION_TYPE:
            raise ValueError(f"KinRT supports action_type={_SUPPORTED_ACTION_TYPE!r}, got {self.action_type!r}.")
        if self.env_cfg_type != _SUPPORTED_ENV_CFG_TYPE:
            raise ValueError(
                f"KinRT supports env_cfg_type={_SUPPORTED_ENV_CFG_TYPE!r}, got {self.env_cfg_type!r}."
            )

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 50))
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive.")

        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self.policy = self._load_policy(model_cfg)
        self.model = self.policy

    def _load_policy(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "kinrt_lora_robodojo")
        repo_id = model_cfg.get("repo_id", "RoboDojo-KinRT-arx_x5-joint")
        model_root = _resolve_model_root(model_cfg)
        config = _config.get_config(train_config_name)
        rank_overrides = {
            key: int(model_cfg[key])
            for key in ("paligemma_lora_rank", "action_expert_lora_rank")
            if model_cfg.get(key) is not None
        }
        if rank_overrides:
            config = dataclasses.replace(config, model=dataclasses.replace(config.model, **rank_overrides))
        norm_stats = _normalize.load(model_root / "assets" / str(repo_id)) if repo_id is not None else None
        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = stack_obs(
            [encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list]
        )

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("Call update_obs or update_obs_batch before requesting an action.")

        requested_env_indices = self._latest_env_idx_list if env_idx_list is None else list(env_idx_list)
        if len(requested_env_indices) != self.observation_window["state"].shape[0]:
            raise ValueError(
                "The requested environment count does not match the latest observation batch: "
                f"{len(requested_env_indices)} != {self.observation_window['state'].shape[0]}."
            )

        action_batch = []
        for batch_index in range(len(requested_env_indices)):
            observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(observation, **kwargs)["actions"][: self.action_chunk_size]
            action_batch.append(
                unpack_robot_state(
                    actions,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return action_batch

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]


def encode_obs(observation: dict[str, Any], action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    if "images" in observation and "state" in observation:
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        return {
            "state": np.asarray(observation["state"], dtype=np.float32),
            "images": images,
            "prompt": observation.get("instruction") or observation.get("instructions"),
        }

    images = {
        "cam_high": ensure_chw_uint8(_extract_image(observation, ("cam_head", "cam_high"))),
        "cam_left_wrist": ensure_chw_uint8(_extract_image(observation, ("cam_left_wrist",))),
        "cam_right_wrist": ensure_chw_uint8(_extract_image(observation, ("cam_right_wrist",))),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    return {
        "state": state,
        "images": images,
        "prompt": observation.get("instruction") or observation.get("instructions"),
    }


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            camera_name: np.stack([obs["images"][camera_name] for obs in obs_list], axis=0)
            for camera_name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {camera_name: image_batch[batch_index] for camera_name, image_batch in obs["images"].items()},
        "prompt": obs["prompt"][batch_index],
    }


def _extract_image(observation: dict[str, Any], candidate_names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        camera = vision[candidate_name]
        if isinstance(camera, dict):
            for image_key in ("color", "rgb"):
                if image_key in camera:
                    return camera[image_key]
        else:
            return camera
    raise KeyError(f"Missing required camera; checked {candidate_names}.")


def ensure_chw_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image = np.transpose(image, (2, 0, 1))
    elif image.shape[0] not in (1, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}.")
    return np.ascontiguousarray(image)
