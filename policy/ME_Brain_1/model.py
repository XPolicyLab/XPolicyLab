"""XPolicyLab adapter for Focus-VLWA closed-loop inference."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .hist_live import hist_from_obs
from .replan import replan_steps_from_env, slice_action_chunk

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _configure_focus_vlwa_import(model_cfg: dict[str, Any]) -> None:
    configured = os.environ.get("FOCUS_VLWA_SOURCE") or model_cfg.get("focus_vlwa_source")
    if not configured:
        linked_source = _POLICY_DIR / "focus-vlwa"
        if linked_source.exists() or linked_source.is_symlink():
            from ._source import configure_source

            configure_source()
        return
    source = Path(configured).expanduser()
    if not source.is_absolute():
        source = _POLICY_DIR / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Focus-VLWA source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_checkpoint(model_cfg: dict[str, Any]) -> Path:
    cli_checkpoint = model_cfg.get("ckpt_name")
    if cli_checkpoint and Path(str(cli_checkpoint)).is_absolute():
        checkpoint = Path(str(cli_checkpoint))
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("model_path or ckpt_name is required for FocusVLWA")
    checkpoint = next((path for path in candidates if (path / "model.safetensors").is_file()), candidates[0])
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Focus-VLWA checkpoint not found: {checkpoint}")
    return checkpoint


class Model(ModelTemplate):
    """Adapt Focus-VLWA chunk inference to the XPolicyLab model contract."""

    def __init__(self, model_cfg: dict[str, Any]):
        if not torch.cuda.is_available():
            raise RuntimeError("FocusVLWA closed-loop inference requires a CUDA GPU")
        _configure_focus_vlwa_import(model_cfg)
        from focus_vlwa.configs.model import load_model_config
        from focus_vlwa.inference import FocusVLWAPolicy

        self.task_name = str(model_cfg["task_name"])
        self.action_type = model_cfg.get("action_type", "joint")
        env_cfg_type = model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg_type) if env_cfg_type else None
        self.observation_window: dict[str, Any] | None = None
        self.latest_env_indices = [0]
        self.replan_steps = replan_steps_from_env()
        checkpoint = _resolve_checkpoint(model_cfg)
        config = load_model_config(checkpoint)
        history_mode = os.environ.get("FOCUS_VLWA_HISTORY_MODE") or model_cfg.get("history_mode")
        if history_mode is not None:
            config = dataclasses.replace(config, history_mode=history_mode)
        max_token_len = os.environ.get("FOCUS_VLWA_MAX_TOKEN_LEN") or model_cfg.get("max_token_len")
        if max_token_len is not None:
            config = dataclasses.replace(config, max_token_len=int(max_token_len))
        self.policy = FocusVLWAPolicy(
            checkpoint,
            config=config,
            device="cuda",
            tokenizer_path=model_cfg.get("tokenizer_path"),
            asset_id=model_cfg.get("asset_id", "arx_x5_sim"),
            num_steps=int(model_cfg.get("num_inference_steps", 10)),
        )
        self.model = self.policy

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self.latest_env_indices = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        encoded: list[dict[str, Any]] = []
        for observation in obs_list:
            item = encode_observation(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                default_prompt=self.task_name.replace("_", " "),
            )
            stamped_history = hist_from_obs(observation)
            if stamped_history is None:
                raise ValueError("Missing client-stamped history; use the supplied deploy.py")
            item.update(stamped_history)
            encoded.append(item)
        self.observation_window = stack_observations(encoded)

    def get_action(self, **kwargs: Any) -> Any:
        return self.get_action_batch(env_idx_list=[self.latest_env_indices[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list: list[int] | None = None, **kwargs: Any) -> list[Any]:
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch must be called before get_action")
        env_idx_list = env_idx_list or self.latest_env_indices
        action_chunks: list[Any] = []
        for env_index in env_idx_list:
            batch_index = self.latest_env_indices.index(env_index)
            observation = slice_stacked_observation(self.observation_window, batch_index)
            actions = self.policy.infer(observation, **kwargs)["actions"]
            if self.robot_action_dim_info is not None:
                actions = unpack_robot_state(
                    actions,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            action_chunks.append(slice_action_chunk(actions, self.replan_steps))
        return action_chunks

    def reset(self) -> None:
        self.observation_window = None
        self.latest_env_indices = [0]

    def reset_obsrvationwindows(self) -> None:
        self.reset()


def encode_observation(
    observation: dict[str, Any],
    action_type: str,
    robot_action_dim_info: Any,
    *,
    default_prompt: str,
) -> dict[str, Any]:
    """Convert canonical or raw RoboDojo observations into Focus-VLWA inputs."""
    if "images" in observation and "state" in observation:
        images = {
            name: ensure_chw_uint8(observation["images"][name])
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }
        state = np.asarray(observation["state"], dtype=np.float32)
    else:
        if robot_action_dim_info is None:
            raise ValueError("env_cfg_type is required for raw environment observations")
        images = {
            "cam_high": ensure_chw_uint8(_extract_image(observation, ("cam_high", "cam_head", "head_camera"))),
            "cam_left_wrist": ensure_chw_uint8(
                _extract_image(observation, ("cam_left_wrist", "left_camera", "left_wrist"))
            ),
            "cam_right_wrist": ensure_chw_uint8(
                _extract_image(observation, ("cam_right_wrist", "right_camera", "right_wrist"))
            ),
        }
        state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction") or observation.get("prompt") or default_prompt
    return {"state": state, "images": images, "prompt": str(prompt)}


def _extract_image(observation: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    vision = observation.get("vision", {})
    for name in candidates:
        if name not in vision:
            continue
        image = vision[name]
        if isinstance(image, dict):
            for key in ("color", "rgb"):
                if key in image:
                    return image[key]
        return image
    raise KeyError(f"No image found for camera names: {candidates}")


def ensure_chw_uint8(image: Any) -> np.ndarray:
    """Return server-decoded RGB pixels as a contiguous CHW uint8 array."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a rank-three image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.shape[-1] in (1, 3):
        array = np.transpose(array, (2, 0, 1))
    elif array.shape[0] not in (1, 3):
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return np.ascontiguousarray(array)


def stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack observations while retaining prompts as Python strings."""
    return {
        "state": np.stack([item["state"] for item in observations]),
        "images": {
            name: np.stack([item["images"][name] for item in observations])
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "prompt": [item["prompt"] for item in observations],
        "hist_images": np.stack([item["hist_images"] for item in observations]),
        "hist_mask": np.stack([item["hist_mask"] for item in observations]),
    }


def slice_stacked_observation(observation: dict[str, Any], index: int) -> dict[str, Any]:
    """Select one environment observation for the single-sample inference API."""
    return {
        "state": observation["state"][index],
        "images": {name: value[index] for name, value in observation["images"].items()},
        "prompt": observation["prompt"][index],
        "hist_images": observation["hist_images"][index],
        "hist_mask": observation["hist_mask"][index],
    }
