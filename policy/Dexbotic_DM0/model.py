"""XPolicyLab Model wrapper for Dexbotic DM0."""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info

from .dm0_infer import load_dm0_infer
from .dm0_state import ACTION_CHUNK_SIZE, pack_dm0_state, unpack_dm0_action_step

_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINTS_DIR = os.path.join(_POLICY_DIR, "checkpoints")


def _normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return os.path.abspath(value)


def _find_latest_checkpoint(run_dir: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None
    ckpts = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, name))
    ]
    if not ckpts:
        if os.path.isfile(os.path.join(run_dir, "config.json")):
            return run_dir
        return None
    ckpts.sort(key=lambda path: int(os.path.basename(path).split("-")[-1]))
    return ckpts[-1]


def _resolve_run_basename(model_cfg: dict) -> str:
    """ckpt_name may already be the full 6-tuple directory name."""
    ckpt_name = str(model_cfg.get("ckpt_name", ""))
    if ckpt_name.count("-") >= 3:
        return ckpt_name
    bench_name = model_cfg.get("bench_name", "")
    action_type = model_cfg.get("action_type", "")
    env_cfg_type = model_cfg.get("env_cfg_type", "")
    expert_data_num = model_cfg.get("expert_data_num", "")
    seed = model_cfg.get("seed", "0")
    return f"{bench_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"


def _list_candidate_run_dirs(run_basename: str) -> list[str]:
    candidates = [
        os.path.join(_CHECKPOINTS_DIR, run_basename),
        os.path.join(_CHECKPOINTS_DIR, f"{run_basename}_bak"),
    ]
    if os.path.isdir(_CHECKPOINTS_DIR):
        prefix = f"{run_basename}-"
        for entry in sorted(os.listdir(_CHECKPOINTS_DIR), key=lambda name: -os.path.getmtime(os.path.join(_CHECKPOINTS_DIR, name))):
            path = os.path.join(_CHECKPOINTS_DIR, entry)
            if entry.startswith(prefix) and os.path.isdir(path) and path not in candidates:
                candidates.append(path)
    return candidates


def _resolve_model_assets(model_cfg: dict) -> tuple[str, Optional[str]]:
    model_path = _normalize_optional_path(model_cfg.get("model_path"))
    norm_stats_path = _normalize_optional_path(model_cfg.get("norm_stats_path"))
    run_dir = None

    if model_path:
        latest = _find_latest_checkpoint(model_path)
        if latest is not None:
            run_dir = model_path if latest != model_path else os.path.dirname(model_path)
            model_path = latest
    else:
        run_basename = _resolve_run_basename(model_cfg)

        for candidate_run_dir in _list_candidate_run_dirs(run_basename):
            candidate_model_path = _find_latest_checkpoint(candidate_run_dir)
            if candidate_model_path is not None:
                run_dir = candidate_run_dir
                model_path = candidate_model_path
                break

    if model_path is None:
        raise FileNotFoundError(
            "No DM0 checkpoint found. Train first or set model_path / MODEL_PATH to a checkpoint-* directory."
        )

    if norm_stats_path is None:
        for candidate_dir in (run_dir, model_path, os.path.dirname(model_path)):
            if not candidate_dir:
                continue
            candidate = os.path.join(candidate_dir, "norm_stats.json")
            if os.path.isfile(candidate):
                norm_stats_path = candidate
                break

    return model_path, norm_stats_path


def _extract_rgb_image(observation: dict, camera_name: str) -> np.ndarray:
    vision = observation.get("vision", {})
    if camera_name not in vision:
        raise KeyError(f"Missing observation['vision']['{camera_name}']")

    cam_data = vision[camera_name]
    img = cam_data.get("color", cam_data) if isinstance(cam_data, dict) else cam_data
    img = np.asarray(img)

    if img.ndim == 1 and img.dtype == np.uint8:
        img = decode_image_bit(img)

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    return img.astype(np.uint8)


def _normalize_prompt(observation: dict, default_prompt: str) -> str:
    instruction = observation.get("instruction", observation.get("instructions", default_prompt))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else default_prompt
    if instruction is None:
        return default_prompt
    return str(instruction)


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.default_prompt = model_cfg.get("prompt") or "Perform the instructed bimanual manipulation task."
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", ACTION_CHUNK_SIZE))

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), (
            "Arm and EE action dimensions must match"
        )

        model_path, norm_stats_path = _resolve_model_assets(model_cfg)
        print(f"[Dexbotic_DM0] model_path={model_path}")
        if norm_stats_path:
            print(f"[Dexbotic_DM0] norm_stats_path={norm_stats_path}")

        self.infer = load_dm0_infer(model_path, norm_stats_path=norm_stats_path)

        self._obs_buffer_batch: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[Dexbotic_DM0] Initialized | action_type={self.action_type} | "
            f"chunk_size={self.action_chunk_size}"
        )

    def update_obs(self, obs: dict):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict]):
        self._latest_env_idx_list = [obs.get("env_idx", idx) for idx, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._obs_buffer_batch[env_idx] = {
                "prompt": _normalize_prompt(obs, self.default_prompt),
                "images": [
                    _extract_rgb_image(obs, "cam_head"),
                    _extract_rgb_image(obs, "cam_left_wrist"),
                    _extract_rgb_image(obs, "cam_right_wrist"),
                ],
                "state": pack_dm0_state(obs),
            }

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list: Optional[list[int]] = None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        action_batch = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_buffer_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")

            payload = self._obs_buffer_batch[env_idx]
            action_chunk = self.infer.predict(
                prompt=payload["prompt"],
                images_rgb=payload["images"],
                state=payload["state"],
            )
            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)

            steps = min(self.action_chunk_size, action_chunk.shape[0])
            action_steps = [
                unpack_dm0_action_step(
                    action_chunk[step_idx],
                    self.action_type,
                    self.robot_action_dim_info,
                )
                for step_idx in range(steps)
            ]
            action_batch.append(action_steps)

        return action_batch

    def reset(self):
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]
        print("[Dexbotic_DM0] Reset")
