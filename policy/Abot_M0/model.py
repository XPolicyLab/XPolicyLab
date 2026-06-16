from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

_CUR_DIR = Path(__file__).resolve().parent
_ABOT_ROOT = _CUR_DIR / "abot_m0"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

if str(_ABOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ABOT_ROOT))

from ABot.model.framework.base_framework import baseframework
from deployment.model_server.tools.image_tools import to_pil_preserve

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_CAMERA_CANDIDATES = {
    "cam_high": ["cam_high", "cam_head", "head_camera", "top_camera"],
    "cam_left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "cam_right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
}


def _normalize_prompt_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_prompt_value(item)
            if normalized is not None:
                return normalized
        return None

    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def resolve_prompt(observation: dict[str, Any], default_prompt: str) -> str:
    for key in ("prompt", "instruction", "instructions", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return prompt

    fallback = _normalize_prompt_value(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or model config.")
    return fallback


def extract_image(observation: dict[str, Any], candidate_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return np.asarray(image[image_key])
        else:
            return np.asarray(image)
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def decode_compressed_image(image_buffer: np.ndarray) -> np.ndarray:
    return decode_image_bit(image_buffer)


def ensure_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """Observations are RGB HWC; compressed JPEG decoded without channel swap."""
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_compressed_image(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


# Training (AgilexDataConfig / action_keys concat): L_j(6), R_j(6), L_g(1), R_g(1)
# Deploy / LeRobot modality.json / pack_robot_state: L_j(6), L_g(1), R_j(6), R_g(1)
def modality_layout_to_train_layout(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.concatenate([x[..., 0:6], x[..., 7:13], x[..., 6:7], x[..., 13:14]], axis=-1)


def train_layout_to_modality_layout(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.concatenate([x[..., 0:6], x[..., 12:13], x[..., 6:12], x[..., 13:14]], axis=-1)


def _looks_like_modality_order_stats(q01: np.ndarray) -> bool:
    q01 = np.asarray(q01)
    if q01.shape[-1] != 14:
        return False
    # Modality layout stores left gripper at index 6 (~0); training layout has right joint 0 there.
    return abs(float(q01[6])) < 1e-6 and abs(float(q01[12])) > 0.05


def _permute_action_norm_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    for key in ("q01", "q99", "mask"):
        if key in out and out[key] is not None:
            arr = modality_layout_to_train_layout(np.asarray(out[key]))
            out[key] = arr.tolist() if key != "mask" else arr.astype(bool).tolist()
    return out


def _unnormalize_actions_train_layout(
    normalized_actions: np.ndarray, action_norm_stats: dict[str, Any]
) -> np.ndarray:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    for grip_idx in (12, 13):
        normalized_actions[:, grip_idx] = np.where(normalized_actions[:, grip_idx] < 0.5, 0, 1)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def _ensure_dataset_statistics(run_dir: Path) -> None:
    stats_path = run_dir / "dataset_statistics.json"
    if stats_path.exists():
        return

    stats_source = Path(
        os.environ.get(
            "ABOT_STATS_JSON",
            "/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot/meta/stats_gr00t.json",
        )
    ).expanduser()
    if not stats_source.is_file():
        raise FileNotFoundError(
            f"Missing `{stats_path}` and fallback stats file `{stats_source}`."
        )

    import json

    with open(stats_source, "r", encoding="utf-8") as handle:
        gr00t_stats = json.load(handle)

    action_stats = gr00t_stats.get("action", gr00t_stats)
    q01 = action_stats.get("q01")
    q99 = action_stats.get("q99")
    if q01 is None or q99 is None:
        raise ValueError(f"Fallback stats `{stats_source}` missing action q01/q99.")

    unnorm_key = os.environ.get("ABOT_UNNORM_KEY", "robodojo_sim")
    q01_train = modality_layout_to_train_layout(np.asarray(q01)).tolist()
    q99_train = modality_layout_to_train_layout(np.asarray(q99)).tolist()
    payload = {
        unnorm_key: {
            "action": {
                "q01": q01_train,
                "q99": q99_train,
                "mask": [True] * len(q01_train),
            }
        }
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Abot_M0] Wrote missing dataset statistics to {stats_path}")


def _resolve_checkpoint_path(model_cfg: dict[str, Any]) -> Path:
    checkpoint_num = model_cfg.get("checkpoint_num")
    if checkpoint_num is not None:
        step_name = f"steps_{int(checkpoint_num)}_pytorch_model.pt"
    else:
        step_name = "steps_60000_pytorch_model.pt"

    tuple_keys = ("dataset_name", "ckpt_name", "env_cfg_type", "expert_data_num", "action_type", "seed")
    if all(model_cfg.get(key) is not None for key in tuple_keys):
        ckpt_setting = "-".join(str(model_cfg[key]) for key in tuple_keys)
        ckpt_dir = (_CHECKPOINTS_DIR / ckpt_setting).expanduser().resolve()
        direct = ckpt_dir / "checkpoints" / step_name
        if direct.is_file():
            return direct
        nested = ckpt_dir / step_name
        if nested.is_file():
            return nested

    for key in ("checkpoint_path", "ckpt_path", "pretrained_path"):
        value = model_cfg.get(key)
        if value:
            return Path(value).expanduser().resolve()

    raise FileNotFoundError(
        "Could not resolve ABot checkpoint. Provide 6-tuple eval args or checkpoint_path."
    )


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.default_prompt = self.model_cfg.get("prompt") or self.task_name
        self.unnorm_key = self.model_cfg.get("unnorm_key")
        self.device = self._get_device(self.model_cfg.get("device", "cuda"))

        self.ckpt_path = _resolve_checkpoint_path(self.model_cfg)
        _ensure_dataset_statistics(self.ckpt_path.parents[1])
        print(f"[Abot_M0] Loading checkpoint: {self.ckpt_path}")

        self.model = baseframework.from_pretrained(str(self.ckpt_path))
        self.model = self.model.to(self.device).eval()
        stats_key = baseframework._check_unnorm_key(self.model.norm_stats, self.unnorm_key)
        self.action_norm_stats = dict(self.model.norm_stats[stats_key]["action"])
        if _looks_like_modality_order_stats(self.action_norm_stats["q01"]):
            self.action_norm_stats = _permute_action_norm_stats(self.action_norm_stats)
            print("[Abot_M0] Permuted action norm stats from modality layout to training layout.")
        self.action_chunk_size = int(
            self.model.config.framework.action_model.future_action_window_size + 1
        )
        image_size = getattr(self.model.config.datasets.vla_data, "image_size", [224, 224])
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            self.image_size = (int(image_size[1]), int(image_size[0]))
        else:
            self.image_size = (224, 224)

        self._latest_env_idx_list = [0]
        self._latest_payloads: dict[int, dict[str, Any]] = {}

    def _get_device(self, device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _encode_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = []
        for camera_key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            rgb = ensure_rgb_hwc(extract_image(observation, _CAMERA_CANDIDATES[camera_key]))
            rgb = resize_image(rgb, self.image_size)
            images.append(to_pil_preserve(rgb))

        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        state = modality_layout_to_train_layout(state).reshape(1, -1)

        return {
            "image": images,
            "lang": resolve_prompt(observation, self.default_prompt),
            "state": state,
        }

    @torch.inference_mode()
    def _infer_payload(self, payload: dict[str, Any]) -> np.ndarray:
        output = self.model.predict_action(examples=[payload])
        normalized_actions = output["normalized_actions"][0]
        actions_train = _unnormalize_actions_train_layout(
            normalized_actions, self.action_norm_stats
        )
        return train_layout_to_modality_layout(actions_train)

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._latest_env_idx_list = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        self._latest_payloads = {
            env_idx: self._encode_obs(obs)
            for env_idx, obs in zip(self._latest_env_idx_list, obs_list)
        }

    def get_action(self, **kwargs: Any) -> list[dict[str, Any]]:
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list: list[int] | None = None, **kwargs: Any) -> list[list[dict[str, Any]]]:
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        else:
            env_idx_list = [int(env_idx) for env_idx in env_idx_list]

        missing_envs = [env_idx for env_idx in env_idx_list if env_idx not in self._latest_payloads]
        if missing_envs:
            raise KeyError(f"Missing observations for env_idx: {missing_envs}")

        results = []
        for env_idx in env_idx_list:
            raw_actions = self._infer_payload(self._latest_payloads[env_idx])
            results.append(
                unpack_robot_state(
                    raw_actions,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return results

    def reset(self) -> None:
        self._latest_env_idx_list = [0]
        self._latest_payloads = {}
