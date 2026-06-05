from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_INTERNVLA_ROOT = _CUR_DIR / "internvla_a1"
_INTERNVLA_SRC = _INTERNVLA_ROOT / "src"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_INTERNVLA_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("COSMOS_PATH", str((_CHECKPOINTS_DIR / "shared" / "Cosmos-Tokenizer-CI8x8").resolve()))
os.environ.setdefault("QWEN3_2B_PATH", str((_CHECKPOINTS_DIR / "shared" / "Qwen3-VL-2B-Instruct").resolve()))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES


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


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)


def ensure_hwc_uint8(image):
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
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


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
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return prompt

    fallback = _normalize_prompt_value(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or model config.")
    return fallback


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    if "images" in observation and "state" in observation:
        images = {
            "cam_high": ensure_hwc_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_hwc_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_hwc_uint8(observation["images"]["cam_right_wrist"]),
        }
        state = np.asarray(observation["state"], dtype=np.float32)
        prompt = resolve_prompt(observation, default_prompt)
        return {"images": images, "state": state, "prompt": prompt}

    images = {
        "cam_high": ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = resolve_prompt(observation, default_prompt)
    return {"images": images, "state": state, "prompt": prompt}


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _resolve_policy_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CUR_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_internvla_ckpt_dir(model_cfg: dict[str, Any]) -> Path:
    ckpt_path = model_cfg.get("ckpt_path")
    if ckpt_path:
        return resolve_ckpt_dir(ckpt_path)

    ckpt_name = model_cfg.get("ckpt_name")
    if not ckpt_name:
        raise ValueError("ckpt_name or ckpt_path is required for InternVLA_A1.")

    checkpoint_root = (_CHECKPOINTS_DIR / str(ckpt_name)).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    candidate_dirs = []
    if (checkpoint_root / "pretrained_model").is_dir():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and (child / "pretrained_model").is_dir()
    )
    if not candidate_dirs:
        raise FileNotFoundError(f"No pretrained_model checkpoint found under {checkpoint_root}")

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return (candidate / "pretrained_model").resolve()

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        selected = max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
        return (selected / "pretrained_model").resolve()
    return (candidate_dirs[0] / "pretrained_model").resolve()


def resolve_ckpt_dir(ckpt_path):
    ckpt = Path(str(ckpt_path)).expanduser()
    if not ckpt.is_absolute():
        ckpt = (_CUR_DIR / ckpt).resolve()
    if ckpt.exists():
        return ckpt.resolve()
    return Path(snapshot_download(repo_id=str(ckpt_path)))

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("InternVLA-A1 in XPolicyLab currently supports only action_type='joint'.")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.float32 if self.model_cfg.get("dtype", "float32") == "float32" else torch.bfloat16
        if self.device.type != "cuda":
            self.dtype = torch.float32

        for env_key, cfg_key in (
            ("COSMOS_PATH", "cosmos_path"),
            ("QWEN3_2B_PATH", "qwen3_2b_path"),
        ):
            value = _resolve_policy_path(self.model_cfg.get(cfg_key))
            if value is not None:
                os.environ[env_key] = str(value)

        self.ckpt_dir = _resolve_internvla_ckpt_dir(self.model_cfg)
        self.policy, self.input_transforms, self.unnormalize_fn = self._build_policy_and_transforms()
        self.image_history_interval = int(self.model_cfg.get("image_history_interval", 15))
        self.infer_horizon = int(self.model_cfg.get("infer_horizon", 30))
        self.action_horizon_size = int(self.model_cfg.get("action_horizon_size", 50))
        self.action_mode = self.model_cfg.get("action_mode", "delta")
        self.decode_image_flag = bool(self.model_cfg.get("decode_image_flag", False))
        self._latest_env_idx_list = [0]
        self.reset()
        self.model = self.policy

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _build_policy_and_transforms(self):
        config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not isinstance(config, QwenA1Config):
            raise ValueError(f"Expected QwenA1Config, got {type(config)}")

        policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=self.ckpt_dir)
        policy.to(self.device).to(self.dtype).eval()

        stats = load_json(self.ckpt_dir / "stats.json")[self.model_cfg.get("stats_key", "aloha")]
        stat_keys = ["min", "max", "mean", "std"]
        state_stat = {"observation.state": {key: np.asarray(stats["observation.state"][key]) for key in stat_keys}}
        action_stat = {"action": {key: np.asarray(stats["action"][key]) for key in stat_keys}}

        unnormalize_fn = UnNormalizeTransformFn(
            selected_keys=["action"],
            mode="mean_std",
            norm_stats=action_stat,
        )

        image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]
        input_transforms = compose(
            [
                ResizeImagesWithPadFn(
                    height=int(self.model_cfg.get("resize_size", 224)),
                    width=int(self.model_cfg.get("resize_size", 224)),
                ),
                RemapImageKeyTransformFn(mapping={key: key for key in image_keys}),
                Qwen3_VLProcessorTransformFn(),
                NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
            ]
        )
        return policy, input_transforms, unnormalize_fn

    def reset(self):
        self.policy.reset()
        self.head_history = []
        self.left_history = []
        self.right_history = []
        self._histories_by_env: dict[int, tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]] = {}
        self._latest_obs_by_env: dict[int, dict[str, Any]] = {}
        self._latest_obs = None
        self._latest_env_idx_list = [0]

    def _to_image_tensor(self, image):
        return torch.as_tensor(image, device=self.device).contiguous().to(self.dtype) / 255.0

    def _append_history(self, encoded_obs, histories=None):
        head_history, left_history, right_history = histories or (
            self.head_history,
            self.left_history,
            self.right_history,
        )
        head_history.append(self._to_image_tensor(encoded_obs["images"]["cam_high"]))
        left_history.append(self._to_image_tensor(encoded_obs["images"]["cam_left_wrist"]))
        right_history.append(self._to_image_tensor(encoded_obs["images"]["cam_right_wrist"]))
        max_history = self.image_history_interval + 1
        while len(head_history) > max_history:
            head_history.pop(0)
            left_history.pop(0)
            right_history.pop(0)

    def _build_image_pair(self, history):
        past_idx = max(len(history) - self.image_history_interval - 1, 0)
        return torch.stack([history[past_idx], history[-1]], dim=0)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        self._latest_obs = encoded_obs_list[0]
        for env_idx, encoded_obs in zip(self._latest_env_idx_list, encoded_obs_list):
            histories = self._histories_by_env.setdefault(env_idx, ([], [], []))
            self._latest_obs_by_env[env_idx] = encoded_obs
            self._append_history(encoded_obs, histories)

    @torch.inference_mode()
    def infer(self, env_idx=None):
        latest_obs = self._latest_obs if env_idx is None else self._latest_obs_by_env.get(env_idx)
        if latest_obs is None:
            raise AssertionError("update_obs must be called before get_action.")
        head_history, left_history, right_history = (
            (self.head_history, self.left_history, self.right_history)
            if env_idx is None
            else self._histories_by_env[env_idx]
        )

        state = torch.from_numpy(latest_obs["state"]).float().to(self.device)
        init_action = state.unsqueeze(0).clone()

        sample = {
            f"{OBS_IMAGES}.image0": self._build_image_pair(head_history),
            f"{OBS_IMAGES}.image1": self._build_image_pair(left_history),
            f"{OBS_IMAGES}.image2": self._build_image_pair(right_history),
            "observation.state": state,
            "task": latest_obs["prompt"],
        }
        for key in list(sample.keys()):
            if OBS_IMAGES in key and "mask" not in key:
                sample[key] = sample[key].permute(0, 3, 1, 2)

        sample = self.input_transforms(sample)
        inputs = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
            elif value.dtype == torch.int64:
                inputs[key] = value[None].to(self.device)
            else:
                inputs[key] = value[None].to(self.device).to(dtype=self.dtype)

        inputs.update(
            {
                f"{OBS_IMAGES}.image0_mask": torch.tensor([True], device=self.device),
                f"{OBS_IMAGES}.image1_mask": torch.tensor([True], device=self.device),
                f"{OBS_IMAGES}.image2_mask": torch.tensor([True], device=self.device),
            }
        )

        action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=self.decode_image_flag)
        action_pred = action_pred[0, : self.infer_horizon, : latest_obs["state"].shape[0]]
        action_pred = self.unnormalize_fn({"action": action_pred})["action"]

        if self.action_mode == "delta":
            left_gripper_idx = self.robot_action_dim_info["arm_dim"][0]
            right_gripper_idx = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"]) - 1
            init_action[:, left_gripper_idx] = 0.0
            init_action[:, right_gripper_idx] = 0.0
            action_pred = action_pred + init_action

        return action_pred.detach().cpu().float().numpy()

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for env_idx in env_idx_list:
            raw_actions = self.infer(env_idx)
            action_list.append(unpack_robot_state(raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs"))
        return action_list
