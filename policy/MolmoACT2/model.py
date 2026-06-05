from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_LEROBOT_SRC = _CUR_DIR / "molmoact2" / "lerobot" / "src"
_LEROBOT_ROOT = _CUR_DIR / "molmoact2" / "lerobot"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_LEROBOT_SRC), str(_LEROBOT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

_IMAGE_SHORT_NAMES = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
    "observation.images.image": "cam_high",
    "observation.images.wrist_image": "cam_left_wrist",
}

_CAMERA_CANDIDATES = {
    "cam_high": ["cam_high", "cam_head", "head_camera", "top_camera"],
    "cam_left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "cam_right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
}


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


def ensure_chw_uint8(image: np.ndarray) -> np.ndarray:
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

    return np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))


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


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _resolve_checkpoint_root(model_cfg: dict[str, Any]) -> Path:
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name:
        direct_path = (_CHECKPOINTS_DIR / str(ckpt_name)).expanduser().resolve()
        if Path(str(ckpt_name)).expanduser().is_absolute() or "/" in str(ckpt_name):
            return direct_path

        tuple_keys = ("dataset_name", "ckpt_name", "env_cfg_type", "expert_data_num", "action_type", "seed")
        if all(model_cfg.get(key) is not None for key in tuple_keys):
            checkpoint_setting = "-".join(str(model_cfg[key]) for key in tuple_keys)
            tuple_path = (_CHECKPOINTS_DIR / checkpoint_setting).expanduser().resolve()
            if tuple_path.exists():
                return tuple_path

        return direct_path

    for key in ("pretrained_path", "model_path", "checkpoint_path"):
        value = model_cfg.get(key)
        if value:
            return Path(value).expanduser().resolve()

    raise FileNotFoundError("ckpt_name or checkpoint_path is required for MolmoACT2.")


def encode_obs(
    observation: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict[str, Any],
    default_prompt: str,
) -> dict[str, Any]:
    images = {
        camera_key: ensure_chw_uint8(extract_image(observation, _CAMERA_CANDIDATES[camera_key]))
        for camera_key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = resolve_prompt(observation, default_prompt)
    return {"state": state, "images": images, "prompt": prompt}


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("MolmoACT2 in XPolicyLab currently supports only action_type='joint'.")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self.default_prompt = self.model_cfg.get("prompt") or self.task_name
        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.pretrained_path = self._resolve_pretrained_path(_resolve_checkpoint_root(self.model_cfg))
        self.policy = self._load_policy()
        self.image_keys = list(getattr(self.policy.config, "image_keys", []))
        self.actions_per_chunk = self._resolve_actions_per_chunk()
        self.preprocessor, self.postprocessor = self._build_processors()
        self._latest_env_idx_list = [0]
        self._latest_payloads: dict[int, dict[str, Any]] = {}
        self.model = self.policy
        print(f"[MolmoACT2] Loaded checkpoint from {self.pretrained_path}")

    def _get_device(self, device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _resolve_actions_per_chunk(self) -> int:
        candidates = (
            self.model_cfg.get("actions_per_chunk"),
            getattr(self.policy.config, "n_action_steps", None),
            getattr(self.policy.config, "chunk_size", None),
            1,
        )
        for value in candidates:
            if value is None:
                continue
            resolved = int(value)
            if resolved <= 0:
                raise ValueError(f"actions_per_chunk must be positive, got {resolved}")
            return resolved
        raise ValueError("Failed to resolve actions_per_chunk from model config or policy config.")

    def _resolve_pretrained_path(self, checkpoint_root: Path) -> str:
        artifact_root = checkpoint_root
        if checkpoint_root.is_dir():
            candidate_dirs = []
            if (checkpoint_root / "model.safetensors").exists() or (checkpoint_root / "pretrained_model").is_dir():
                candidate_dirs.append(checkpoint_root)
            candidate_dirs.extend(
                child
                for child in sorted(checkpoint_root.iterdir())
                if child.is_dir()
                and ((child / "model.safetensors").exists() or (child / "pretrained_model").is_dir())
            )
            checkpoint_num = self.model_cfg.get("checkpoint_num")
            desired_step = _extract_step_number(checkpoint_num)
            if desired_step is not None:
                step_dirs = [
                    candidate
                    for candidate in candidate_dirs
                    if _extract_step_number(candidate.name) is not None
                ]
                exact = [
                    candidate
                    for candidate in step_dirs
                    if _extract_step_number(candidate.name) in {desired_step, desired_step * 10}
                    or str(candidate.name).lstrip("0") == str(desired_step).lstrip("0")
                ]
                if exact:
                    artifact_root = exact[0]
                elif step_dirs:
                    artifact_root = max(
                        step_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1
                    )
            elif candidate_dirs:
                numeric_dirs = [
                    candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None
                ]
                artifact_root = (
                    max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
                    if numeric_dirs
                    else candidate_dirs[0]
                )

        candidates = [
            artifact_root,
            artifact_root / "pretrained_model",
            artifact_root / "checkpoints" / "last" / "pretrained_model",
        ]
        for candidate in candidates:
            if (candidate / "model.safetensors").is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not find a LeRobot pretrained policy under `{artifact_root}`. "
            "Expected `model.safetensors` in the path itself or `pretrained_model/`."
        )

    def _load_molmoact2_config(self):
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(self.pretrained_path)
        override = self.model_cfg.get("inference_action_mode")
        if override is not None:
            config.inference_action_mode = override
        elif getattr(config, "inference_action_mode", None) is None:
            action_mode = getattr(config, "action_mode", "continuous")
            config.inference_action_mode = (
                action_mode if action_mode in ("continuous", "discrete") else "continuous"
            )
        return config

    def _load_policy(self):
        policy_class = get_policy_class("molmoact2")
        config = self._load_molmoact2_config()
        policy = policy_class.from_pretrained(self.pretrained_path, config=config)
        policy.to(self.device)
        return policy

    def _build_processors(self):
        device_override = {"device": str(self.device)}
        return make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.pretrained_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._latest_env_idx_list = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        self._latest_payloads = {
            env_idx: encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt)
            for env_idx, obs in zip(self._latest_env_idx_list, obs_list)
        }

    def _postprocess_action_chunk(self, action_tensor: torch.Tensor) -> np.ndarray:
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]
        batch_size, chunk_size, action_dim = action_tensor.shape
        flat_actions = action_tensor.reshape(batch_size * chunk_size, action_dim)
        processed_actions = self.postprocessor(flat_actions)
        return processed_actions.reshape(batch_size, chunk_size, -1).detach().cpu().float().numpy()

    def _stack_observations(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        if len(observations) == 1:
            return observations[0]

        stacked = {"task": [observation["task"] for observation in observations]}
        for key in observations[0]:
            if key == "task":
                continue
            values = [observation[key] for observation in observations]
            if all(isinstance(value, torch.Tensor) for value in values):
                stacked[key] = torch.cat(values, dim=0)
            else:
                stacked[key] = values
        return stacked

    def _payload_to_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = {
            OBS_STATE: torch.as_tensor(payload["state"], dtype=torch.float32).unsqueeze(0),
            "task": payload["prompt"],
        }
        for image_key in self.image_keys:
            short_name = _IMAGE_SHORT_NAMES.get(image_key, image_key.split(".")[-1])
            observation[image_key] = prepare_image(torch.as_tensor(payload["images"][short_name])).unsqueeze(0)
        return observation

    @torch.inference_mode()
    def infer_batch_payloads(self, payloads: list[dict[str, Any]]) -> np.ndarray:
        if not payloads:
            raise ValueError("infer_batch_payloads requires at least one payload.")

        observations = [self._payload_to_observation(payload) for payload in payloads]
        observation = self._stack_observations(observations)
        observation = self.preprocessor(observation)
        action_tensor = self.policy.predict_action_chunk(observation)
        return self._postprocess_action_chunk(action_tensor)

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

        payloads = [self._latest_payloads[env_idx] for env_idx in env_idx_list]
        raw_action_batch = self.infer_batch_payloads(payloads)
        return [
            unpack_robot_state(raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs")
            for raw_actions in raw_action_batch
        ]

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self._latest_env_idx_list = [0]
        self._latest_payloads = {}


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    image = image.type(torch.float32) / 255
    return image.contiguous()
