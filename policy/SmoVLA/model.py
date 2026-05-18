from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_SMOVLA_ROOT = _REPO_ROOT.parent / "smo_vla"
_LEROBOT_SRC = _SMOVLA_ROOT / "lerobot" / "src"
_LEROBOT_ROOT = _SMOVLA_ROOT / "lerobot"

for _path in (str(_REPO_ROOT), str(_LEROBOT_SRC), str(_LEROBOT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame


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
    decoded = cv2.imdecode(np.asarray(image_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode compressed image buffer.")
    return decoded


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
            "camera1": ensure_hwc_uint8(observation["images"]["cam_high"]),
            "camera2": ensure_hwc_uint8(observation["images"]["cam_left_wrist"]),
            "camera3": ensure_hwc_uint8(observation["images"]["cam_right_wrist"]),
        }
        state = np.asarray(observation["state"], dtype=np.float32)
        prompt = resolve_prompt(observation, default_prompt)
    else:
        images = {
            "camera1": ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
            "camera2": ensure_hwc_uint8(
                extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
            ),
            "camera3": ensure_hwc_uint8(
                extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
            ),
        }
        state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
        prompt = resolve_prompt(observation, default_prompt)

    payload = {"task": prompt}
    payload.update(images)
    for idx, value in enumerate(state.tolist()):
        payload[f"state_{idx}"] = value
    return payload


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("SmoVLA in XPolicyLab currently supports only action_type='joint'.")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)
        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.pretrained_path = self._resolve_pretrained_path(self.model_cfg.get("pretrained_path") or self.model_cfg.get("model_path"))
        self.policy = self._load_policy()
        self.actions_per_chunk = int(
            self.model_cfg.get(
                "actions_per_chunk",
                getattr(self.policy.config, "n_action_steps", getattr(self.policy.config, "chunk_size", 1)),
            )
        )
        self.preprocessor, self.postprocessor = self._build_processors()
        self._latest_env_idx_list = [0]
        self._latest_payload = None
        self._lerobot_features = None
        self.model = self.policy

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _resolve_pretrained_path(self, pretrained_path):
        if pretrained_path is None:
            raise ValueError("pretrained_path or model_path is required for SmoVLA.")
        root = Path(pretrained_path).expanduser().resolve()
        candidates = [
            root,
            root / "pretrained_model",
            root / "checkpoints" / "last" / "pretrained_model",
        ]
        for candidate in candidates:
            if (candidate / "model.safetensors").is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not find a LeRobot pretrained policy under `{pretrained_path}`. "
            "Expected `model.safetensors` in the path itself, `pretrained_model/`, "
            "or `checkpoints/last/pretrained_model/`."
        )

    def _load_policy(self):
        policy_class = get_policy_class("smolvla")
        policy = policy_class.from_pretrained(self.pretrained_path)
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

    def _ensure_lerobot_features(self, payload):
        if self._lerobot_features is not None:
            return
        image_shape = payload["camera1"].shape
        state_dim = len([key for key in payload.keys() if key.startswith("state_")])
        state_names = [f"state_{idx}" for idx in range(state_dim)]
        self._lerobot_features = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
            "observation.images.camera1": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.camera2": {
                "dtype": "image",
                "shape": payload["camera2"].shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.camera3": {
                "dtype": "image",
                "shape": payload["camera3"].shape,
                "names": ["height", "width", "channels"],
            },
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        if len(encoded_obs_list) != 1:
            raise NotImplementedError("SmoVLA currently supports single-env inference in XPolicyLab.")
        self._latest_payload = encoded_obs_list[0]
        self._ensure_lerobot_features(self._latest_payload)

    @torch.inference_mode()
    def infer(self):
        if self._latest_payload is None:
            raise AssertionError("update_obs must be called before get_action.")

        observation = raw_observation_to_observation(
            self._latest_payload,
            self._lerobot_features,
            self.policy.config.image_features,
        )
        observation = self.preprocessor(observation)
        action_tensor = self.policy.predict_action_chunk(observation)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]

        processed_actions = []
        for idx in range(action_tensor.shape[1]):
            single_action = action_tensor[:, idx, :]
            processed_actions.append(self.postprocessor(single_action))

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu().float().numpy()
        return action_tensor

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        if len(env_idx_list) != 1:
            raise NotImplementedError("SmoVLA currently supports single-env inference in XPolicyLab.")

        raw_actions = self.infer()
        return [unpack_robot_state(raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs")]

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
        self._latest_env_idx_list = [0]
        self._latest_payload = None
        self._lerobot_features = None


def is_image_key(key: str) -> bool:
    return key.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: torch.Tensor, resize_dims) -> torch.Tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    image_batched = image.unsqueeze(0)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    image = image.type(torch.float32) / 255
    image = image.contiguous()
    return image


def extract_state_from_raw_observation(lerobot_obs):
    state = torch.tensor(lerobot_obs[OBS_STATE])
    if state.ndim == 1:
        state = state.unsqueeze(0)
    return state


def make_lerobot_observation(robot_obs, lerobot_features):
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(robot_obs, lerobot_features, policy_image_features):
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)
    image_keys = list(filter(is_image_key, lerobot_obs))
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }
    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]
    return {**state_dict, **image_dict}


def raw_observation_to_observation(raw_observation, lerobot_features, policy_image_features):
    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for key, value in observation.items():
        if isinstance(value, torch.Tensor) and "image" in key:
            observation[key] = prepare_image(value).unsqueeze(0)
    return observation
