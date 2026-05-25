from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

_POLICY_DIR = Path(__file__).resolve().parent
_MOTUS_ROOT = _POLICY_DIR / "motus" / "inference" / "robotwin" / "Motus"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_DEFAULT_WAN_PATH = "/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B"
_DEFAULT_VLM_PATH = "/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct"
if str(_MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOTUS_ROOT))

from deploy_policy import MotusPolicy, get_model as get_motus_model, reset_model


def extract_image(observation: dict[str, Any], candidate_names: list[str]) -> Any:
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

    images = observation.get("images", {})
    for candidate_name in candidate_names:
        if candidate_name in images:
            return images[candidate_name]

    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def decode_compressed_image(image_buffer: np.ndarray) -> np.ndarray:
    decoded = cv2.imdecode(np.asarray(image_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode compressed image buffer.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def ensure_hwc_uint8(image: Any) -> np.ndarray:
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


def resolve_prompt(observation: dict[str, Any], default_prompt: str | None) -> str:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return prompt

    fallback = _normalize_prompt_value(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or model config.")
    return fallback


def encode_obs(observation: dict[str, Any], action_type: str, robot_action_dim_info: dict[str, Any]) -> dict[str, Any]:
    if "observation" in observation and "joint_action" in observation:
        return observation

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations for Motus.")

    head = ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]))
    head = cv2.resize(head, (320, 240), interpolation=cv2.INTER_AREA)
    left = ensure_hwc_uint8(extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]))
    right = ensure_hwc_uint8(extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]))
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    return {
        "observation": {
            "head_camera": {"rgb": head},
            "left_camera": {"rgb": left},
            "right_camera": {"rgb": right},
        },
        "joint_action": {"vector": state},
    }


def _as_int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def patch_motus_runtime_config(model_cfg: dict[str, Any]) -> None:
    # The RoboDojo checkpoint was trained with the LeRobot config, while the
    # bundled RobotWin inference config has a shorter action sequence.
    common_overrides = {
        "global_downsample_rate": _as_int_or_default(model_cfg.get("global_downsample_rate"), 1),
        "video_action_freq_ratio": _as_int_or_default(model_cfg.get("video_action_freq_ratio"), 6),
    }

    if not getattr(MotusPolicy, "_xpolicylab_config_patched", False):
        original_create_model_config = MotusPolicy._create_model_config

        def _create_model_config_with_xpolicylab_overrides(self):
            common_cfg = self.config_dict.setdefault("common", {})
            for key, value in getattr(MotusPolicy, "_xpolicylab_common_overrides", {}).items():
                common_cfg[key] = value
            return original_create_model_config(self)

        MotusPolicy._xpolicylab_original_create_model_config = original_create_model_config
        MotusPolicy._create_model_config = _create_model_config_with_xpolicylab_overrides
        MotusPolicy._xpolicylab_config_patched = True

    MotusPolicy._xpolicylab_common_overrides = common_overrides


def normalize_motus_image_layout(encoded_observation: dict[str, Any]) -> None:
    obs_data = encoded_observation.get("observation", {})
    head_camera = obs_data.get("head_camera", {})
    head_image = head_camera.get("rgb")
    if head_image is None:
        return

    head_image = ensure_hwc_uint8(head_image)
    if head_image.shape[:2] != (240, 320):
        head_image = cv2.resize(head_image, (320, 240), interpolation=cv2.INTER_AREA)
    head_camera["rgb"] = head_image


def patch_motus_qwen_rope_index(policy: Any) -> None:
    vlm_model = getattr(getattr(getattr(policy, "model", None), "vlm_model", None), "model", None)
    if vlm_model is None or getattr(vlm_model, "_xpolicylab_rope_index_patched", False):
        return

    original_get_rope_index = vlm_model.get_rope_index
    if "mm_token_type_ids" in inspect.signature(original_get_rope_index).parameters:
        return

    def _get_rope_index_compat(*args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return original_get_rope_index(*args, **kwargs)

    vlm_model.get_rope_index = _get_rope_index_compat
    vlm_model._xpolicylab_rope_index_patched = True


def _resolve_path(value: Any, base_dir: Path = _POLICY_DIR) -> Path | None:
    if value is None or value == "":
        return None

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def resolve_motus_checkpoint(model_cfg: dict[str, Any]) -> str:
    for key in ("ckpt_setting", "checkpoint_path", "model_path"):
        explicit_path = _resolve_path(model_cfg.get(key))
        if explicit_path is not None:
            return str(explicit_path)

    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name:
        raw_ckpt_name = Path(str(ckpt_name)).expanduser()
        if raw_ckpt_name.is_absolute() or "/" in str(ckpt_name):
            return str(_resolve_path(ckpt_name))

        tuple_keys = ("dataset_name", "ckpt_name", "env_cfg_type", "expert_data_num", "action_type", "seed")
        if all(model_cfg.get(key) is not None for key in tuple_keys):
            checkpoint_setting = "-".join(str(model_cfg[key]) for key in tuple_keys)
            tuple_path = (_CHECKPOINTS_DIR / checkpoint_setting).resolve()
            if tuple_path.exists():
                return str(tuple_path)

        return str((_CHECKPOINTS_DIR / str(ckpt_name)).resolve())

    raise ValueError("ckpt_name, ckpt_setting, checkpoint_path, or model_path is required for Motus.")


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("Motus in XPolicyLab currently supports only action_type='joint'.")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)
        self._latest_env_idx_list: list[int] = [0]
        self.observation_window: list[dict[str, Any]] | None = None

        ckpt_setting = resolve_motus_checkpoint(self.model_cfg)

        model_args = dict(self.model_cfg)
        model_args["ckpt_setting"] = ckpt_setting
        model_args["wan_path"] = str(_resolve_path(model_args.get("wan_path")) or Path(_DEFAULT_WAN_PATH))
        model_args["vlm_path"] = str(_resolve_path(model_args.get("vlm_path")) or Path(_DEFAULT_VLM_PATH))
        model_args["prompt"] = model_args.get("prompt") or self.default_prompt
        patch_motus_runtime_config(model_args)
        self.policy = get_motus_model(model_args)
        patch_motus_qwen_rope_index(self.policy)
        self.model = self.policy

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = [
            {
                "observation": encode_obs(obs, self.action_type, self.robot_action_dim_info),
                "instruction": resolve_prompt(obs, self.default_prompt),
            }
            for obs in obs_list
        ]

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            cached = self.observation_window[batch_index]
            normalize_motus_image_layout(cached["observation"])
            self.policy.set_instruction(cached["instruction"])
            self.policy.update_obs(cached["observation"])
            action_chunk = self.policy.get_action()
            action_list.append(
                unpack_robot_state(
                    action_chunk,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        reset_model(self.policy)

    def reset_obsrvationwindows(self):
        self.reset()