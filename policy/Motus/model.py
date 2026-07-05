from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_MOTUS_ROOT = _POLICY_DIR / "motus" / "inference" / "robotwin" / "Motus"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_DEFAULT_WAN_PATH = "/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B"
_DEFAULT_VLM_PATH = "/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct"
_DEFAULT_ROBOT_ACTION_DIM_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
if str(_MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOTUS_ROOT))

from deploy_policy import MotusPolicy, get_model as get_motus_model, reset_model

# RoboDojo v1.0 observation keys (see test.pkl): vision uses cam_head /
# cam_left_wrist / cam_right_wrist; state holds per-arm joint + ee fields.
_ROBODOJO_HEAD_CAMERA_NAMES = ("cam_head", "cam_high", "cam_third_view", "head_camera", "top_camera")
_ROBODOJO_LEFT_CAMERA_NAMES = ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left")
_ROBODOJO_RIGHT_CAMERA_NAMES = ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right")
_DEBUG_INSTRUCTION_PLACEHOLDERS = frozenset(
    {
        "language instruction",
        "instruction",
        "task",
        "default_task",
    }
)
_STAT_JSON_PATH = _MOTUS_ROOT / "utils" / "stat.json"


def _gripper_indices(robot_action_dim_info: dict[str, Any]) -> list[int]:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    indices: list[int] = []
    offset = 0
    for arm_dim, ee_dim in zip(arm_dims, ee_dims):
        offset += arm_dim
        indices.append(offset)
        offset += ee_dim
    return indices


def _gripper_state_keys(robot_action_dim_info: dict[str, Any]) -> list[str]:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) == 1:
        return ["ee_joint_state"]
    if len(arm_dims) == 2:
        return ["left_ee_joint_state", "right_ee_joint_state"]
    raise ValueError(f"Unsupported arm count for gripper mapping: {len(arm_dims)}")


def _load_embodiment_action_stats(embodiment_type: str) -> tuple[np.ndarray, np.ndarray]:
    with _STAT_JSON_PATH.open("r", encoding="utf-8") as f:
        stat_data = json.load(f)

    stats = stat_data.get(embodiment_type)
    if stats is None:
        raise ValueError(
            f"Normalization stats for embodiment '{embodiment_type}' not found in {_STAT_JSON_PATH}. "
            f"Available: {list(stat_data.keys())}"
        )

    action_min = np.asarray(stats["min"], dtype=np.float32)
    action_max = np.asarray(stats["max"], dtype=np.float32)
    return action_min, action_max


def _robodojo_gripper_to_training_scale(
    packed_state: np.ndarray,
    gripper_indices: list[int],
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> np.ndarray:
    """Map RoboDojo ee_joint_state in [0, 1] to Motus training qpos scale."""
    converted = np.asarray(packed_state, dtype=np.float32).copy()
    for idx in gripper_indices:
        value = float(converted[idx])
        stat_max = float(action_max[idx])
        if value > stat_max + 1e-4:
            stat_min = float(action_min[idx])
            converted[idx] = stat_min + np.clip(value, 0.0, 1.0) * (stat_max - stat_min)
    return converted


def _training_gripper_to_robodojo(
    value: float,
    action_min: np.ndarray,
    action_max: np.ndarray,
    index: int,
) -> float:
    stat_min = float(action_min[index])
    stat_max = float(action_max[index])
    stat_range = stat_max - stat_min
    if stat_range <= 0:
        return value
    return float(np.clip((value - stat_min) / stat_range, 0.0, 1.0))


def _format_action_chunk(
    action_chunk: Any,
    action_type: str,
    robot_action_dim_info: dict[str, Any],
    *,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
    gripper_indices: list[int] | None = None,
    gripper_state_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert MotusPolicy output (denormalized packed joints) to RoboDojo action dicts.

    deploy_policy.get_action returns a [chunk, dim] float32 array in real joint scale.
    RoboDojo expects each step as a dict of per-arm fields, so we unpack here.
    """
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        steps = [actions]
    elif actions.ndim == 2:
        steps = [step for step in actions]
    else:
        raise ValueError(f"Unexpected Motus action shape: {actions.shape}")

    action_dicts = [
        unpack_robot_state(step, action_type, robot_action_dim_info, source_type="obs")
        for step in steps
    ]

    if (
        action_min is None
        or action_max is None
        or gripper_indices is None
        or gripper_state_keys is None
    ):
        return action_dicts

    for action_dict in action_dicts:
        for key, idx in zip(gripper_state_keys, gripper_indices):
            gripper_value = np.asarray(action_dict[key], dtype=np.float32).reshape(-1)
            gripper_value[0] = _training_gripper_to_robodojo(
                float(gripper_value[0]), action_min, action_max, idx
            )
            action_dict[key] = gripper_value.astype(np.float32)

    return action_dicts


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
    return decode_image_bit(image_buffer)


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


def _load_task_instruction_index(t5_cache_dir: Path | None) -> dict[str, str]:
    if t5_cache_dir is None:
        return {}

    manifest_path = t5_cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    task_index: dict[str, str] = {}
    for item in manifest.get("tasks", []):
        if not isinstance(item, dict):
            continue
        task = _as_optional_str(item.get("task"))
        instruction = _normalize_prompt_value(item.get("instruction"))
        if task and instruction:
            task_index[task] = instruction
    return task_index


def _is_debug_instruction_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return not normalized or normalized in _DEBUG_INSTRUCTION_PLACEHOLDERS


def resolve_prompt(
    observation: dict[str, Any],
    default_prompt: str | None,
    task_name: str | None = None,
    task_instructions: dict[str, str] | None = None,
) -> str:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None and not _is_debug_instruction_placeholder(prompt):
            return prompt

    task_key = _as_optional_str(task_name)
    if task_key and task_instructions and task_key in task_instructions:
        return task_instructions[task_key]

    fallback = _normalize_prompt_value(default_prompt)
    if fallback is not None and not _is_debug_instruction_placeholder(fallback):
        return fallback

    raise ValueError("No valid prompt found in observation, task_name, or model config.")


def encode_obs(
    observation: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict[str, Any],
    *,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
    gripper_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Convert RoboDojo v1.0 obs into the layout expected by MotusPolicy.update_obs."""
    if "observation" in observation and "joint_action" in observation:
        return observation

    head = ensure_hwc_uint8(extract_image(observation, list(_ROBODOJO_HEAD_CAMERA_NAMES)))
    left = ensure_hwc_uint8(extract_image(observation, list(_ROBODOJO_LEFT_CAMERA_NAMES)))
    right = ensure_hwc_uint8(extract_image(observation, list(_ROBODOJO_RIGHT_CAMERA_NAMES)))
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    if action_min is not None and action_max is not None and gripper_indices is not None:
        state = _robodojo_gripper_to_training_scale(state, gripper_indices, action_min, action_max)
    return {
        "observation": {
            "head_camera": {"rgb": head},
            "left_camera": {"rgb": left},
            "right_camera": {"rgb": right},
        },
        "joint_action": {"vector": state},
    }


def _run_policy_step(policy: MotusPolicy, encoded_observation: dict[str, Any], instruction: str) -> Any:
    """Mirror deploy_policy.eval: set_instruction -> update_obs -> get_action."""
    normalize_motus_image_layout(encoded_observation)
    policy.set_instruction(instruction)
    policy.update_obs(encoded_observation)
    return policy.get_action()


def _as_int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_t5_cache_dir(cache_dir: Path) -> Path:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"t5_cache_dir is set but manifest.json was not found: {manifest_path}"
        )
    return cache_dir


def build_motus_model_args(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize XPolicyLab deploy config into MotusPolicy constructor args."""
    model_args = dict(model_cfg)
    model_args["ckpt_setting"] = resolve_motus_checkpoint(model_cfg)
    model_args["wan_path"] = str(_resolve_path(model_args.get("wan_path")) or Path(_DEFAULT_WAN_PATH))
    model_args["vlm_path"] = str(_resolve_path(model_args.get("vlm_path")) or Path(_DEFAULT_VLM_PATH))

    t5_cache_dir = _resolve_path(model_args.get("t5_cache_dir"))
    if t5_cache_dir is not None:
        model_args["t5_cache_dir"] = str(_validate_t5_cache_dir(t5_cache_dir))

    log_dir = _resolve_path(model_args.get("log_dir"))
    if log_dir is not None:
        model_args["log_dir"] = str(log_dir)

    # RoboDojo checkpoints are trained with the LeRobot pipeline, which feeds raw
    # task strings to T5/VLM (no scene prefix). Default to False unless overridden.
    model_args["use_scene_prefix"] = _as_bool(model_args.get("use_scene_prefix"), default=False)

    default_prompt = _as_optional_str(model_args.get("prompt")) or _as_optional_str(model_args.get("task_name"))
    if default_prompt is not None:
        model_args["prompt"] = default_prompt

    patch_motus_runtime_config(model_args)
    return model_args


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
    for camera_name in ("head_camera", "left_camera", "right_camera"):
        camera = obs_data.get(camera_name, {})
        image = camera.get("rgb")
        if image is not None:
            camera["rgb"] = ensure_hwc_uint8(image)


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

        tuple_keys = ("bench_name", "ckpt_name", "env_cfg_type", "expert_data_num", "action_type", "seed")
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
        self.robot_action_dim_info = (
            get_robot_action_dim_info(env_cfg)
            if env_cfg is not None
            else dict(_DEFAULT_ROBOT_ACTION_DIM_INFO)
        )
        self.embodiment_type = self.model_cfg.get("embodiment_type") or "aloha_agilex_2"
        self.action_min, self.action_max = _load_embodiment_action_stats(self.embodiment_type)
        self.gripper_indices = _gripper_indices(self.robot_action_dim_info)
        self.gripper_state_keys = _gripper_state_keys(self.robot_action_dim_info)
        self.default_prompt = (
            _as_optional_str(self.model_cfg.get("prompt"))
            or _as_optional_str(self.model_cfg.get("task_name"))
        )
        self._latest_env_idx_list: list[int] = [0]
        self.observation_window: list[dict[str, Any]] | None = None
        t5_cache_dir = _resolve_path(self.model_cfg.get("t5_cache_dir"))
        self._task_instructions = _load_task_instruction_index(t5_cache_dir)

        model_args = build_motus_model_args(self.model_cfg)
        self.policy = get_motus_model(model_args)
        patch_motus_qwen_rope_index(self.policy)
        self.model = self.policy
        initial_instruction = resolve_prompt(
            {},
            self.default_prompt,
            task_name=_as_optional_str(self.task_name),
            task_instructions=self._task_instructions,
        )
        self.policy.set_instruction(initial_instruction)

    def _policy_instruction(self, observation: dict[str, Any]) -> str:
        return resolve_prompt(
            observation,
            self.default_prompt,
            task_name=_as_optional_str(self.task_name),
            task_instructions=self._task_instructions,
        )

    def _encode_robodojo_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        return encode_obs(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            action_min=self.action_min,
            action_max=self.action_max,
            gripper_indices=self.gripper_indices,
        )

    def _format_policy_actions(self, action_chunk: Any) -> list[dict[str, Any]]:
        return _format_action_chunk(
            action_chunk,
            self.action_type,
            self.robot_action_dim_info,
            action_min=self.action_min,
            action_max=self.action_max,
            gripper_indices=self.gripper_indices,
            gripper_state_keys=self.gripper_state_keys,
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = [
            {
                "observation": self._encode_robodojo_obs(obs),
                "instruction": self._policy_instruction(obs),
            }
            for obs in obs_list
        ]

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        if env_idx_list is None:
            env_idx_list = kwargs.get("obs")
        env_idx_list = env_idx_list or self._latest_env_idx_list
        if len(self.observation_window) != len(env_idx_list):
            raise ValueError(
                f"observation_window size ({len(self.observation_window)}) "
                f"does not match env_idx_list size ({len(env_idx_list)})"
            )

        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            cached = self.observation_window[batch_index]
            action_chunk = _run_policy_step(self.policy, cached["observation"], cached["instruction"])
            action_list.append(self._format_policy_actions(action_chunk))

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        reset_model(self.policy)
        self.policy.current_state = None
        self.policy.current_state_norm = None
        instruction = resolve_prompt(
            {},
            self.default_prompt,
            task_name=_as_optional_str(self.task_name),
            task_instructions=self._task_instructions,
        )
        self.policy.set_instruction(instruction)