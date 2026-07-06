from __future__ import annotations

import importlib.util
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit
from XPolicyLab.policy.Spirit_v15.spirit_v15.model import SpiritVLAPolicy

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

_TASK_INFO_MODULE_PATH = (
    Path(__file__).resolve().parent
    / "spirit_v15"
    / "robochallenge"
    / "runner"
    / "task_info.py"
)

_TASK_INFO_SPEC = importlib.util.spec_from_file_location("spirit_v15_task_info", _TASK_INFO_MODULE_PATH)
if _TASK_INFO_SPEC is None or _TASK_INFO_SPEC.loader is None:
    raise ImportError(f"Failed to load Spirit task info module from {_TASK_INFO_MODULE_PATH}")
_TASK_INFO_MODULE = importlib.util.module_from_spec(_TASK_INFO_SPEC)
_TASK_INFO_SPEC.loader.exec_module(_TASK_INFO_MODULE)

TASK_INFO = _TASK_INFO_MODULE.TASK_INFO
TASKS_USE_LESS_CHUNK_SIZE = _TASK_INFO_MODULE.TASKS_USE_LESS_CHUNK_SIZE
TASTS_APPLY_GRIPPER_BINARIZATION = _TASK_INFO_MODULE.TASTS_APPLY_GRIPPER_BINARIZATION


TASK_NAME_ALIASES = {
    "stack_bowls_three": "stack_bowls",
    "stack_bowls_two": "stack_bowls",
}


def extract_image(observation: dict[str, Any], candidate_names: list[str]) -> Any:
    if "vision" in observation:
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


def resolve_prompt(observation: dict[str, Any], default_prompt: str | None) -> str | None:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return prompt
    return _normalize_prompt_value(default_prompt)


def quat_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[0] != 7:
        raise ValueError(f"Expected 7-dim pose, got {pose.shape}")
    return np.concatenate([pose[:3], pose[4:7], pose[3:4]], axis=0).astype(np.float32)


def quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[0] != 4:
        raise ValueError(f"Expected 4-dim quaternion, got {quaternion.shape}")
    return np.concatenate([quaternion[3:4], quaternion[:3]], axis=0).astype(np.float32)


def _to_scalar_gripper(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Empty gripper state is not supported.")
    return float(array[0])


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _resolve_spirit_checkpoint_dir(model_cfg: dict[str, Any]) -> Path:
    ckpt_name = model_cfg.get("ckpt_name")
    if not ckpt_name:
        checkpoint_path = model_cfg.get("checkpoint_path") or model_cfg.get("model_path")
        if checkpoint_path is None:
            raise ValueError("ckpt_name, checkpoint_path, or model_path is required for Spirit_v15.")
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (_POLICY_DIR / checkpoint_path).resolve()
        return checkpoint_path.resolve()

    checkpoint_root = (_CHECKPOINTS_DIR / str(ckpt_name)).expanduser().resolve()
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "config.json").exists() or (checkpoint_root / "model.safetensors").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "config.json").exists() or (child / "model.safetensors").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

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
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


def _resolve_policy_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_POLICY_DIR / path).resolve()
    return path.resolve()


def _post_process_action(
    action_np: np.ndarray,
    state_np: np.ndarray,
    robot_type: str,
    used_chunk_size: int,
    raw_embodiment_stats: dict[str, Any] | None,
    binarization_threshold: Optional[float] = None,
) -> list[list[float]]:
    result_list = []
    if raw_embodiment_stats is not None:
        left_gripper_min, left_gripper_max = (
            raw_embodiment_stats[robot_type]["action"]["min"][6],
            raw_embodiment_stats[robot_type]["action"]["max"][6],
        )
        right_gripper_min, right_gripper_max = (
            raw_embodiment_stats[robot_type]["action"]["min"][13],
            raw_embodiment_stats[robot_type]["action"]["max"][13],
        )
    eps = 1e-8

    for index in range(min(action_np.shape[0], used_chunk_size)):
        action_item = action_np[index]

        if robot_type == "ARX5":
            target_xyz = action_item[:3] + state_np[:3]
            target_rot = (Rotation.from_rotvec(action_item[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_euler = Rotation.from_rotvec(target_rot).as_euler("xyz", degrees=False)
            target_gripper = action_item[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min

            result_list.append(target_xyz.tolist() + target_euler.tolist() + [target_gripper])
            continue

        if robot_type == "UR5":
            target_joint = action_item[:6] + state_np[:6]
            target_gripper = 0.1 - action_item[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min
            else:
                target_gripper = target_gripper / 0.1 * 255

            result_list.append(target_joint.tolist() + [target_gripper])
            continue

        if robot_type == "Franka":
            target_xyz = action_item[:3] + state_np[:3]
            target_rot = (Rotation.from_rotvec(action_item[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_quat = Rotation.from_rotvec(target_rot).as_quat()
            target_gripper = action_item[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min

            result_list.append(target_xyz.tolist() + target_quat.tolist() + [target_gripper])
            continue

        if robot_type == "aloha":
            target_left_xyz = action_item[:3] + state_np[:3]
            target_left_rot = (Rotation.from_rotvec(action_item[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_left_quat = Rotation.from_rotvec(target_left_rot).as_quat()
            target_left_gripper = action_item[6].item()
            if raw_embodiment_stats is not None:
                target_left_gripper = (
                    target_left_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min
                )
            if binarization_threshold is not None:
                target_left_gripper = left_gripper_max if target_left_gripper > binarization_threshold else left_gripper_min

            target_right_xyz = action_item[7:10] + state_np[7:10]
            target_right_rot = (
                Rotation.from_rotvec(action_item[10:13]) * Rotation.from_rotvec(state_np[10:13])
            ).as_rotvec()
            target_right_quat = Rotation.from_rotvec(target_right_rot).as_quat()
            target_right_gripper = action_item[13].item()
            if raw_embodiment_stats is not None:
                target_right_gripper = (
                    target_right_gripper / 0.1 * (right_gripper_max - right_gripper_min + eps) + right_gripper_min
                )
            if binarization_threshold is not None:
                target_right_gripper = right_gripper_max if target_right_gripper > binarization_threshold else right_gripper_min

            result_list.append(
                target_left_xyz.tolist()
                + target_left_quat.tolist()
                + [target_left_gripper]
                + target_right_xyz.tolist()
                + target_right_quat.tolist()
                + [target_right_gripper]
            )
            continue

        raise ValueError(f"Unsupported robot_type: {robot_type}")

    return result_list


def _is_complete_hf_snapshot(path: Path) -> bool:
    if (path / "model.safetensors").exists():
        return True
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    return bool(shard_names) and all((path / shard_name).exists() for shard_name in shard_names)


def _resolve_hf_local_path(model_id: str) -> str | None:
    candidate = Path(model_id).expanduser()
    if candidate.exists() and _is_complete_hf_snapshot(candidate):
        return str(candidate.resolve())

    cache_roots = []
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    cache_roots.append(Path.home() / ".cache/huggingface/hub")
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    repo_cache_name = f"models--{model_id.replace('/', '--')}"
    for cache_root in cache_roots:
        snapshots = cache_root / repo_cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        for candidate_dir in reversed(candidates):
            if _is_complete_hf_snapshot(candidate_dir):
                return str(candidate_dir.resolve())
    return None


def _patch_spirit_backbone_config(config_path: Path, backbone_override: str | None) -> str | None:
    if not config_path.exists():
        return None

    original = config_path.read_text(encoding="utf-8")
    cfg_data = json.loads(original)
    backbone = backbone_override or cfg_data.get("backbone")
    local_backbone = _resolve_hf_local_path(str(backbone)) if backbone else None
    if not local_backbone or cfg_data.get("backbone") == local_backbone:
        return None

    cfg_data["backbone"] = local_backbone
    config_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    return original


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.device = self._get_device(self.model_cfg.get("device", "auto"))

        checkpoint_path = _resolve_spirit_checkpoint_dir(self.model_cfg)
        spirit_base_weights = _resolve_policy_path(self.model_cfg.get("spirit_base_weights"))
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Spirit_v15 checkpoint directory not found: {checkpoint_path}")
        if not (checkpoint_path / "config.json").exists():
            base_config = spirit_base_weights / "config.json" if spirit_base_weights else None
            if base_config and base_config.exists():
                (checkpoint_path / "config.json").symlink_to(base_config)
            else:
                raise FileNotFoundError(
                    "Spirit_v15 checkpoint is missing config.json and no usable "
                    f"spirit_base_weights config was found: {spirit_base_weights}"
                )
        checkpoint_path = str(checkpoint_path)

        self.default_task_name = None
        self.fallback_task_name = self._resolve_known_task_name(
            self.model_cfg.get("fallback_task_name") or "stack_bowls"
        )
        self.force_default_task_name = bool(self.model_cfg.get("force_default_task_name", True))
        task_name = self.model_cfg.get("task_name")
        self.default_task_name = self._resolve_task_name(task_name)
        self.default_prompt = self.model_cfg.get("prompt", TASK_INFO[self.default_task_name]["task"])
        self.used_chunk_size = int(self.model_cfg.get("used_chunk_size", 60))
        self.raw_embodiment_stats = None

        raw_stats_path = self._resolve_raw_stats_path(Path(checkpoint_path))
        if raw_stats_path:
            with open(raw_stats_path, "r", encoding="utf-8") as file:
                self.raw_embodiment_stats = json.load(file)

        config_path = Path(checkpoint_path) / "config.json"
        config_backup = _patch_spirit_backbone_config(
            config_path,
            str(_resolve_policy_path(self.model_cfg.get("spirit_backbone_path")) or ""),
        )
        qwen_device_map = "auto" if self.device.type == "cuda" and torch.cuda.device_count() > 1 else None
        if qwen_device_map is not None:
            print(f"[Spirit_v15] Using multi-GPU Qwen device_map={qwen_device_map} across {torch.cuda.device_count()} visible GPUs.")
        try:
            self.policy = SpiritVLAPolicy.from_pretrained(
                checkpoint_path,
                qwen_device_map=qwen_device_map,
            )
        finally:
            if config_backup is not None:
                config_path.write_text(config_backup, encoding="utf-8")
        self._assert_norm_stats_loaded()
        self._move_policy_to_device()
        self.policy.eval()
        self.model = self.policy

        self._latest_obs_list: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _move_policy_to_device(self) -> None:
        if getattr(getattr(self.policy, "qwen", None), "hf_device_map", None):
            for name, module in self.policy.named_children():
                if name != "qwen":
                    module.to(self.device)
            return
        self.policy.to(self.device)

    def _resolve_raw_stats_path(self, checkpoint_dir: Path) -> str | None:
        configured_path = self.model_cfg.get("raw_embodiment_stats_json_path")
        if configured_path:
            return str(Path(configured_path).expanduser().resolve())

        checkpoint_root = (_CHECKPOINTS_DIR / str(self.model_cfg.get("ckpt_name"))).expanduser().resolve() if self.model_cfg.get("ckpt_name") else checkpoint_dir
        candidates = [
            checkpoint_dir / "raw_embodiment_stats.json",
            checkpoint_dir / "embodiment_stats.json",
            checkpoint_dir / "stats" / "raw_embodiment_stats.json",
            checkpoint_dir / "assets" / "raw_embodiment_stats.json",
            checkpoint_root / "raw_embodiment_stats.json",
            checkpoint_root / "embodiment_stats.json",
            checkpoint_root / "stats" / "raw_embodiment_stats.json",
            checkpoint_root / "assets" / "raw_embodiment_stats.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    def _assert_norm_stats_loaded(self):
        required_buffers = {
            "normalize_inputs.buffer_observation_state": self.policy.normalize_inputs.buffer_observation_state,
            "normalize_targets.buffer_action": self.policy.normalize_targets.buffer_action,
            "unnormalize_outputs.buffer_action": self.policy.unnormalize_outputs.buffer_action,
        }
        for name, buffer in required_buffers.items():
            if torch.isinf(buffer["min"]).any() or torch.isinf(buffer["max"]).any():
                raise RuntimeError(
                    f"Spirit_v15 norm stats not loaded for {name}; checkpoint is missing normalization buffers or load_state_dict did not restore them."
                )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._latest_obs_list = list(obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self._latest_obs_list is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            observation = self._latest_obs_list[batch_index]
            result = self.infer(
                observation=observation,
                instruction=resolve_prompt(observation, self.default_prompt),
                task_name=observation.get("task_name"),
            )
            
            action_list.append(
                self._decode_action_chunk(
                    result["actions"],
                    result["task_name"],
                    result["action_type"],
                )
            )

        return action_list

    def reset(self):
        self._latest_obs_list = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()

    def infer(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        task_name: str | None = None,
    ) -> dict[str, Any]:
        resolved_task_name = self._resolve_task_name(task_name)
        batch = self._prepare_batch(observation, resolved_task_name, instruction)

        used_chunk_size = self.used_chunk_size
        if resolved_task_name in TASKS_USE_LESS_CHUNK_SIZE:
            used_chunk_size = 40

        binarization_threshold = TASTS_APPLY_GRIPPER_BINARIZATION.get(resolved_task_name)
        with torch.inference_mode():
            action_tensor = self.policy.select_action(batch).cpu()

        actions = _post_process_action(
            action_tensor.squeeze(0).numpy(),
            batch["observation.state.before_norm"].numpy(),
            TASK_INFO[resolved_task_name]["robot_type"],
            used_chunk_size,
            self.raw_embodiment_stats,
            binarization_threshold,
        )
        return {
            "actions": actions,
            "action_type": self._resolve_action_type(resolved_task_name),
            "task_name": resolved_task_name,
            "instruction": instruction,
        }

    def _resolve_known_task_name(self, task_name: str | None) -> str | None:
        if not task_name:
            return None
        if task_name in TASK_INFO:
            return task_name
        alias = TASK_NAME_ALIASES.get(task_name)
        if alias in TASK_INFO:
            return alias
        return None

    def _resolve_task_name(self, task_name: str | None) -> str:
        if self.force_default_task_name and self.fallback_task_name is not None:
            return self.fallback_task_name

        for candidate in (task_name, self.default_task_name, self.fallback_task_name):
            if not candidate:
                continue
            resolved = self._resolve_known_task_name(candidate)
            if resolved is not None:
                return resolved
        available = ", ".join(sorted(TASK_INFO.keys()))
        raise KeyError(f"unsupported Spirit task name: {task_name!r}; available tasks: {available}")

    def _prepare_batch(
        self,
        observation: dict[str, Any],
        task_name: str,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        spirit_observation = self._normalize_observation(observation)
        robot_type = TASK_INFO[task_name]["robot_type"]
        task_text = _normalize_prompt_value(instruction) or TASK_INFO[task_name]["task"]
        item: dict[str, Any] = {
            "task": [task_text],
            "normalized_in_getitem": torch.tensor([False]),
            "batch_source": "rb",
            "robot_type": [robot_type],
        }

        state_tensor = self._extract_internal_state(spirit_observation, robot_type)
        item["observation.state.before_norm"] = state_tensor.clone()
        item["observation.state"] = state_tensor.unsqueeze(0).to(self.device)

        semantic_images = {
            "high": spirit_observation["observation"]["head_camera"]["rgb"],
            "left_hand": spirit_observation["observation"]["left_camera"]["rgb"],
            "right_hand": spirit_observation["observation"]["right_camera"]["rgb"],
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            image = semantic_images[TASK_INFO[task_name][key]]
            item[key] = self._image_to_tensor(image).unsqueeze(0).to(self.device)
        return item

    def _normalize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "observation" in observation:
            normalized = dict(observation)
            if "endpose" not in normalized:
                state_dict = observation.get("state")
                if isinstance(state_dict, dict):
                    endpose = self._build_endpose_from_state(state_dict)
                    if endpose is not None:
                        normalized["endpose"] = endpose
            return normalized

        state_dict = observation.get("state")
        normalized = {
            "observation": {
                "head_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
                    )
                },
                "left_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
                    )
                },
                "right_camera": {
                    "rgb": ensure_hwc_uint8(
                        extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
                    )
                },
            }
        }

        if isinstance(state_dict, dict):
            endpose = self._build_endpose_from_state(state_dict)
            if endpose is not None:
                normalized["endpose"] = endpose

            joint_vector = self._build_joint_action_from_state(state_dict)
            if joint_vector is not None:
                normalized["joint_action"] = {"vector": joint_vector}

        return normalized

    def _build_endpose_from_state(self, state_dict: dict[str, Any]) -> dict[str, Any] | None:
        if {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}.issubset(state_dict):
            return {
                "left_endpose": quat_wxyz_to_xyzw(np.asarray(state_dict["left_ee_pose"], dtype=np.float32)),
                "left_gripper": _to_scalar_gripper(state_dict["left_ee_joint_state"]),
                "right_endpose": quat_wxyz_to_xyzw(np.asarray(state_dict["right_ee_pose"], dtype=np.float32)),
                "right_gripper": _to_scalar_gripper(state_dict["right_ee_joint_state"]),
            }

        pose = state_dict.get("ee_pose")
        gripper = state_dict.get("ee_joint_state")
        if pose is None or gripper is None:
            pose = state_dict.get("left_ee_pose")
            gripper = state_dict.get("left_ee_joint_state")

        if pose is None or gripper is None:
            return None

        return {
            "left_endpose": quat_wxyz_to_xyzw(np.asarray(pose, dtype=np.float32)),
            "left_gripper": _to_scalar_gripper(gripper),
        }

    def _build_joint_action_from_state(self, state_dict: dict[str, Any]) -> np.ndarray | None:
        if "arm_joint_state" in state_dict and "ee_joint_state" in state_dict:
            arm = np.asarray(state_dict["arm_joint_state"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(state_dict["ee_joint_state"], dtype=np.float32).reshape(-1)
            return np.concatenate([arm, gripper], axis=0).astype(np.float32)

        if {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"}.issubset(
            state_dict
        ):
            left_arm = np.asarray(state_dict["left_arm_joint_state"], dtype=np.float32).reshape(-1)
            left_gripper = np.asarray(state_dict["left_ee_joint_state"], dtype=np.float32).reshape(-1)
            right_arm = np.asarray(state_dict["right_arm_joint_state"], dtype=np.float32).reshape(-1)
            right_gripper = np.asarray(state_dict["right_ee_joint_state"], dtype=np.float32).reshape(-1)
            return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=0).astype(np.float32)

        return None

    def _extract_internal_state(self, observation: dict[str, Any], robot_type: str) -> torch.Tensor:
        endpose = observation.get("endpose") or {}
        if robot_type == "ARX5":
            if "joint_action" in observation and "vector" in observation["joint_action"]:
                return self._robotwin_joint_state_to_internal(
                    np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                    robot_type,
                )
            if "action" in observation:
                return self._robotwin_joint_state_to_internal(np.asarray(observation["action"], dtype=np.float32), robot_type)

        if robot_type == "aloha":
            if all(key in endpose for key in ("left_endpose", "left_gripper", "right_endpose", "right_gripper")):
                return self._dual_ee_to_internal_state(
                    left_endpose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    left_gripper=float(endpose["left_gripper"]),
                    right_endpose=np.asarray(endpose["right_endpose"], dtype=np.float32),
                    right_gripper=float(endpose["right_gripper"]),
                )

        if robot_type in {"ARX5", "Franka", "UR5"}:
            if "left_endpose" in endpose and "left_gripper" in endpose:
                return self._single_ee_to_internal_state(
                    ee_pose=np.asarray(endpose["left_endpose"], dtype=np.float32),
                    gripper=float(endpose["left_gripper"]),
                )

        if "joint_action" in observation and "vector" in observation["joint_action"]:
            return self._robotwin_joint_state_to_internal(
                np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                robot_type,
            )
        if "action" in observation:
            return self._robotwin_joint_state_to_internal(np.asarray(observation["action"], dtype=np.float32), robot_type)
        raise KeyError("missing usable state in endpose, joint_action.vector, or action")

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected RGB image with shape [H, W, 3], got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0) if np.issubdtype(array.dtype, np.floating) else array
            if array.max() <= 1.0:
                array = (array * 255.0).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        resized = cv2.resize(array, (320, 240), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0

    @staticmethod
    def _robotwin_joint_state_to_internal(state: np.ndarray, robot_type: str) -> torch.Tensor:
        state_tensor = torch.zeros(14, dtype=torch.float32)
        if robot_type == "ARX5":
            if state.shape[0] != 7:
                raise ValueError(f"expected 7-dim ARX5 state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_euler("xyz", state[3:6], degrees=False).as_rotvec())
            state_tensor[6] = torch.tensor(state[6], dtype=torch.float32)
            return state_tensor
        if robot_type == "UR5":
            if state.shape[0] not in {7, 14}:
                raise ValueError(f"expected 7-dim or 14-dim UR5 state, got {state.shape}")
            state_tensor[:7] = torch.from_numpy(state[:7])
            return state_tensor
        if robot_type == "Franka":
            if state.shape[0] not in {8, 16}:
                raise ValueError(f"expected 8-dim or 16-dim Franka state, got {state.shape}")
            state_tensor[:3] = torch.from_numpy(state[:3])
            state_tensor[3:6] = torch.tensor(Rotation.from_quat(state[3:7]).as_rotvec())
            state_tensor[6] = torch.tensor(state[7], dtype=torch.float32)
            return state_tensor
        if robot_type == "aloha":
            raise ValueError(
                f"aloha requires endpose fields in observation; received only joint state with shape {state.shape}"
            )
        raise ValueError(f"unsupported robot type: {robot_type}")

    @staticmethod
    def _single_ee_to_internal_state(ee_pose: np.ndarray, gripper: float) -> torch.Tensor:
        if ee_pose.shape[0] != 7:
            raise ValueError(f"expected 7-dim end-effector pose, got {ee_pose.shape}")

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(ee_pose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(ee_pose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _dual_ee_to_internal_state(
        left_endpose: np.ndarray,
        left_gripper: float,
        right_endpose: np.ndarray,
        right_gripper: float,
    ) -> torch.Tensor:
        if left_endpose.shape[0] != 7 or right_endpose.shape[0] != 7:
            raise ValueError(
                f"expected 7-dim dual-arm endpose, got left={left_endpose.shape}, right={right_endpose.shape}"
            )

        state_tensor = torch.zeros(14, dtype=torch.float32)
        state_tensor[:3] = torch.from_numpy(left_endpose[:3])
        state_tensor[3:6] = torch.tensor(Rotation.from_quat(left_endpose[3:]).as_rotvec())
        state_tensor[6] = torch.tensor(left_gripper, dtype=torch.float32)
        state_tensor[7:10] = torch.from_numpy(right_endpose[:3])
        state_tensor[10:13] = torch.tensor(Rotation.from_quat(right_endpose[3:]).as_rotvec())
        state_tensor[13] = torch.tensor(right_gripper, dtype=torch.float32)
        return state_tensor

    @staticmethod
    def _resolve_action_type(task_name: str) -> str:
        action_type = TASK_INFO[task_name].get("action_type")
        if action_type == "leftjoint":
            return "joint"
        return "ee"

    def _decode_action_chunk(self, action_chunk: list[list[float]], task_name: str, action_type: str):
        robot_type = TASK_INFO[task_name]["robot_type"]
        if action_type == "joint":
            return [self._decode_joint_action(action, robot_type) for action in action_chunk]
        return [self._decode_ee_action(action, robot_type) for action in action_chunk]

    @staticmethod
    def _decode_joint_action(action: list[float], robot_type: str) -> dict[str, np.ndarray]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if robot_type == "UR5":
            return {
                "arm_joint_state": action_array[:6],
                "ee_joint_state": action_array[6:7],
            }
        if robot_type == "aloha":
            return {
                "left_arm_joint_state": action_array[:6],
                "left_ee_joint_state": action_array[6:7],
                "right_arm_joint_state": action_array[7:13],
                "right_ee_joint_state": action_array[13:14],
            }
        raise ValueError(f"Unsupported joint action robot type: {robot_type}")

    @staticmethod
    def _decode_ee_action(action: list[float], robot_type: str) -> dict[str, np.ndarray]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)

        if robot_type == "ARX5":
            quat_xyzw = Rotation.from_euler("xyz", action_array[3:6], degrees=False).as_quat().astype(np.float32)
            return {
                "ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(quat_xyzw)], axis=0),
                "ee_joint_state": action_array[6:7],
            }

        if robot_type == "Franka":
            return {
                "ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(action_array[3:7])], axis=0),
                "ee_joint_state": action_array[7:8],
            }

        if robot_type == "aloha":
            return {
                "left_ee_pose": np.concatenate([action_array[:3], quat_xyzw_to_wxyz(action_array[3:7])], axis=0),
                "left_ee_joint_state": action_array[7:8],
                "right_ee_pose": np.concatenate([action_array[8:11], quat_xyzw_to_wxyz(action_array[11:15])], axis=0),
                "right_ee_joint_state": action_array[15:16],
            }

        if robot_type == "UR5":
            return {
                "arm_joint_state": action_array[:6],
                "ee_joint_state": action_array[6:7],
            }

        raise ValueError(f"Unsupported ee action robot type: {robot_type}")