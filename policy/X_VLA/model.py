from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_XVLA_ROOT = _CUR_DIR / "xvla"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_CUR_DIR), str(_XVLA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info

from xvla.models.modeling_xvla import XVLA
from xvla.models.processing_xvla import XVLAProcessor


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


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)


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


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CUR_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_checkpoint_root(model_cfg: dict[str, Any]) -> Path | None:
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name:
        checkpoint_root = (_CHECKPOINTS_DIR / str(ckpt_name)).expanduser().resolve()
        if not checkpoint_root.is_dir():
            return checkpoint_root

        candidate_dirs = []
        if any((checkpoint_root / marker).exists() for marker in ("config.json", "model.safetensors", "preprocessor_config.json")):
            candidate_dirs.append(checkpoint_root)
        candidate_dirs.extend(
            child
            for child in sorted(checkpoint_root.iterdir())
            if child.is_dir() and any((child / marker).exists() for marker in ("config.json", "model.safetensors", "preprocessor_config.json"))
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

    for key in ("model_path", "checkpoint_path", "processor_path"):
        resolved = _resolve_path(model_cfg.get(key))
        if resolved is not None:
            return resolved
    return None


def _build_candidate_dirs(checkpoint_root: Path | None, *explicit_paths: str | None) -> list[Path]:
    candidates: list[Path] = []
    for explicit_path in explicit_paths:
        resolved = _resolve_path(explicit_path)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)

    if checkpoint_root is not None:
        for candidate in (
            checkpoint_root,
            checkpoint_root / "processor",
            checkpoint_root / "model",
            checkpoint_root / "base",
            checkpoint_root / "base_model",
            checkpoint_root / "checkpoint",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def quat_to_rotate6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 dims, got shape {quat.shape}.")
    quat = quat.copy()
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    zero_norm_mask = norm.squeeze(-1) < 1e-8
    if np.any(zero_norm_mask):
        quat[zero_norm_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)
    xyzw = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)
    rot = R.from_quat(xyzw).as_matrix()
    return rot[..., :, :2].reshape(quat.shape[:-1] + (6,)).astype(np.float32)


def rotate6d_to_quat(vec6: np.ndarray) -> np.ndarray:
    vec6 = np.asarray(vec6, dtype=np.float32)
    if vec6.shape[-1] != 6:
        raise ValueError(f"Expected last dim to be 6, got {vec6.shape[-1]}.")

    a1 = vec6[..., 0:5:2]
    a2 = vec6[..., 1:6:2]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    rot = np.stack((b1, b2, b3), axis=-1)

    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    trace = m00 + m11 + m22
    quat = np.empty(rot.shape[:-2] + (4,), dtype=np.float32)

    positive = trace > 0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-8)) * 2
    quat[positive, 3] = 0.25 * s
    quat[positive, 0] = (m21[positive] - m12[positive]) / s
    quat[positive, 1] = (m02[positive] - m20[positive]) / s
    quat[positive, 2] = (m10[positive] - m01[positive]) / s

    cond1 = (~positive) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00[cond1] - m11[cond1] - m22[cond1], 1e-8)) * 2
    quat[cond1, 3] = (m21[cond1] - m12[cond1]) / s
    quat[cond1, 0] = 0.25 * s
    quat[cond1, 1] = (m01[cond1] + m10[cond1]) / s
    quat[cond1, 2] = (m02[cond1] + m20[cond1]) / s

    cond2 = (~positive) & (~cond1) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[cond2] - m00[cond2] - m22[cond2], 1e-8)) * 2
    quat[cond2, 3] = (m02[cond2] - m20[cond2]) / s
    quat[cond2, 0] = (m01[cond2] + m10[cond2]) / s
    quat[cond2, 1] = 0.25 * s
    quat[cond2, 2] = (m12[cond2] + m21[cond2]) / s

    cond3 = (~positive) & (~cond1) & (~cond2)
    s = np.sqrt(np.maximum(1.0 + m22[cond3] - m00[cond3] - m11[cond3], 1e-8)) * 2
    quat[cond3, 3] = (m10[cond3] - m01[cond3]) / s
    quat[cond3, 0] = (m02[cond3] + m20[cond3]) / s
    quat[cond3, 1] = (m12[cond3] + m21[cond3]) / s
    quat[cond3, 2] = 0.25 * s

    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1).astype(np.float32)




def build_xvla_proprio(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    left_ee = np.asarray(state["left_ee_pose"], dtype=np.float32)
    right_ee = np.asarray(state["right_ee_pose"], dtype=np.float32)
    left_grip_joint = np.asarray(state["left_ee_joint_state"], dtype=np.float32)[-1]
    right_grip_joint = np.asarray(state["right_ee_joint_state"], dtype=np.float32)[-1]

    left_grip = 1 - left_grip_joint * 2
    right_grip = 1 - right_grip_joint * 2

    return np.concatenate(
        [
            left_ee[:3],
            quat_to_rotate6d(left_ee[3:]),
            np.array([left_grip_joint], dtype=np.float32),
            right_ee[:3],
            quat_to_rotate6d(right_ee[3:]),
            np.array([right_grip_joint], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def encode_obs(observation, default_prompt):
    if "images" in observation and "state" in observation:
        head = ensure_hwc_uint8(observation["images"]["cam_high"])
        prompt = resolve_prompt(observation, default_prompt)
        return {
            "images": [head],
            "proprio": build_xvla_proprio(observation),
            "prompt": prompt,
            "output_format": "xpolicylab",
        }

    images = [ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]))]
    prompt = resolve_prompt(observation, default_prompt)
    return {
        "images": images,
        "proprio": build_xvla_proprio(observation),
        "prompt": prompt,
        "output_format": "xpolicylab",
    }


def action_chunk_to_ee_dict_list(action_chunk: np.ndarray):
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]

    left_xyz = action_chunk[:, :3]
    left_rotate6d = action_chunk[:, 3:9]
    left_gripper = action_chunk[:, 9:10]
    left_quat = rotate6d_to_quat(left_rotate6d)
    left_grip = 1 - 2 * (left_gripper > 0.7)

    right_xyz = action_chunk[:, 10:13]
    right_rotate6d = action_chunk[:, 13:19]
    right_quat = rotate6d_to_quat(right_rotate6d)
    right_gripper = action_chunk[:, 19:20]
    right_grip = 1 - 2 * (right_gripper > 0.7)

    actions = []
    for idx in range(action_chunk.shape[0]):
        actions.append(
            {
                "left_ee_pose": np.concatenate([left_xyz[idx], left_quat[idx]], axis=0).astype(np.float32),
                "left_ee_joint_state": np.asarray([left_gripper[idx, 0]], dtype=np.float32),
                "right_ee_pose": np.concatenate([right_xyz[idx], right_quat[idx]], axis=0).astype(np.float32),
                "right_ee_joint_state": np.asarray([right_gripper[idx, 0]], dtype=np.float32),
            }
        )
    return actions


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "ee")
        if self.action_type != "ee":
            raise ValueError("X-VLA in XPolicyLab currently supports only action_type='ee'.")

        self.default_prompt = self.model_cfg.get("prompt", self.task_name)
        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self._latest_env_idx_list: list[int] = [0]
        self.observation_window: list[dict[str, Any]] | None = None

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.processor = self._load_processor(self.model_cfg)
        self.model = self._load_model(self.model_cfg)
        self.model.eval()

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _load_processor(self, model_cfg):
        checkpoint_root = _resolve_checkpoint_root(model_cfg)
        candidate_paths = _build_candidate_dirs(
            checkpoint_root,
            model_cfg.get("processor_path"),
            model_cfg.get("model_path"),
            model_cfg.get("checkpoint_path"),
        )
        
        processor_path = None
        for candidate in candidate_paths:
            if (candidate / "preprocessor_config.json").exists():
                processor_path = str(candidate)
                break
        if processor_path is None:
            searched = ", ".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(
                "Could not find XVLA processor files. "
                f"Searched: {searched}"
            )
        return XVLAProcessor.from_pretrained(processor_path)

    def _load_model(self, model_cfg):
        checkpoint_root = _resolve_checkpoint_root(model_cfg)
        candidate_paths = _build_candidate_dirs(
            checkpoint_root,
            model_cfg.get("model_path"),
            model_cfg.get("checkpoint_path"),
        )
        model_path = None
        for candidate in candidate_paths:
            if (candidate / "config.json").exists():
                model_path = str(candidate)
                break
        if model_path is None:
            raise ValueError("ckpt_name, model_path, or checkpoint_path is required for X-VLA.")

        model = XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device).to(torch.float32)

        lora_path = model_cfg.get("lora_path") or model_cfg.get("LoRA_path")
        if not lora_path and checkpoint_root is not None and (checkpoint_root / "adapter_config.json").exists():
            lora_path = str(checkpoint_root)
        if lora_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float32,
            ).to(self.device)
        return model

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            obs.get("env_idx", index) if isinstance(obs, dict) else index
            for index, obs in enumerate(obs_list)
        ]
        self.observation_window = [encode_obs(obs, self.default_prompt) for obs in obs_list]

    def infer(self, observation: dict[str, Any], steps: int | None = None):
        pil_images = [Image.fromarray(image) for image in observation["images"]]
        prompt = resolve_prompt(observation, self.default_prompt)
        inputs = self.processor(images=pil_images, language_instruction=prompt)
        missing_inputs = {"input_ids", "image_input", "image_mask"} - set(inputs)
        if missing_inputs:
            raise ValueError(
                f"Processor returned incomplete inputs: missing {sorted(missing_inputs)} for prompt={prompt!r}."
            )
        proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32).unsqueeze(0)
        domain_id = torch.tensor([int(self.model_cfg.get("domain_id", 0))], dtype=torch.long)

        def to_model(tensor: torch.Tensor):
            if tensor.is_floating_point():
                return tensor.to(device=self.device, dtype=torch.float32)
            return tensor.to(device=self.device)

        inputs = {key: to_model(value) for key, value in inputs.items()}
        inputs["proprio"] = to_model(proprio)
        inputs["domain_id"] = domain_id.to(self.device)

        denoise_steps = int(steps if steps is not None else self.model_cfg.get("steps", 10))
        with torch.no_grad():
            action = self.model.generate_actions(**inputs, steps=denoise_steps)
        return action.squeeze(0).float().cpu().numpy()

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for batch_index, _ in enumerate(env_idx_list):
            encoded_obs = self.observation_window[batch_index]
            action_chunk = self.infer(encoded_obs)
            action_list.append(action_chunk_to_ee_dict_list(action_chunk))
        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]