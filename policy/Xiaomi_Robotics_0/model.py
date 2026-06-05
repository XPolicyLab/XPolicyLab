from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmengine import Config
from PIL import Image
from transformers import AutoProcessor

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info

_POLICY_DIR = Path(__file__).resolve().parent
_XR0_ROOT = _POLICY_DIR / "xiaomi_robotics_0" / "xr0"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

if str(_XR0_ROOT) not in sys.path:
    sys.path.insert(0, str(_XR0_ROOT))

from mibot.models import MIMODEL  # noqa: E402
from mibot.utils.io import (  # noqa: E402
    build_action_mask,
    compose_state,
    denormalize_action,
    recover_action,
    resize_image,
    validate_stats,
)

PROMPT_RE = re.compile(r"(<image>|<video>)")
XR0_PROMPT_TEMPLATE = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n"
    "# Left-Wrist View\n<image>\n"
    "# Right-Wrist View\n<image>\n"
    "Generate robot actions for the task:\n{task}"
)


def _resolve_relative_path(raw_path: str | Path, base_dir: Path) -> Path:
    """Resolve a deploy.yml path relative to base_dir (policy or xr0 root)."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        raise ValueError(
            f"Absolute paths are not supported: {path}. "
            f"Use a path relative to {base_dir} or set it in deploy.yml."
        )
    return (base_dir / path).resolve()


DEFAULT_VLM_PROCESSOR_REPO = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"


def _is_hf_repo_id(value: str) -> bool:
    """True for HuggingFace repo ids like org/model (not a filesystem path)."""
    if value.startswith((".", "/")) or "://" in value:
        return False
    parts = value.split("/")
    return len(parts) >= 2 and all(parts)


def _resolve_processor_source(model_cfg: dict[str, Any]) -> str | Path:
    """Return HF repo id (download on load) or a local path relative to xr0/."""
    raw_path = model_cfg.get("vlm_processor_path")
    if raw_path is None or raw_path == "":
        return DEFAULT_VLM_PROCESSOR_REPO

    raw = str(raw_path)
    if _is_hf_repo_id(raw):
        local_dir = (_XR0_ROOT / raw).resolve()
        if (local_dir / "processor_config.json").is_file():
            return _resolve_relative_path(raw, _XR0_ROOT)
        return raw

    processor_dir = _resolve_relative_path(raw, _XR0_ROOT)
    if (processor_dir / "processor_config.json").is_file():
        return processor_dir

    raise FileNotFoundError(
        f"VLM processor not found: {processor_dir} (missing processor_config.json). "
        f"Set vlm_processor_path in deploy.yml to a HuggingFace repo id "
        f"(default: {DEFAULT_VLM_PROCESSOR_REPO}) or a directory under {_XR0_ROOT}."
    )


def _resolve_ckpt_setting(model_cfg: dict[str, Any]) -> str:
    ckpt_name = str(model_cfg.get("ckpt_name", ""))
    if ckpt_name.count("-") >= 3:
        return ckpt_name
    dataset_name = model_cfg["dataset_name"]
    env_cfg_type = model_cfg["env_cfg_type"]
    expert_data_num = model_cfg["expert_data_num"]
    action_type = model_cfg["action_type"]
    seed = model_cfg["seed"]
    return f"{dataset_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"


def _resolve_ckpt_dir(model_cfg: dict[str, Any]) -> Path:
    if model_cfg.get("model_dir"):
        return _resolve_relative_path(model_cfg["model_dir"], _POLICY_DIR)

    ckpt_setting = _resolve_ckpt_setting(model_cfg)
    return (_CHECKPOINTS_DIR / ckpt_setting).resolve()


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _force_sdpa_attn(node: Any) -> None:
    """Deploy hosts may lack flash_attn; fall back to PyTorch SDPA."""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in ("attn_implementation", "_attn_implementation") and value == "flash_attention_2":
                node[key] = "sdpa"
            else:
                _force_sdpa_attn(value)
    elif isinstance(node, list):
        for item in node:
            _force_sdpa_attn(item)


def _patch_xr0_vlm_attn_to_sdpa() -> None:
    """XR0 hardcodes flash_attention_2; force sdpa when flash_attn is unavailable."""
    from mibot.models.VLA import XR0 as xr0_module

    if getattr(xr0_module.Qwen3VLForConditionalGeneration, "_xpolicylab_sdpa_patch", False):
        return

    original_from_pretrained = xr0_module.Qwen3VLForConditionalGeneration.from_pretrained

    @classmethod
    def from_pretrained_sdpa(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return original_from_pretrained(*args, **kwargs)

    xr0_module.Qwen3VLForConditionalGeneration.from_pretrained = from_pretrained_sdpa  # type: ignore[method-assign]
    xr0_module.Qwen3VLForConditionalGeneration._xpolicylab_sdpa_patch = True


def _load_xr0_model(model_dir: Path, checkpoint_tag: str, device: torch.device):
    cfg = Config.fromfile(str(model_dir / "config.py"))
    _force_sdpa_attn(cfg.model.params.model)
    _patch_xr0_vlm_attn_to_sdpa()
    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)

    ckpt_file = model_dir / f"{checkpoint_tag}.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location="cpu")["module"]
    missing, unexpected = model.load_state_dict(_strip_prefix(ckpt, "model."), assign=True)
    if missing:
        print(f"[Xiaomi_Robotics_0] Missing keys when loading checkpoint: {missing[:5]}...")
    if unexpected:
        print(f"[Xiaomi_Robotics_0] Unexpected keys when loading checkpoint: {unexpected[:5]}...")

    data_cfg = cfg.data.params.train_datasets
    action_length = int(data_cfg.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = validate_stats(data_cfg.mean, data_cfg.std, action_length)
    action_mask = build_action_mask(action_length)

    return cfg, model.eval().to(device), mean, std, action_mask, action_length


def _decode_compressed_image(image_buffer: np.ndarray) -> np.ndarray:
    return decode_image_bit(image_buffer)


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return _decode_compressed_image(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        return _decode_compressed_image(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image(observation: dict[str, Any], candidate_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "colors", "rgb"):
                if image_key in image:
                    return _ensure_hwc_uint8(image[image_key])
        else:
            return _ensure_hwc_uint8(image)
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def _quat_wxyz_to_rotm(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _rotm_to_quat_wxyz(rotm: np.ndarray) -> np.ndarray:
    rotm = np.asarray(rotm, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(rotm))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotm[2, 1] - rotm[1, 2]) / s
        y = (rotm[0, 2] - rotm[2, 0]) / s
        z = (rotm[1, 0] - rotm[0, 1]) / s
    elif rotm[0, 0] > rotm[1, 1] and rotm[0, 0] > rotm[2, 2]:
        s = np.sqrt(1.0 + rotm[0, 0] - rotm[1, 1] - rotm[2, 2]) * 2.0
        w = (rotm[2, 1] - rotm[1, 2]) / s
        x = 0.25 * s
        y = (rotm[0, 1] + rotm[1, 0]) / s
        z = (rotm[0, 2] + rotm[2, 0]) / s
    elif rotm[1, 1] > rotm[2, 2]:
        s = np.sqrt(1.0 + rotm[1, 1] - rotm[0, 0] - rotm[2, 2]) * 2.0
        w = (rotm[0, 2] - rotm[2, 0]) / s
        x = (rotm[0, 1] + rotm[1, 0]) / s
        y = 0.25 * s
        z = (rotm[1, 2] + rotm[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotm[2, 2] - rotm[0, 0] - rotm[1, 1]) * 2.0
        w = (rotm[1, 0] - rotm[0, 1]) / s
        x = (rotm[0, 2] + rotm[2, 0]) / s
        y = (rotm[1, 2] + rotm[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / (np.linalg.norm(quat) + 1e-8)


def _pose_wxyz_to_rotm(pose_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_wxyz = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
    pos = pose_wxyz[:3]
    rotm = _quat_wxyz_to_rotm(pose_wxyz[3:7])
    return pos, rotm


def _rotm_to_pose_wxyz(pos: np.ndarray, rotm: np.ndarray) -> np.ndarray:
    quat_wxyz = _rotm_to_quat_wxyz(rotm)
    return np.concatenate([np.asarray(pos, dtype=np.float32).reshape(3), quat_wxyz], axis=0)


def _as_1d(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length}, got {arr.shape}")
    return arr


def _extract_robot_state(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    state = observation.get("state", {})
    left_pose = _as_1d(state["left_ee_pose"], 7)
    right_pose = _as_1d(state["right_ee_pose"], 7)
    left_pos, left_rotm = _pose_wxyz_to_rotm(left_pose)
    right_pos, right_rotm = _pose_wxyz_to_rotm(right_pose)

    return {
        "left_ee_pos": left_pos,
        "left_ee_rotm": left_rotm,
        "left_gripper_pos": _as_1d(state["left_ee_joint_state"], 1),
        "left_arm_joint": _as_1d(state["left_arm_joint_state"], 6),
        "right_ee_pos": right_pos,
        "right_ee_rotm": right_rotm,
        "right_gripper_pos": _as_1d(state["right_ee_joint_state"], 1),
        "right_arm_joint": _as_1d(state["right_arm_joint_state"], 6),
    }


def _extract_prompt(observation: dict[str, Any], default_prompt: str) -> str:
    for key in ("instruction", "instructions"):
        if key not in observation:
            continue
        value = observation[key]
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_prompt


def _build_messages(prompt_text: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    human_value = XR0_PROMPT_TEMPLATE.format(task=prompt_text) + " /no_cot"
    conversations = [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": "<cot></cot>"},
    ]

    image_pool = [{"type": "image", "image": image} for image in images]
    messages: list[dict[str, Any]] = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        if role == "assistant":
            messages.append({"role": role, "content": [{"type": "text", "text": turn["value"]}]})
            continue

        content: list[dict[str, Any]] = []
        for part in PROMPT_RE.split(turn["value"]):
            if part == "<image>":
                if not image_pool:
                    raise ValueError("number of <image> placeholders exceeds provided images")
                content.append(image_pool.pop(0))
            elif part == "<video>":
                raise ValueError("video placeholders are not supported")
            elif part:
                content.append({"type": "text", "text": part})
        messages.append({"role": role, "content": content})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused")
    return messages


def _targets_to_action_dict(targets: dict[str, np.ndarray], action_type: str) -> dict[str, np.ndarray]:
    if action_type == "joint":
        return {
            "left_arm_joint_state": targets["left_arm_joint"][0].astype(np.float32),
            "right_arm_joint_state": targets["right_arm_joint"][0].astype(np.float32),
            "left_ee_joint_state": targets["left_gripper_pos"][0].astype(np.float32),
            "right_ee_joint_state": targets["right_gripper_pos"][0].astype(np.float32),
        }

    return {
        "left_ee_pose": _rotm_to_pose_wxyz(targets["left_ee_pos"][0], targets["left_ee_rotm"][0]),
        "right_ee_pose": _rotm_to_pose_wxyz(targets["right_ee_pos"][0], targets["right_ee_rotm"][0]),
        "left_ee_joint_state": targets["left_gripper_pos"][0].astype(np.float32),
        "right_ee_joint_state": targets["right_gripper_pos"][0].astype(np.float32),
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "ee")
        self.default_prompt = model_cfg.get("default_prompt", model_cfg.get("task_name", "Perform the task."))
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint_tag = model_cfg.get("checkpoint_tag", "last")
        model_dir = _resolve_ckpt_dir(model_cfg)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

        _, self.model, mean, std, action_mask, self.action_length = _load_xr0_model(
            model_dir, checkpoint_tag, self.device
        )
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.bfloat16)
        self.std = torch.tensor(std, device=self.device, dtype=torch.bfloat16)
        self.action_mask = torch.from_numpy(action_mask).to(self.device, dtype=torch.bfloat16)

        processor_source = _resolve_processor_source(model_cfg)
        local_only = isinstance(processor_source, Path)
        self.processor = AutoProcessor.from_pretrained(
            str(processor_source),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_only,
        )
        self.processor.tokenizer.padding_side = "right"

        self._obs_list: list[dict[str, Any]] = []
        self._latest_env_idx_list: list[int] = [0]

        print(f"[Xiaomi_Robotics_0] Loaded checkpoint from {model_dir} ({checkpoint_tag})")
        print(f"[Xiaomi_Robotics_0] Loaded processor from {processor_source}")

    def _encode_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = [
            resize_image(Image.fromarray(_extract_image(obs, ["cam_head", "cam_high", "head_camera"])), factor=32, max_pixels=90000),
            resize_image(
                Image.fromarray(_extract_image(obs, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])),
                factor=32,
                max_pixels=90000,
            ),
            resize_image(
                Image.fromarray(_extract_image(obs, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])),
                factor=32,
                max_pixels=90000,
            ),
        ]
        robot_state = _extract_robot_state(obs)
        prompt = _extract_prompt(obs, self.default_prompt)
        state = compose_state(
            left_gripper=robot_state["left_gripper_pos"],
            left_joint=robot_state["left_arm_joint"],
            right_gripper=robot_state["right_gripper_pos"],
            right_joint=robot_state["right_arm_joint"],
        )
        return {
            "messages": _build_messages(prompt, images),
            "state": torch.from_numpy(state),
            "robot_state": robot_state,
        }

    def _build_batch(self, encoded_obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages = [item["messages"] for item in encoded_obs_list]
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        states = torch.stack([item["state"] for item in encoded_obs_list], dim=0).to(self.device, dtype=torch.bfloat16)
        batch["state"] = states
        batch["action"] = torch.zeros(
            (len(encoded_obs_list), *self.mean.shape),
            device=self.device,
            dtype=torch.bfloat16,
        )
        mask = self.action_mask.unsqueeze(0).expand(len(encoded_obs_list), -1, -1)
        batch["action_mask"] = mask
        return batch

    def _predict_action_chunk(self, encoded_obs: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        batch = self._build_batch([encoded_obs])
        with torch.no_grad():
            pred = self.model.generate(batch)
        mask = self.action_mask.unsqueeze(0)
        pred = denormalize_action(pred * mask, self.mean, self.std) * mask
        actions_np = pred[0].detach().float().cpu().numpy()
        robot_state = encoded_obs["robot_state"]

        action_list: list[dict[str, np.ndarray]] = []
        for step in range(self.action_length):
            targets = recover_action(actions_np[step : step + 1], robot_state)
            action_list.append(_targets_to_action_dict(targets, self.action_type))
        return action_list

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._obs_list = [self._encode_observation(obs) for obs in obs_list]

    def get_action(self, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return self._predict_action_chunk(self._obs_list[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return [self._predict_action_chunk(encoded_obs) for encoded_obs in self._obs_list]

    def reset(self):
        self._obs_list = []
        self._latest_env_idx_list = [0]
