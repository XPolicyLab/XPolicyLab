from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit

_POLICY_DIR = Path(__file__).resolve().parent
_GR00T_ROOT = _POLICY_DIR / "gr00t_n17"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

if str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy import Gr00tPolicy  # noqa: E402

VIDEO_KEY_CANDIDATES = {
    "front": ["cam_head", "cam_high", "head_camera", "top_camera"],
    "left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
}


def _load_modality_config(env_cfg_type: str) -> None:
    config_path = _POLICY_DIR / "configs" / f"{env_cfg_type}_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Modality config not found: {config_path}. Run process_data.sh for env_cfg_type={env_cfg_type} first."
        )
    spec = importlib.util.spec_from_file_location(f"gr00t_modality_{env_cfg_type}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


DEFAULT_COSMOS_MODEL_REPO = "nvidia/Cosmos-Reason2-2B"


def _resolve_relative_path(raw_path: str | Path, base_dir: Path) -> Path:
    """Resolve a deploy.yml path relative to base_dir."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        raise ValueError(
            f"Absolute paths are not supported: {path}. "
            f"Use a path relative to {base_dir} or set it in deploy.yml."
        )
    return (base_dir / path).resolve()


def _is_hf_repo_id(value: str) -> bool:
    if value.startswith((".", "/")) or "://" in value:
        return False
    parts = value.split("/")
    return len(parts) >= 2 and all(parts)


def _resolve_cosmos_model(model_cfg: dict[str, Any]) -> str:
    """Return HuggingFace repo id or a local path for Cosmos (processor backbone)."""
    raw_path = model_cfg.get("cosmos_model_path")
    if raw_path is None or raw_path == "":
        return DEFAULT_COSMOS_MODEL_REPO

    raw = str(raw_path)
    for candidate in (
        _POLICY_DIR / raw,
        _CHECKPOINTS_DIR / raw,
        _POLICY_DIR / "checkpoints" / raw,
    ):
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())

    if _is_hf_repo_id(raw):
        return raw

    return str(_resolve_relative_path(raw, _POLICY_DIR))


@contextmanager
def _override_processor_cosmos_model(checkpoint_dir: Path, cosmos_model: str) -> Iterator[None]:
    """Replace baked-in absolute Cosmos paths in processor_config.json during load."""
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        yield
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    processor_kwargs = data.setdefault("processor_kwargs", {})
    previous = processor_kwargs.get("model_name")
    if previous == cosmos_model:
        yield
        return

    processor_kwargs["model_name"] = cosmos_model
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    try:
        yield
    finally:
        if previous is not None:
            processor_kwargs["model_name"] = previous
        else:
            processor_kwargs.pop("model_name", None)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _resolve_checkpoint_dir(model_cfg: dict[str, Any]) -> Path:
    if model_cfg.get("model_dir"):
        return _resolve_relative_path(model_cfg["model_dir"], _POLICY_DIR)

    bench_name = model_cfg["bench_name"]
    ckpt_name = model_cfg["ckpt_name"]
    env_cfg_type = model_cfg["env_cfg_type"]
    expert_data_num = model_cfg["expert_data_num"]
    action_type = model_cfg["action_type"]
    seed = model_cfg["seed"]
    ckpt_setting = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"
    root = (_CHECKPOINTS_DIR / ckpt_setting).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    search_roots = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("checkpoint-"):
            search_roots.append(child)

    candidates = []
    for search_root in search_roots:
        candidates.extend(sorted(search_root.glob("checkpoint-*"), key=lambda p: p.name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories under {root}")

    checkpoint_num = model_cfg.get("checkpoint_num")
    if checkpoint_num in (None, "last"):
        return max(candidates, key=lambda p: _extract_step_number(p.name) or -1)

    desired = _extract_step_number(checkpoint_num)
    if desired is not None:
        for candidate in candidates:
            if _extract_step_number(candidate.name) == desired:
                return candidate.resolve()

    explicit = root / f"checkpoint-{checkpoint_num}"
    if explicit.is_dir():
        return explicit.resolve()
    for search_root in search_roots:
        nested = search_root / f"checkpoint-{checkpoint_num}"
        if nested.is_dir():
            return nested.resolve()

    raise FileNotFoundError(
        f"Checkpoint step {checkpoint_num!r} not found under {root}. "
        f"Available: {[p.name for p in candidates]}"
    )


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


def _to_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """XPolicyLab obs images are RGB; match LeRobot video training."""
    return _ensure_hwc_uint8(image)


def _as_1d(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length}, got {arr.shape}")
    return arr


def _extract_prompt(observation: dict[str, Any], default_prompt: str) -> str:
    for key in ("instruction", "instructions"):
        if key not in observation:
            continue
        value = observation[key]
        if isinstance(value, dict):
            general = value.get("general")
            if isinstance(general, list) and general:
                first = general[0]
                if isinstance(first, dict):
                    conversations = first.get("conversations", [])
                    for turn in conversations:
                        if turn.get("from") == "human" and turn.get("value"):
                            text = str(turn["value"])
                            marker = "Generate robot actions for the task:\n"
                            if marker in text:
                                text = text.split(marker, 1)[1]
                            return text.replace(" /no_cot", "").strip()
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_prompt


def _pack_arm_state(observation: dict[str, Any], side: str) -> np.ndarray:
    state = observation.get("state", {})
    prefix = f"{side}_"
    joint = _as_1d(state[f"{prefix}arm_joint_state"], 6)
    gripper = _as_1d(state[f"{prefix}ee_joint_state"], 1)
    return np.concatenate([joint, gripper], axis=0).astype(np.float32)


def _encode_observation(obs: dict[str, Any], default_prompt: str) -> dict[str, Any]:
    images = {
        video_key: _to_rgb_hwc(_extract_image(obs, candidates))
        for video_key, candidates in VIDEO_KEY_CANDIDATES.items()
    }
    prompt = _extract_prompt(obs, default_prompt)
    left_arm = _pack_arm_state(obs, "left")
    right_arm = _pack_arm_state(obs, "right")

    return {
        "video": {
            key: np.asarray(image, dtype=np.uint8)[None, None, ...]
            for key, image in images.items()
        },
        "state": {
            "left_arm": left_arm[None, None, :],
            "right_arm": right_arm[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[prompt]],
        },
    }


def _gr00t_action_to_env(action: dict[str, np.ndarray], action_type: str) -> list[dict[str, np.ndarray]]:
    left_arm = np.asarray(action["left_arm"][0], dtype=np.float32)
    right_arm = np.asarray(action["right_arm"][0], dtype=np.float32)
    horizon = left_arm.shape[0]

    if action_type != "joint":
        raise ValueError(
            f"GR00T_N17 RoboDojo arx_x5 is trained with joint-space relative actions (action_type=joint). "
            f"Got action_type={action_type!r}."
        )

    action_list: list[dict[str, np.ndarray]] = []
    for step in range(horizon):
        left = left_arm[step]
        right = right_arm[step]
        action_list.append(
            {
                "left_arm_joint_state": left[:6].astype(np.float32),
                "left_ee_joint_state": left[6:7].astype(np.float32),
                "right_arm_joint_state": right[:6].astype(np.float32),
                "right_ee_joint_state": right[6:7].astype(np.float32),
            }
        )
    return action_list


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get("default_prompt", model_cfg.get("task_name", "Perform the robot manipulation task."))
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")

        _load_modality_config(self.env_cfg_type)
        checkpoint_dir = _resolve_checkpoint_dir(model_cfg)
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        cosmos_model = _resolve_cosmos_model(model_cfg)

        with _override_processor_cosmos_model(checkpoint_dir, cosmos_model):
            self.policy = Gr00tPolicy(
                model_path=str(checkpoint_dir),
                embodiment_tag=embodiment_tag,
                device=self.device,
                strict=True,
            )
        self.model = self.policy
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)

        self._obs_list: list[dict[str, Any]] = []
        self._latest_env_idx_list: list[int] = [0]

        print(f"[GR00T_N17] Loaded checkpoint from {checkpoint_dir}")
        print(f"[GR00T_N17] cosmos_model={cosmos_model}")
        print(f"[GR00T_N17] action_horizon={self.action_horizon}, embodiment_tag={embodiment_tag}")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._obs_list = [_encode_observation(obs, self.default_prompt) for obs in obs_list]

    def get_action(self, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")

        action_list = []
        for encoded_obs in self._obs_list:
            gr00t_action, _ = self.policy.get_action(encoded_obs, **kwargs)
            action_list.append(_gr00t_action_to_env(gr00t_action, self.action_type))
        return action_list

    def reset(self):
        self._obs_list = []
        self._latest_env_idx_list = [0]
        self.policy.reset()
