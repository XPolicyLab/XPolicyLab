from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DREAMZERO_DIR = SCRIPT_DIR / "dreamzero"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_MODEL_PATH = CHECKPOINTS_DIR / "DreamZero-AgiBot"
LEGACY_FLAT_MODEL_PATH = CHECKPOINTS_DIR
DEFAULT_TOKENIZER_PATHS = (
    CHECKPOINTS_DIR / "umt5-xxl",
    CHECKPOINTS_DIR / "Wan2.1-I2V-14B-480P" / "google" / "umt5-xxl",
)

if str(DREAMZERO_DIR) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_DIR))

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


AGIBOT_STATE_DIM = 20
AGIBOT_ACTION_DIM = 22
INFERENCE_IMAGE_SIZE = (640, 480)
AGIBOT_6DOF_ARM_INDICES = np.array([0, 1, 3, 4, 5, 6], dtype=np.int64)


def _configure_torch_dynamo_for_eval() -> None:
    """Match DreamZero's original serving defaults for autoregressive inference."""
    try:
        dynamo_cfg = torch._dynamo.config
    except Exception:
        return

    settings = {
        "cache_size_limit": int(os.environ.get("DREAMZERO_DYNAMO_CACHE_SIZE_LIMIT", "1000")),
        "recompile_limit": int(os.environ.get("DREAMZERO_DYNAMO_RECOMPILE_LIMIT", "800")),
        "accumulated_cache_size_limit": int(os.environ.get("DREAMZERO_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT", "1000")),
        "accumulated_recompile_limit": int(os.environ.get("DREAMZERO_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT", "2000")),
    }
    for key, value in settings.items():
        if hasattr(dynamo_cfg, key):
            setattr(dynamo_cfg, key, value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _pad_or_trim(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == dim:
        return values
    if values.shape[-1] > dim:
        return values[..., :dim]
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, dim - values.shape[-1])])


def _extract_action_value(action: dict[str, Any], key: str, dim: int) -> np.ndarray:
    value = action.get(key)
    if value is None:
        return np.zeros((1, dim), dtype=np.float32)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.reshape(-1, dim) if value.size % dim == 0 else value.reshape(1, -1)
    else:
        value = value.reshape(-1, value.shape[-1])
    return _pad_or_trim(value, dim)


def _ensure_dist_initialized() -> None:
    if dist.is_available() and not dist.is_initialized():
        rendezvous_file = Path(os.environ.get("DREAMZERO_DIST_INIT_FILE", f"/tmp/dreamzero_dist_{os.getpid()}"))
        rendezvous_file.parent.mkdir(parents=True, exist_ok=True)
        if rendezvous_file.exists():
            rendezvous_file.unlink()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_file}",
            world_size=1,
            rank=0,
        )


def _latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    checkpoints = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        if any((run_dir / name).exists() for name in ("config.json", "model.safetensors", "pytorch_model.bin")):
            return run_dir
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return checkpoints[-1]


def _candidate_run_dirs(checkpoints_dir: Path, run_basename: str) -> list[Path]:
    candidates: list[Path] = []

    latest_file = checkpoints_dir / f"{run_basename}.latest"
    if latest_file.is_file():
        latest_dir = Path(latest_file.read_text(encoding="utf-8").strip()).expanduser()
        if latest_dir.is_dir():
            candidates.append(latest_dir)

    preferred_dir = checkpoints_dir / run_basename
    if preferred_dir.is_dir() and preferred_dir not in candidates:
        candidates.append(preferred_dir)

    prefix = f"{run_basename}-"
    if checkpoints_dir.is_dir():
        legacy_dirs = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
        legacy_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in legacy_dirs:
            if path not in candidates:
                candidates.append(path)

    return candidates


def _resolve_model_path(model_cfg: dict[str, Any]) -> Path:
    explicit = model_cfg.get("model_path") or os.environ.get("MODEL_PATH")
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        resolved = _latest_checkpoint(explicit_path)
        return (resolved or explicit_path).resolve()

    dataset_name = model_cfg.get("dataset_name", "")
    ckpt_name = model_cfg.get("ckpt_name", "")
    env_cfg_type = model_cfg.get("env_cfg_type", "")
    expert_data_num = model_cfg.get("expert_data_num", "")
    action_type = model_cfg.get("action_type", "")
    seed = model_cfg.get("seed", "0")
    run_basename = f"{dataset_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"
    checkpoints_dir = CHECKPOINTS_DIR

    for candidate in _candidate_run_dirs(checkpoints_dir, run_basename):
        resolved = _latest_checkpoint(candidate)
        if resolved is not None:
            return resolved.resolve()

    pretrained_model_path = model_cfg.get("pretrained_model_path")
    if pretrained_model_path:
        return Path(pretrained_model_path).expanduser().resolve()

    for candidate in (DEFAULT_MODEL_PATH, LEGACY_FLAT_MODEL_PATH):
        if (candidate / "experiment_cfg" / "conf.yaml").is_file() and any(
            (candidate / name).exists()
            for name in ("config.json", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
        ):
            return candidate.resolve()

    return DEFAULT_MODEL_PATH.resolve()


def _resolve_tokenizer_path(model_cfg: dict[str, Any]) -> str | None:
    tokenizer_path = model_cfg.get("tokenizer_path") or os.environ.get("TOKENIZER_DIR")
    if tokenizer_path:
        return str(Path(tokenizer_path).expanduser().resolve())
    for default_path in DEFAULT_TOKENIZER_PATHS:
        if default_path.exists():
            return str(default_path.resolve())
    return None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.task_name = model_cfg.get("task_name", "")
        self.default_prompt = model_cfg.get("prompt") or "Do your job."
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.expected_action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        configured_action_dim = model_cfg.get("action_dim")
        if configured_action_dim is not None and int(configured_action_dim) != self.expected_action_dim:
            raise ValueError(
                f"DreamZero action_dim mismatch for env_cfg_type={self.env_cfg_type}: "
                f"deploy config has {configured_action_dim}, robot config expects {self.expected_action_dim}."
            )
        self.action_horizon = int(model_cfg.get("action_horizon", 24))
        self.video_history = int(model_cfg.get("video_history", 4))
        self.inference_method = model_cfg.get("inference_method", "lazy_joint_forward_causal")
        self.skip_img_transform = _as_bool(model_cfg.get("skip_img_transform"), False)
        self.tokenizer_path = _resolve_tokenizer_path(model_cfg)

        self.model_path = _resolve_model_path(model_cfg)
        _configure_torch_dynamo_for_eval()
        _ensure_dist_initialized()
        print(f"[DreamZero Model] Loading model from: {self.model_path}")
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag.AGIBOT,
            model_path=str(self.model_path),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            tokenizer_path_override=self.tokenizer_path,
            skip_img_transform=self.skip_img_transform,
        )

        self._obs_batch: dict[int, dict[str, Any]] = {}
        self._frame_buffers: dict[int, dict[str, list[np.ndarray]]] = {}
        self._latest_env_idx_list = [0]
        print(f"[DreamZero Model] Initialized | action_type={self.action_type}")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._obs_batch[env_idx] = self._encode_obs(obs, env_idx)

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        batch_actions = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")
            batch = Batch(obs=self._obs_batch[env_idx])
            with torch.inference_mode():
                if self.inference_method == "lazy_joint_forward_causal":
                    result, _ = self.policy.lazy_joint_forward_causal(batch)
                elif self.inference_method == "lazy_joint_forward":
                    result, _ = self.policy.lazy_joint_forward(batch)
                else:
                    result = self.policy.forward(batch)
            batch_actions.append(self._decode_actions(result.act))
        return batch_actions

    def reset(self):
        self._obs_batch = {}
        self._frame_buffers = {}
        self._latest_env_idx_list = [0]
        print("[DreamZero Model] Reset")

    def _encode_obs(self, observation: dict[str, Any], env_idx: int) -> dict[str, Any]:
        buffers = self._frame_buffers.setdefault(
            env_idx,
            {"video.top_head": [], "video.hand_left": [], "video.hand_right": []},
        )
        for camera_key, dreamzero_key in [
            ("cam_head", "video.top_head"),
            ("cam_left_wrist", "video.hand_left"),
            ("cam_right_wrist", "video.hand_right"),
        ]:
            frame = _extract_image(observation.get("vision", {}), camera_key)
            buffers[dreamzero_key].append(frame)
            buffers[dreamzero_key] = buffers[dreamzero_key][-self.video_history :]

        state = _xpolicylab_obs_to_agibot_state(observation, self.action_type, self.robot_action_dim_info)
        prompt = observation.get("instruction", observation.get("instructions", self.default_prompt))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else self.default_prompt

        return {
            "video.top_head": _stack_recent_frames(buffers["video.top_head"], self.video_history),
            "video.hand_left": _stack_recent_frames(buffers["video.hand_left"], self.video_history),
            "video.hand_right": _stack_recent_frames(buffers["video.hand_right"], self.video_history),
            "state.left_arm_joint_position": state[0:7].reshape(1, 7),
            "state.right_arm_joint_position": state[7:14].reshape(1, 7),
            "state.left_effector_position": state[14:15].reshape(1, 1),
            "state.right_effector_position": state[15:16].reshape(1, 1),
            "state.head_position": state[16:18].reshape(1, 2),
            "state.waist_pitch": state[18:19].reshape(1, 1),
            "state.waist_lift": state[19:20].reshape(1, 1),
            "annotation.language.action_text": str(prompt),
        }

    def _decode_actions(self, action: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        left_arm = _extract_action_value(action, "action.left_arm_joint_position", 7)
        right_arm = _extract_action_value(action, "action.right_arm_joint_position", 7)
        left_ee = _extract_action_value(action, "action.left_effector_position", 1)
        right_ee = _extract_action_value(action, "action.right_effector_position", 1)
        horizon = min(self.action_horizon, left_arm.shape[0], right_arm.shape[0], left_ee.shape[0], right_ee.shape[0])

        action_steps = []
        for idx in range(max(horizon, 1)):
            packed = _agibot_to_xpolicylab_packed(
                left_arm[min(idx, left_arm.shape[0] - 1)],
                right_arm[min(idx, right_arm.shape[0] - 1)],
                left_ee[min(idx, left_ee.shape[0] - 1)],
                right_ee[min(idx, right_ee.shape[0] - 1)],
                self.action_type,
                self.robot_action_dim_info,
            )
            action_steps.append(
                unpack_robot_state(
                    packed,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return action_steps


def _extract_image(vision: dict[str, Any], camera_key: str) -> np.ndarray:
    camera = vision.get(camera_key, {})
    image = camera.get("color") if isinstance(camera, dict) else camera
    if image is None:
        return np.zeros((INFERENCE_IMAGE_SIZE[1], INFERENCE_IMAGE_SIZE[0], 3), dtype=np.uint8)
    image = np.asarray(image)
    image = cv2.resize(image, INFERENCE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


def _stack_recent_frames(frames: list[np.ndarray], history: int) -> np.ndarray:
    if not frames:
        frames = [np.zeros((INFERENCE_IMAGE_SIZE[1], INFERENCE_IMAGE_SIZE[0], 3), dtype=np.uint8)]
    padded = list(frames)
    while len(padded) < history:
        padded.insert(0, padded[0])
    return np.stack(padded[-history:], axis=0)


def _xpolicylab_obs_to_agibot_state(observation: dict[str, Any], action_type: str, robot_info: dict) -> np.ndarray:
    packed = pack_robot_state(observation, action_type, robot_info, source_type="obs").astype(np.float32)
    out = np.zeros(AGIBOT_STATE_DIM, dtype=np.float32)
    offset = 0
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(robot_info["arm_dim"], robot_info["ee_dim"])):
        arm = packed[offset : offset + arm_dim]
        offset += arm_dim
        ee = packed[offset : offset + ee_dim]
        offset += ee_dim
        if arm_idx == 0:
            out[_agibot_arm_indices(arm_dim, arm_idx)] = _pad_or_trim(arm.reshape(1, -1), arm_dim)[0]
            out[14:15] = _pad_or_trim(ee.reshape(1, -1), 1)[0]
        elif arm_idx == 1:
            out[_agibot_arm_indices(arm_dim, arm_idx)] = _pad_or_trim(arm.reshape(1, -1), arm_dim)[0]
            out[15:16] = _pad_or_trim(ee.reshape(1, -1), 1)[0]
    return out


def _agibot_arm_indices(arm_dim: int, arm_idx: int) -> np.ndarray:
    base = 0 if arm_idx == 0 else 7
    if arm_dim == 6:
        return base + AGIBOT_6DOF_ARM_INDICES
    return base + np.arange(min(arm_dim, 7), dtype=np.int64)


def _agibot_arm_to_robot_arm(agibot_arm: np.ndarray, arm_dim: int) -> np.ndarray:
    values = np.asarray(agibot_arm, dtype=np.float32).reshape(-1)
    if arm_dim == 6 and values.shape[0] >= 7:
        return values[AGIBOT_6DOF_ARM_INDICES]
    return _pad_or_trim(values.reshape(1, -1), arm_dim)[0]


def _agibot_to_xpolicylab_packed(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    action_type: str,
    robot_info: dict,
) -> np.ndarray:
    parts = []
    arms = [left_arm, right_arm]
    ees = [left_ee, right_ee]
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(robot_info["arm_dim"], robot_info["ee_dim"])):
        arm_source = arms[min(arm_idx, 1)]
        ee_source = ees[min(arm_idx, 1)]
        parts.append(_agibot_arm_to_robot_arm(arm_source, arm_dim))
        parts.append(_pad_or_trim(np.asarray(ee_source).reshape(1, -1), ee_dim)[0])
    return np.concatenate(parts).astype(np.float32)
