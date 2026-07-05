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
DEFAULT_IMAGE_SIZE = (640, 480)
DEFAULT_CAMERA_GROUPS = (
    ("cam_head", "cam_high"),
    ("cam_left_wrist", "cam_hand_left"),
    ("cam_right_wrist", "cam_hand_right"),
)


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


def _parse_image_resize(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"none", "null"}:
            return None
        value = [int(x) for x in value.replace(" ", "").split(",") if x]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"image_resize must be null or [width, height], got {value!r}")


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

    bench_name = model_cfg.get("bench_name", "")
    ckpt_name = model_cfg.get("ckpt_name", "")
    env_cfg_type = model_cfg.get("env_cfg_type", "")
    expert_data_num = model_cfg.get("expert_data_num", "")
    action_type = model_cfg.get("action_type", "")
    seed = model_cfg.get("seed", "0")
    run_basename = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"
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
        self.image_resize = _parse_image_resize(model_cfg.get("image_resize"))
        self.inference_method = model_cfg.get("inference_method", "lazy_joint_forward_causal")
        self.skip_img_transform = _as_bool(model_cfg.get("skip_img_transform"), False)
        self.tokenizer_path = _resolve_tokenizer_path(model_cfg)
        self.native_dojo_action = _as_bool(model_cfg.get("native_dojo_action"), False)

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
        self.native_dojo_action = self.native_dojo_action or _policy_uses_native_dojo_action(self.policy)

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
        for camera_keys, dreamzero_key in zip(
            DEFAULT_CAMERA_GROUPS,
            ("video.top_head", "video.hand_left", "video.hand_right"),
        ):
            frame = _extract_image(observation.get("vision", {}), camera_keys, self.image_resize)
            buffers[dreamzero_key].append(frame)
            buffers[dreamzero_key] = buffers[dreamzero_key][-self.video_history :]

        packed_state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        prompt = observation.get("instruction", observation.get("instructions", self.default_prompt))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else self.default_prompt

        if self.native_dojo_action:
            state_fields = _xpolicylab_packed_to_native_dojo_fields(packed_state, self.robot_action_dim_info)
        else:
            state = _xpolicylab_packed_to_agibot_state(packed_state, self.robot_action_dim_info)
            state_fields = {
                "state.left_arm_joint_position": state[0:7].reshape(1, 7),
                "state.right_arm_joint_position": state[7:14].reshape(1, 7),
                "state.left_effector_position": state[14:15].reshape(1, 1),
                "state.right_effector_position": state[15:16].reshape(1, 1),
                "state.head_position": state[16:18].reshape(1, 2),
                "state.waist_pitch": state[18:19].reshape(1, 1),
                "state.waist_lift": state[19:20].reshape(1, 1),
            }

        encoded = {
            "video.top_head": _stack_recent_frames(buffers["video.top_head"], self.video_history),
            "video.hand_left": _stack_recent_frames(buffers["video.hand_left"], self.video_history),
            "video.hand_right": _stack_recent_frames(buffers["video.hand_right"], self.video_history),
            "annotation.language.action_text": str(prompt),
        }
        encoded.update(state_fields)
        return encoded

    def _decode_actions(self, action: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        left_arm_dim = int(self.robot_action_dim_info["arm_dim"][0])
        right_arm_dim = int(self.robot_action_dim_info["arm_dim"][1])
        left_ee_dim = int(self.robot_action_dim_info["ee_dim"][0])
        right_ee_dim = int(self.robot_action_dim_info["ee_dim"][1])
        if self.native_dojo_action:
            left_arm = _extract_action_value(action, "action.left_arm_joint_position", left_arm_dim)
            right_arm = _extract_action_value(action, "action.right_arm_joint_position", right_arm_dim)
            left_ee = _extract_action_value(action, "action.left_effector_position", left_ee_dim)
            right_ee = _extract_action_value(action, "action.right_effector_position", right_ee_dim)
        else:
            left_arm = _extract_action_value(action, "action.left_arm_joint_position", 7)
            right_arm = _extract_action_value(action, "action.right_arm_joint_position", 7)
            left_ee = _extract_action_value(action, "action.left_effector_position", 1)
            right_ee = _extract_action_value(action, "action.right_effector_position", 1)
        horizon = min(self.action_horizon, left_arm.shape[0], right_arm.shape[0], left_ee.shape[0], right_ee.shape[0])

        action_steps = []
        for idx in range(max(horizon, 1)):
            if self.native_dojo_action:
                packed = _native_dojo_fields_to_xpolicylab_packed(
                    left_arm[min(idx, left_arm.shape[0] - 1)],
                    left_ee[min(idx, left_ee.shape[0] - 1)],
                    right_arm[min(idx, right_arm.shape[0] - 1)],
                    right_ee[min(idx, right_ee.shape[0] - 1)],
                )
            else:
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


def _extract_image(
    vision: dict[str, Any],
    camera_keys: str | tuple[str, ...],
    image_resize: tuple[int, int] | None,
) -> np.ndarray:
    if isinstance(camera_keys, str):
        camera_keys = (camera_keys,)
    image = None
    for camera_key in camera_keys:
        camera = vision.get(camera_key)
        if camera is None:
            continue
        if isinstance(camera, dict):
            image = camera.get("color")
            if image is None:
                image = camera.get("rgb")
            if image is None:
                image = camera.get("colors")
        else:
            image = camera
        if image is not None:
            break
    if image is None:
        width, height = image_resize or DEFAULT_IMAGE_SIZE
        return np.zeros((height, width, 3), dtype=np.uint8)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        width, height = image_resize or DEFAULT_IMAGE_SIZE
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image_resize is not None:
        image = cv2.resize(image, image_resize, interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


def _stack_recent_frames(frames: list[np.ndarray], history: int) -> np.ndarray:
    if not frames:
        frames = [np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), dtype=np.uint8)]
    padded = list(frames)
    while len(padded) < history:
        padded.insert(0, padded[0])
    return np.stack(padded[-history:], axis=0)


# AgiBot G1 7-DOF arm structure (per-arm slot indices):
#   0 = J1 (shoulder), 1 = J2 (shoulder), 2 = J3 (upper-arm roll),
#   3 = J4 (elbow),    4 = J5 (wrist),    5 = J6 (wrist),     6 = J7 (wrist)
# arx_x5 / dual_x5 / similar 6-DOF arms map onto AgiBot's [J1, J2, J4, J5, J6, J7]
# (per-arm slots [0, 1, 3, 4, 5, 6]); J3 (slot 2) is the redundant upper-arm roll DOF
# absent on these arms and stays = 0.
_AGIBOT_J3_LOCKED_PER_ARM_INDEX = 2
_ARX_X5_ARM_TO_AGIBOT_ARM_SLOTS = [
    s for s in range(7) if s != _AGIBOT_J3_LOCKED_PER_ARM_INDEX
]  # [0, 1, 3, 4, 5, 6]


def _agibot_arm_slot_targets(arm_dim: int) -> list[int]:
    """Per-arm AgiBot slot indices for a source `arm_dim`-DOF arm.
    For 6-DOF arms, returns [0,1,3,4,5,6] (lock AgiBot J3 = slot 2).
    For 7-DOF arms (AgiBot native), returns [0..6] (identity).
    For other dims, falls back to the first arm_dim slots."""
    if arm_dim == 6:
        return _ARX_X5_ARM_TO_AGIBOT_ARM_SLOTS
    if arm_dim == 7:
        return list(range(7))
    # Fallback: dense fill from slot 0
    return list(range(min(arm_dim, 7)))


def _xpolicylab_packed_to_agibot_state(packed: np.ndarray, robot_info: dict) -> np.ndarray:
    out = np.zeros(AGIBOT_STATE_DIM, dtype=np.float32)
    offset = 0
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(robot_info["arm_dim"], robot_info["ee_dim"])):
        arm = packed[offset : offset + arm_dim]
        offset += arm_dim
        ee = packed[offset : offset + ee_dim]
        offset += ee_dim
        # Route 6-DOF arm joints into AgiBot 7-DOF arm slots, locking J3 (slot 2)
        arm_slot_offset = arm_idx * 7  # left = 0, right = 7
        for src_i, tgt_slot in enumerate(_agibot_arm_slot_targets(arm_dim)):
            out[arm_slot_offset + tgt_slot] = arm[src_i]
        # Effector slot
        ee_slot = 14 + arm_idx  # left = 14, right = 15
        out[ee_slot : ee_slot + ee_dim] = _pad_or_trim(ee.reshape(1, -1), ee_dim)[0]
    return out


def _xpolicylab_packed_to_native_dojo_fields(packed: np.ndarray, robot_info: dict) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    offset = 0
    for arm_idx, side in enumerate(("left", "right")):
        arm_dim = int(robot_info["arm_dim"][arm_idx])
        ee_dim = int(robot_info["ee_dim"][arm_idx])
        arm = packed[offset : offset + arm_dim]
        offset += arm_dim
        ee = packed[offset : offset + ee_dim]
        offset += ee_dim
        fields[f"state.{side}_arm_joint_position"] = arm.reshape(1, arm_dim)
        fields[f"state.{side}_effector_position"] = ee.reshape(1, ee_dim)
    return fields


def _native_dojo_fields_to_xpolicylab_packed(
    left_arm: np.ndarray,
    left_ee: np.ndarray,
    right_arm: np.ndarray,
    right_ee: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(left_arm).reshape(-1),
            np.asarray(left_ee).reshape(-1),
            np.asarray(right_arm).reshape(-1),
            np.asarray(right_ee).reshape(-1),
        ]
    ).astype(np.float32)


def _agibot_to_xpolicylab_packed(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    action_type: str,
    robot_info: dict,
) -> np.ndarray:
    """Inverse of _xpolicylab_packed_to_agibot_state: decode AgiBot per-arm 7-dim
    vectors back into the source robot's packed layout, dropping the locked J3
    slot for 6-DOF arms."""
    parts = []
    arms = [np.asarray(left_arm).reshape(-1), np.asarray(right_arm).reshape(-1)]
    ees = [np.asarray(left_ee).reshape(-1), np.asarray(right_ee).reshape(-1)]
    for arm_idx, (arm_dim, ee_dim) in enumerate(zip(robot_info["arm_dim"], robot_info["ee_dim"])):
        agibot_arm = arms[min(arm_idx, 1)]
        agibot_ee = ees[min(arm_idx, 1)]
        # Extract source-arm-dim joints from AgiBot 7-slot per-arm vector by index
        src_slots = _agibot_arm_slot_targets(arm_dim)
        arm_pick = np.array(
            [agibot_arm[s] if s < agibot_arm.shape[0] else 0.0 for s in src_slots],
            dtype=np.float32,
        )
        parts.append(arm_pick)
        parts.append(_pad_or_trim(agibot_ee.reshape(1, -1), ee_dim)[0])
    return np.concatenate(parts).astype(np.float32)


def _policy_uses_native_dojo_action(policy: Any) -> bool:
    try:
        action_head_cfg = policy.train_cfg.get("action_head_cfg", {})
        config = action_head_cfg.get("config", action_head_cfg)
        return _as_bool(config.get("native_dojo_action"), False)
    except Exception:
        return False
