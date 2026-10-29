"""XPolicyLab adapter for EWAM (dual 6+1 arm joint control: RoboTwin aloha_agilex, RoboDojo arx_x5).

Preprocessing mirrors EWAM's RoboTwin dim16 training pipeline
(``EWAM/data/robotwin2/robotwin_data_convert/robotwin_converter.py`` +
``robotwin_agilex_dataset_dim16.py``) and the official RoboTwin deploy wrapper
(``inference/robotwin/deploy_policy.py`` upstream). RoboDojo checkpoints use the same
model config but the stage-2 RoboDojo loader: no scene prefix, each camera resized once
onto the canvas, and 25 Hz data with global_downsample_rate=3, so every model action
spans 3 env steps (see ``_BENCH_DEFAULTS``):

    image   T-shape layout: cam_head on top, cam_left_wrist | cam_right_wrist below.
            robotwin: wrists resized to half the head size, stitched, then aspect-
            preserving resize + black padding to (video_height, video_width).
            robodojo: each camera resized once onto the (video_height, video_width)
            canvas, block centered vertically. RGB in [0, 1]
    text    scene prefix + task instruction (prefix must match training metas)
    state   [L arm 6, L grip 1, R arm 6, R grip 1] -> unified 16-dim
            [L arm 6, pad, L grip, R arm 6, pad, R grip], raw (no normalization)
    action  16-dim chunk -> keep dims [0..5, 7, 8..13, 15] -> linear interpolation
            from the current state to action_stride env steps per model action
            -> XPolicyLab joint dicts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
# Vendored EWAM inference subset (models/, wan/, utils/, data/utils/, configs/).
VENDORED_EWAM_ROOT = POLICY_DIR / "EWAM"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
CHECKPOINT_FILE = "mp_rank_00_model_states.pt"

# Must match robotwin_converter.py meta_prefix used to build the training metas.
ROBOTWIN_SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

# Per-benchmark defaults; each must match how that benchmark's released checkpoint was trained.
_BENCH_DEFAULTS = {
    "RoboTwin": {
        "scene_prefix": ROBOTWIN_SCENE_PREFIX, "image_layout": "robotwin", "action_stride": 1, "clip_gripper": False,
    },
    "RoboDojo": {"scene_prefix": "", "image_layout": "robodojo", "action_stride": 3, "clip_gripper": False},
}
_IMAGE_LAYOUTS = ("robotwin", "robodojo")

_EXPECTED_ARM_DIMS = [6, 6]
_EXPECTED_EE_DIMS = [1, 1]
_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_VALID_ACTION_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _is_unset(value: Any) -> bool:
    """Like _is_none_like, but an empty string is an explicit value."""
    return value is None or (isinstance(value, str) and value.strip().lower() in {"none", "null"})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_int(value: Any) -> int | None:
    return None if _is_none_like(value) else int(value)


def _resolve_path(value: Any, default: Path, name: str) -> Path:
    path = default if _is_none_like(value) else Path(str(value)).expanduser()
    if not path.is_absolute():
        path = POLICY_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _activate_ewam_root(ewam_root: Any) -> Path:
    """Put the EWAM source root first on sys.path.

    EWAM imports its code as top-level ``models`` / ``wan`` / ``utils`` / ``data``,
    so the configured tree must win over any same-named module on sys.path.
    """
    root = _resolve_path(ewam_root, VENDORED_EWAM_ROOT, "ewam_root")
    if not (root / "models" / "ewam.py").is_file():
        raise FileNotFoundError(f"models/ewam.py not found under ewam_root: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    for name in ("models", "wan", "utils", "data"):
        module = sys.modules.get(name)
        loaded = getattr(module, "__file__", None) if module is not None else None
        if loaded and not str(Path(loaded).resolve()).startswith(str(root.resolve())):
            raise RuntimeError(
                f"module '{name}' was already imported from {loaded}, not from ewam_root {root}; "
                "start the server in a fresh process."
            )
    return root


def _resolve_checkpoint_file(model_cfg: dict) -> Path:
    """Resolve the EWAM checkpoint via the shared XPolicyLab rules.

    The resolved root may be the ``mp_rank_00_model_states.pt`` file itself, a
    directory holding it, or a DeepSpeed save dir (``<dir>/pytorch_model/``).
    """
    root = resolve_checkpoint_root(model_cfg, CHECKPOINTS_DIR, policy_dir=POLICY_DIR, must_exist=True)
    if root.is_file():
        return root
    for candidate in (root / CHECKPOINT_FILE, root / "pytorch_model" / CHECKPOINT_FILE):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {CHECKPOINT_FILE} under checkpoint root: {root}")


def _instruction(obs: dict, fallback: str) -> str:
    value = obs.get("instruction")
    for key in ("instructions", "task_instruction"):
        if value is None:
            value = obs.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else (value.reshape(-1)[0] if value.size else None)
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    text = "" if value is None else str(value).strip()
    return text or fallback


def _color(obs: dict, camera: str) -> np.ndarray:
    try:
        image = obs["vision"][camera]["color"]
    except KeyError as exc:
        raise KeyError(f"observation is missing vision.{camera}.color") from exc
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"vision.{camera}.color must be an HxWx3 RGB array, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"vision.{camera}.color must be uint8, got {image.dtype}")
    return image


def _t_layout(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Head on top; wrists resized to half the head size side by side below."""
    head_h, head_w = head.shape[:2]
    half_h, half_w = head_h // 2, head_w // 2
    left = cv2.resize(left, (half_w, half_h))
    right = cv2.resize(right, (head_w - half_w, half_h))
    return np.vstack([head, np.hstack([left, right])])


def _aspect_t_layout(
    head: np.ndarray, left: np.ndarray, right: np.ndarray, video_size: tuple[int, int], resize_with_padding
) -> np.ndarray:
    """RoboDojo stage-2 stitch: each camera resized once onto a (H, W) canvas, block centered vertically."""
    th, tw = video_size
    split_w = tw // 2
    right_w = tw - split_w

    def natural_h(image: np.ndarray, target_w: int) -> int:
        h, w = image.shape[:2]
        return max(1, int(round(h * target_w / w)))

    bot_h = min(max(natural_h(left, split_w), natural_h(right, right_w)), th - 1)
    top_h = min(natural_h(head, tw), th - bot_h)
    y0 = (th - top_h - bot_h) // 2
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[y0 : y0 + top_h] = resize_with_padding(head, (top_h, tw))
    yb = y0 + top_h
    canvas[yb : yb + bot_h, :split_w] = resize_with_padding(left, (bot_h, split_w))
    canvas[yb : yb + bot_h, split_w:] = resize_with_padding(right, (bot_h, right_w))
    return canvas


def _upsample_chunk(state14: np.ndarray, actions14: np.ndarray, stride: int) -> np.ndarray:
    """Model actions are `stride` env steps apart; interpolate linearly from the current state."""
    if stride <= 1:
        return actions14
    anchors = np.concatenate([state14[None], actions14], axis=0)
    alphas = np.arange(1, stride + 1, dtype=np.float32)[:, None] / stride
    segments = [(1 - alphas) * anchors[k] + alphas * anchors[k + 1] for k in range(len(actions14))]
    return np.concatenate(segments, axis=0)


def _joint_state(obs: dict) -> np.ndarray:
    state = obs.get("state", {})
    parts = []
    for key, dim in (
        ("left_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_arm_joint_state", 6),
        ("right_ee_joint_state", 1),
    ):
        if key not in state:
            raise KeyError(f"observation is missing state.{key}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (dim,):
            raise ValueError(f"state.{key} must have shape ({dim},), got {value.shape}")
        parts.append(value)
    return np.concatenate(parts)


def _unify_state_to_16(state14: np.ndarray) -> np.ndarray:
    pad = np.zeros(1, dtype=np.float32)
    return np.concatenate([state14[:6], pad, state14[6:7], state14[7:13], pad, state14[13:14]])


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise ValueError(f"EWAM only supports action_type=joint, got {self.action_type!r}")

        dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if list(dim_info["arm_dim"]) != _EXPECTED_ARM_DIMS or list(dim_info["ee_dim"]) != _EXPECTED_EE_DIMS:
            raise ValueError(
                f"EWAM checkpoints expect a dual 6+1 arm robot, got {dim_info} for {self.env_cfg_type}"
            )

        bench = str(model_cfg.get("bench_name") or "")
        prefix, clip = model_cfg.get("scene_prefix"), model_cfg.get("clip_gripper")
        if bench not in _BENCH_DEFAULTS and (_is_unset(prefix) or _is_unset(clip)):
            raise ValueError(
                f"no EWAM defaults for bench_name={bench!r} (known: {list(_BENCH_DEFAULTS)}); "
                "set scene_prefix and clip_gripper explicitly in deploy.yml"
            )
        defaults = _BENCH_DEFAULTS.get(bench, {})
        self.scene_prefix = defaults["scene_prefix"] if _is_unset(prefix) else str(prefix)
        self.clip_gripper = defaults["clip_gripper"] if _is_unset(clip) else _as_bool(clip)
        layout = model_cfg.get("image_layout")
        self.image_layout = defaults.get("image_layout", "robotwin") if _is_none_like(layout) else str(layout)
        if self.image_layout not in _IMAGE_LAYOUTS:
            raise ValueError(f"image_layout must be one of {_IMAGE_LAYOUTS}, got {self.image_layout!r}")
        stride = _optional_int(model_cfg.get("action_stride"))
        self.action_stride = defaults.get("action_stride", 1) if stride is None else stride
        if self.action_stride < 1:
            raise ValueError(f"action_stride must be >= 1, got {self.action_stride}")
        self.default_instruction = str(model_cfg.get("default_instruction") or "follow the instruction")
        self.replan_steps = _optional_int(model_cfg.get("replan_steps"))
        self.max_batch_size = _optional_int(model_cfg.get("max_batch_size"))
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        self.allow_dummy_policy = _as_bool(model_cfg.get("allow_dummy_policy"))

        ewam_root = _activate_ewam_root(model_cfg.get("ewam_root"))
        config_path = _resolve_path(
            model_cfg.get("ewam_config"), ewam_root / "configs" / "ewam_robotwin.yaml", "ewam_config"
        )
        import yaml

        with open(config_path, "r") as f:
            self.ewam_cfg = yaml.safe_load(f)
        common = self.ewam_cfg["common"]
        self.video_size = (int(common["video_height"]), int(common["video_width"]))
        self.chunk_size = int(common["num_video_frames"]) * int(common["video_action_freq_ratio"])
        steps = _optional_int(model_cfg.get("num_inference_steps"))
        self.num_inference_steps = steps or int(self.ewam_cfg["model"]["inference"]["num_inference_timesteps"])

        self._obs = None
        self._obs_batch: dict[int, dict] = {}

        from data.utils.image_utils import resize_with_padding

        self._resize_with_padding = resize_with_padding

        if self.allow_dummy_policy:
            print("\033[33m[EWAM] allow_dummy_policy=true: no weights loaded, returning hold-position chunks\033[0m")
            return

        self._load(model_cfg)

    def _load(self, model_cfg: dict) -> None:
        import torch
        from transformers import AutoProcessor

        from data.utils.image_utils import tensor_to_pil
        from models.ewam import Ewam, EwamConfig
        from utils.vlm_utils import preprocess_vlm_messages
        from wan.modules.t5 import T5EncoderModel

        self._torch = torch
        self._tensor_to_pil = tensor_to_pil
        self._preprocess_vlm_messages = preprocess_vlm_messages

        self.device = str(model_cfg.get("device") or "cuda")
        checkpoint_file = _resolve_checkpoint_file(model_cfg)
        wan_path = _resolve_path(model_cfg.get("wan_path"), CHECKPOINTS_DIR / "Wan2.2-TI2V-5B", "wan_path")
        vlm_path = _resolve_path(model_cfg.get("vlm_path"), CHECKPOINTS_DIR / "Qwen3-VL-2B-Instruct", "vlm_path")
        print(f"[EWAM] checkpoint={checkpoint_file}\n[EWAM] wan_path={wan_path}\n[EWAM] vlm_path={vlm_path}")

        common = self.ewam_cfg["common"]
        mcfg = self.ewam_cfg["model"]
        config = EwamConfig(
            wan_checkpoint_path=str(wan_path),
            vae_path=str(wan_path / "Wan2.2_VAE.pth"),
            wan_config_path=str(wan_path),
            video_precision=mcfg["wan"].get("precision", "bfloat16"),
            vlm_checkpoint_path=str(vlm_path),
            vlm_frozen=mcfg["vlm"].get("frozen", False),
            vlm_dim=mcfg["qwen3_expert"]["vlm_dim"],
            qwen3_expert_head_dim=mcfg["qwen3_expert"]["head_dim"],
            qwen3_expert_num_heads=mcfg["qwen3_expert"]["num_heads"],
            qwen3_expert_num_layers=mcfg["qwen3_expert"]["num_layers"],
            # PyYAML reads "1e-5" (no decimal point) as a string.
            qwen3_expert_norm_eps=float(mcfg["qwen3_expert"]["norm_eps"]),
            action_state_dim=common["state_dim"],
            action_dim=common["action_dim"],
            action_expert_dim=mcfg["action_expert"]["hidden_size"],
            action_expert_ffn_dim_multiplier=mcfg["action_expert"]["ffn_dim_multiplier"],
            action_expert_norm_eps=float(mcfg["action_expert"]["norm_eps"]),
            global_downsample_rate=common["global_downsample_rate"],
            video_action_freq_ratio=common["video_action_freq_ratio"],
            num_video_frames=common["num_video_frames"],
            video_height=common["video_height"],
            video_width=common["video_width"],
            batch_size=1,
            training_mode=mcfg.get("training_mode", "finetune"),
            load_pretrained_backbones=False,
        )
        model = Ewam(config).to(self.device)
        model.load_checkpoint(str(checkpoint_file), strict=False)
        model.dtype = torch.bfloat16
        self.model = model.eval()

        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=str(wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(wan_path / "google" / "umt5-xxl"),
        )
        self.vlm_processor = AutoProcessor.from_pretrained(str(vlm_path), trust_remote_code=True)
        print(f"[EWAM] model loaded, chunk_size={self.chunk_size}, steps={self.num_inference_steps}")

    def _prepare(self, obs: dict) -> tuple[np.ndarray, np.ndarray, str]:
        state14 = _joint_state(obs)
        head, left, right = (_color(obs, camera) for camera in _CAMERAS)
        if self.image_layout == "robodojo":
            image = _aspect_t_layout(head, left, right, self.video_size, self._resize_with_padding)
        else:
            image = _t_layout(head, left, right)
            if image.shape[:2] != self.video_size:
                image = self._resize_with_padding(image, self.video_size)
        instruction = f"{self.scene_prefix}{_instruction(obs, self.default_instruction)}"
        return state14, image, instruction

    def _infer(self, obs: dict) -> list[dict]:
        return self._infer_batch([obs])[0]

    def _infer_batch(self, obs_list: list[dict]) -> list[list[dict]]:
        prepared = [self._prepare(obs) for obs in obs_list]
        if self.allow_dummy_policy:
            chunks = [np.repeat(state14[None], self.chunk_size, axis=0) for state14, _, _ in prepared]
        else:
            chunks = self._predict_batch(prepared)
        return [self._to_env_actions(state14, actions14) for (state14, _, _), actions14 in zip(prepared, chunks)]

    def _to_env_actions(self, state14: np.ndarray, actions14: np.ndarray) -> list[dict]:
        if self.replan_steps is not None:
            actions14 = actions14[: max(1, self.replan_steps)]
        if self.clip_gripper:
            actions14 = actions14.copy()
            actions14[:, [6, 13]] = np.clip(actions14[:, [6, 13]], 0.0, 1.0)
        actions14 = _upsample_chunk(state14, actions14, self.action_stride)
        return [
            {
                "left_arm_joint_state": step[0:6].astype(np.float32),
                "left_ee_joint_state": step[6:7].astype(np.float32),
                "right_arm_joint_state": step[7:13].astype(np.float32),
                "right_ee_joint_state": step[13:14].astype(np.float32),
            }
            for step in actions14
        ]

    def _predict_batch(self, prepared: list[tuple[np.ndarray, np.ndarray, str]]) -> list[np.ndarray]:
        """One forward per group of envs whose VLM prompts have the same token length.

        The joint attention has no padding mask for VLM tokens, so only equal-length
        prompts may share a forward; a batch of one is exactly the single-env path.
        """
        torch = self._torch
        samples = []
        for state14, image, instruction in prepared:
            frame = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)
            vlm_inputs = self._preprocess_vlm_messages(instruction, self._tensor_to_pil(frame), self.vlm_processor)
            vlm_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in vlm_inputs.items()}
            samples.append((frame, _unify_state_to_16(state14), instruction, vlm_inputs))

        groups: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            groups.setdefault(int(sample[3]["input_ids"].shape[1]), []).append(i)
        cap = self.max_batch_size or len(samples)
        chunks: list[np.ndarray | None] = [None] * len(samples)
        for indices in groups.values():
            for start in range(0, len(indices), cap):
                part = indices[start : start + cap]
                for i, actions14 in zip(part, self._forward([samples[i] for i in part])):
                    chunks[i] = actions14
        return chunks

    def _forward(self, samples: list[tuple]) -> list[np.ndarray]:
        torch = self._torch
        first_frame = torch.stack([s[0] for s in samples]).to(self.device)
        state = torch.from_numpy(np.stack([s[1] for s in samples])).float().to(self.device)
        t5_list = list(self.t5_encoder([s[2] for s in samples], self.device))
        # Ewam precomputes the per-sample video grid for config.batch_size rows; RoPE and
        # unpatchify only cover those rows, so it must match the forward's batch size.
        video_module = self.model.video_module
        if video_module.grid_sizes.shape[0] != len(samples):
            video_module.grid_sizes = video_module.grid_sizes[:1].expand(len(samples), -1)

        with torch.no_grad():
            _, actions = self.model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=self.num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[s[3] for s in samples],
                skip_decode=True,
            )
        actions16 = actions.float().cpu().numpy()
        return [a[..., _VALID_ACTION_DIMS] for a in actions16]

    def update_obs(self, obs):
        self._obs = obs

    def update_obs_batch(self, obs_list):
        for i, obs in enumerate(obs_list):
            self._obs_batch[int(obs.get("env_idx", i))] = obs

    def get_action(self):
        if self._obs is None:
            raise RuntimeError("get_action called before update_obs")
        return self._infer(self._obs)

    def get_action_batch(self, env_idx_list=None):
        env_ids = list(self._obs_batch) if env_idx_list is None else [int(i) for i in env_idx_list]
        missing = [i for i in env_ids if i not in self._obs_batch]
        if missing:
            raise RuntimeError(f"get_action_batch called before update_obs_batch for envs {missing}")
        return self._infer_batch([self._obs_batch[i] for i in env_ids])

    def reset(self):
        self._obs = None
        self._obs_batch.clear()
