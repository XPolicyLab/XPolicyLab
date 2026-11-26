from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

_CUR_DIR = Path(__file__).resolve().parent
# RoboDojo root — needed for `from XPolicyLab.* import ...`.
_XPOLICYLAB_PARENT = _CUR_DIR.parents[2]
_INTERNVLA_ROOT = _CUR_DIR / "internvla_a1_5"
# Keep the policy self-contained like InternVLA_A1 while retaining an override
# for local development against another InternVLA-A1.5 checkout.
_LEROBOT_SRC = Path(
    os.environ.get("LEROBOT_SRC_PATH", str(_INTERNVLA_ROOT / "src"))
).resolve()
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_XPOLICYLAB_PARENT), str(_LEROBOT_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.batch_indices import observation_env_ids, selected_env_ids
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.datasets.utils import load_json
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


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

    image = np.asarray(image)

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
            "cam_high": ensure_hwc_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_hwc_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_hwc_uint8(observation["images"]["cam_right_wrist"]),
        }
        state = np.asarray(observation["state"], dtype=np.float32)
        prompt = resolve_prompt(observation, default_prompt)
        return {"images": images, "state": state, "prompt": prompt}

    images = {
        "cam_high": ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = resolve_prompt(observation, default_prompt)
    return {"images": images, "state": state, "prompt": prompt}


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _resolve_internvla_ckpt_dir(model_cfg: dict[str, Any]) -> Path:
    ckpt_path = model_cfg.get("ckpt_path")
    if ckpt_path:
        return resolve_ckpt_dir(ckpt_path)

    checkpoint_root = resolve_checkpoint_root(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_CUR_DIR,
        explicit_keys=("model_dir", "model_path"),
        must_exist=False,
    ).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    candidate_dirs = []
    if (checkpoint_root / "pretrained_model").is_dir():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and (child / "pretrained_model").is_dir()
    )
    if not candidate_dirs:
        raise FileNotFoundError(f"No pretrained_model checkpoint found under {checkpoint_root}")

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
                return (candidate / "pretrained_model").resolve()

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        selected = max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
        return (selected / "pretrained_model").resolve()
    return (candidate_dirs[0] / "pretrained_model").resolve()


def resolve_ckpt_dir(ckpt_path):
    ckpt = Path(str(ckpt_path)).expanduser()
    if not ckpt.is_absolute():
        ckpt = (_CUR_DIR / ckpt).resolve()
    if ckpt.exists():
        return ckpt.resolve()
    return Path(snapshot_download(repo_id=str(ckpt_path)))


def invert_action_reorder(
    reordered: torch.Tensor,
    reorder_spec: list[list[int]] | None,
    native_dim: int,
) -> torch.Tensor:
    """Invert a ReorderStateActionTransform mapping back to the native layout.

    The forward spec is [[src_start, src_end, dst_start, dst_end], ...], meaning
    native[src_start:src_end] -> reordered[dst_start:dst_end]. To invert we copy
    reordered[dst_start:dst_end] back into native[src_start:src_end].
    """
    if reordered.shape[-1] < native_dim and reorder_spec is None:
        return reordered

    if reorder_spec is None:
        return reordered[..., :native_dim]

    out = torch.zeros(
        *reordered.shape[:-1],
        native_dim,
        dtype=reordered.dtype,
        device=reordered.device,
    )
    for spec in reorder_spec:
        src_start, src_end, dst_start, dst_end = spec
        length = min(src_end - src_start, dst_end - dst_start)
        out[..., src_start : src_start + length] = reordered[..., dst_start : dst_start + length]
    return out


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("InternVLA-A1.5 in XPolicyLab currently supports only action_type='joint'.")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = self._load_robot_action_dim_info(env_cfg)
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.float32 if self.model_cfg.get("dtype", "float32") == "float32" else torch.bfloat16
        if self.device.type != "cuda":
            self.dtype = torch.float32

        self.stats_key = self.model_cfg.get("stats_key", "aloha")
        self.resize_size = int(self.model_cfg.get("resize_size", 224))
        self.infer_horizon = int(self.model_cfg.get("infer_horizon", 20))
        self.action_horizon_size = int(self.model_cfg.get("action_horizon_size", 50))
        self.action_mode = self.model_cfg.get("action_mode", "delta")
        self.inference_backend = self.model_cfg.get("inference_backend", "standard")
        self.tokenize_state = bool(self.model_cfg.get("tokenize_state", True))
        self.max_state_dim = int(self.model_cfg.get("max_state_dim", 32))
        self.max_action_dim = int(self.model_cfg.get("max_action_dim", 32))

        self.ckpt_dir = _resolve_internvla_ckpt_dir(self.model_cfg)
        self.policy, self.input_transforms, self.unnormalize_fn, self.schema, self.native_action_dim = (
            self._build_policy_and_transforms()
        )
        self._latest_env_idx_list = [0]
        self.reset()
        self.model = self.policy

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    def _load_robot_action_dim_info(self, env_cfg):
        # setup_eval_policy_server.sh always overrides env_cfg_type (arg 4), so
        # a missing value means the server was launched off the bare deploy.yml,
        # where the key is null. Fail loudly: silently defaulting to arx_x5
        # would mis-pack the robot state for any non-14-dim-dual-arm robot
        # without raising anywhere.
        if env_cfg is None:
            raise ValueError(
                "env_cfg_type is not set. Launch through setup_eval_policy_server.sh "
                "(which passes it as arg 4), or supply env_cfg_type via --overrides "
                "or deploy.yml."
            )
        try:
            return get_robot_action_dim_info(env_cfg)
        except (FileNotFoundError, KeyError, OSError):
            # Fallback: when env_cfg/*.yml is absent (e.g. running from the
            # lerobot_lab copy without a RoboDojo env_cfg tree), look up
            # XPolicyLab/utils/robot/_robot_info.json using env_cfg as the key,
            # matching get_action_dim.sh semantics.
            import json
            robot_info_path = _XPOLICYLAB_PARENT / "XPolicyLab" / "utils" / "robot" / "_robot_info.json"
            if robot_info_path.exists():
                info = json.loads(robot_info_path.read_text(encoding="utf-8"))
                if env_cfg in info:
                    return info[env_cfg]
            raise

    def _build_policy_and_transforms(self):
        config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not isinstance(config, InternVLAA15Config):
            raise ValueError(f"Expected InternVLAA15Config, got {type(config)}")

        config.action_loss_only = True
        config.inference_backend = self.inference_backend
        config.device = str(self.device)

        policy = InternVLAA15Policy.from_pretrained(
            pretrained_name_or_path=self.ckpt_dir, config=config
        )
        policy.to(self.device).to(self.dtype).eval()

        stats = load_json(self.ckpt_dir / "stats.json")[self.stats_key]
        stat_keys = ["min", "max", "mean", "std"]
        state_stat = {OBS_STATE: {key: np.asarray(stats[OBS_STATE][key]) for key in stat_keys}}
        action_stat = {ACTION: {key: np.asarray(stats[ACTION][key]) for key in stat_keys}}
        native_action_dim = action_stat[ACTION]["min"].shape[0]

        unnormalize_fn = UnNormalizeTransformFn(
            selected_keys=[ACTION],
            mode="mean_std",
            norm_stats=action_stat,
        )

        schema = get_schema(self.stats_key)

        processor_pretrained = getattr(config, "vlm_model_name_or_path", None) or "Qwen/Qwen3.5-2B"
        input_transforms = compose(
            [
                ResizeImagesWithPadFn(
                    height=self.resize_size,
                    width=self.resize_size,
                    mapping=schema.image_mapping,
                ),
                RemapImageKeyTransformFn(mapping=schema.image_mapping),
                NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=state_stat),
                InternVLAA15ChatProcessorTransformFn(
                    mode="eval",
                    tokenize_state=self.tokenize_state,
                    max_state_dim=self.max_state_dim,
                    pretrained_model_name_or_path=processor_pretrained,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=self.max_state_dim,
                    max_action_dim=self.max_action_dim,
                ),
                ReorderStateActionTransform(
                    state_reorder=schema.state_reorder,
                    action_reorder=schema.action_reorder,
                ),
            ]
        )
        return policy, input_transforms, unnormalize_fn, schema, native_action_dim

    def reset(self):
        self.policy.reset()
        self._latest_obs_by_env: dict[int, dict[str, Any]] = {}
        self._latest_obs = None
        self._latest_env_idx_list = [0]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_ids = observation_env_ids(obs_list)
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        self._latest_obs = encoded_obs_list[0] if encoded_obs_list else None
        self._latest_obs_by_env = dict(zip(env_ids, encoded_obs_list))
        self._latest_env_idx_list = env_ids

    def _to_image_tensor(self, image):
        return torch.as_tensor(image).contiguous().to(self.dtype) / 255.0

    @torch.inference_mode()
    def infer(self, env_idx=None):
        latest_obs = self._latest_obs if env_idx is None else self._latest_obs_by_env.get(env_idx)
        if latest_obs is None:
            raise AssertionError("update_obs must be called before get_action.")

        state = torch.from_numpy(latest_obs["state"]).float()

        sample = {
            f"{OBS_IMAGES}.cam_high": self._to_image_tensor(latest_obs["images"]["cam_high"]).permute(2, 0, 1),
            f"{OBS_IMAGES}.cam_left_wrist": self._to_image_tensor(latest_obs["images"]["cam_left_wrist"]).permute(2, 0, 1),
            f"{OBS_IMAGES}.cam_right_wrist": self._to_image_tensor(latest_obs["images"]["cam_right_wrist"]).permute(2, 0, 1),
            OBS_STATE: state,
            ACTION: torch.zeros(self.action_horizon_size, self.native_action_dim, dtype=torch.float32),
            "task": latest_obs["prompt"],
        }

        sample = self.input_transforms(sample)

        inputs = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue
            if not isinstance(value, torch.Tensor):
                inputs[key] = value
                continue
            value = value.unsqueeze(0)
            if value.dtype.is_floating_point:
                value = value.to(device=self.device, dtype=self.dtype)
            else:
                value = value.to(device=self.device)
            inputs[key] = value

        for i in range(3):
            mask_key = f"{OBS_IMAGES}.image{i}_mask"
            if mask_key not in inputs:
                inputs[mask_key] = torch.tensor([True], device=self.device)

        action_pred = self.policy.predict_action_chunk(inputs)
        if action_pred.ndim == 3:
            action_pred = action_pred[0]
        action_pred = action_pred[: self.infer_horizon, :]

        action_pred = invert_action_reorder(
            action_pred, self.schema.action_reorder, self.native_action_dim
        )
        action_pred = self.unnormalize_fn({ACTION: action_pred})[ACTION]

        if self.action_mode == "delta":
            init_action = state[: self.native_action_dim].to(action_pred)
            if self.robot_action_dim_info is not None:
                arm_dim = self.robot_action_dim_info["arm_dim"]
                ee_dim = self.robot_action_dim_info["ee_dim"]
                left_gripper_idx = arm_dim[0]
                right_gripper_idx = sum(arm_dim) + sum(ee_dim) - 1
                if 0 <= left_gripper_idx < init_action.shape[-1]:
                    init_action[..., left_gripper_idx] = 0.0
                if 0 <= right_gripper_idx < init_action.shape[-1]:
                    init_action[..., right_gripper_idx] = 0.0
            action_pred = action_pred + init_action

        return action_pred.detach().cpu().float().numpy()

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = selected_env_ids(self._latest_env_idx_list, env_idx_list)
        action_list = []
        for env_idx in env_idx_list:
            raw_actions = self.infer(env_idx)
            action_list.append(
                unpack_robot_state(
                    raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
            )
        return action_list
