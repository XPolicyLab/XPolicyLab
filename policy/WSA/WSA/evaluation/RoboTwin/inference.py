#!/usr/bin/env python

import copy
import sys
import os
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

ROOT_PATH = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT_PATH / "src"
for candidate in [str(SRC_ROOT), str(ROOT_PATH)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.factory import get_policy_class
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import (
    Qwen3_VLProcessorTransformFn as QwenA1ProcessorTransformFn,
)
from lerobot.policies.qwenaction.transform_qwenaction import (
    QwenActionProcessorTransformFn,
)
from lerobot.policies.WSA_Base.transform_wsa_base import (
    Qwen3_VLProcessorTransformFn as WSABaseProcessorTransformFn,
)
from lerobot.transforms.core import (
    NormalizeTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    RemapImageKeyTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES
from lerobot.policies.WSA_Large.core.data.lerobot.utils.normalizer import (
    SingleFieldLinearNormalizer,
    load_dataset_stats_from_json,
)
from lerobot.policies.WSA_Large.dataset_wsa_large import (
    resolve_wsa_large_concat_layout,
    resolve_wsa_large_video_size,
)
from lerobot.policies.WSA_Large.stats_adapter import ensure_wsa_large_stats_format
from lerobot.policies.WSA_Large.text_cache import build_wsa_large_prompt
from lerobot.policies.names import (
    WSA_LARGE,
    WSA_LARGE_LEGACY_ALIASES,
    canonical_policy_type,
    is_wsa_base,
    is_wsa_large,
)

# RoboTwin dependencies
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "third_party" / "RoboTwin"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "policy"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "description" / "utils"),
    ]
)

from envs import CONFIGS_PATH 
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
import image_tools


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    """
    Resolve a checkpoint path to a local directory.

    Supports:
    - Local directory path
    - HuggingFace repo id (e.g., "org/repo"), downloaded to HF cache via snapshot_download
    """
    ckpt_str = str(ckpt_path)
    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        if (local_dir / "config.json").exists():
            return local_dir.resolve()
        pretrained_dir = local_dir / "pretrained_model"
        if (pretrained_dir / "config.json").exists():
            return pretrained_dir.resolve()
        return local_dir.resolve()

    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


def resolve_optional_env(value: str | None, *env_names: str) -> str | None:
    if value is not None and str(value).strip():
        return str(value)
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value.strip():
            return env_value
    return None


def resolve_bool_env(default: bool, *env_names: str) -> bool:
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        normalized = env_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Environment variable {env_name} must be boolean-like, got {env_value!r}.")
    return default


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        if device == "cpu":
            logging.warning("Requested float16 on CPU; falling back to float32.")
            return torch.float32
        return torch.float16
    if dtype_name == "bfloat16":
        if device == "cpu":
            logging.warning("Requested bfloat16 on CPU; falling back to float32.")
            return torch.float32
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={dtype_name!r}. Expected float32, float16, or bfloat16.")


def resolve_policy_components(config: PreTrainedConfig):
    policy_cls = get_policy_class(config.type)
    if config.type in {"qwena1", "internvla_a1_3b"}:
        processor_transform_cls = QwenA1ProcessorTransformFn
    elif config.type == "qwenaction":
        processor_transform_cls = QwenActionProcessorTransformFn
    elif is_wsa_base(config.type):
        processor_transform_cls = WSABaseProcessorTransformFn
    elif is_wsa_large(config.type):
        processor_transform_cls = None
    else:
        raise ValueError(
            "RoboTwin inference currently supports qwena1/internvla_a1_3b/qwenaction/"
            f"WSA_Base/WSA_Large checkpoints, got {config.type!r}."
        )
    return policy_cls, processor_transform_cls


def apply_runtime_config_overrides(config: PreTrainedConfig, args: "InferenceArgs") -> None:
    if args.policy_type is not None:
        if canonical_policy_type(config.type) != canonical_policy_type(args.policy_type):
            raise ValueError(
                f"Checkpoint policy type is {config.type!r}, but --args.policy_type={args.policy_type!r}. "
                "Use a matching checkpoint or leave --args.policy_type unset."
            )

    if args.qwen3_vl_pretrained_path is not None and hasattr(config, "qwen3_vl_pretrained_path"):
        config.qwen3_vl_pretrained_path = args.qwen3_vl_pretrained_path
    if args.qwen3_vl_processor_path is not None and hasattr(config, "qwen3_vl_processor_path"):
        config.qwen3_vl_processor_path = args.qwen3_vl_processor_path
    if args.cosmos_tokenizer_path_or_name is not None and hasattr(config, "cosmos_tokenizer_path_or_name"):
        config.cosmos_tokenizer_path_or_name = args.cosmos_tokenizer_path_or_name
    if args.da3_model_path_or_name is not None and hasattr(config, "da3_model_path_or_name"):
        config.da3_model_path_or_name = args.da3_model_path_or_name
    if args.da3_code_root is not None and hasattr(config, "da3_code_root"):
        config.da3_code_root = args.da3_code_root

    if is_wsa_large(config.type):
        model_id = resolve_optional_env(args.wsa_large_model_id, "WAN_MODEL_ID")
        tokenizer_model_id = resolve_optional_env(args.wsa_large_tokenizer_model_id, "WAN_TOKENIZER_MODEL_ID")
        action_dit_path = resolve_optional_env(
            args.wsa_large_action_dit_pretrained_path,
            "ACTION_DIT_PRETRAINED_PATH",
        )
        future_3d_path = resolve_optional_env(
            args.wsa_large_future_3d_pretrained_path,
            "FUTURE_3D_PRETRAINED_PATH",
        )
        if model_id is not None and hasattr(config, "model_id"):
            config.model_id = model_id
        if tokenizer_model_id is not None and hasattr(config, "tokenizer_model_id"):
            config.tokenizer_model_id = tokenizer_model_id
        if action_dit_path is not None and hasattr(config, "action_dit_pretrained_path"):
            config.action_dit_pretrained_path = action_dit_path
        if future_3d_path is not None and hasattr(config, "future_3d_pretrained_path"):
            config.future_3d_pretrained_path = future_3d_path
        if hasattr(config, "load_text_encoder"):
            config.load_text_encoder = resolve_bool_env(
                args.wsa_large_load_text_encoder,
                "WSA_LARGE_LOAD_TEXT_ENCODER",
                "LOAD_TEXT_ENCODER",
            )
        if hasattr(config, "redirect_common_files"):
            config.redirect_common_files = resolve_bool_env(
                args.wsa_large_redirect_common_files,
                "WSA_LARGE_REDIRECT_COMMON_FILES",
            )
        if hasattr(config, "skip_dit_load_from_pretrain"):
            config.skip_dit_load_from_pretrain = resolve_bool_env(
                args.wsa_large_skip_dit_load_from_pretrain,
                "WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN",
            )
        if hasattr(config, "dtype"):
            config.dtype = args.dtype

    if (is_wsa_base(config.type) or is_wsa_large(config.type)) and args.disable_3d_teacher_for_eval and hasattr(config, "lambda_3d"):
        # 3D queries remain enabled so the checkpoint architecture still matches,
        # but the frozen DA3 teacher is not instantiated for action-only inference.
        config.lambda_3d = 0.0

    if args.num_inference_steps is not None and hasattr(config, "num_inference_steps"):
        config.num_inference_steps = int(args.num_inference_steps)


# Task list matching eval_robotwin.py
TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def get_embodiment_config(robot_file: str):
    """Load robot embodiment configuration from YAML file."""
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def class_decorator(task_name: str):
    """Dynamically import and instantiate task environment class."""
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(task_config: str, task_name: str):
    """Build task arguments from configuration files."""
    task_cfg_file = ROOT_PATH / "third_party" / "RoboTwin" / "task_config" / f"{task_config}.yml"
    with open(task_cfg_file, "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files found")
        return robot_file

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0])
        )
        task_args["right_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[1])
        )
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(f"Invalid embodiment type length: {len(embodiment_type)}, expected 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    return task_args


def load_train_config_or_none(ckpt_dir: Path) -> TrainPipelineConfig | None:
    try:
        return TrainPipelineConfig.from_pretrained(ckpt_dir)
    except Exception as exc:
        logging.info("Could not load train_config.json from %s: %s", ckpt_dir, exc)
        return None


def wsa_large_shape_meta(config: PreTrainedConfig) -> dict[str, list[dict[str, object]]]:
    action_dim = int(getattr(config, "action_dim", 14))
    proprio_dim = int(getattr(config, "proprio_dim", action_dim))
    return {
        "action": [{"key": "default", "raw_shape": action_dim, "shape": action_dim}],
        "state": [{"key": "default", "raw_shape": proprio_dim, "shape": proprio_dim}],
    }


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _candidate_relative_roots(path_value: str | Path, ckpt_dir: Path) -> list[Path]:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return [raw_path]

    candidates = [raw_path, ROOT_PATH / raw_path, ckpt_dir / raw_path]
    candidates.extend(parent / raw_path for parent in list(ckpt_dir.parents)[:5])
    return _dedupe_paths(candidates)


def _select_wsa_large_stats_payload(stats_payload: dict[str, Any], stats_key: str) -> dict[str, Any]:
    stats_aliases = (WSA_LARGE, "wsa_large", *WSA_LARGE_LEGACY_ALIASES)
    if any(stats_alias in stats_payload for stats_alias in stats_aliases):
        return stats_payload
    keyed_payload = stats_payload.get(stats_key)
    if isinstance(keyed_payload, dict):
        return keyed_payload
    return stats_payload


def _iter_wsa_large_stats_candidates(
    args: "InferenceArgs",
    ckpt_dir: Path,
    train_config: TrainPipelineConfig | None,
) -> Iterable[tuple[Path, str, bool]]:
    explicit_stats_path = resolve_optional_env(args.wsa_large_stats_path, "WSA_LARGE_STATS_PATH")
    if explicit_stats_path is not None:
        explicit_candidates = _candidate_relative_roots(explicit_stats_path, ckpt_dir)
        for explicit_candidate in explicit_candidates:
            if explicit_candidate.is_file():
                yield explicit_candidate, "explicit WSA_Large stats path", True
                return
        yield explicit_candidates[0], "explicit WSA_Large stats path", True
        return

    checkpoint_stats_path = ckpt_dir / "stats.json"
    if checkpoint_stats_path.is_file():
        yield checkpoint_stats_path, "checkpoint stats", False

    dataset_config = None if train_config is None else getattr(train_config, "dataset", None)
    normalization_stats_path = None if dataset_config is None else getattr(dataset_config, "normalization_stats_path", None)
    if normalization_stats_path is not None and str(normalization_stats_path).strip():
        for candidate_file in _candidate_relative_roots(normalization_stats_path, ckpt_dir):
            if candidate_file.is_file():
                yield candidate_file, "train_config dataset normalization stats", False
                break

    external_stats_root = None if dataset_config is None else getattr(dataset_config, "external_stats_root", None)
    if external_stats_root is not None and str(external_stats_root).strip():
        action_modes: list[str] = []
        for action_mode in (args.action_mode, getattr(dataset_config, "action_mode", None)):
            if action_mode is not None and str(action_mode) not in action_modes:
                action_modes.append(str(action_mode))

        candidate_files: list[Path] = []
        for stats_root in _candidate_relative_roots(external_stats_root, ckpt_dir):
            for action_mode in action_modes:
                candidate_files.append(stats_root / args.stats_key / action_mode / "stats.json")
        for candidate_file in _dedupe_paths(candidate_files):
            if candidate_file.is_file():
                yield candidate_file, "external per-embodiment stats", False


def load_wsa_large_runtime_stats(
    args: "InferenceArgs",
    config: PreTrainedConfig,
    ckpt_dir: Path,
    train_config: TrainPipelineConfig | None,
    *,
    require_state: bool,
) -> tuple[dict[str, Any] | None, Path | None]:
    shape_meta = wsa_large_shape_meta(config)
    errors: list[str] = []
    for stats_path, source, explicit in _iter_wsa_large_stats_candidates(args, ckpt_dir, train_config):
        if not stats_path.is_file():
            message = f"{source} not found: {stats_path}"
            if explicit:
                raise FileNotFoundError(message)
            errors.append(message)
            continue
        try:
            raw_payload = load_dataset_stats_from_json(str(stats_path))
            selected_payload = _select_wsa_large_stats_payload(raw_payload, args.stats_key)
            converted_payload = ensure_wsa_large_stats_format(
                selected_payload,
                shape_meta=shape_meta,
                require_state=require_state,
            )
        except Exception as exc:
            message = f"{source} at {stats_path} is not usable for WSA_Large stats: {exc}"
            if explicit:
                raise ValueError(message) from exc
            errors.append(message)
            continue
        logging.info("Loaded WSA_Large runtime stats from %s (%s).", stats_path, source)
        return converted_payload, stats_path

    if errors:
        logging.warning("Could not load WSA_Large runtime stats. Last error: %s", errors[-1])
    return None, None


def maybe_load_wsa_large_action_postprocess(
    policy,
    args: "InferenceArgs",
    config: PreTrainedConfig,
    ckpt_dir: Path,
    train_config: TrainPipelineConfig | None,
) -> None:
    if getattr(policy, "_action_denorm_specs", None):
        return
    stats_payload, stats_path = load_wsa_large_runtime_stats(
        args,
        config,
        ckpt_dir,
        train_config,
        require_state=False,
    )
    if stats_payload is None:
        logging.warning(
            "WSA_Large action denormalization stats were not loaded. "
            "For pretraining checkpoints, set WSA_LARGE_STATS_PATH or keep train_config.dataset.external_stats_root "
            "available so RoboTwin can resolve %s/%s/stats.json.",
            args.stats_key,
            args.action_mode,
        )
        return
    policy.set_action_postprocess_from_stats(stats_payload)
    if hasattr(policy, "_action_stats_source"):
        policy._action_stats_source = str(stats_path)


def resize_wsa_large_video_view(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(video, size=size, mode="bilinear", align_corners=False, antialias=True)
    except TypeError:
        return F.interpolate(video, size=size, mode="bilinear", align_corners=False)


def concat_wsa_large_camera_views(
    video: torch.Tensor,
    target_video_size: tuple[int, int],
    concat_layout: str,
) -> torch.Tensor:
    num_cameras = int(video.shape[0])
    target_h, target_w = target_video_size
    if concat_layout == "single":
        if num_cameras != 1:
            raise ValueError(f"`single` camera layout requires 1 camera, got {num_cameras}.")
        return resize_wsa_large_video_view(video[0], target_video_size)

    if concat_layout == "horizontal":
        if target_w % num_cameras != 0:
            raise ValueError(
                "horizontal camera layout requires target width divisible by camera count: "
                f"width={target_w}, num_cameras={num_cameras}."
            )
        tile_w = target_w // num_cameras
        views = [
            resize_wsa_large_video_view(video[view_idx], (target_h, tile_w))
            for view_idx in range(num_cameras)
        ]
        return torch.cat(views, dim=-1)

    if concat_layout == "vertical":
        if target_h % num_cameras != 0:
            raise ValueError(
                "vertical camera layout requires target height divisible by camera count: "
                f"height={target_h}, num_cameras={num_cameras}."
            )
        tile_h = target_h // num_cameras
        views = [
            resize_wsa_large_video_view(video[view_idx], (tile_h, target_w))
            for view_idx in range(num_cameras)
        ]
        return torch.cat(views, dim=-2)

    if concat_layout == "robotwin":
        if num_cameras != 3:
            raise ValueError(f"`robotwin` camera layout requires exactly 3 cameras, got {num_cameras}.")
        if target_h % 3 != 0 or target_w % 2 != 0:
            raise ValueError(
                "robotwin camera layout requires target height divisible by 3 and width divisible by 2: "
                f"height={target_h}, width={target_w}."
            )
        bottom_h = target_h // 3
        top_h = target_h - bottom_h
        half_w = target_w // 2
        cam_top = resize_wsa_large_video_view(video[0], (top_h, target_w))
        cam_left = resize_wsa_large_video_view(video[1], (bottom_h, half_w))
        cam_right = resize_wsa_large_video_view(video[2], (bottom_h, half_w))
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        return torch.cat([cam_top, bottom], dim=-2)

    raise ValueError(
        f"Invalid WSA_Large concat layout: {concat_layout}. "
        "Expected one of: single, horizontal, vertical, robotwin."
    )


class WSALargeRuntimeAdapter:
    def __init__(
        self,
        args: "InferenceArgs",
        config: PreTrainedConfig,
        ckpt_dir: Path,
        train_config: TrainPipelineConfig | None,
    ):
        self.args = args
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.train_config = train_config
        self.target_proprio_dim = int(getattr(config, "proprio_dim", 14))
        self.state_normalizer = self._build_state_normalizer()

    @staticmethod
    def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1).contiguous()
        if tensor.dtype == torch.uint8:
            return tensor.to(torch.float32) / 255.0
        return tensor.to(torch.float32)

    def _build_state_normalizer(self) -> SingleFieldLinearNormalizer | None:
        stats_payload, stats_file = load_wsa_large_runtime_stats(
            self.args,
            self.config,
            self.ckpt_dir,
            self.train_config,
            require_state=True,
        )
        if stats_payload is None:
            logging.warning("WSA_Large state stats were not loaded; proprio will be passed without normalization.")
            return None

        state_stats = stats_payload.get("state", {})
        if not state_stats:
            logging.warning("WSA_Large stats contain no state section; proprio will be passed raw.")
            return None

        state_key = self.args.wsa_large_state_key
        if state_key not in state_stats:
            if len(state_stats) == 1:
                state_key = next(iter(state_stats))
            else:
                raise KeyError(
                    f"WSA_Large state stats key {state_key!r} not found. Available keys: {list(state_stats.keys())}"
                )
        selected_stats = {
            key.removeprefix("global_"): value
            for key, value in state_stats[state_key].items()
            if key.startswith("global_")
        }
        mode = str(getattr(self.config, "action_norm_default_mode", "z-score"))
        exception_mode = getattr(self.config, "action_norm_exception_mode", None) or {}
        mode = exception_mode.get("state", {}).get(state_key, mode)
        return SingleFieldLinearNormalizer(stats=selected_stats, mode=mode)

    def _normalize_proprio(self, state: torch.Tensor) -> torch.Tensor:
        proprio = state.detach().cpu().to(torch.float32).flatten()
        if self.state_normalizer is not None:
            proprio = self.state_normalizer.forward(proprio)
        if proprio.numel() < self.target_proprio_dim:
            proprio = F.pad(proprio, (0, self.target_proprio_dim - proprio.numel()))
        elif proprio.numel() > self.target_proprio_dim:
            proprio = proprio[: self.target_proprio_dim]
        return proprio.unsqueeze(0)

    def build_inputs(
        self,
        *,
        head_image: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        state: torch.Tensor,
        task: str,
    ) -> dict[str, object]:
        camera_images = [
            self._image_to_chw_float(head_image),
            self._image_to_chw_float(left_wrist_image),
            self._image_to_chw_float(right_wrist_image),
        ]
        video = torch.stack(camera_images, dim=0).unsqueeze(1)
        target_video_size = resolve_wsa_large_video_size(
            len(camera_images),
            (self.args.wsa_large_video_height, self.args.wsa_large_video_width),
            resolve_bool_env(
                self.args.wsa_large_standardize_video_size_by_cameras,
                "WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS",
                "STANDARDIZE_VIDEO_SIZE_BY_CAMERAS",
            ),
        )
        concat_layout = resolve_wsa_large_concat_layout(
            len(camera_images),
            self.args.wsa_large_concat_multi_camera,
        )
        input_image = concat_wsa_large_camera_views(video, target_video_size, concat_layout)
        input_image = (input_image - 0.5) / 0.5
        return {
            "input_image": input_image,
            "proprio": self._normalize_proprio(state),
            "prompt": [build_wsa_large_prompt(task)],
        }


def build_policy_and_transforms(args: "InferenceArgs", dtype: torch.dtype):
    """Load policy and build input/output transforms."""
    ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    apply_runtime_config_overrides(config, args)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cls, processor_transform_cls = resolve_policy_components(config)
    train_config = load_train_config_or_none(ckpt_dir) if is_wsa_large(config.type) else None
    policy_config = config
    if is_wsa_large(config.type) and hasattr(config, "action_stats_path"):
        # Keep checkpoint stats portable across machines by not letting the saved config override them.
        policy_config = copy.deepcopy(config)
        policy_config.action_stats_path = None
    policy = policy_cls.from_pretrained(config=policy_config, pretrained_name_or_path=ckpt_dir)
    if is_wsa_base(config.type) and hasattr(policy, "model"):
        setattr(
            policy.model,
            "omit_visual_tokens_in_causal_inference",
            resolve_bool_env(
                args.omit_visual_tokens_in_causal_inference,
                "OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE",
            ),
        )
    policy.to(device=config.device, dtype=dtype).eval()

    logging.info(f"Resolved policy type: {config.type}")
    if (is_wsa_base(config.type) or is_wsa_large(config.type)) and args.disable_3d_teacher_for_eval:
        logging.info("%s eval mode: disabled DA3 teacher instantiation.", config.type)

    if is_wsa_large(config.type):
        dataset_config = None if train_config is None else getattr(train_config, "dataset", None)
        if dataset_config is not None:
            dataset_name = getattr(dataset_config, "type", dataset_config.__class__.__name__)
            logging.info("Loaded WSA_Large train_config dataset: %s", dataset_name)
        maybe_load_wsa_large_action_postprocess(policy, args, config, ckpt_dir, train_config)
        return policy, WSALargeRuntimeAdapter(args, config, ckpt_dir, train_config), None, config

    stats = load_json(ckpt_dir / "stats.json")[args.stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    state_concat = {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}
    state_stat = {"observation.state": state_concat}

    action_concat = {k: np.asarray(stats["action"][k]) for k in stat_keys}
    action_stat = {"action": action_concat}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]
    processor_path = (
        args.qwen3_vl_processor_path
        or getattr(config, "qwen3_vl_processor_path", None)
        or getattr(config, "qwen3_vl_pretrained_path", None)
    )
    if processor_path is None:
        raise ValueError("Failed to resolve a Qwen3-VL processor path for RoboTwin inference.")
    tokenizer_max_length = int(getattr(config, "tokenizer_max_length", 48))

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=args.resize_size, width=args.resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            processor_transform_cls(
                pretrained_model_name_or_path=processor_path,
                max_length=tokenizer_max_length,
            ),
            NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        ]
    )

    return policy, input_transforms, unnormalize_fn, config


@dataclass
class InferenceArgs:
    """Configuration arguments for inference."""

    task_idx: int = 0
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    seed: int = 0
    ckpt_path: Union[str, Path] = "zaleni/WSA-Base-RoboTwin"
    stats_key: str = "aloha"
    resize_size: int = 224
    image_history_interval: int = 15
    action_mode: str = "delta"  # delta | abs
    binarize_gripper: bool = True
    skip_get_obs_within_replan: bool = False
    dtype: str = "float32"  # float32 | float16 | bfloat16
    video_dir: Path = Path("videos")
    fps: int = 30
    decode_image_flag: bool = False
    debug: bool = False
    log_level: str = "WARNING"  # DEBUG | INFO | WARNING | ERROR
    infer_horizon: int = 30
    action_horizon_size: int = 50
    num_inference_steps: int | None = None
    test_num: int = 100
    robot_type: tuple[int, ...] = (6, 1, 6, 1)
    policy_type: str | None = None
    qwen3_vl_pretrained_path: str | None = None
    qwen3_vl_processor_path: str | None = None
    cosmos_tokenizer_path_or_name: str | None = None
    da3_model_path_or_name: str | None = None
    da3_code_root: str | None = None
    disable_3d_teacher_for_eval: bool = False
    omit_visual_tokens_in_causal_inference: bool = True
    wsa_large_model_id: str | None = None
    wsa_large_tokenizer_model_id: str | None = None
    wsa_large_action_dit_pretrained_path: str | None = None
    wsa_large_future_3d_pretrained_path: str | None = None
    wsa_large_load_text_encoder: bool = True
    wsa_large_redirect_common_files: bool = True
    wsa_large_skip_dit_load_from_pretrain: bool = True
    wsa_large_stats_path: str | None = None
    wsa_large_state_key: str = "default"
    wsa_large_video_height: int = 384
    wsa_large_video_width: int = 320
    wsa_large_standardize_video_size_by_cameras: bool = True
    wsa_large_concat_multi_camera: str = "robotwin"


def write_task_summary(args: InferenceArgs, task_name: str, success_count: int, test_num: int) -> None:
    """Persist a per-task summary for downstream aggregation."""
    args.video_dir.mkdir(parents=True, exist_ok=True)
    success_rate = round((success_count / test_num) * 100, 2) if test_num else 0.0
    summary = {
        "task_idx": args.task_idx,
        "task_name": task_name,
        "task_config": args.task_config,
        "success_count": int(success_count),
        "test_num": int(test_num),
        "success_rate": success_rate,
        "ckpt_path": str(args.ckpt_path),
        "instruction_type": args.instruction_type,
        "action_mode": args.action_mode,
        "stats_key": args.stats_key,
    }

    (args.video_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.video_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"task_idx: {args.task_idx}",
                f"task_name: {task_name}",
                f"task_config: {args.task_config}",
                f"success_count: {success_count}",
                f"test_num: {test_num}",
                f"success_rate: {success_rate:.2f}%",
                f"ckpt_path: {args.ckpt_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def infer_once(args: InferenceArgs):
    """Run inference on a single task."""
    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name)
    TASK_ENV = class_decorator(task_args["task_name"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype, device)
    policy, input_transforms, unnormalize_fn, policy_config = build_policy_and_transforms(args, dtype)
    policy_is_wsa_large = is_wsa_large(policy_config.type)
    binarize_gripper = resolve_bool_env(args.binarize_gripper, "BINARIZE_GRIPPER")
    skip_get_obs_within_replan = policy_is_wsa_large and resolve_bool_env(
        args.skip_get_obs_within_replan,
        "SKIP_GET_OBS_WITHIN_REPLAN",
    )

    logging.info("=" * 80)
    logging.info("Initializing environment...")
    logging.info(f"Task: {task_name}, seed: {args.seed}")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    expert_check = True

    now_id = 0
    succ_seed = 0
    seed = args.seed
    st_seed = 100000 * (1 + seed)
    now_seed = st_seed
    test_num = args.test_num
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    succ_seeds = list(range(st_seed, st_seed * 2))

    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
                )
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except (UnStableError, Exception):
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        TASK_ENV.setup_demo(
            now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][args.instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        succ = False
        policy.reset()
        action_plan = deque([], maxlen=args.action_horizon_size)
        replay_images = []
        head_color_list = []
        left_wrist_color_list = []
        right_wrist_color_list = []
        image_history_interval = args.image_history_interval
        action_dim = sum(args.robot_type)
        left_gripper_idx = sum(args.robot_type[0:2])-1
        right_gripper_idx = sum(args.robot_type[0:4])-1

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = not (skip_get_obs_within_replan and action_plan)
            observation = TASK_ENV.get_obs() if need_obs else None
            img = None
            if observation is not None:
                img = observation["observation"]["head_camera"]["rgb"]

                # Record frames only when RGB is rendered.
                replay_images.append(img.copy())

            if not policy_is_wsa_large and len(action_plan) <= image_history_interval:
                if observation is None or img is None:
                    raise RuntimeError("Non-WSA_Large policies require an observation at every eval step.")
                left_wrist_img = observation["observation"]["left_camera"]["rgb"]
                right_wrist_img = observation["observation"]["right_camera"]["rgb"]

                head_color_list.append(torch.as_tensor(img).contiguous().cuda().to(dtype) / 255.0)
                left_wrist_color_list.append(torch.as_tensor(left_wrist_img).contiguous().cuda().to(dtype) / 255.0)
                right_wrist_color_list.append(torch.as_tensor(right_wrist_img).contiguous().cuda().to(dtype) / 255.0)

                while len(head_color_list) > image_history_interval + 1:
                    head_color_list.pop(0)
                    left_wrist_color_list.pop(0)
                    right_wrist_color_list.pop(0)

                past_idx = max(len(head_color_list) - image_history_interval - 1, 0)
                image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
                image_hand_left_with_history = torch.stack(
                    [left_wrist_color_list[past_idx], left_wrist_color_list[-1]], dim=0
                )
                image_hand_right_with_history = torch.stack(
                    [                    right_wrist_color_list[past_idx], right_wrist_color_list[-1]], dim=0
                )

            if not action_plan:
                if observation is None:
                    raise RuntimeError("Observation is required when planning a new WSA_Large action chunk.")
                init_action = torch.as_tensor(observation["joint_action"]["vector"][None]).contiguous().cuda()
                state = torch.from_numpy(observation["joint_action"]["vector"]).float().cuda()
                task = TASK_ENV.get_instruction()

                if policy_is_wsa_large:
                    left_wrist_img = observation["observation"]["left_camera"]["rgb"]
                    right_wrist_img = observation["observation"]["right_camera"]["rgb"]
                    inputs = input_transforms.build_inputs(
                        head_image=img,
                        left_wrist_image=left_wrist_img,
                        right_wrist_image=right_wrist_img,
                        state=state,
                        task=task,
                    )
                    with torch.no_grad():
                        action_pred = policy.predict_action_chunk(inputs, seed=args.seed)
                    action_pred = action_pred[0, : args.infer_horizon, :action_dim]
                else:
                    sample = {
                        f"{OBS_IMAGES}.image0": image_head_with_history,
                        f"{OBS_IMAGES}.image1": image_hand_left_with_history,
                        f"{OBS_IMAGES}.image2": image_hand_right_with_history,
                        "observation.state": state,
                        "task": task,
                    }
                    for key in sample.keys():
                        if OBS_IMAGES in key and "mask" not in key:
                            image = sample[key].permute(0, 3, 1, 2)
                            sample[key] = image

                    sample = input_transforms(sample)

                    inputs = {}
                    for key in sample.keys():
                        if key == "task":
                            inputs[key] = [sample[key]]
                        elif sample[key].dtype == torch.int64:
                            inputs[key] = sample[key][None].cuda()
                        else:
                            inputs[key] = sample[key][None].cuda().to(dtype=dtype)

                    inputs.update({
                        f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                        f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                        f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
                    })

                    with torch.no_grad():
                        action_pred, _ = policy.predict_action_chunk(inputs, decode_image=args.decode_image_flag)

                    action_pred = action_pred[0, : args.infer_horizon, :action_dim]
                    action_pred = unnormalize_fn({"action": action_pred})["action"]

                if args.action_mode == "delta":
                    init_action_for_delta = init_action.to(device=action_pred.device, dtype=action_pred.dtype)
                    init_action_for_delta[:, left_gripper_idx] = 0.0
                    init_action_for_delta[:, right_gripper_idx] = 0.0
                    action_pred += init_action_for_delta
                action_plan.extend(action_pred.cpu().numpy())

            action = action_plan.popleft()
            if binarize_gripper:
                action[left_gripper_idx] = 0 if action[left_gripper_idx] < 0.5 else 1
                action[right_gripper_idx] = 0 if action[right_gripper_idx] < 0.5 else 1
            TASK_ENV.take_action(action, action_type="qpos")

            if TASK_ENV.eval_success:
                succ = True
                break

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Save a replay video of the episode
        args.video_dir.mkdir(parents=True, exist_ok=True)
        suffix = "success" if succ else "failure"
        imageio.mimwrite(
            args.video_dir / f"{suffix}_{succ_seed}.mp4",
            replay_images,  # Already in HWC format (uint8 numpy arrays)
            fps=args.fps,
        )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m |  \033[92m{task_args['task_config']}\033[0m \033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    write_task_summary(args, task_name, TASK_ENV.suc, TASK_ENV.test_num)
    logging.info("Saved task summary to %s", args.video_dir / "summary.json")


def main(args: InferenceArgs):
    """Main entry point for inference."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Suppress curobo INFO logs
    logging.getLogger("curobo").setLevel(logging.WARNING)

    logging.info("=" * 80)
    logging.info("Starting inference...")
    logging.info(f"Debug mode: {args.debug}, Log level: {args.log_level}")
    logging.info(f"Task index: {args.task_idx}, Checkpoint: {args.ckpt_path}")

    infer_once(args)


if __name__ == "__main__":
    tyro.cli(main)
