#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in [THIS_DIR, SRC_ROOT, REPO_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import (
    Qwen3_VLProcessorTransformFn as QwenA1ProcessorTransformFn,
)
from lerobot.policies.WSA_Base.modeling_wsa_base_rtc import WSABaseRTCPolicy
from lerobot.policies.WSA_Base.transform_wsa_base import (
    Qwen3_VLProcessorTransformFn as WSABaseProcessorTransformFn,
)
from lerobot.policies.factory import get_policy_class
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
from lerobot.policies.WSA_Large.text_cache import build_text_embedding_cache_path
from lerobot.policies.names import (
    WSA_LARGE,
    WSA_BASE_ALIASES,
    WSA_LARGE_ALIASES,
    WSA_LARGE_LEGACY_ALIASES,
    is_wsa_base,
    is_wsa_large,
)
from lerobot.policies.rtc import RTCConfig, RTCProcessor
from lerobot.transforms.constants import get_mask_mapping
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

try:
    from .websocket_server import WebsocketPolicyServer
except ImportError:
    from websocket_server import WebsocketPolicyServer


CAMERA_ALIASES = {
    f"{OBS_IMAGES}.image0": ("cam_high", "head", "image0"),
    f"{OBS_IMAGES}.image1": ("cam_left_wrist", "left_wrist", "left", "image1"),
    f"{OBS_IMAGES}.image2": ("cam_right_wrist", "right_wrist", "right", "image2"),
}

SUPPORTED_POLICY_TYPES = {*WSA_BASE_ALIASES, *WSA_LARGE_ALIASES, "qwena1", "internvla_a1_3b"}


@dataclass
class ServeArgs:
    ckpt_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "Clear the junk and items off the desktop."
    stats_key: str | None = None
    stats_path: str | None = None
    infer_horizon: int | None = None
    num_inference_steps: int | None = None
    resize_size: int = 224
    request_image_height: int = 480
    request_image_width: int = 640
    dtype: str = "bfloat16"
    device: str = "auto"
    load_device: str | None = None
    cosmos_device: str | None = None
    qwen3_vl_processor_path: str | None = None
    qwen3_vl_pretrained_path: str | None = None
    cosmos_tokenizer_path_or_name: str | None = None
    da3_model_path_or_name: str | None = None
    da3_code_root: str | None = None
    action_mode: str | None = None
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"
    disable_3d_teacher_for_eval: bool = True
    omit_visual_tokens_in_causal_inference: bool = True
    wsa_large_model_id: str | None = None
    wsa_large_tokenizer_model_id: str | None = None
    wsa_large_action_dit_pretrained_path: str | None = None
    wsa_large_future_3d_pretrained_path: str | None = None
    wsa_large_load_text_encoder: bool = True
    wsa_large_redirect_common_files: bool = True
    wsa_large_skip_dit_load_from_pretrain: bool = True
    wsa_large_state_key: str = "default"
    wsa_large_video_height: int = 384
    wsa_large_video_width: int = 320
    wsa_large_standardize_video_size_by_cameras: bool = True
    wsa_large_concat_multi_camera: str = "robotwin"
    wsa_large_text_embedding_cache_dir: str | None = None
    wsa_large_context_len: int | None = None


def _env_fallback(value: str | None, env_name: str) -> str | None:
    return value if value is not None else os.environ.get(env_name)


def _bool_env_fallback(value: bool, env_name: str) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return value
    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like, got {env_value!r}.")


def add_bool_arg(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    default: bool,
    help_text: str | None = None,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(*flags, dest=dest, action="store_true", help=help_text)
    no_flags = [f"--no-{flag[2:]}" for flag in flags if flag.startswith("--")]
    group.add_argument(*no_flags, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> ServeArgs:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned WSABase policy for the Real Lift2 example.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint step dir or pretrained_model dir.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default_prompt",
        default="Clear the junk and items off the desktop.",
        help="Fallback prompt when the request does not include `prompt` or `task`.",
    )
    parser.add_argument("--stats_key", default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--infer_horizon", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--request_image_height", type=int, default=480)
    parser.add_argument("--request_image_width", type=int, default=640)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="`auto`, `cpu`, `cuda`, or `cuda:N`.")
    parser.add_argument("--load_device", default=None, help="Device used for initial checkpoint loading.")
    parser.add_argument("--cosmos_device", default=None, help="Device used by the Cosmos tokenizer.")
    parser.add_argument("--qwen3_vl_processor_path", default=None)
    parser.add_argument("--qwen3_vl_pretrained_path", default=None)
    parser.add_argument("--cosmos_tokenizer_path_or_name", default=None)
    parser.add_argument("--da3_model_path_or_name", default=None)
    parser.add_argument("--da3_code_root", default=None)
    parser.add_argument("--action_mode", choices=["abs", "delta"], default=None)
    add_bool_arg(
        parser,
        "--rtc_enabled",
        "--rtc-enabled",
        dest="rtc_enabled",
        default=False,
        help_text="Enable runtime-only Real-Time Chunking guidance for WSABase inference.",
    )
    parser.add_argument("--rtc_execution_horizon", type=int, default=10)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        choices=["zeros", "ones", "linear", "exp"],
        default="linear",
    )
    add_bool_arg(
        parser,
        "--disable_3d_teacher_for_eval",
        "--disable-3d-teacher-for-eval",
        dest="disable_3d_teacher_for_eval",
        default=True,
    )
    add_bool_arg(
        parser,
        "--omit_visual_tokens_in_causal_inference",
        "--omit-visual-tokens-in-causal-inference",
        dest="omit_visual_tokens_in_causal_inference",
        default=True,
        help_text="Skip causal visual-generation middle tokens for action-only WSABase inference.",
    )
    parser.add_argument("--wsa_large_model_id", default=None)
    parser.add_argument("--wsa_large_tokenizer_model_id", default=None)
    parser.add_argument("--wsa_large_action_dit_pretrained_path", default=None)
    parser.add_argument("--wsa_large_future_3d_pretrained_path", default=None)
    add_bool_arg(
        parser,
        "--wsa_large_load_text_encoder",
        dest="wsa_large_load_text_encoder",
        default=True,
        help_text="Load WSA_Large text encoder so sync requests can send plain text prompts.",
    )
    add_bool_arg(
        parser,
        "--wsa_large_redirect_common_files",
        dest="wsa_large_redirect_common_files",
        default=True,
    )
    add_bool_arg(
        parser,
        "--wsa_large_skip_dit_load_from_pretrain",
        dest="wsa_large_skip_dit_load_from_pretrain",
        default=True,
    )
    parser.add_argument("--wsa_large_state_key", default="default")
    parser.add_argument("--wsa_large_video_height", type=int, default=384)
    parser.add_argument("--wsa_large_video_width", type=int, default=320)
    add_bool_arg(
        parser,
        "--wsa_large_standardize_video_size_by_cameras",
        dest="wsa_large_standardize_video_size_by_cameras",
        default=True,
    )
    parser.add_argument(
        "--wsa_large_concat_multi_camera",
        choices=["single", "horizontal", "vertical", "robotwin"],
        default="robotwin",
    )
    parser.add_argument("--wsa_large_text_embedding_cache_dir", default=None)
    parser.add_argument("--wsa_large_context_len", type=int, default=None)
    parsed = ServeArgs(**vars(parser.parse_args()))
    parsed.stats_key = _env_fallback(parsed.stats_key, "STATS_KEY")
    parsed.stats_path = _env_fallback(parsed.stats_path, "STATS_PATH")
    parsed.load_device = _env_fallback(parsed.load_device, "LOAD_DEVICE")
    parsed.cosmos_device = _env_fallback(parsed.cosmos_device, "COSMOS_DEVICE")
    parsed.qwen3_vl_processor_path = _env_fallback(parsed.qwen3_vl_processor_path, "QWEN3_VL_PROCESSOR_PATH")
    parsed.qwen3_vl_pretrained_path = _env_fallback(parsed.qwen3_vl_pretrained_path, "QWEN3_VL_PRETRAINED_PATH")
    parsed.cosmos_tokenizer_path_or_name = _env_fallback(
        parsed.cosmos_tokenizer_path_or_name, "COSMOS_TOKENIZER_PATH_OR_NAME"
    )
    parsed.da3_model_path_or_name = _env_fallback(parsed.da3_model_path_or_name, "DA3_MODEL_PATH_OR_NAME")
    parsed.da3_code_root = _env_fallback(parsed.da3_code_root, "DA3_CODE_ROOT")
    parsed.action_mode = _env_fallback(parsed.action_mode, "ACTION_MODE")
    parsed.omit_visual_tokens_in_causal_inference = _bool_env_fallback(
        parsed.omit_visual_tokens_in_causal_inference,
        "OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE",
    )
    parsed.wsa_large_model_id = _env_fallback(parsed.wsa_large_model_id, "WAN_MODEL_ID")
    parsed.wsa_large_tokenizer_model_id = _env_fallback(
        parsed.wsa_large_tokenizer_model_id,
        "WAN_TOKENIZER_MODEL_ID",
    )
    parsed.wsa_large_action_dit_pretrained_path = _env_fallback(
        parsed.wsa_large_action_dit_pretrained_path,
        "ACTION_DIT_PRETRAINED_PATH",
    )
    parsed.wsa_large_future_3d_pretrained_path = _env_fallback(
        parsed.wsa_large_future_3d_pretrained_path,
        "FUTURE_3D_PRETRAINED_PATH",
    )
    parsed.wsa_large_load_text_encoder = _bool_env_fallback(
        parsed.wsa_large_load_text_encoder,
        "WSA_LARGE_LOAD_TEXT_ENCODER",
    )
    parsed.wsa_large_redirect_common_files = _bool_env_fallback(
        parsed.wsa_large_redirect_common_files,
        "WSA_LARGE_REDIRECT_COMMON_FILES",
    )
    parsed.wsa_large_skip_dit_load_from_pretrain = _bool_env_fallback(
        parsed.wsa_large_skip_dit_load_from_pretrain,
        "WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN",
    )
    parsed.wsa_large_text_embedding_cache_dir = _env_fallback(
        parsed.wsa_large_text_embedding_cache_dir,
        "WSA_LARGE_TEXT_EMBED_CACHE_DIR",
    )
    parsed.wsa_large_text_embedding_cache_dir = _env_fallback(
        parsed.wsa_large_text_embedding_cache_dir,
        "TEXT_EMBED_CACHE_DIR",
    )
    context_len_env = os.environ.get("WSA_LARGE_CONTEXT_LEN")
    if parsed.wsa_large_context_len is None and context_len_env is not None:
        parsed.wsa_large_context_len = int(context_len_env)
    return parsed


def resolve_ckpt_dir(ckpt_path: str | Path) -> Path:
    ckpt_dir = Path(ckpt_path).expanduser()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_dir}")
    if (ckpt_dir / "config.json").exists():
        return ckpt_dir.resolve()
    pretrained_dir = ckpt_dir / "pretrained_model"
    if (pretrained_dir / "config.json").exists():
        return pretrained_dir.resolve()
    raise FileNotFoundError(
        f"Could not find config.json under {ckpt_dir} or {pretrained_dir}. "
        "Pass either the `pretrained_model` directory or the containing checkpoint step directory."
    )


def resolve_runtime_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        if device == "cpu":
            logging.warning("Requested bfloat16 on CPU; falling back to float32.")
            return torch.float32
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_train_config_or_none(ckpt_dir: Path) -> TrainPipelineConfig | None:
    try:
        return TrainPipelineConfig.from_pretrained(ckpt_dir)
    except Exception as exc:
        logging.warning("Failed to load train_config.json from %s: %s", ckpt_dir, exc)
        return None


def apply_runtime_config_overrides(config: PreTrainedConfig, args: ServeArgs) -> None:
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
    if is_wsa_base(config.type) and args.disable_3d_teacher_for_eval and hasattr(config, "lambda_3d"):
        config.lambda_3d = 0.0
    if is_wsa_large(config.type):
        if args.wsa_large_model_id is not None and hasattr(config, "model_id"):
            config.model_id = args.wsa_large_model_id
        if args.wsa_large_tokenizer_model_id is not None and hasattr(config, "tokenizer_model_id"):
            config.tokenizer_model_id = args.wsa_large_tokenizer_model_id
        if (
            args.wsa_large_action_dit_pretrained_path is not None
            and hasattr(config, "action_dit_pretrained_path")
        ):
            config.action_dit_pretrained_path = args.wsa_large_action_dit_pretrained_path
        if (
            args.wsa_large_future_3d_pretrained_path is not None
            and hasattr(config, "future_3d_pretrained_path")
        ):
            config.future_3d_pretrained_path = args.wsa_large_future_3d_pretrained_path
        if hasattr(config, "load_text_encoder"):
            config.load_text_encoder = bool(args.wsa_large_load_text_encoder)
        if hasattr(config, "redirect_common_files"):
            config.redirect_common_files = bool(args.wsa_large_redirect_common_files)
        if hasattr(config, "skip_dit_load_from_pretrain"):
            config.skip_dit_load_from_pretrain = bool(args.wsa_large_skip_dit_load_from_pretrain)
        if args.disable_3d_teacher_for_eval and hasattr(config, "lambda_3d"):
            config.lambda_3d = 0.0
    if args.num_inference_steps is not None and hasattr(config, "num_inference_steps"):
        config.num_inference_steps = int(args.num_inference_steps)


def resolve_policy_components(config: PreTrainedConfig, *, rtc_enabled: bool):
    if config.type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            "Expected a WSA_Base/WSA_Large or InternVLA-A1 checkpoint, "
            f"got config.type={config.type!r}. Supported types: {sorted(SUPPORTED_POLICY_TYPES)}."
        )

    if config.type in {"qwena1", "internvla_a1_3b"}:
        if rtc_enabled:
            raise ValueError(
                "RTC serving is only supported for WSABase checkpoints. "
                "Set RTC_ENABLED=false for InternVLA-A1."
            )
        return get_policy_class(config.type), QwenA1ProcessorTransformFn

    if is_wsa_large(config.type):
        if rtc_enabled:
            raise NotImplementedError("WSA_Large Real Lift2 example serving currently supports sync mode only.")
        return get_policy_class(config.type), None

    policy_cls = WSABaseRTCPolicy if rtc_enabled else get_policy_class(config.type)
    return policy_cls, WSABaseProcessorTransformFn


def resolve_stats(stats_path: Path, requested_key: str | None) -> tuple[str, dict[str, Any]]:
    stats_root = load_json(stats_path)
    is_flat_stats = OBS_STATE in stats_root and "action" in stats_root
    if requested_key is not None:
        if requested_key not in stats_root:
            if is_flat_stats:
                return requested_key, stats_root
            raise KeyError(f"stats_key={requested_key!r} not found in {stats_path}")
        return requested_key, stats_root[requested_key]

    if is_flat_stats:
        return "default", stats_root

    if len(stats_root) == 1:
        key = next(iter(stats_root))
        return key, stats_root[key]

    if "real_lift2" in stats_root:
        return "real_lift2", stats_root["real_lift2"]

    raise ValueError(
        f"stats.json contains multiple keys {list(stats_root.keys())}; please pass --stats_key explicitly."
    )


def to_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an image tensor with 3 dims, got shape={array.shape}")

    if array.shape[-1] == 3:
        hwc = array
    elif array.shape[0] == 3:
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}. Expected HWC or CHW with 3 channels.")

    if np.issubdtype(hwc.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        hwc = np.clip(hwc * scale, 0.0, 255.0)
    else:
        hwc = np.clip(hwc, 0, 255)

    return np.ascontiguousarray(hwc.astype(np.uint8))


def coerce_history(image_value: Any) -> np.ndarray:
    array = np.asarray(image_value)
    if array.ndim == 3:
        frame = to_hwc_uint8(array)
        return np.stack([frame, frame], axis=0)
    if array.ndim != 4:
        raise ValueError(
            f"Unsupported history tensor shape: {array.shape}. Expected [H,W,C], [C,H,W], [T,H,W,C], or [T,C,H,W]."
        )

    if array.shape[-1] == 3:
        frames = [to_hwc_uint8(array[idx]) for idx in range(array.shape[0])]
    elif array.shape[1] == 3:
        frames = [to_hwc_uint8(array[idx]) for idx in range(array.shape[0])]
    else:
        raise ValueError(f"Unsupported history tensor shape: {array.shape}")

    if len(frames) == 1:
        return np.stack([frames[0], frames[0]], axis=0)
    return np.stack([frames[0], frames[-1]], axis=0)


def wsa_large_shape_meta(config: PreTrainedConfig) -> dict[str, list[dict[str, object]]]:
    action_dim = int(getattr(config, "action_dim", 14))
    proprio_dim = int(getattr(config, "proprio_dim", action_dim))
    return {
        "action": [{"key": "default", "raw_shape": action_dim, "shape": action_dim}],
        "state": [{"key": "default", "raw_shape": proprio_dim, "shape": proprio_dim}],
    }


def _select_wsa_large_stats_payload(
    stats_root: dict[str, Any],
    requested_key: str | None,
) -> tuple[str, dict[str, Any]]:
    stats_aliases = (WSA_LARGE, "wsa_large", *WSA_LARGE_LEGACY_ALIASES)
    if any(stats_alias in stats_root for stats_alias in stats_aliases):
        return requested_key or "real_lift2", stats_root

    if requested_key is not None:
        if requested_key in stats_root and isinstance(stats_root[requested_key], dict):
            return requested_key, stats_root[requested_key]
        return requested_key, stats_root

    if "real_lift2" in stats_root and isinstance(stats_root["real_lift2"], dict):
        return "real_lift2", stats_root["real_lift2"]

    if len(stats_root) == 1:
        key = next(iter(stats_root))
        value = stats_root[key]
        if isinstance(value, dict):
            return key, value

    return "real_lift2", stats_root


def resolve_wsa_large_stats(
    stats_path: Path,
    requested_key: str | None,
    config: PreTrainedConfig,
) -> tuple[str, dict[str, Any]]:
    raw_payload = load_dataset_stats_from_json(str(stats_path))
    stats_key, selected_payload = _select_wsa_large_stats_payload(raw_payload, requested_key)
    return stats_key, ensure_wsa_large_stats_format(
        selected_payload,
        shape_meta=wsa_large_shape_meta(config),
        require_state=True,
    )


def _infer_wsa_large_action_dim(stats_payload: dict[str, Any], config: PreTrainedConfig) -> int:
    action_stats = stats_payload.get("action", {})
    if isinstance(action_stats, dict):
        for key_stats in action_stats.values():
            if not isinstance(key_stats, dict):
                continue
            for stat_name in ("global_mean", "mean", "global_min", "min"):
                stat_value = key_stats.get(stat_name)
                if stat_value is None:
                    continue
                array = np.asarray(stat_value)
                if array.ndim == 0:
                    return 1
                return int(array.shape[-1])
    return int(getattr(config, "action_dim", 14))


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
        return torch.cat(
            [resize_wsa_large_video_view(video[view_idx], (target_h, tile_w)) for view_idx in range(num_cameras)],
            dim=-1,
        )

    if concat_layout == "vertical":
        if target_h % num_cameras != 0:
            raise ValueError(
                "vertical camera layout requires target height divisible by camera count: "
                f"height={target_h}, num_cameras={num_cameras}."
            )
        tile_h = target_h // num_cameras
        return torch.cat(
            [resize_wsa_large_video_view(video[view_idx], (tile_h, target_w)) for view_idx in range(num_cameras)],
            dim=-2,
        )

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
        return torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)

    raise ValueError(
        f"Invalid WSA_Large concat layout: {concat_layout}. "
        "Expected one of: single, horizontal, vertical, robotwin."
    )


class WSALargeRealLift2Adapter:
    def __init__(
        self,
        args: ServeArgs,
        config: PreTrainedConfig,
        stats_payload: dict[str, Any],
        device: str,
        dtype: torch.dtype,
    ):
        self.args = args
        self.config = config
        self.device = device
        self.dtype = dtype
        self.target_proprio_dim = int(getattr(config, "proprio_dim", 14))
        self.state_normalizer = self._build_state_normalizer(stats_payload)
        self.text_embedding_cache_dir = (
            Path(args.wsa_large_text_embedding_cache_dir).expanduser()
            if args.wsa_large_text_embedding_cache_dir
            else None
        )
        self.context_len = int(
            args.wsa_large_context_len
            if args.wsa_large_context_len is not None
            else getattr(config, "tokenizer_max_len", 128)
        )
        if self.text_embedding_cache_dir is not None and not self.text_embedding_cache_dir.is_dir():
            raise FileNotFoundError(
                f"WSA_Large text embedding cache dir does not exist: {self.text_embedding_cache_dir}"
            )
        if self.text_embedding_cache_dir is None and not bool(args.wsa_large_load_text_encoder):
            raise ValueError(
                "WSA_LARGE_LOAD_TEXT_ENCODER=false requires WSA_LARGE_TEXT_EMBED_CACHE_DIR. "
                "Precompute the prompt embedding first, or enable the text encoder."
            )

    @staticmethod
    def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {tuple(tensor.shape)}")
        return tensor.permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

    def _build_state_normalizer(self, stats_payload: dict[str, Any]) -> SingleFieldLinearNormalizer | None:
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

    def _normalize_proprio(self, state: np.ndarray) -> torch.Tensor:
        proprio = torch.as_tensor(state, dtype=torch.float32).flatten()
        if self.state_normalizer is not None:
            proprio = self.state_normalizer.forward(proprio)
        if proprio.numel() < self.target_proprio_dim:
            proprio = F.pad(proprio, (0, self.target_proprio_dim - proprio.numel()))
        elif proprio.numel() > self.target_proprio_dim:
            proprio = proprio[: self.target_proprio_dim]
        return proprio.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _load_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("WSA_Large text embedding cache dir is not set.")
        cache_path = build_text_embedding_cache_path(
            self.text_embedding_cache_dir,
            prompt,
            self.context_len,
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing WSA_Large text embedding cache: {cache_path}. "
                "Precompute text embeddings with the same task prompt, "
                "or set WSA_LARGE_LOAD_TEXT_ENCODER=true."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context len mismatch: expected {self.context_len}, "
                f"got {context.shape[0]} and {context_mask.shape[0]} in {cache_path}"
            )
        context = context.clone()
        context[~context_mask] = 0.0
        return (
            context.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            context_mask.unsqueeze(0).to(device=self.device, dtype=torch.bool),
        )

    def build_inputs(
        self,
        *,
        camera_histories: list[np.ndarray] | None = None,
        head_history: np.ndarray | None = None,
        left_history: np.ndarray | None = None,
        right_history: np.ndarray | None = None,
        state: np.ndarray,
        task: str,
    ) -> dict[str, Any]:
        if camera_histories is None:
            camera_histories = [
                history
                for history in (head_history, left_history, right_history)
                if history is not None
            ]
        if not camera_histories:
            raise ValueError("WSA_Large serving requires at least one valid camera view.")

        camera_images = [self._image_to_chw_float(history[-1]) for history in camera_histories]
        video = torch.stack(camera_images, dim=0).unsqueeze(1)
        target_video_size = resolve_wsa_large_video_size(
            len(camera_images),
            (self.args.wsa_large_video_height, self.args.wsa_large_video_width),
            bool(self.args.wsa_large_standardize_video_size_by_cameras),
        )
        concat_layout = resolve_wsa_large_concat_layout(
            len(camera_images),
            self.args.wsa_large_concat_multi_camera,
        )
        input_image = concat_wsa_large_camera_views(video, target_video_size, concat_layout)
        input_image = (input_image - 0.5) / 0.5
        inputs = {
            "input_image": input_image.to(device=self.device, dtype=self.dtype),
            "proprio": self._normalize_proprio(state),
        }
        prompt = build_wsa_large_prompt(task)
        if self.text_embedding_cache_dir is not None:
            context, context_mask = self._load_cached_text_context(prompt)
            inputs["context"] = context
            inputs["context_mask"] = context_mask
        else:
            inputs["prompt"] = [prompt]
        return inputs


class WSABaseRemotePolicy:
    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)

        config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        apply_runtime_config_overrides(config, args)
        policy_cls, processor_transform_cls = resolve_policy_components(config, rtc_enabled=args.rtc_enabled)
        self.is_wsa_large = is_wsa_large(config.type)
        self.wsa_large_adapter: WSALargeRealLift2Adapter | None = None
        self.device = resolve_device(args.device)
        self.load_device = resolve_device(args.load_device) if args.load_device else ("cpu" if self.device != "cpu" else "cpu")
        self.cosmos_device = resolve_device(args.cosmos_device) if args.cosmos_device else self.device
        config.device = self.load_device
        setattr(config, "cosmos_device", self.cosmos_device)
        self.runtime_dtype = resolve_runtime_dtype(args.dtype, self.device)
        config.dtype = "float32" if self.runtime_dtype == torch.float32 else "bfloat16"
        logging.info(
            "Resolved runtime backbone paths: qwen_pretrained=%s | qwen_processor=%s | cosmos=%s | da3=%s",
            getattr(config, "qwen3_vl_pretrained_path", None),
            getattr(config, "qwen3_vl_processor_path", None),
            getattr(config, "cosmos_tokenizer_path_or_name", None),
            getattr(config, "da3_model_path_or_name", None),
        )
        logging.info(
            "Resolved runtime devices: runtime_device=%s | load_device=%s | cosmos_device=%s | runtime_dtype=%s",
            self.device,
            self.load_device,
            self.cosmos_device,
            self.runtime_dtype,
        )

        action_horizon = int(getattr(config, "action_horizon", getattr(config, "chunk_size", 1)))
        if args.infer_horizon is not None:
            config.n_action_steps = min(args.infer_horizon, action_horizon)
        self.infer_horizon = int(args.infer_horizon or getattr(config, "n_action_steps", action_horizon))

        self.policy = policy_cls.from_pretrained(config=config, pretrained_name_or_path=self.ckpt_dir)
        if is_wsa_base(config.type) and hasattr(self.policy, "model"):
            setattr(
                self.policy.model,
                "omit_visual_tokens_in_causal_inference",
                bool(args.omit_visual_tokens_in_causal_inference),
            )
        self.policy.config.device = self.device
        setattr(self.policy.config, "cosmos_device", self.cosmos_device)
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()
        self.policy.requires_grad_(False)

        stats_path = Path(args.stats_path).expanduser() if args.stats_path else self.ckpt_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"stats.json not found at {stats_path}. Pass --stats_path if it was saved elsewhere."
            )
        if self.is_wsa_large:
            self.stats_key, wsa_large_stats = resolve_wsa_large_stats(stats_path, args.stats_key, config)
            self.policy.set_action_postprocess_from_stats(wsa_large_stats)
            self.target_action_dim = _infer_wsa_large_action_dim(wsa_large_stats, config)
            self.action_mean = np.zeros((self.target_action_dim,), dtype=np.float32)
            self.action_std = np.ones((self.target_action_dim,), dtype=np.float32)
            self.wsa_large_adapter = WSALargeRealLift2Adapter(
                args=args,
                config=config,
                stats_payload=wsa_large_stats,
                device=self.device,
                dtype=self.runtime_dtype,
            )
            self.resize_fn = None
            self.normalize_state_fn = None
            self.unnormalize_action_fn = None
            self.processor_fn = None
        else:
            self.stats_key, stats = resolve_stats(stats_path, args.stats_key)
            stat_keys = ["min", "max", "mean", "std"]
            self.state_stats = {
                OBS_STATE: {key: np.asarray(stats[OBS_STATE][key]) for key in stat_keys}
            }
            self.action_stats = {
                "action": {key: np.asarray(stats["action"][key]) for key in stat_keys}
            }
            self.action_mean = np.asarray(self.action_stats["action"]["mean"], dtype=np.float32)
            self.action_std = np.asarray(self.action_stats["action"]["std"], dtype=np.float32)
            self.target_action_dim = int(self.action_stats["action"]["mean"].shape[0])

            self.resize_fn = ResizeImagesWithPadFn(height=args.resize_size, width=args.resize_size)
            self.normalize_state_fn = NormalizeTransformFn(
                selected_keys=[OBS_STATE],
                mode="mean_std",
                norm_stats=self.state_stats,
            )
            self.unnormalize_action_fn = UnNormalizeTransformFn(
                selected_keys=["action"],
                mode="mean_std",
                norm_stats=self.action_stats,
            )

            processor_path = (
                args.qwen3_vl_processor_path
                or getattr(config, "qwen3_vl_processor_path", None)
                or getattr(config, "qwen3_vl_pretrained_path", None)
            )
            if processor_path is None:
                raise ValueError("Failed to resolve a Qwen3-VL processor path for WSABase serving.")
            self.processor_fn = processor_transform_cls(
                pretrained_model_name_or_path=processor_path,
                max_length=int(getattr(config, "tokenizer_max_length", 48)),
            )

        train_action_mode = None if self.train_cfg is None else getattr(self.train_cfg.dataset, "action_mode", None)
        if train_action_mode is not None:
            train_action_mode = str(train_action_mode).lower()
        requested_action_mode = None if args.action_mode is None else str(args.action_mode).lower()
        if (
            requested_action_mode is not None
            and train_action_mode is not None
            and requested_action_mode != train_action_mode
        ):
            raise RuntimeError(
                "Requested action_mode does not match the checkpoint training config: "
                f"requested={requested_action_mode!r}, checkpoint={train_action_mode!r}. "
                "Update ACTION_MODE/--action_mode to match the checkpoint, or omit it to follow train_config."
            )

        self.action_mode = requested_action_mode or train_action_mode or "abs"
        if self.action_mode not in {"abs", "delta"}:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        self.rtc_config: RTCConfig | None = None
        self.rtc_processor: RTCProcessor | None = None
        if args.rtc_enabled:
            self.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=int(args.rtc_execution_horizon),
                max_guidance_weight=float(args.rtc_max_guidance_weight),
                prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule.upper()),
            )
            self.rtc_processor = RTCProcessor(self.rtc_config)

        self.delta_mask = None
        if self.action_mode == "delta":
            try:
                self.delta_mask = get_mask_mapping(self.stats_key).detach().cpu().numpy().astype(np.float32)
            except KeyError:
                if self.is_wsa_large:
                    self.delta_mask = get_mask_mapping("real_lift2").detach().cpu().numpy().astype(np.float32)
                else:
                    raise

        self._metadata = {
            "model_type": config.type,
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "target_action_dim": int(self.target_action_dim),
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "num_inference_steps": int(getattr(self.policy.config, "num_inference_steps", -1)),
            "default_prompt": args.default_prompt,
            "rtc_enabled": bool(args.rtc_enabled),
            "rtc_execution_horizon": int(args.rtc_execution_horizon),
            "rtc_max_guidance_weight": float(args.rtc_max_guidance_weight),
            "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule,
            "omit_visual_tokens_in_causal_inference": bool(
                getattr(getattr(self.policy, "model", None), "omit_visual_tokens_in_causal_inference", True)
            ),
            "wsa_large_sync_only": bool(self.is_wsa_large),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        prompt = obs.get("prompt") or obs.get("task") or self.args.default_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            return self.args.default_prompt
        return prompt

    def _resolve_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state")
        if state is None:
            state = obs.get("qpos")
        if state is None:
            raise KeyError("Request is missing `state` (or `qpos`).")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(state)

    def _resolve_image_history(self, images: dict[str, Any], standardized_key: str) -> tuple[np.ndarray, bool]:
        aliases = CAMERA_ALIASES[standardized_key]
        value = None
        for alias in aliases:
            if alias in images:
                value = images[alias]
                break

        if value is None:
            blank = np.zeros(
                (2, self.args.request_image_height, self.args.request_image_width, 3),
                dtype=np.uint8,
            )
            return blank, False
        return coerce_history(value), True

    def _prepare_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise KeyError("Request is missing `images` dictionary.")

        head_history, head_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image0")
        left_history, left_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image1")
        right_history, right_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image2")
        state = self._resolve_state(obs)
        prompt = self._resolve_prompt(obs)

        sample = {
            f"{OBS_IMAGES}.image0": torch.from_numpy(head_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image1": torch.from_numpy(left_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image2": torch.from_numpy(right_history).permute(0, 3, 1, 2).float() / 255.0,
            OBS_STATE: torch.from_numpy(state),
            "task": prompt,
        }

        sample = self.resize_fn(sample)
        sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(head_mask)
        sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(left_mask)
        sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(right_mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)

        inputs: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key == "task" or not isinstance(value, torch.Tensor):
                continue

            if value.dtype == torch.bool:
                inputs[key] = value.reshape(1).to(self.device)
            elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                inputs[key] = value[None].to(self.device)
            elif value.is_floating_point():
                inputs[key] = value[None].to(device=self.device, dtype=self.runtime_dtype)
            else:
                inputs[key] = value[None].to(self.device)

        return inputs, state

    def _prepare_wsa_large_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise KeyError("Request is missing `images` dictionary.")
        if self.wsa_large_adapter is None:
            raise RuntimeError("WSA_Large adapter is not initialized.")

        head_history, head_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image0")
        left_history, left_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image1")
        right_history, right_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image2")
        state = self._resolve_state(obs)
        prompt = self._resolve_prompt(obs)
        camera_histories = [
            history
            for history, mask in (
                (head_history, head_mask),
                (left_history, left_mask),
                (right_history, right_mask),
            )
            if mask
        ]
        return self.wsa_large_adapter.build_inputs(
            camera_histories=camera_histories,
            state=state,
            task=prompt,
        ), state

    def _coerce_prev_chunk_array(self, prev_chunk_value: Any) -> np.ndarray:
        prev_chunk_np = np.asarray(prev_chunk_value, dtype=np.float32)
        if prev_chunk_np.ndim == 2:
            prev_chunk_np = prev_chunk_np[None]
        elif prev_chunk_np.ndim != 3:
            raise ValueError(
                "prev_chunk_left_over must be shaped as [T, A] or [B, T, A], "
                f"got {prev_chunk_np.shape}."
            )
        return np.ascontiguousarray(prev_chunk_np)

    def _normalize_action_array(self, action_array: np.ndarray) -> np.ndarray:
        eps = 1e-6
        action_array = np.asarray(action_array, dtype=np.float32).copy()
        action_dim = action_array.shape[-1]

        mean = np.zeros((action_dim,), dtype=np.float32)
        std = np.ones((action_dim,), dtype=np.float32)
        usable_dims = min(action_dim, self.action_mean.shape[0], self.action_std.shape[0])
        mean[:usable_dims] = self.action_mean[:usable_dims]
        std[:usable_dims] = self.action_std[:usable_dims]

        return (action_array - mean.reshape((1,) * (action_array.ndim - 1) + (-1,))) / (
            std.reshape((1,) * (action_array.ndim - 1) + (-1,)) + eps
        )

    def _prepare_rtc_prefix(self, obs: dict[str, Any], state: np.ndarray) -> torch.Tensor | None:
        prev_chunk_processed_value = obs.get("prev_chunk_left_over_processed")
        prev_chunk_value = obs.get("prev_chunk_left_over")
        if prev_chunk_processed_value is None and prev_chunk_value is None:
            return None

        if prev_chunk_processed_value is not None:
            prev_chunk_np = self._coerce_prev_chunk_array(prev_chunk_processed_value)
            if self.action_mode == "delta":
                action_dim = prev_chunk_np.shape[-1]
                delta_mask = np.zeros((action_dim,), dtype=np.float32)
                if self.delta_mask is not None:
                    usable_mask_dims = min(action_dim, self.delta_mask.shape[0])
                    delta_mask[:usable_mask_dims] = self.delta_mask[:usable_mask_dims]

                state_pad = np.zeros((action_dim,), dtype=np.float32)
                usable_state_dims = min(action_dim, state.shape[0])
                state_pad[:usable_state_dims] = state[:usable_state_dims]
                prev_chunk_np = prev_chunk_np - (state_pad * delta_mask).reshape(1, 1, -1)

            prev_chunk_np = self._normalize_action_array(prev_chunk_np)
        else:
            prev_chunk_np = self._coerce_prev_chunk_array(prev_chunk_value)

        return torch.from_numpy(prev_chunk_np).to(device=self.device, dtype=torch.float32)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()

        inputs, state = (
            self._prepare_wsa_large_inputs(obs)
            if self.is_wsa_large
            else self._prepare_inputs(obs)
        )
        if self.rtc_processor is not None:
            if self.is_wsa_large:
                raise NotImplementedError("WSA_Large Real Lift2 example serving currently supports sync mode only.")
            inference_delay = None
            prev_chunk_left_over = None
            if obs.get("inference_delay") is not None:
                inference_delay = int(obs["inference_delay"])
            prev_chunk_left_over = self._prepare_rtc_prefix(obs, state)

            with torch.no_grad():
                action_pred, _ = self.policy.predict_action_chunk(
                    inputs,
                    decode_image=False,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_chunk_left_over,
                    rtc_processor=self.rtc_processor,
                    execution_horizon=self.rtc_config.execution_horizon,
                )
        else:
            with torch.no_grad():
                if self.is_wsa_large:
                    action_pred = self.policy.predict_action_chunk(inputs)
                else:
                    action_pred, _ = self.policy.predict_action_chunk(
                        inputs,
                        decode_image=False,
                    )

        if action_pred.ndim != 3:
            raise RuntimeError(f"Unexpected action prediction shape: {tuple(action_pred.shape)}")
        model_action_pred = action_pred[0, : self.infer_horizon, : self.target_action_dim]
        if self.is_wsa_large:
            action_pred = model_action_pred
        else:
            action_pred = self.unnormalize_action_fn({"action": model_action_pred})["action"]
        model_action_np = model_action_pred.detach().cpu().numpy().astype(np.float32)
        action_np = action_pred.detach().cpu().numpy().astype(np.float32)

        if self.action_mode == "delta":
            if self.delta_mask is None:
                raise RuntimeError("delta_mask is not initialized for delta inference.")
            state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
            usable_dims = min(len(state_pad), len(state))
            state_pad[:usable_dims] = state[:usable_dims]
            action_dims = min(action_np.shape[-1], len(self.delta_mask))
            action_np[:, :action_dims] += state_pad[None, :action_dims] * self.delta_mask[None, :action_dims]

        return {
            "actions": action_np,
            "action": action_np[0],
            "model_actions": model_action_np,
            "model_action": model_action_np[0],
        }


def main(args: ServeArgs) -> None:
    logging.info("Serve args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = WSABaseRemotePolicy(args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("Failed to resolve hostname %s to an IP address: %s", hostname, exc)
    logging.info("Creating WSABase server (host=%s, ip=%s, port=%s)", hostname, local_ip, args.port)
    logging.info("Server metadata: %s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
