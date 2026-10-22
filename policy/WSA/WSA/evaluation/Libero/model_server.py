#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from collections import OrderedDict
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
from lerobot.policies.WSA_Large.dataset_wsa_large import resolve_wsa_large_video_size
from lerobot.policies.WSA_Large.modeling_wsa_large import WSALargePolicy
from lerobot.policies.WSA_Large.stats_adapter import ensure_wsa_large_stats_format
from lerobot.policies.WSA_Large.text_cache import build_wsa_large_prompt, build_text_embedding_cache_path
from lerobot.policies.WSA_Large.core.data.lerobot.utils.normalizer import (
    SingleFieldLinearNormalizer,
    load_dataset_stats_from_json,
)
from lerobot.policies.WSA_Base.modeling_wsa_base_rtc import WSABaseRTCPolicy
from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
from lerobot.policies.factory import get_policy_class
from lerobot.policies.names import (
    WSA_BASE,
    WSA_LARGE,
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


@dataclass
class ServeArgs:
    ckpt_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "Execute the LIBERO task."
    stats_key: str | None = None
    stats_path: str | None = None
    infer_horizon: int | None = None
    resize_size: int = 224
    request_image_height: int = 256
    request_image_width: int = 256
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
    load_text_encoder: bool = False
    text_embed_cache_dir: str | None = None
    text_embed_context_len: int = 128
    text_embed_cache_max_entries: int = 0
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"
    disable_3d_teacher_for_eval: bool = True


def _env_fallback(value: str | None, env_name: str) -> str | None:
    return value if value is not None else os.environ.get(env_name)


def _env_bool_fallback(value: bool | None, env_name: str, default: bool) -> bool:
    if value is not None:
        return bool(value)
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean env {env_name}={raw_value!r}")


def _env_int_fallback(value: int | None, env_name: str, default: int) -> int:
    if value is not None:
        return int(value)
    raw_value = os.environ.get(env_name)
    if raw_value is None or not str(raw_value).strip():
        return default
    return int(raw_value)


def parse_args() -> ServeArgs:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned WSABase policy for LIBERO evaluation.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint step dir or pretrained_model dir.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default_prompt",
        default="Execute the LIBERO task.",
        help="Fallback prompt when the request does not include `prompt` or `task`.",
    )
    parser.add_argument("--stats_key", default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--infer_horizon", type=int, default=None)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--request_image_height", type=int, default=256)
    parser.add_argument("--request_image_width", type=int, default=256)
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
    parser.add_argument(
        "--load_text_encoder",
        "--load-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load the Wan text encoder/tokenizer and encode prompts on the fly instead of using cached text embeddings.",
    )
    parser.add_argument("--text_embed_cache_dir", default=None)
    parser.add_argument("--text_embed_context_len", type=int, default=None)
    parser.add_argument("--text_embed_cache_max_entries", type=int, default=None)
    parser.add_argument(
        "--rtc_enabled",
        "--rtc-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable runtime-only Real-Time Chunking guidance for WSABase inference.",
    )
    parser.add_argument("--rtc_execution_horizon", type=int, default=10)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        choices=["zeros", "ones", "linear", "exp"],
        default="linear",
    )
    parser.add_argument(
        "--disable_3d_teacher_for_eval",
        "--disable-3d-teacher-for-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parsed.load_text_encoder = _env_bool_fallback(parsed.load_text_encoder, "LOAD_TEXT_ENCODER", default=False)
    parsed.text_embed_cache_dir = _env_fallback(parsed.text_embed_cache_dir, "TEXT_EMBED_CACHE_DIR")
    parsed.text_embed_context_len = _env_int_fallback(
        parsed.text_embed_context_len,
        "TEXT_EMBED_CONTEXT_LEN",
        default=128,
    )
    parsed.text_embed_cache_max_entries = _env_int_fallback(
        parsed.text_embed_cache_max_entries,
        "TEXT_EMBED_CACHE_MAX_ENTRIES",
        default=0,
    )
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
    if args.disable_3d_teacher_for_eval and hasattr(config, "lambda_3d"):
        config.lambda_3d = 0.0


def resolve_stats(stats_path: Path, requested_key: str | None) -> tuple[str, dict[str, Any]]:
    stats_root = load_json(stats_path)
    if requested_key is not None:
        if requested_key not in stats_root:
            raise KeyError(f"stats_key={requested_key!r} not found in {stats_path}")
        return requested_key, stats_root[requested_key]

    if len(stats_root) == 1:
        key = next(iter(stats_root))
        return key, stats_root[key]

    if "franka" in stats_root:
        return "franka", stats_root["franka"]

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


def select_wsa_large_stats_payload(stats_payload: dict[str, Any], stats_key: str | None) -> dict[str, Any]:
    for stats_alias in (WSA_LARGE, "wsa_large", *WSA_LARGE_LEGACY_ALIASES):
        if stats_alias in stats_payload and isinstance(stats_payload[stats_alias], dict):
            return stats_payload[stats_alias]
    if stats_key is not None and stats_key in stats_payload and isinstance(stats_payload[stats_key], dict):
        return stats_payload[stats_key]
    return stats_payload


def resolve_optional_path(path_value: str | None) -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None
    return Path(path_value).expanduser()


class WSABaseLiberoPolicy:
    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)

        config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not is_wsa_base(config.type):
            raise ValueError(f"Expected a WSA_Base checkpoint, got config.type={config.type!r}")
        apply_runtime_config_overrides(config, args)
        self.device = resolve_device(args.device)
        self.load_device = resolve_device(args.load_device) if args.load_device else ("cpu" if self.device != "cpu" else "cpu")
        self.cosmos_device = resolve_device(args.cosmos_device) if args.cosmos_device else self.device
        config.device = self.load_device
        setattr(config, "cosmos_device", self.cosmos_device)
        self.runtime_dtype = resolve_runtime_dtype(args.dtype, self.device)
        checkpoint_dtype = getattr(config, "dtype", None)
        logging.info(
            "Resolved runtime backbone paths: qwen_pretrained=%s | qwen_processor=%s | cosmos=%s | da3=%s",
            getattr(config, "qwen3_vl_pretrained_path", None),
            getattr(config, "qwen3_vl_processor_path", None),
            getattr(config, "cosmos_tokenizer_path_or_name", None),
            getattr(config, "da3_model_path_or_name", None),
        )
        logging.info(
            "Resolved runtime devices: runtime_device=%s | load_device=%s | cosmos_device=%s | runtime_dtype=%s | checkpoint_dtype=%s",
            self.device,
            self.load_device,
            self.cosmos_device,
            self.runtime_dtype,
            checkpoint_dtype,
        )

        if args.infer_horizon is not None:
            config.n_action_steps = min(args.infer_horizon, config.chunk_size)
        self.infer_horizon = int(args.infer_horizon or getattr(config, "n_action_steps", config.chunk_size))

        policy_cls = WSABaseRTCPolicy if args.rtc_enabled else get_policy_class(config.type)
        self.policy = policy_cls.from_pretrained(config=config, pretrained_name_or_path=self.ckpt_dir)
        self.policy.config.device = self.device
        setattr(self.policy.config, "cosmos_device", self.cosmos_device)
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()

        stats_path = Path(args.stats_path).expanduser() if args.stats_path else self.ckpt_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"stats.json not found at {stats_path}. Pass --stats_path if it was saved elsewhere."
            )
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
        self.processor_fn = Qwen3_VLProcessorTransformFn(
            pretrained_model_name_or_path=processor_path,
            max_length=int(getattr(config, "tokenizer_max_length", 48)),
        )

        train_action_mode = None if self.train_cfg is None else getattr(self.train_cfg.dataset, "action_mode", None)
        self.action_mode = args.action_mode or train_action_mode or "abs"
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
            self.delta_mask = get_mask_mapping(self.stats_key).detach().cpu().numpy().astype(np.float32)

        self._metadata = {
            "model_type": WSA_BASE,
            "deployment": "libero_eval",
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "default_prompt": args.default_prompt,
            "rtc_enabled": bool(args.rtc_enabled),
            "rtc_execution_horizon": int(args.rtc_execution_horizon),
            "rtc_max_guidance_weight": float(args.rtc_max_guidance_weight),
            "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule,
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

        inputs, state = self._prepare_inputs(obs)
        if self.rtc_processor is not None:
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
                action_pred, _ = self.policy.predict_action_chunk(
                    inputs,
                    decode_image=False,
                )

        if action_pred.ndim != 3:
            raise RuntimeError(f"Unexpected action prediction shape: {tuple(action_pred.shape)}")
        model_action_pred = action_pred[0, : self.infer_horizon, : self.target_action_dim]
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


class WSALargeLiberoPolicy:
    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)

        config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not is_wsa_large(config.type):
            raise ValueError(f"Expected a WSA_Large checkpoint, got config.type={config.type!r}")
        apply_runtime_config_overrides(config, args)
        self.load_text_encoder = bool(args.load_text_encoder)
        config.load_text_encoder = self.load_text_encoder
        self.config = config

        self.device = resolve_device(args.device)
        self.load_device = (
            resolve_device(args.load_device)
            if args.load_device
            else ("cpu" if self.device != "cpu" else "cpu")
        )
        self.runtime_dtype = resolve_runtime_dtype(args.dtype, self.device)
        config.device = self.load_device

        self.policy = WSALargePolicy.from_pretrained(
            config=config,
            pretrained_name_or_path=self.ckpt_dir,
        )
        self.policy.config.device = self.device
        self.config.device = self.device
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()
        self.policy.model.device = self.device
        self.policy.model.torch_dtype = self.runtime_dtype

        train_action_mode = None if self.train_cfg is None else getattr(self.train_cfg.dataset, "action_mode", None)
        self.action_mode = args.action_mode or train_action_mode or "abs"
        if self.action_mode != "abs":
            raise NotImplementedError("WSA_Large LIBERO serving currently expects abs action mode.")

        self.action_dim = int(getattr(config, "action_dim", 24))
        self.target_proprio_dim = int(getattr(config, "proprio_dim", 24))
        self.text_embed_cache_dir = self._resolve_text_embed_cache_dir()
        self.text_embed_context_len = int(args.text_embed_context_len or 128)
        self.text_embed_cache_max_entries = int(args.text_embed_cache_max_entries or 0)
        if self.text_embed_cache_max_entries < 0:
            raise ValueError("text_embed_cache_max_entries must be non-negative.")
        if not self.load_text_encoder and not self.text_embed_cache_dir.is_dir():
            raise FileNotFoundError(
                "WSA_Large serving defaults to cached LIBERO text embeddings, but "
                f"TEXT_EMBED_CACHE_DIR does not exist: {self.text_embed_cache_dir}. "
                "Precompute the cache there, set TEXT_EMBED_CACHE_DIR to the correct directory, "
                "or set LOAD_TEXT_ENCODER=true to encode prompts on the fly."
            )
        self._cached_text_contexts: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.input_height, self.input_width = resolve_wsa_large_video_size(
            2,
            (args.request_image_height, args.request_image_width),
            True,
        )
        self.tile_width = self.input_width // 2
        self.state_normalizer, self.state_stats_key = self._build_state_normalizer()

        self._metadata = {
            "model_type": WSA_LARGE,
            "deployment": "libero_eval",
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.state_stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "default_prompt": args.default_prompt,
            "load_text_encoder": self.load_text_encoder,
            "text_embed_cache_dir": str(self.text_embed_cache_dir),
            "text_embed_context_len": self.text_embed_context_len,
            "text_embed_cache_max_entries": self.text_embed_cache_max_entries,
        }

    @property
    def infer_horizon(self) -> int:
        return int(self.args.infer_horizon or getattr(self.config, "n_action_steps", getattr(self.config, "action_horizon", 1)))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _latest_frame(image_value: Any) -> np.ndarray:
        return coerce_history(image_value)[-1]

    @staticmethod
    def _image_to_chw_float(image_value: Any) -> torch.Tensor:
        frame = WSALargeLiberoPolicy._latest_frame(image_value)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        return tensor.to(torch.float32) / 255.0

    @staticmethod
    def _resize_view(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        image = image.unsqueeze(0)
        try:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True)
        except TypeError:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        return image.squeeze(0)

    @staticmethod
    def _resolve_camera(images: dict[str, Any], aliases: tuple[str, ...], *, label: str) -> Any:
        for alias in aliases:
            if alias in images:
                return images[alias]
        raise KeyError(f"Request is missing `{label}` image. Tried aliases: {aliases}")

    def _resolve_text_embed_cache_dir(self) -> Path:
        if self.args.text_embed_cache_dir is not None and str(self.args.text_embed_cache_dir).strip():
            return Path(self.args.text_embed_cache_dir).expanduser()
        new_cache_dir = REPO_ROOT / "outputs" / WSA_LARGE / "text_embeds" / "libero"
        legacy_cache_dir = REPO_ROOT / "outputs" / "MagicBot_R0" / "text_embeds" / "libero"
        return new_cache_dir if new_cache_dir.is_dir() else legacy_cache_dir

    def _build_state_normalizer(self) -> tuple[SingleFieldLinearNormalizer, str]:
        stats_candidates: list[Path] = []
        explicit_stats_path = resolve_optional_path(self.args.stats_path)
        if explicit_stats_path is not None:
            stats_candidates.append(explicit_stats_path)
        stats_candidates.append(self.ckpt_dir / "stats.json")
        if self.train_cfg is not None:
            dataset_cfg = getattr(self.train_cfg, "dataset", None)
            if dataset_cfg is not None and getattr(dataset_cfg, "normalization_stats_path", None):
                stats_candidates.append(Path(dataset_cfg.normalization_stats_path).expanduser())

        shape_meta = {
            "action": [{"key": "default", "raw_shape": self.action_dim, "shape": self.action_dim}],
            "state": [{"key": "default", "raw_shape": self.target_proprio_dim, "shape": self.target_proprio_dim}],
        }

        errors: list[str] = []
        for stats_path in stats_candidates:
            if not stats_path.is_file():
                errors.append(f"Missing WSA_Large stats file: {stats_path}")
                continue
            try:
                raw_payload = load_dataset_stats_from_json(str(stats_path))
                selected_payload = select_wsa_large_stats_payload(raw_payload, self.args.stats_key)
                converted_payload = ensure_wsa_large_stats_format(
                    selected_payload,
                    shape_meta=shape_meta,
                    require_state=True,
                )
                state_stats = converted_payload["state"]
                if "default" in state_stats:
                    state_key = "default"
                elif self.args.stats_key is not None and self.args.stats_key in state_stats:
                    state_key = self.args.stats_key
                elif len(state_stats) == 1:
                    state_key = next(iter(state_stats))
                else:
                    state_key = next(iter(state_stats))
                selected_state_stats = state_stats[state_key]
                selected_stats = {
                    key.removeprefix("global_"): value
                    for key, value in selected_state_stats.items()
                    if key.startswith("global_")
                }
                mode = str(getattr(self.config, "action_norm_default_mode", "z-score"))
                exception_mode = getattr(self.config, "action_norm_exception_mode", None) or {}
                mode = exception_mode.get("state", {}).get(state_key, mode)
                return SingleFieldLinearNormalizer(stats=selected_stats, mode=mode), state_key
            except Exception as exc:
                errors.append(f"{stats_path}: {exc}")

        raise FileNotFoundError(
            "Could not load WSA_Large normalization stats. "
            f"Tried: {', '.join(errors) if errors else '(none)'}"
        )

    def _load_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = prompt
        if self.text_embed_cache_max_entries > 0:
            cached = self._cached_text_contexts.get(cache_key)
            if cached is not None:
                self._cached_text_contexts.move_to_end(cache_key)
                return cached

        cache_path = build_text_embedding_cache_path(self.text_embed_cache_dir, prompt, self.text_embed_context_len)
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Missing WSA_Large cached text embedding for prompt. "
                f"Expected {cache_path}. "
                "Either precompute the cache under `outputs/WSA_Large/text_embeds/libero` "
                "(legacy `outputs/MagicBot_R0/text_embeds/libero` is also checked), "
                "or set LOAD_TEXT_ENCODER=true to encode prompts on the fly."
            )

        payload = torch.load(cache_path, map_location="cpu")
        context = payload.get("context")
        if context is None:
            raise KeyError(f"Cached text embedding is missing `context`: {cache_path}")
        mask = payload.get("mask")
        if mask is None:
            mask = payload.get("context_mask")
        if mask is None:
            raise KeyError(f"Cached text embedding is missing `mask`/`context_mask`: {cache_path}")

        context = torch.as_tensor(context, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}")
        if mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got shape {tuple(mask.shape)} in {cache_path}")
        if context.shape[0] != self.text_embed_context_len or mask.shape[0] != self.text_embed_context_len:
            raise ValueError(
                "Cached text embedding length mismatch: "
                f"expected {self.text_embed_context_len}, got context={context.shape[0]} and mask={mask.shape[0]} "
                f"in {cache_path}"
            )

        context = context.unsqueeze(0).contiguous()
        mask = mask.unsqueeze(0).contiguous()
        if self.text_embed_cache_max_entries > 0:
            self._cached_text_contexts[cache_key] = (context, mask)
            self._cached_text_contexts.move_to_end(cache_key)
            while len(self._cached_text_contexts) > self.text_embed_cache_max_entries:
                self._cached_text_contexts.popitem(last=False)
        return context, mask

    def _normalize_proprio(self, state: np.ndarray) -> torch.Tensor:
        proprio = np.asarray(state, dtype=np.float32).reshape(-1)
        proprio_tensor = torch.from_numpy(proprio)
        proprio_tensor = self.state_normalizer.forward(proprio_tensor)
        if proprio_tensor.numel() < self.target_proprio_dim:
            proprio_tensor = F.pad(proprio_tensor, (0, self.target_proprio_dim - proprio_tensor.numel()))
        elif proprio_tensor.numel() > self.target_proprio_dim:
            proprio_tensor = proprio_tensor[: self.target_proprio_dim]
        return proprio_tensor.unsqueeze(0)

    def _to_runtime_tensor(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=dtype, non_blocking=True)

    def _build_input_image(self, images: dict[str, Any]) -> torch.Tensor:
        head_value = self._resolve_camera(images, ("head", "cam_high", "image0"), label="head")
        wrist_value = self._resolve_camera(images, ("left_wrist", "cam_left_wrist", "left", "image1"), label="left_wrist")

        head = self._resize_view(self._image_to_chw_float(head_value), (self.input_height, self.tile_width))
        wrist = self._resize_view(self._image_to_chw_float(wrist_value), (self.input_height, self.tile_width))
        input_image = torch.cat([head, wrist], dim=-1)
        return (input_image - 0.5) / 0.5

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        prompt = obs.get("prompt") or obs.get("task") or self.args.default_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = self.args.default_prompt
        prefix = "A video recorded from a robot's point of view executing the following instruction:"
        if prompt.startswith(prefix):
            return prompt
        return build_wsa_large_prompt(prompt)

    def _resolve_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state")
        if state is None:
            state = obs.get("qpos")
        if state is None:
            raise KeyError("Request is missing `state` (or `qpos`).")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(state)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()

        images = obs.get("images")
        if not isinstance(images, dict):
            raise KeyError("Request is missing `images` dictionary.")

        input_image = self._build_input_image(images)
        state = self._resolve_state(obs)
        proprio = self._normalize_proprio(state)
        prompt = self._resolve_prompt(obs)

        batch: dict[str, Any] = {
            "input_image": self._to_runtime_tensor(input_image, dtype=self.runtime_dtype),
            "proprio": self._to_runtime_tensor(proprio, dtype=self.runtime_dtype),
        }
        if self.load_text_encoder:
            batch["prompt"] = [prompt]
        else:
            context, context_mask = self._load_cached_text_context(prompt)
            batch["context"] = self._to_runtime_tensor(context, dtype=self.runtime_dtype)
            batch["context_mask"] = self._to_runtime_tensor(context_mask, dtype=torch.bool)
        with torch.no_grad():
            action_pred = self.policy.predict_action_chunk(batch)

        if action_pred.ndim != 3:
            raise RuntimeError(f"Unexpected WSA_Large action prediction shape: {tuple(action_pred.shape)}")
        model_action_pred = action_pred[0, : self.infer_horizon, : self.action_dim]
        action_np = model_action_pred.detach().cpu().numpy().astype(np.float32)

        return {
            "actions": action_np,
            "action": action_np[0],
            "model_actions": action_np,
            "model_action": action_np[0],
        }


def main(args: ServeArgs) -> None:
    logging.info("Serve args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    if is_wsa_base(config.type):
        policy = WSABaseLiberoPolicy(args)
    elif is_wsa_large(config.type):
        policy = WSALargeLiberoPolicy(args)
    else:
        raise ValueError(f"Unsupported LIBERO checkpoint type: {config.type!r}")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("Failed to resolve hostname %s to an IP address: %s", hostname, exc)
    logging.info("Creating LIBERO WSABase server (host=%s, ip=%s, port=%s)", hostname, local_ip, args.port)
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
