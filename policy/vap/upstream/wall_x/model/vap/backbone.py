"""Wan2.2 TI2V 5B video-only loading boundary for VAP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch

from .config import VAPModelConfig


WAN22_TI2V_5B_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_TI2V_5B_PATTERNS = (
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
)
WAN22_TI2V_5B_TOKENIZER_SUBDIR = Path("google/umt5-xxl")
WAN22_TI2V_5B_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _split_source(source: str) -> tuple[str, str]:
    try:
        model_id, pattern = source.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"model source must be 'model_id:file_pattern', got {source!r}"
        ) from exc
    if not model_id or not pattern:
        raise ValueError(f"invalid model source {source!r}")
    return model_id, pattern


def wan22_5b_model_configs(config: VAPModelConfig):
    """Return existing VGA ModelConfig objects without loading weights."""

    from wall_x.model.vga.pipeline.config import ModelConfig

    if config.base_model_id != WAN22_TI2V_5B_ID:
        raise ValueError(
            f"VAP currently requires {WAN22_TI2V_5B_ID}, got {config.base_model_id}"
        )
    configs = []
    if config.model_paths:
        configs.extend(ModelConfig(path=path) for path in config.model_paths)
    sources: Iterable[str] = config.model_id_with_origin_paths or (
        f"{config.base_model_id}:{pattern}" for pattern in WAN22_TI2V_5B_PATTERNS
    )
    for source in sources:
        model_id, pattern = _split_source(source)
        if model_id != config.base_model_id:
            raise ValueError(
                f"VAP source {source!r} does not match base_model_id "
                f"{config.base_model_id!r}"
            )
        configs.append(ModelConfig(model_id=model_id, origin_file_pattern=pattern))
    return configs


def wan22_5b_tokenizer_config():
    """Resolve the local UMT5 tokenizer from the Wan2.2 asset directory."""

    from wall_x.model.vga.pipeline.config import ModelConfig

    model_root = Path(os.environ["WALLX_MODEL_ROOT"])
    default_path = model_root / WAN22_TI2V_5B_ID / WAN22_TI2V_5B_TOKENIZER_SUBDIR
    tokenizer_path = Path(os.environ.get("VAP_WAN22_TOKENIZER_PATH", default_path))
    missing = [
        name
        for name in WAN22_TI2V_5B_TOKENIZER_FILES
        if not (tokenizer_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Wan2.2 UMT5 tokenizer is incomplete at "
            f"{tokenizer_path}: missing {missing}"
        )
    return ModelConfig(path=str(tokenizer_path))


def load_wan22_5b_video_base(
    config: VAPModelConfig,
    *,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    skip_text_encoder: bool | None = None,
):
    """Load Wan2.2 5B DiT/VAE/T5 while disabling legacy VGA action.

    The loader reuses refactor_2604's checked ModelManager path. It does not
    download in training: weights resolve below ``WALLX_MODEL_ROOT`` exactly as
    in the source branch. The pipeline is an initialization source for VAP.
    """

    from wall_x.model.vga.modeling.wan_video import WanVideoPipeline

    qwen_cache = None
    qwen_online = getattr(config, "text_encoder_backend", "t5") == "qwen35_online"
    if qwen_online:
        from .text_conditioning import validate_text_config
        validate_text_config(config)
        if skip_text_encoder is False:
            raise ValueError("qwen35_online must skip the T5 encoder")
        skip_text_encoder = True
    if getattr(config, "text_encoder_backend", "t5") == "qwen35_cache":
        from .text_conditioning import Qwen35TextCache, validate_text_config

        validate_text_config(config)
        if skip_text_encoder is False:
            raise ValueError("qwen35_cache requires skipping the online text encoder")
        qwen_cache = Qwen35TextCache(
            config.text_feature_cache_path,
            expected_id=config.text_feature_cache_id,
            hidden_dim=config.text_feature_hidden_dim,
            seq_len=config.text_feature_seq_len,
        )
        skip_text_encoder = True
    text_feature_cache_path = getattr(config, "text_feature_cache_path", None)
    online_cache_fallback = bool(
        getattr(config, "text_feature_cache_online_fallback", False)
    )
    if online_cache_fallback and not text_feature_cache_path:
        raise ValueError(
            "text_feature_cache_online_fallback=True requires "
            "model.text_feature_cache_path"
        )
    if skip_text_encoder is None:
        skip_text_encoder = bool(text_feature_cache_path) and not online_cache_fallback
    if skip_text_encoder and online_cache_fallback:
        raise ValueError(
            "skip_text_encoder=True is incompatible with online cache fallback"
        )
    if skip_text_encoder and not text_feature_cache_path and not qwen_online:
        raise ValueError(
            "skip_text_encoder=True requires model.text_feature_cache_path; "
            "otherwise VAP would silently train without text conditioning"
        )

    model_configs = wan22_5b_model_configs(config)
    if skip_text_encoder:
        model_configs = [
            source
            for source in model_configs
            if not (
                source.origin_file_pattern
                and "t5" in source.origin_file_pattern.lower()
            )
        ]
    tokenizer_config = None if skip_text_encoder else wan22_5b_tokenizer_config()
    if qwen_cache is not None or qwen_online:
        # Also remove explicit T5 paths; do not materialize unused UMT5 weights.
        model_configs = [source for source in model_configs
                         if "t5" not in str(source.path).lower()]
    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        # The shared Wan loader otherwise redirects the common T5 filename to
        # Wan2.1-T2V-1.3B.  VAP is pinned to the configured Wan2.2 TI2V 5B
        # assets, so every source must resolve under that model id.
        redirect_common_files=False,
        enable_action=False,
        num_views=1,
        scheduler_shift=float(config.video_flow["shift"]),
        timestep_sample_mode=str(config.video_flow["sample_mode"]),
        skip_text_encoder=skip_text_encoder,
    )
    if pipeline.dit is None:
        raise RuntimeError("Wan2.2 loader returned no video DiT")
    if pipeline.dit.dim != 3072 or len(pipeline.dit.blocks) != 30:
        raise RuntimeError(
            "resolved checkpoint is not the expected Wan2.2 TI2V 5B "
            f"(dim={pipeline.dit.dim}, layers={len(pipeline.dit.blocks)})"
        )
    if getattr(pipeline, "action_transformer", None) is not None:
        raise RuntimeError("legacy VGA ActionTransformer must stay disabled for VAP")
    if qwen_online:
        from .qwen35_online import Qwen35OnlinePrompter
        pipeline.prompter = Qwen35OnlinePrompter(config)
    elif qwen_cache is not None:
        pipeline.prompter = qwen_cache
    elif text_feature_cache_path:
        pipeline.prompter.load_text_feature_cache(text_feature_cache_path)
        if not pipeline.prompter.text_feature_cache:
            raise RuntimeError(
                "VAP text feature cache is empty or unavailable: "
                f"{text_feature_cache_path}"
            )
        pipeline.prompter.cache_online_encoded_features = online_cache_fallback
        if online_cache_fallback and pipeline.text_encoder is None:
            raise RuntimeError(
                "online text cache fallback requires the Wan T5 text encoder"
            )
    elif pipeline.text_encoder is None:
        raise RuntimeError(
            "VAP requires either the Wan T5 text encoder or a non-empty text feature cache"
        )
    return pipeline


__all__ = [
    "WAN22_TI2V_5B_ID",
    "WAN22_TI2V_5B_PATTERNS",
    "WAN22_TI2V_5B_TOKENIZER_FILES",
    "WAN22_TI2V_5B_TOKENIZER_SUBDIR",
    "load_wan22_5b_video_base",
    "wan22_5b_model_configs",
    "wan22_5b_tokenizer_config",
]
