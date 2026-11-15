"""Construction of inference-time VAE and text encoders."""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from falcon.utils.locate import locate
from soma.inference.config import TextEncoderConfig
from soma.inference.config import VaeConfig
from soma.inference.runtime.artifact_cache import cache_locally
from soma.inference.types import TextEncoder
from soma.inference.types import VideoVAE


def _resolve_factory(
    model_type: object,
    namespace: str,
) -> Callable[..., nn.Module]:
    if isinstance(model_type, str):
        path = model_type if "." in model_type else f"{namespace}.{model_type}"
        model_type = locate(path)
    if not isinstance(model_type, Callable):
        raise TypeError(f"model type must resolve to a callable, got {model_type!r}")
    return cast(Callable[..., nn.Module], model_type)


def build_text_encoder(
    config: TextEncoderConfig,
    device: str | torch.device,
) -> TextEncoder:
    """Build and freeze the configured text encoder."""
    model_class = _resolve_factory(
        config.type,
        "modelzoo.models.text_encoders",
    )
    path = config.pretrained_model_name_or_path
    if path:
        path = cache_locally(path)
    model = model_class(
        device=str(device),
        pretrained_model_name_or_path=path,
        max_length=config.max_length,
    )
    return cast(TextEncoder, model.eval().requires_grad_(False))


@torch.no_grad()
def encode_text(model: TextEncoder, prompt: str) -> torch.Tensor:
    """Match offline text encoding's IEEE FP32 math, including rotary phases."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        embedding = model([prompt])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if isinstance(embedding, (tuple, list)):
        embedding = embedding[0]
    if embedding.dim() == 3:
        embedding = embedding.squeeze(0)
    return embedding


def build_vae(
    config: VaeConfig,
    device: str | torch.device,
) -> VideoVAE:
    """Build and freeze the configured video VAE.

    Match training: computation precision initializes the VAE's scale buffers,
    while model parameters retain their native storage dtype.
    """
    model_class = _resolve_factory(config.type, "modelzoo.models.video_vae")
    path = cache_locally(config.vae_pth) if config.vae_pth else None
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[config.precision]
    model = model_class(vae_pth=path, dtype=dtype).to(device)
    model = model.eval().requires_grad_(False)
    return cast(VideoVAE, model)
