from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from flexwam.utils.logging_config import get_logger

from flexwam.models.backbone.lingbot_video.lingbot_video import LingBotVideoTransformer3DModel
from flexwam.models.backbone.lingbot_video.vae import LingBotWanVAEWrapper

logger = get_logger(__name__)


@dataclass
class LingBotLoadedComponents:
    transformer: LingBotVideoTransformer3DModel
    vae: LingBotWanVAEWrapper
    transformer_path: str
    vae_path: str


def load_lingbot_components(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    load_vae: bool = True,
    load_transformer: bool = True,
) -> LingBotLoadedComponents:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"LingBot model path does not exist: {model_path}")

    transformer = None
    if load_transformer:
        transformer_path = model_path / "transformer"
        logger.info("Loading LingBot transformer from %s", transformer_path)
        transformer = LingBotVideoTransformer3DModel.from_pretrained(
            str(transformer_path),
            torch_dtype=torch_dtype,
        ).to(device=device, dtype=torch_dtype)
        transformer.eval()
    else:
        transformer_path = model_path / "transformer"

    vae = None
    if load_vae:
        vae_path = model_path / "vae"
        logger.info("Loading LingBot VAE from %s", vae_path)
        vae = AutoencoderKLWan.from_pretrained(
            str(vae_path),
            torch_dtype=torch_dtype,
        ).to(device=device, dtype=torch_dtype)
        vae.eval().requires_grad_(False)
        vae = LingBotWanVAEWrapper(vae).to(device=device, dtype=torch_dtype)
    else:
        vae_path = model_path / "vae"

    if transformer is None or vae is None:
        raise ValueError("Both `load_transformer` and `load_vae` must be true for LingBotLoadedComponents.")

    return LingBotLoadedComponents(
        transformer=transformer,
        vae=vae,
        transformer_path=str(transformer_path),
        vae_path=str(vae_path),
    )
