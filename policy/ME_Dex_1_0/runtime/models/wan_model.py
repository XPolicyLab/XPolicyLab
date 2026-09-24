"""WAN video backbone and VAE wrapper for ME-Dex-1.0 inference."""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging
import os
import json

from wan.modules.model import WanModel
from wan.modules.vae import Wan2_2_VAE

logger = logging.getLogger(__name__)

class WanVideoModel(nn.Module):
    """Build the released WAN architecture and its external VAE."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        vae_path: str,
        device: str = "cuda",
        precision: str = "bfloat16"
    ):
        super().__init__()

        self.device = torch.device(device)
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]

        # Initialize WAN model
        self.wan_model = WanModel(**model_config)
        # WAN uses its own attention path, so the base PyTorch transfer avoids
        # importing unrelated optional Diffusers attention backends.
        nn.Module.to(self.wan_model, device=self.device, dtype=self.precision)

        # Initialize VAE
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)

        logger.info(f"WAN Video Model initialized with {sum(p.numel() for p in self.wan_model.parameters()):,} parameters")

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode video pixels to latent space.

        Args:
            video_pixels: Video in pixel space [B, C, T, H, W], range [-1, 1]

        Returns:
            Video latents [B, C', T', H', W']
        """
        with torch.no_grad():
            return self.vae.encode(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode video latents to pixel space.

        Args:
            video_latents: Video latents [B, C, T, H, W]

        Returns:
            Video pixels [B, C', T', H', W'], range [-1, 1]
        """
        with torch.no_grad():
            video_pixels = []
            for i in range(video_latents.shape[0]):
                pixels = self.vae.decode([video_latents[i]])[0]
                video_pixels.append(pixels)
            result = torch.stack(video_pixels, dim=0)
            return result

    @classmethod
    def from_config(
        cls,
        config_path: str,
        vae_path: str,
        device: str = "cuda",
        precision: str = "bfloat16"
    ) -> 'WanVideoModel':
        """Initialize WAN and VAE; ME-Dex weights are loaded by checkpoint.py."""
        # Load WAN model config
        config_json_path = os.path.join(config_path, 'config.json')
        with open(config_json_path, 'r') as f:
            model_config = json.load(f)
        # Create model without loading WAN weights
        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision
        )
        logger.info("Initialized WAN architecture and VAE")
        return model
