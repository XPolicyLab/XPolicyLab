from __future__ import annotations

import torch
from torch import nn

from worldscape_policy.memory.visual.normalization import (
    VisualInputRange,
    normalize_visual,
)
from worldscape_policy.types import WanI2VCondition


class Wan22ImageConditioner:
    """Numerically preserve Wan2.2's raw-frame I2V conditioning path.

    This helper intentionally owns no ``nn.Module``. The plugin owns the image
    encoder and the visual-memory codec owns the VAE, so native checkpoint
    prefixes remain unambiguous.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        visual_input_range: VisualInputRange = "zero_one",
    ) -> None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.num_frames = int(num_frames)
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.visual_input_range = visual_input_range

    def encode(
        self,
        *,
        reference_frame: torch.Tensor,
        image_encoder: nn.Module,
        vae: nn.Module,
        normalized: bool = False,
    ) -> tuple[WanI2VCondition, torch.Tensor]:
        if reference_frame.ndim != 5 or reference_frame.shape[1] != 1:
            raise ValueError("reference_frame must have shape [B, 1, C, H, W]")
        if reference_frame.shape[2] not in (1, 3, 4):
            raise ValueError("reference_frame must be channel-first")
        if not hasattr(image_encoder, "encode_image"):
            raise TypeError("image_encoder must implement encode_image(reference_frame)")
        if not hasattr(vae, "encode"):
            raise TypeError("vae must implement encode(video, ...)")

        image = (
            reference_frame.float()
            if normalized
            else normalize_visual(
                reference_frame,
                input_range=self.visual_input_range,
            )
        )
        if normalized and image.numel() and (
            bool(image.detach().amin() < -1) or bool(image.detach().amax() > 1)
        ):
            raise ValueError("normalized reference frame must stay within [-1, 1]")
        vae_parameter = next(vae.parameters(), None)
        if vae_parameter is not None:
            image = image.to(device=vae_parameter.device, dtype=vae_parameter.dtype)

        clip_features = image_encoder.encode_image(image)
        image_input = image.transpose(1, 2)
        zeros = torch.zeros(
            (
                image_input.shape[0],
                image_input.shape[1],
                self.num_frames - 1,
                image_input.shape[3],
                image_input.shape[4],
            ),
            dtype=image_input.dtype,
            device=image_input.device,
        )
        latent_y = vae.encode(
            torch.cat([image_input, zeros], dim=2),
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        mask = torch.zeros(
            latent_y.shape[0],
            4,
            latent_y.shape[2],
            latent_y.shape[3],
            latent_y.shape[4],
            dtype=latent_y.dtype,
            device=latent_y.device,
        )
        mask[:, :, :1] = 1
        anchor_latent = latent_y[:, :, :1]
        condition = WanI2VCondition(
            clip_features=clip_features,
            masked_latent_y=torch.cat([mask, latent_y], dim=1),
        )
        return condition, anchor_latent
