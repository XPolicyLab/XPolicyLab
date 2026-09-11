from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from worldscape_policy.memory.visual.normalization import (
    VisualInputRange,
    normalize_visual,
)
from worldscape_policy.visual_mosaic import (
    prepare_diffusion_mosaic,
    prepare_diffusion_robotwin_concat,
)

DiffusionViewLayout = Literal["mosaic_2x2", "robotwin_concat"]


class WanVisualCodec(nn.Module):
    """Normalize public video layouts and delegate to the pretrained Wan VAE."""

    def __init__(
        self,
        vae: nn.Module,
        *,
        view_index: int = 0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        visual_input_range: VisualInputRange = "zero_one",
        diffusion_view_layout: DiffusionViewLayout = "mosaic_2x2",
    ) -> None:
        super().__init__()
        if not hasattr(vae, "encode"):
            raise TypeError("vae must implement encode(video, ...)")
        self.vae = vae
        self.view_index = int(view_index)
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.visual_input_range = visual_input_range
        if diffusion_view_layout not in {"mosaic_2x2", "robotwin_concat"}:
            raise ValueError(
                "diffusion_view_layout must be 'mosaic_2x2' or "
                "'robotwin_concat'"
            )
        self.diffusion_view_layout = diffusion_view_layout

    def encode_visual(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 6:
            if not 0 <= self.view_index < video.shape[2]:
                raise ValueError(
                    f"view_index={self.view_index} is invalid for {video.shape[2]} views"
                )
            video = video[:, :, self.view_index]
        if video.ndim != 5:
            raise ValueError(
                "visual input must have shape [B,T,C,H,W] or [B,T,V,C,H,W]"
            )
        if video.shape[2] not in (1, 3, 4):
            raise ValueError("expected channel-first visual input")

        return self.encode_normalized(
            normalize_visual(video, input_range=self.visual_input_range)
        )

    def prepare_diffusion_video(self, views: torch.Tensor) -> torch.Tensor:
        if self.diffusion_view_layout == "robotwin_concat":
            return prepare_diffusion_robotwin_concat(
                views,
                input_range=self.visual_input_range,
            )
        return prepare_diffusion_mosaic(
            views,
            input_range=self.visual_input_range,
        )

    def encode_normalized(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] not in (1, 3, 4):
            raise ValueError("normalized visual input must have shape [B,T,C,H,W]")
        if not video.is_floating_point():
            raise TypeError("normalized visual input must be floating point")
        if video.numel() and (
            bool(video.detach().amin() < -1) or bool(video.detach().amax() > 1)
        ):
            raise ValueError("normalized visual input must stay within [-1, 1]")
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        vae_parameter = next(self.vae.parameters(), None)
        if vae_parameter is not None:
            video = video.to(
                device=vae_parameter.device,
                dtype=vae_parameter.dtype,
            )
        return self.vae.encode(
            video,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
