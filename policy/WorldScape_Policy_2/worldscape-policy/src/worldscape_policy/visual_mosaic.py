from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from worldscape_policy.memory.visual.normalization import (
    VisualInputRange,
    normalize_visual,
)


def compose_three_view_mosaic(views: Tensor) -> Tensor:
    """Place ``[head, left, right]`` views in DreamZero's 2x2 layout."""

    if views.ndim != 6:
        raise ValueError("views must have shape [B,T,V,C,H,W]")
    if views.shape[2] != 3:
        raise ValueError("diffusion video requires exactly [head, left, right]")
    batch, frames, _, channels, height, width = views.shape
    mosaic = views.new_zeros((batch, frames, channels, 2 * height, 2 * width))
    mosaic[..., :height, :width] = views[:, :, 0]
    mosaic[..., height:, :width] = views[:, :, 1]
    mosaic[..., :height, width:] = views[:, :, 2]
    return mosaic


def prepare_diffusion_mosaic(
    views: Tensor,
    *,
    input_range: VisualInputRange,
) -> Tensor:
    """Compose, normalize, then resize the full mosaic to one-view resolution."""

    height, width = views.shape[-2:]
    mosaic = normalize_visual(
        compose_three_view_mosaic(views),
        input_range=input_range,
    )
    batch, frames, channels, mosaic_height, mosaic_width = mosaic.shape
    resized = F.interpolate(
        mosaic.reshape(batch * frames, channels, mosaic_height, mosaic_width),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, height, width)


def compose_three_view_robotwin(views: Tensor) -> Tensor:
    """Resize raw ``[head, left, right]`` views directly into a 384x320 image."""

    if views.ndim != 6:
        raise ValueError("views must have shape [B,T,V,C,H,W]")
    if views.shape[2] != 3:
        raise ValueError("diffusion video requires exactly [head, left, right]")
    batch, frames, _, channels, height, width = views.shape
    flat = views.reshape(batch * frames, 3, channels, height, width)

    def resize(view: Tensor, size: tuple[int, int]) -> Tensor:
        if not view.is_floating_point():
            view = view.float()
        return F.interpolate(
            view,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    head = resize(flat[:, 0], (256, 320))
    left = resize(flat[:, 1], (128, 160))
    right = resize(flat[:, 2], (128, 160))
    bottom = torch.cat((left, right), dim=-1)
    result = torch.cat((head, bottom), dim=-2)
    return result.reshape(batch, frames, channels, 384, 320)


def prepare_diffusion_robotwin_concat(
    views: Tensor,
    *,
    input_range: VisualInputRange,
) -> Tensor:
    """Normalize raw views, then resize each view once into the final layout."""

    normalized = normalize_visual(views, input_range=input_range)
    # Antialiased interpolation may overshoot the normalized range by a few
    # ulps around saturated edges, while the VAE input contract is strict.
    return compose_three_view_robotwin(normalized).clamp_(-1.0, 1.0)


__all__ = [
    "compose_three_view_mosaic",
    "compose_three_view_robotwin",
    "prepare_diffusion_mosaic",
    "prepare_diffusion_robotwin_concat",
]
