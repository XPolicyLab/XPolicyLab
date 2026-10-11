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


def compose_three_view_robotwin(
    views: Tensor, *, output_size: tuple[int, int] = (384, 320),
) -> Tensor:
    """Resize ``[head, left, right]`` into a configurable (height, width) canvas.

    Head occupies the top two-thirds (rounded); wrists share the remaining row.
    The default preserves the native RoboTwin2 layout exactly.
    """

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

    canvas_height, canvas_width = output_size
    if (type(canvas_height) is not int or canvas_height < 3
            or type(canvas_width) is not int or canvas_width < 2 or canvas_width % 2):
        raise ValueError("Concat output requires integer height >= 3 and even width >= 2")
    head_height = round(canvas_height * 2 / 3)
    wrist_size = (canvas_height - head_height, canvas_width // 2)
    head = resize(flat[:, 0], (head_height, canvas_width))
    left = resize(flat[:, 1], wrist_size)
    right = resize(flat[:, 2], wrist_size)
    bottom = torch.cat((left, right), dim=-1)
    result = torch.cat((head, bottom), dim=-2)
    return result.reshape(batch, frames, channels, canvas_height, canvas_width)


def prepare_diffusion_robotwin_concat(
    views: Tensor,
    *,
    input_range: VisualInputRange,
    output_size: tuple[int, int] = (384, 320),
) -> Tensor:
    """Normalize raw views, then resize each view once into the final layout."""

    normalized = normalize_visual(views, input_range=input_range)
    # Antialiased interpolation may overshoot the normalized range by a few
    # ulps around saturated edges, while the VAE input contract is strict.
    return compose_three_view_robotwin(normalized, output_size=output_size).clamp_(-1.0, 1.0)


__all__ = [
    "compose_three_view_mosaic",
    "compose_three_view_robotwin",
    "prepare_diffusion_mosaic",
    "prepare_diffusion_robotwin_concat",
]
