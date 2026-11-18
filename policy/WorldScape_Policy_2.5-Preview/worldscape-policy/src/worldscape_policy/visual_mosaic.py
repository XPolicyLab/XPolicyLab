from __future__ import annotations

import random
from typing import Literal, cast
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from worldscape_policy.memory.visual.normalization import (
    VisualInputRange,
    normalize_visual,
)


DiffusionViewLayout = Literal["mosaic_2x2", "mosaic_T"]


def validate_diffusion_view_layout(
    value: object, *, path: str = "diffusion_view_layout"
) -> DiffusionViewLayout:
    """Validate a supported diffusion layout without rewriting its name."""

    if not isinstance(value, str):
        raise ValueError(f"{path} must be 'mosaic_2x2' or 'mosaic_T'")
    if value not in {"mosaic_2x2", "mosaic_T"}:
        raise ValueError(f"{path} must be 'mosaic_2x2' or 'mosaic_T'")
    return cast(DiffusionViewLayout, value)


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


def compose_three_view_mosaic_T(
    views: Tensor, *, image_size: tuple[int, int] = (384, 320),
) -> Tensor:
    """Resize raw views once into a T mosaic of the requested (height, width)."""

    if views.ndim != 6:
        raise ValueError("views must have shape [B,T,V,C,H,W]")
    if views.shape[2] != 3:
        raise ValueError("diffusion video requires exactly [head, left, right]")
    output_h, output_w = image_size
    if output_h <= 0 or output_w <= 0 or output_h % 3 or output_w % 2:
        raise ValueError("T mosaic height must be divisible by 3 and width by 2")
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

    head = resize(flat[:, 0], (output_h * 2 // 3, output_w))
    left = resize(flat[:, 1], (output_h // 3, output_w // 2))
    right = resize(flat[:, 2], (output_h // 3, output_w // 2))
    bottom = torch.cat((left, right), dim=-1)
    result = torch.cat((head, bottom), dim=-2)
    return result.reshape(batch, frames, channels, output_h, output_w)


def prepare_diffusion_mosaic_T(
    views: Tensor,
    *,
    input_range: VisualInputRange,
    image_size: tuple[int, int] = (384, 320),
) -> Tensor:
    """Normalize raw views, then resize each view once into the final layout."""

    from worldscape_policy.memory.visual.normalization import normalize_visual

    normalized = normalize_visual(views, input_range=input_range)
    # Antialiased interpolation may overshoot the normalized range by a few
    # ulps around saturated edges, while the VAE input contract is strict.
    return compose_three_view_mosaic_T(normalized, image_size=image_size).clamp_(-1.0, 1.0)


def color_jitter_composed_video(
    video: Tensor,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float = 0.0,
    probability: float = 1.0,
    per_clip_factors: Sequence[tuple[float, float, float, float] | None] | None = None,
) -> Tensor:
    """Jitter selected RGB clips with factors shared across their frames."""

    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError("video must have shape [B,T,3,H,W]")
    for name, strength in (
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ):
        if not 0 <= strength <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0 <= hue <= 0.5:
        raise ValueError("hue must be between 0 and 0.5")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if per_clip_factors is not None and len(per_clip_factors) != video.shape[0]:
        raise ValueError("per_clip_factors must have one entry per sample")
    if per_clip_factors is None and probability == 0:
        return video
    augmented = []
    for index, clip in enumerate(video):
        if per_clip_factors is None:
            if probability < 1 and random.random() >= probability:
                augmented.append(clip)
                continue
            factors = (
                random.uniform(1.0 - brightness, 1.0 + brightness),
                random.uniform(1.0 - contrast, 1.0 + contrast),
                random.uniform(1.0 - saturation, 1.0 + saturation),
                random.uniform(-hue, hue) if hue else 0.0,
            )
        else:
            factors = per_clip_factors[index]
            if factors is None:
                augmented.append(clip)
                continue
        rgb = (clip.float() + 1.0) * 0.5
        rgb = TF.adjust_brightness(rgb, factors[0])
        rgb = TF.adjust_contrast(rgb, factors[1])
        rgb = TF.adjust_saturation(rgb, factors[2])
        if factors[3]:
            rgb = TF.adjust_hue(rgb, factors[3])
        augmented.append(rgb.clamp(0.0, 1.0) * 2.0 - 1.0)
    return torch.stack(augmented, dim=0).to(dtype=video.dtype)


__all__ = [
    "DiffusionViewLayout",
    "validate_diffusion_view_layout",
    "compose_three_view_mosaic",
    "compose_three_view_mosaic_T",
    "color_jitter_composed_video",
    "prepare_diffusion_mosaic",
    "prepare_diffusion_mosaic_T",
]
