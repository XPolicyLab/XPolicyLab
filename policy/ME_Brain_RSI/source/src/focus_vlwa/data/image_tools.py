"""Torch image operations used by Focus-VLWA."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize channels-first or channels-last images while preserving aspect ratio."""
    original_rank = images.ndim
    channels_last = images.shape[-1] <= 4
    if original_rank == 3:
        images = images.unsqueeze(0)
    if channels_last:
        images = images.permute(0, 3, 1, 2)

    _, _, current_height, current_width = images.shape
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = functional.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized = resized.round().clamp(0, 255).to(torch.uint8)
        fill_value = 0
    elif images.is_floating_point():
        resized = resized.clamp(-1.0, 1.0)
        fill_value = -1.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    top, extra_height = divmod(height - resized_height, 2)
    left, extra_width = divmod(width - resized_width, 2)
    resized = functional.pad(
        resized,
        (left, left + extra_width, top, top + extra_height),
        mode="constant",
        value=fill_value,
    )
    if channels_last:
        resized = resized.permute(0, 2, 3, 1)
    if original_rank == 3:
        resized = resized.squeeze(0)
    return resized


def resize_with_pad_pil(image, height, width):
    """Match the reference wrist-camera PIL bilinear letterbox."""
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.asarray(image))
    ratio = max(image.width / width, image.height / height)
    resized = image.resize((int(image.width / ratio), int(image.height / ratio)), Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, (width, height), 0)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas)


def resize_with_pad(image, height, width):
    """Match the reference head-camera antialiased linear resize without JAX."""
    import numpy as np

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Head-frame resize requires an HWC uint8 image")
    old_height, old_width = image.shape[:2]
    ratio = max(old_width / width, old_height / height)
    new_height, new_width = int(old_height / ratio), int(old_width / ratio)

    def weights(old_size, new_size):
        scale = np.float32(new_size / old_size)
        sample = (np.arange(new_size, dtype=np.float32) + np.float32(0.5)) / scale - np.float32(0.5)
        distance = np.abs(sample[None, :] - np.arange(old_size, dtype=np.float32)[:, None])
        kernel = np.maximum(np.float32(0), np.float32(1) - distance * np.minimum(scale, np.float32(1)))
        return kernel / kernel.sum(axis=0, keepdims=True)

    values = image.astype(np.float32)
    if old_height != new_height:
        values = np.einsum("hwc,hy->ywc", values, weights(old_height, new_height))
    if old_width != new_width:
        values = np.einsum("hwc,wx->hxc", values, weights(old_width, new_width))
    resized = np.rint(values).clip(0, 255).astype(np.uint8)
    output = np.zeros((height, width, image.shape[-1]), dtype=np.uint8)
    top, left = (height - new_height) // 2, (width - new_width) // 2
    output[top:top + new_height, left:left + new_width] = resized
    return output
