"""Tri-view composite assembly and PIL conversion, shared by training and serving.

Upstream kept two copies of this logic: a deployment-safe module that the
simulator and the policy server imported, and a thin re-export inside the
training dataloader package so that simulator-only environments would not
execute the training package initializer.  With a single package there is no
such split any more, so both copies collapse into this module and training,
evaluation and the policy server are guaranteed to see identical pixels.

``to_pil_preserve`` comes from the policy server's image tools and is kept here
for the same reason: the model's example-to-PIL conversion must be the one the
dataloader produced arrays for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

FASTWAM_COMPOSITE_LAYOUT = "fastwam_composite"
FASTWAM_COMPOSITE_VIEW_KEY = "video.fastwam_composite"
TRI_VIEW_COMPOSITE_LAYOUT = "tri_view_composite"
TRI_VIEW_COMPOSITE_VIEW_KEY = "video.tri_view_composite"
# PIL/config order is (width, height); numpy/tensor order is (height, width).
TRI_VIEW_COMPOSITE_SIZE = (320, 384)
TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]


def _to_chw_float(image: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image, copy=True)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image with 1/3/4 channels, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    tensor = torch.from_numpy(array[..., :3]).permute(2, 0, 1).to(torch.float32)
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp_(0.0, 1.0)


def build_tri_view_composite(images: Sequence[Image.Image | np.ndarray]) -> Image.Image:
    """Reproduce the two-stage three-camera layout exactly.

    Every camera is first normalized to 240x320, then the head view is resized
    to 256x320 and the two wrist views to 128x160 and stitched side by side
    underneath it.  The intermediate 240x320 step is not redundant: it is what
    the released checkpoints were trained against, and skipping it changes the
    resampling kernel's support and therefore the pixels.
    """

    if len(images) != 3:
        raise ValueError(f"Tri-view composite requires [head, left_wrist, right_wrist], got {len(images)} views")
    frames = [
        tvf.resize(
            _to_chw_float(image),
            [240, 320],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        for image in images
    ]
    head = tvf.resize(frames[0], [256, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    left = tvf.resize(frames[1], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    right = tvf.resize(frames[2], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    composite = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
    array = composite.mul(255.0).round().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def build_per_view_images(
    images: Sequence[Image.Image | np.ndarray],
    size: Sequence[int],
) -> list[Image.Image]:
    """Resize each camera on its own, for a planner that reads separate images.

    Shares ``_to_chw_float`` and the same bilinear+antialias resize as
    :func:`build_tri_view_composite`, so training and evaluation see identical
    pixels here too.  ``size`` is (width, height) to match ``obs_image_size``
    and PIL, while ``tvf.resize`` wants (height, width).
    """

    width, height = (int(value) for value in size)
    resized = []
    for image in images:
        frame = tvf.resize(
            _to_chw_float(image),
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        array = (
            frame.mul(255.0)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        resized.append(Image.fromarray(array, mode="RGB"))
    return resized


def to_pil_preserve(images: Any, scale_float: bool = True):
    """
    Convert (possibly nested) numpy image arrays back to PIL.Image WITHOUT changing spatial shape
    or nesting structure.

    Accepts:
      - np.ndarray with shape (H, W, C), C in {1,3,4}, dtype uint8 or float
      - PIL.Image.Image (returned as-is)
      - Nested list / tuple structures containing the above

    Guarantees:
      - No resize / pad / crop performed
      - Returns an object with the SAME nesting layout (list -> list, tuple -> tuple)
      - Only dtype (float -> uint8) and channel-mode adaptation may happen
        * float arrays assumed in [0,1] if scale_float=True (scaled *255 + clip)
    Args:
      images: input object / sequence
      scale_float: whether to scale float images in [0,1] to uint8
    Returns:
      Mirrored structure with all leaf nodes as PIL.Image.Image
    """

    def _convert(obj):
        # Nested containers
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)

        # PIL stays
        if isinstance(obj, Image.Image):
            return obj

        # numpy -> PIL
        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D array (H,W,C), got shape={arr.shape}")
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Channel count must be 1/3/4, got {arr.shape[2]}")
            if np.issubdtype(arr.dtype, np.floating):
                if scale_float:
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    raise TypeError("Float array provided but scale_float=False")
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # Single channel -> 'L'
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
                return Image.fromarray(arr, mode="L")
            # 3 channels -> RGB, 4 -> RGBA
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            return Image.fromarray(arr, mode=mode)

        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)


__all__ = [
    "FASTWAM_COMPOSITE_LAYOUT",
    "FASTWAM_COMPOSITE_VIEW_KEY",
    "TRI_VIEW_COMPOSITE_LAYOUT",
    "TRI_VIEW_COMPOSITE_SIZE",
    "TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS",
    "TRI_VIEW_COMPOSITE_VIEW_KEY",
    "build_per_view_images",
    "build_tri_view_composite",
    "to_pil_preserve",
]
