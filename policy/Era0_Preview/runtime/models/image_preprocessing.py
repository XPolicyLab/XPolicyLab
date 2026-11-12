from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


IMAGE_RESIZE_STRETCH = "stretch"
IMAGE_RESIZE_LETTERBOX = "letterbox"
QWEN_POLICY_PAD_RGB = (128, 128, 128)


def normalize_image_resize_mode(
    mode: str | None,
    *,
    vlm_backend: str,
) -> str:
    normalized = str(mode or "auto").strip().lower()
    backend = str(vlm_backend or "florence2").strip().lower()
    if normalized == "auto":
        return (
            IMAGE_RESIZE_LETTERBOX
            if backend == "qwen3_5"
            else IMAGE_RESIZE_STRETCH
        )
    if normalized not in {IMAGE_RESIZE_STRETCH, IMAGE_RESIZE_LETTERBOX}:
        raise ValueError(
            "image_resize_mode must be auto, stretch, or letterbox; "
            f"got {mode!r}."
        )
    if backend == "qwen3_5" and normalized != IMAGE_RESIZE_LETTERBOX:
        raise ValueError(
            "The formal Qwen3.5 policy requires aspect-preserving letterbox "
            "preprocessing."
        )
    return normalized


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(
                "Tensor image must be [C,H,W] or [H,W,C], got "
                f"{tuple(tensor.shape)}."
            )
        if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)
        return tvf.to_pil_image(tensor).convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(
            "Array image must be [H,W,C], got "
            f"{tuple(array.shape)}."
        )
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def letterbox_image(
    image: Any,
    *,
    size: tuple[int, int],
    fill: tuple[int, int, int] = QWEN_POLICY_PAD_RGB,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> Image.Image:
    """Resize without geometric distortion and pad to an exact HxW canvas."""
    target_h, target_w = (int(size[0]), int(size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Letterbox target must be positive, got {size}.")
    pil = _to_pil_rgb(image)
    source_w, source_h = pil.size
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Image has invalid size {pil.size}.")
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = min(target_w, max(1, int(round(source_w * scale))))
    resized_h = min(target_h, max(1, int(round(source_h * scale))))
    resized = tvf.resize(
        pil,
        [resized_h, resized_w],
        interpolation=interpolation,
        antialias=True,
    )
    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = pad_w // 2
    top = pad_h // 2
    return tvf.pad(
        resized,
        [left, top, pad_w - left, pad_h - top],
        fill=fill,
    )


@dataclass(frozen=True)
class LetterboxResize:
    size: tuple[int, int]
    fill: tuple[int, int, int] = QWEN_POLICY_PAD_RGB
    interpolation: InterpolationMode = InterpolationMode.BICUBIC

    def __call__(self, image: Any) -> Image.Image:
        return letterbox_image(
            image,
            size=self.size,
            fill=self.fill,
            interpolation=self.interpolation,
        )
