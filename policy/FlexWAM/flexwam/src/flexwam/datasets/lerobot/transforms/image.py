import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as transforms_F
from PIL import Image


def pil_bilinear_resize(images: torch.Tensor, size_hw) -> torch.Tensor:
    """Stretch [..., 3, H, W] frames to ``size_hw`` with PIL ``Image.BILINEAR``.

    PIL works on uint8, so float input must be in [0, 1]; it is quantized to
    uint8 and returned as float [0, 1]. Training and deployment share this
    function so their pixels match exactly.
    """
    height, width = int(size_hw[0]), int(size_hw[1])
    is_float = images.is_floating_point()
    if is_float:
        frames = (images.detach().cpu().float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
    elif images.dtype == torch.uint8:
        frames = images.detach().cpu()
    else:
        raise TypeError(f"pil_bilinear_resize expects uint8 or float [0,1] input, got {images.dtype}.")
    if frames.ndim < 3 or int(frames.shape[-3]) != 3:
        raise ValueError(f"pil_bilinear_resize expects [...,3,H,W], got shape {tuple(frames.shape)}.")
    lead_shape = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]).permute(0, 2, 3, 1).contiguous().numpy()
    resized = np.stack(
        [np.asarray(Image.fromarray(frame, mode="RGB").resize((width, height), Image.BILINEAR)) for frame in flat],
        axis=0,
    )
    out = torch.from_numpy(resized).permute(0, 3, 1, 2).reshape(*lead_shape, 3, height, width)
    if is_float:
        out = out.to(torch.float32) / 255.0
    return out.to(device=images.device)


class PILBilinearResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = [int(size[0]), int(size[1])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pil_bilinear_resize(x, self.size)


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class VideoColorJitter(nn.Module):
    """Apply one color-jitter sample consistently to a whole multiview clip.

    The processor samples the factors once per dataset item and reuses them for
    every camera and frame. Input and output keep the same tensor dtype and
    shape. ``hue`` is retained for OpenWAM config compatibility but is not
    applied, matching OpenWAM's PIL implementation.
    """

    def __init__(
        self,
        enabled: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.0,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        for name in ("brightness", "contrast", "saturation", "hue"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"`{name}` must be non-negative, got {value}.")

    def sample_factors(self) -> tuple[float, float, float] | None:
        if not self.enabled:
            return None
        return (
            1.0 + random.uniform(-self.brightness, self.brightness),
            1.0 + random.uniform(-self.contrast, self.contrast),
            1.0 + random.uniform(-self.saturation, self.saturation),
        )

    def forward(
        self,
        x: torch.Tensor,
        factors: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.ndim < 4 or int(x.shape[-3]) != 3:
            raise ValueError(
                "VideoColorJitter expects [...,3,H,W], "
                f"got shape {tuple(x.shape)}."
            )
        if factors is None:
            factors = self.sample_factors()
        if factors is None:
            return x
        brightness, contrast, saturation = factors
        original_shape = x.shape
        frames = x.reshape(-1, *x.shape[-3:])
        frames = transforms_F.adjust_brightness(frames, brightness)
        frames = transforms_F.adjust_contrast(frames, contrast)
        frames = transforms_F.adjust_saturation(frames, saturation)
        return frames.reshape(original_shape)
