# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import torch
import torch.nn as nn
from torchvision.transforms import Resize
from typing import Optional

def rgb2hsv_torch(img: torch.Tensor) -> torch.Tensor:
    """[..., 3, H, W] float [0,1] -> [..., 3, H ,W] HSV [0,1]"""
    r, g, b = img.unbind(dim=-3)
    maxc = img.max(dim=-3).values
    minc = img.min(dim=-3).values
    v = maxc
    delta = (maxc - minc).clamp(min=1e-8)
    s = torch.where(maxc > 0, (maxc - minc) / maxc, torch.zeros_like(maxc))
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta
    h = torch.where(r == maxc, bc - gc,
                    torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(maxc == minc, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=-3)

def hsv2rgb_torch(img: torch.Tensor) -> torch.Tensor:
    """[..., 3, H, W] HSV [0,1] → [..., 3, H, W] float [0,1]"""
    h, s, v = img.unbind(dim=-3)
    h6 = h * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
        torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
        torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
        torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack([r, g, b], dim=-3).clamp(0, 1)

def adjust_brightness(channel: torch.Tensor, factor: float, invert: bool = False) -> torch.Tensor:
    if invert:
        factor = -factor
    return (channel + factor).clamp(0, 1)

def adjust_contrast(channel: torch.Tensor, factor: float, invert: bool = False) -> torch.Tensor:
    if invert:
        factor = -factor
    mean = channel.mean()
    return ((channel - mean) * (1 + factor) + mean).clamp(0, 1)


class ColorJitter(nn.Module):
    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.1,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.brightness = brightness if brightness > 0 else None
        self.contrast = contrast if contrast > 0 else None
        self.saturation = saturation if saturation > 0 else None
        self.hue = hue if hue > 0 else None
        self.probability = p

    @staticmethod
    def get_params(
        brightness: Optional[float],
        contrast: Optional[float],
        saturation: Optional[float],
        hue: Optional[float],
    ):
        b = None if brightness is None else float(torch.empty(1).uniform_(-brightness, brightness))
        c = None if contrast is None else float(torch.empty(1).uniform_(-contrast, contrast))
        s = None if saturation is None else float(torch.empty(1).uniform_(-saturation, saturation))
        h = None if hue is None else float(torch.empty(1).uniform_(-hue, hue))
        return b, c, s, h

    def forward(self, img, invert=False):
        brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        fn_idx = [0, 1, 2, 3]
        hsv = rgb2hsv_torch(img[None, ...])[0]
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                hsv[2:] = adjust_brightness(hsv[2:], brightness_factor, invert)
            elif fn_id == 1 and contrast_factor is not None:
                hsv[2:] = adjust_contrast(hsv[2:], contrast_factor, invert)
            elif fn_id == 2 and hue_factor is not None:
                if invert:
                    hue_factor = -hue_factor
                hsv[0:1] += hue_factor
            elif fn_id == 3 and saturation_factor is not None:
                # official pi0 do this, maybe bug
                adjust_brightness(hsv[1:2], saturation_factor, invert)
        transformed_img = hsv2rgb_torch(hsv[None, ...])[0]

        if self.probability < 1:
            do_apply = torch.bernoulli(torch.tensor(self.probability)).bool()
            transformed_img = torch.where(do_apply, transformed_img, img)

        return transformed_img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
            f", probability={self.probability}"
        )
        return s


def process_images(
    images: dict[str, torch.Tensor],
    resize: Resize,
    jitter: Optional[ColorJitter] = None,
    augment: bool = False,
) -> dict[str, torch.Tensor]:
    processed = {}
    for key, img in images.items():
        img = resize(img)
        if augment:
            if jitter is not None:
                img = jitter(img)
        processed[key] = img

    return processed