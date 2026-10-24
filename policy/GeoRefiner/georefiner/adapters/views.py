"""View adapter for lightweight RGB preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class ViewAdapterOutput:
    """Adapted global and wrist RGB views."""

    global_rgb: Tensor
    wrist_rgb: Tensor


class ViewAdapter(nn.Module):
    """Validate, optionally resize, and optionally normalize two RGB views.

    This adapter does not extract visual features. In pretrained mode it must be
    a no-op because DINOv2 owns resize, rescale, and normalization.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        normalize: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if image_size is not None:
            if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
                raise ValueError(f"image_size must be a positive (H, W); got {image_size}.")

        self.image_size = image_size
        self.normalize = normalize
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain 3 values.")
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        if torch.any(std_tensor <= 0):
            raise ValueError("std values must be positive.")
        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor, persistent=False)

    def forward(self, global_rgb: Tensor, wrist_rgb: Tensor) -> ViewAdapterOutput:
        self._validate_rgb("global_rgb", global_rgb)
        self._validate_rgb("wrist_rgb", wrist_rgb)
        if global_rgb.shape[0] != wrist_rgb.shape[0]:
            raise ValueError(
                "global_rgb and wrist_rgb batch sizes must match; "
                f"got {global_rgb.shape[0]} and {wrist_rgb.shape[0]}."
            )

        return ViewAdapterOutput(
            global_rgb=self._process(global_rgb),
            wrist_rgb=self._process(wrist_rgb),
        )

    def _process(self, image: Tensor) -> Tensor:
        output = image
        if self.image_size is not None and tuple(output.shape[-2:]) != self.image_size:
            output = F.interpolate(
                output,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        if self.normalize:
            mean = self.mean.to(device=output.device, dtype=output.dtype)
            std = self.std.to(device=output.device, dtype=output.dtype)
            output = (output - mean) / std
        return output

    @staticmethod
    def _validate_rgb(name: str, image: Tensor) -> None:
        if image.ndim != 4:
            raise ValueError(f"{name} must have shape [B, 3, H, W]; got {tuple(image.shape)}.")
        if image.shape[1] != 3:
            raise ValueError(f"{name} must have 3 channels; got shape {tuple(image.shape)}.")
        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError(f"{name} spatial dimensions must be positive.")
