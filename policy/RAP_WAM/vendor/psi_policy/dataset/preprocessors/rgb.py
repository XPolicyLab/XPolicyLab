from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from psi_policy.common.normalize_util import get_image_range_normalizer
from psi_policy.dataset.preprocessors.base import BaseObsPreprocessor
from psi_policy.dataset.preprocessors.utils import normalize_image_size


class RGBPreprocessor(BaseObsPreprocessor):
    """将原始时序图像规范化为 [T,C,H,W] 的 float RGB。"""

    def __init__(
        self,
        *,
        image_size: int | Sequence[int] | None = None,
        default_image_size: Sequence[int] | None = None,
    ) -> None:
        self.image_size = normalize_image_size(
            image_size,
            default_image_size=default_image_size,
        )

    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        del obs_shape
        rgb = torch.from_numpy(np.asarray(value))
        if rgb.ndim != 4:
            raise ValueError(f"{key} 期望 4D rgb tensor [T,H,W,C] 或 [T,C,H,W]，实际 {tuple(rgb.shape)}")

        if rgb.shape[-1] in (1, 3, 4):
            rgb = rgb.permute(0, 3, 1, 2)
        elif rgb.shape[1] in (1, 3, 4):
            pass
        else:
            raise ValueError(f"{key} 无法推断通道维，实际 shape={tuple(rgb.shape)}")

        rgb = rgb[:, :3].float()
        if rgb.numel() > 0 and rgb.max() > 1.0:
            rgb = rgb / 255.0

        return F.interpolate(
            rgb,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

    def get_normalizer(self, *, key: str):
        del key
        return get_image_range_normalizer()


__all__ = ["RGBPreprocessor"]
