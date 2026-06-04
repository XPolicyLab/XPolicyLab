# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from typing import TypeVar

import torch
from robo_orchard_core.datatypes.camera_data import (
    ImageChannelLayout,
)
from torchvision.transforms import ColorJitter as TorchColorJitter

from robo_orchard_lab.dataset.datatypes import BatchCameraData, BatchImageData
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "ColorJitter",
    "ColorJitterConfig",
]

BatchImageDataType = TypeVar(
    "BatchImageDataType", bound=BatchImageData | BatchCameraData
)


class ColorJitter(DictTransform[dict[str, BatchImageData | BatchCameraData]]):
    cfg: ColorJitterConfig

    def __init__(self, cfg: ColorJitterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._color_jitter = TorchColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        )

    def transform(
        self, **kwargs: BatchImageData | BatchCameraData
    ) -> dict[str, BatchImageData | BatchCameraData]:
        """Apply color jitter to the input columns."""
        if self.cfg.batching_all:
            return self._transform_together(**kwargs)
        else:
            return self._transform_seperately(**kwargs)

    def _transform_seperately(
        self, **kwargs: BatchImageData | BatchCameraData
    ) -> dict[str, BatchImageData | BatchCameraData]:
        """Apply color jitter to the input columns."""
        ret = {}
        for k, v in kwargs.items():
            if isinstance(v, (BatchCameraData, BatchImageData)):
                ret[k] = self._impl(v, k)
            else:
                raise TypeError(
                    "Expected BatchCameraData or BatchImageData "
                    f"for column '{k}', but got {type(v).__name__}."
                )
        return ret

    def _transform_together(
        self, **kwargs: BatchImageData | BatchCameraData
    ) -> dict[str, BatchImageData | BatchCameraData]:
        """Apply color jitter to the input columns."""
        ret = {}
        all_images: list[torch.Tensor] = []
        all_batchsize: list[int] = []

        for k, v in kwargs.items():
            if isinstance(v, (BatchCameraData, BatchImageData)):
                if v.channel_layout != ImageChannelLayout.CHW:
                    raise ValueError(
                        f"Expected channel layout {ImageChannelLayout.CHW}, "
                        f"but got {v.channel_layout} for {k}"
                    )
                all_images.append(v.sensor_data)
                all_batchsize.append(v.sensor_data.shape[0])
                if (
                    len(all_images) > 1
                    and all_images[-1].shape != all_images[0].shape
                ):
                    raise ValueError(
                        "All images must have the same shape for "
                        "batch processing."
                    )

            else:
                raise TypeError(
                    "Expected BatchCameraData or BatchImageData "
                    f"for column '{k}', but got {type(v).__name__}."
                )
        # concatenate all images
        all_tensor = self._color_jitter(torch.concat(all_images, dim=0))
        # split back to original batch sizes
        start = 0
        for i, k in enumerate(kwargs.keys()):
            batchsize = all_batchsize[i]
            end = start + batchsize
            in_data: BatchImageDataType = kwargs[k]  # type: ignore
            out_data = in_data.model_copy(deep=False)
            out_data.sensor_data = all_tensor[start:end]
            start = end
            ret[k] = out_data

        if start != all_tensor.shape[0]:
            raise ValueError(
                "Batch size mismatch after processing. "
                f"Expected {start}, but got {all_tensor.shape[0]}."
            )

        return ret

    def _impl(self, data: BatchImageDataType, name: str) -> BatchImageDataType:
        """Apply color jitter to the images in the batch."""
        if data.channel_layout != ImageChannelLayout.CHW:
            raise ValueError(
                f"Expected channel layout {ImageChannelLayout.CHW}, "
                f"but got {data.channel_layout} for {name}"
            )
        if data.sensor_data.shape[-3] not in (1, 3):
            raise ValueError(
                "Expected 1 or 3 channels, but got "
                f"{data.sensor_data.shape[-3]} for {name}."
            )
        out = data.model_copy(deep=False)
        out.sensor_data = self._color_jitter(data.sensor_data)

        return out


class ColorJitterConfig(DictTransformConfig[ColorJitter]):
    """Configuration class for ColorJitter transform."""

    class_type: ClassType[ColorJitter] = ColorJitter

    batching_all: bool = False
    """Whether to apply color jitter to all images in the batch together.

    If True, all images in the batch will be processed together.
    If False, each image will be processed separately.
    """

    brightness: float | tuple[float, float] = 0
    """The brightness adjustment factor.

    See `torchvision.transforms.ColorJitter` for details.
    """
    contrast: float | tuple[float, float] = 0
    """The contrast adjustment factor.

    See `torchvision.transforms.ColorJitter` for details.
    """
    saturation: float | tuple[float, float] = 0
    """The saturation adjustment factor.

    See `torchvision.transforms.ColorJitter` for details.
    """
    hue: float | tuple[float, float] = 0
    """The hue adjustment factor.

    See `torchvision.transforms.ColorJitter` for details.
    """
