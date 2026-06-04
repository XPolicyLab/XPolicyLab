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
from abc import ABCMeta, abstractmethod
from typing import Generic, Literal, TypeVar

from robo_orchard_core.datatypes.camera_data import ImageChannelLayout
from robo_orchard_core.utils.math.transform import Transform2D_M

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchImageData,
)
from robo_orchard_lab.transforms.base import (
    ClassType,
    Config,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "ImageTransform2D",
    "ImageTransform2DConfig",
    "ImageTransform2DConfigType_co",
    "Transform2DGenMixin",
    "Transform2DGenMixinType_co",
]

ImageTransform2DConfigType_co = TypeVar(
    "ImageTransform2DConfigType_co",
    bound="ImageTransform2DConfig",
    covariant=True,
)


BatchImageDataType = TypeVar(
    "BatchImageDataType", bound=BatchImageData | BatchCameraData
)


class ImageTransform2D(DictTransform, Generic[ImageTransform2DConfigType_co]):
    cfg: ImageTransform2DConfigType_co
    is_variadic: bool = True

    def __init__(self, cfg: ImageTransform2DConfigType_co) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(
        self, **kwargs: dict[str, BatchImageDataType]
    ) -> dict[str, BatchImageDataType]:
        """Apply the affine transformation to the image data."""
        ret = {}
        if self.cfg.batching_groups is not None:
            for group in self.cfg.batching_groups:
                if not all(k in kwargs for k in group):
                    raise ValueError(
                        f"Not all keys in group {group} are "
                        "present in the input."
                    )
                group_data = {k: kwargs.pop(k) for k in group}
                ret.update(self.transform_batch(**group_data))

        for key, value in kwargs.items():
            if isinstance(value, (BatchCameraData, BatchImageData)):
                ret[key] = self.transform_single(value)
            else:
                raise TypeError(
                    f"Expected BatchCameraData for key '{key}', "
                    f"but got {type(value)}."
                )
        return ret

    def transform_single(
        self, value: BatchImageDataType
    ) -> BatchImageDataType:
        if value.channel_layout != ImageChannelLayout.HWC:
            raise ValueError(
                f"Expected input layout {ImageChannelLayout.HWC}, "
                f"but got {value.channel_layout}."
            )
        src_hw: tuple[int, int] = value.sensor_data.shape[-3:-1]  # type: ignore
        target_hw = self.cfg.target_hw
        ts = self.cfg.generate_affine_transform(src_hw)
        # Apply the affine transformation
        return value.apply_transform2d(
            ts,
            target_hw=target_hw,
            inter_mode=self.cfg.inter_mode,
        )

    def transform_batch(
        self, **kwargs: dict[str, BatchImageDataType]
    ) -> dict[str, BatchImageDataType]:
        """Apply the same affine transformation to all inputs."""

        src_hw: tuple[int, int] = (0, 0)
        # check all should have same shape
        for key, value in kwargs.items():
            if not isinstance(value, (BatchCameraData, BatchImageData)):
                raise TypeError(
                    f"Expected BatchCameraData or BatchImageData for key"
                    f"'{key}', but got {type(value)}."
                )
            if value.channel_layout != ImageChannelLayout.HWC:
                raise ValueError(
                    f"Expected input layout {ImageChannelLayout.HWC}, "
                    f"but got {value.channel_layout} for {key}."
                )
            if src_hw == (0, 0):
                src_hw = value.sensor_data.shape[-3:-1]  # type: ignore
            elif src_hw != value.sensor_data.shape[-3:-1]:  # type: ignore
                raise ValueError(
                    f"All images must have the same shape, but got {src_hw} "
                    f"and {value.sensor_data.shape[-3:-1]} for key '{key}'."
                )

        ts = self.cfg.generate_affine_transform(src_hw)
        target_hw = self.cfg.target_hw

        ret = {}
        for key, value in kwargs.items():
            assert isinstance(value, (BatchCameraData, BatchImageData))
            ret[key] = value.apply_transform2d(
                ts,
                target_hw=target_hw,
                inter_mode=self.cfg.inter_mode,
            )
        return ret


class Transform2DGenMixin(Config, metaclass=ABCMeta):
    @abstractmethod
    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )


Transform2DGenMixinType_co = TypeVar(
    "Transform2DGenMixinType_co", bound=Transform2DGenMixin, covariant=True
)


class ImageTransform2DConfig(
    DictTransformConfig[ImageTransform2D], Transform2DGenMixin
):
    class_type: ClassType[ImageTransform2D] = ImageTransform2D

    target_hw: tuple[int, int] | list[int]
    """The target height and width of the image after transformation."""

    inter_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
    """The interpolation mode to use for the transformation."""

    padding_mode: Literal["zeros"] = "zeros"

    # batching_all: bool = False
    # """Whether to apply the transformation to all inputs using the same transform."""  # noqa: E501

    batching_groups: list[list[str]] | None = None
    """Whether to apply the same transformation to a group of inputs.

    If None, the transformation will be applied to each input separately.
    If a list of lists is provided, each inner list will be treated as a group
    of inputs that should be transformed together.
    """

    @abstractmethod
    def generate_affine_transform(
        self, input_hw: tuple[int, int] | list[int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )
