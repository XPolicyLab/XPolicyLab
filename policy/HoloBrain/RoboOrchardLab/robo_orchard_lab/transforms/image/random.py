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

import random
from typing import Generic, Sequence

import numpy as np
from pydantic import Field
from robo_orchard_core.utils.config import ConfigInstanceOf
from robo_orchard_core.utils.math.transform import (
    Rotate2D,
    Scale2D,
    Shear2D,
    Transform2D_M,
    Translate2D,
)

from robo_orchard_lab.transforms.image.transform2d import (
    ImageTransform2DConfig,
    Transform2DGenMixin,
    Transform2DGenMixinType_co,
)

__all__ = [
    "RandomAffineConfig",
    "RandomApplyTransform2DConfig",
    "RandomChooseTransform2DConfig",
    "RandomImageTransform2DConfig",
]


def get_affine_transform(
    center: tuple[float, float],
    rad: float = 0,
    scale: tuple[float, float] = (1.0, 1.0),
    shear: tuple[float, float] = (0.0, 0.0),
    translation: tuple[float, float] = (0.0, 0.0),
) -> Transform2D_M:
    """Generate an affine transformation matrix."""
    center_ts = Translate2D([-center[0], -center[1]])
    return (
        center_ts
        @ Rotate2D(rad, axis="-Z")
        @ center_ts.inverse()
        @ Scale2D(scale)
        @ Shear2D(shear)
        @ Translate2D(translation)
    )


def _1d_random2range(
    v: float | tuple[float, ...],
    name: str,
    range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Convert a single float to a range tuple."""

    if isinstance(v, (int, float)):
        ret = (-v, v)
    elif isinstance(v, tuple) and len(v) == 2:
        ret = v
    else:
        raise ValueError(
            f"{name} must be a float or a tuple of two floats, got {v}."
        )

    if range is not None:
        if ret[0] < range[0] or ret[1] > range[1]:
            raise ValueError(f"{name} range {ret} is out of bounds {range}.")

    return ret


class RandomAffineConfig(Transform2DGenMixin):
    angle: float | tuple[float, float] = 0.0
    """The range of rotation angle in degrees or a range of angles.

    If a single float is provided, it will be used as (-angle, angle).
    If a tuple is provided, it should be a range of angles (min, max).

    """

    center: tuple[float, float] | None = None
    """The center point of the rotation. If None, the center of the input
    image will be used."""

    scale: tuple[float, float] = (1.0, 1.0)
    """The rangeo of scaling factor or a range of scaling factors.

    The scaling factor will be sampled uniformly from the range
    defined by the tuple.
    """

    shear: float | tuple[float, float] = 0.0
    """The rangeo of shear factor or a range of shear factors.

    If a single float is provided, it will be used as (-shear, shear).
    If a tuple is provided, both X and Y shear will be sampled uniformly
    from the range defined by the tuple.

    """
    translation: tuple[float, float] = (0.0, 0.0)
    """The range of fraction for translation in the X and Y directions.

    The translation will be sampled uniformly from the range defined by
    the tuple, and the values will be multiplied by the input image width
    and height respectively.

    For example, a value of (a, b) means the translation in X direction
    will be sampled from (-a * width, a * width), and the translation in
    Y direction will be sampled from (-b * height, b * height).
    """

    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        if self.center is None:
            center = (input_hw[1] / 2.0, input_hw[0] / 2.0)
        else:
            center = self.center

        angle_range = _1d_random2range(self.angle, "angle")
        scale_range = _1d_random2range(self.scale, "scale")
        shear_range = _1d_random2range(self.shear, "shear")
        translation_range = _1d_random2range(self.translation, "translation")

        angle = random.uniform(angle_range[0], angle_range[1])
        scale = random.uniform(scale_range[0], scale_range[1])
        shear = (
            random.uniform(shear_range[0], shear_range[1]),
            random.uniform(shear_range[0], shear_range[1]),
        )
        translation = (
            random.uniform(-translation_range[0], translation_range[0])
            * input_hw[1],
            random.uniform(-translation_range[1], translation_range[1])
            * input_hw[0],
        )
        ret = get_affine_transform(
            center=center,
            rad=np.deg2rad(angle),
            scale=(scale, scale),
            shear=shear,
            translation=translation,
        )
        return ret


class RandomApplyTransform2DConfig(
    Transform2DGenMixin, Generic[Transform2DGenMixinType_co]
):
    prob: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
    )
    """The probability of applying the transformation."""

    transform: Transform2DGenMixinType_co

    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        if random.random() < self.prob:
            return self.transform.generate_affine_transform(input_hw)
        else:
            # Return an identity transform if not applying the transformation
            return Transform2D_M()


class RandomChooseTransform2DConfig(Transform2DGenMixin):
    transforms: Sequence[ConfigInstanceOf[Transform2DGenMixin]]
    prob: Sequence[float] | None = None

    def __post_init__(self):
        if self.prob is not None and len(self.prob) != len(self.transforms):
            raise ValueError(
                "The length of prob must match the number of transforms."
            )

    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        transform = random.choices(self.transforms, weights=self.prob)[0]
        return transform.generate_affine_transform(input_hw)


class RandomImageTransform2DConfig(
    ImageTransform2DConfig, Generic[Transform2DGenMixinType_co]
):
    transform: Transform2DGenMixinType_co

    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        """Generate the affine transformation matrix."""
        return self.transform.generate_affine_transform(input_hw)
