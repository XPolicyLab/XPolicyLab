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
from typing import Generic, Literal

import cv2
import numpy as np
import torch
from robo_orchard_core.utils.math.transform import (
    Rotate2D,
    Scale2D,
    Transform2D_M,
    Translate2D,
)

from robo_orchard_lab.transforms.image.transform2d import (
    ImageTransform2DConfig,
    Transform2DGenMixinType_co,
)

__all__ = [
    "ImageResizeConfig",
    "ImageRotateConfig",
]


def _get_affine_transform(
    pts_before: np.ndarray, pts_after: np.ndarray
) -> Transform2D_M:
    """Calculates an affine transform from two point sets.

    Args:
        pts_before (np.ndarray): The points before transformation.
            It should be in the shape of (3, 2)
        pts_after (np.ndarray): The points after transformation.
            It should be in the shape of (3, 2)

    Returns:
        Transform2D_M: The affine transformation matrix.

    """
    assert isinstance(pts_before, np.ndarray)
    assert isinstance(pts_after, np.ndarray)
    assert pts_before.shape == (3, 2)
    assert pts_after.shape == (3, 2)
    mat = cv2.getAffineTransform(
        pts_before.astype(np.float32), pts_after.astype(np.float32)
    )

    # M is a 2x3 matrix, we need to convert it to a 3x3 matrix
    mat_3x3 = np.eye(3, dtype=np.float32)
    mat_3x3[:2, :] = mat
    return Transform2D_M(matrix=torch.asarray(mat_3x3))


def rotate(
    center: tuple[float, float],
    rad: float,
    axis: Literal["Z", "-Z"] = "Z",
) -> Transform2D_M:
    """Create a rotation transformation around a center point.

    Args:
        center (tuple[float, float]): The center point of the rotation.
        rad (float): The rotation angle in radians.
    """
    t = Translate2D([-center[0], -center[1]])
    r = Rotate2D(rad, axis=axis)
    return t @ r @ t.inverse()


def aspect_ratio(center: tuple[float, float], x_ratio: float) -> Transform2D_M:
    """Create a transformation that apply aspect ratio scaling.

    Args:
        center (tuple[float, float]): The center point of the scaling.
        x_ratio (float): The scaling ratio for the x-axis.
    """
    t = Translate2D([-center[0], -center[1]])
    s = Scale2D([x_ratio, 1.0 / x_ratio])
    return t @ s @ t.inverse()


def resize(
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    align_type: Literal[
        "center", "left-top", "left-bottom", "full"
    ] = "center",
    align_scale: Literal["max", "min"] = "max",
) -> Transform2D_M:
    """Create a resize transformation from source to destination size.

    The scale is calculated based on the source and destination sizes. When
    the aspect ratio of the source and destination sizes are different,
    the transformation will align the image according to the specified
    alignment type. The scale is determined by either the maximum or minimum
    scale factor, depending on the `align_scale` parameter.

    If the aspect ratio of the source and destination sizes are different,
    the transformation will align the image using the max or min scale
    according to the specified alignment type:

    - "center": The image will be centered in the destination size.
    - "left-top": The image will be aligned to the left-top corner of the
      destination size.
    - "left-bottom": The image will be aligned to the left-bottom corner of
      the destination size.
    - "full": The image will be stretched to fill the entire destination size.
      This will not preserve the aspect ratio.

    Args:
        src_hw (tuple[int, int]): The source height and width.
        dst_hw (tuple[int, int]): The destination height and width.
        align_type (Literal["center", "left-top", "left-bottom", "full"]):
            The alignment type for the transformation. Defaults to "center".
    """
    s_h, s_w = src_hw
    t_h, t_w = dst_hw
    if align_scale == "max":
        scale = max(t_w / float(s_w), t_h / float(s_h))
    else:
        scale = min(t_w / float(s_w), t_h / float(s_h))
    r_w, r_h = (int(s_w * scale), int(s_h * scale))

    if align_type == "center":
        # use left-top, right-top, center as reference points
        ptr_before = np.array([[0, 0], [s_w, 0], [s_w / 2.0, s_h / 2.0]])
        ptr_after = np.array(
            [
                [t_w / 2.0 - r_w / 2.0, t_h / 2.0 - r_h / 2.0],
                [t_w / 2.0 + r_w / 2.0, t_h / 2.0 - r_h / 2.0],
                [t_w / 2.0, t_h / 2.0],
            ]
        )
    elif align_type == "left-top":
        # use left-top, right-top, left-bottom as reference points
        ptr_before = np.array([[0, 0], [s_w, 0], [0, s_h]])
        ptr_after = np.array([[0, 0], [r_w, 0], [0, r_h]])
    elif align_type == "left-bottom":
        # use left-bottom, right-bottom, left-top as reference points
        ptr_before = np.array([[0, s_h], [s_w, s_h], [0, 0]])
        ptr_after = np.array([[0, t_h], [r_w, t_h], [0, t_h - r_h]])
    elif align_type == "full":
        ptr_before = np.array([[0, 0], [s_w, 0], [0, s_h]])
        ptr_after = np.array([[0, 0], [t_w, 0], [0, t_h]])
    else:
        raise NotImplementedError(f"not supported align_type: {align_type}")

    return _get_affine_transform(ptr_before, ptr_after)


class ImageResizeConfig(
    ImageTransform2DConfig, Generic[Transform2DGenMixinType_co]
):
    align_type: Literal["center", "left-top", "left-bottom", "full"] = "center"
    """The alignment type for the resizing transformation.

    - "center": The image will be centered in the destination size.
    - "left-top": The image will be aligned to the left-top corner of the
      destination size.
    - "left-bottom": The image will be aligned to the left-bottom corner of
      the destination size.
    - "full": The image will be stretched to fill the entire destination size.
      This will not preserve the aspect ratio.
    """
    align_scale: Literal["max", "min"] = "max"
    """The scale factor to use if the aspect ratio of the source and
    destination sizes are different.
    """

    augmentation: Transform2DGenMixinType_co | None = None
    """The augmentation to apply to the image after resizing."""

    def generate_affine_transform(
        self, input_hw: tuple[int, int] | list[int]
    ) -> Transform2D_M:
        ret = resize(
            src_hw=input_hw,
            dst_hw=self.target_hw,
            align_type=self.align_type,
            align_scale=self.align_scale,
        )
        if self.augmentation is not None:
            # Apply the augmentation if provided
            ret = self.augmentation.generate_affine_transform(input_hw) @ ret

        return ret


class ImageRotateConfig(
    ImageTransform2DConfig, Generic[Transform2DGenMixinType_co]
):
    angle: float = 0.0
    """The rotation angle in degree."""

    center: tuple[float, float] | None = None
    """The center point of the rotation."""

    augmentation: Transform2DGenMixinType_co | None = None
    """The augmentation to apply to the image after resizing."""

    def generate_affine_transform(
        self, input_hw: tuple[int, int]
    ) -> Transform2D_M:
        angle = np.deg2rad(self.angle)
        if self.center is None:
            center = (input_hw[1] / 2.0, input_hw[0] / 2.0)
        else:
            center = self.center
        ret = rotate(center=center, rad=angle, axis="-Z")

        if self.augmentation is not None:
            # Apply the augmentation if provided
            ret = self.augmentation.generate_affine_transform(input_hw) @ ret

        return ret
