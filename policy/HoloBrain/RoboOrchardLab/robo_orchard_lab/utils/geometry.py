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

from typing import Optional, Tuple

import numpy as np

__all__ = ["depth_to_range_image", "mask_points"]


def depth_to_range_image(
    depth: np.ndarray, camera_intrinsic: np.ndarray, depth_scale: float = 1.0
) -> np.ndarray:
    """Depth to range image.

    Converts a depth map to a range image (3D point cloud representation)
    using the camera's intrinsic parameters.

    Args:
        depth (np.ndarray): Depth map with shape (H, W), where each element
            represents depth in the image plane.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix with shape
            (3, 3). Contains focal lengths and optical center:
            [[fx,  0, cx], [0, fy, cy], [0,  0,  1]]
        depth_scale (float): Scale factor to convert raw depth values to
            metric units. For example, if the depth map values are in
            millimeters and you want meters, set `depth_scale=1000.0`.

    Returns:
        np.ndarray: Range image (3D point cloud) with shape (H, W, 3).
            Each pixel contains an (x, y, z) coordinate in the camera's
            coordinate system, corresponding to the 3D position of the
            point in space.

    Example:
        >>> depth = np.random.rand(480, 640) * 1000  # Depth map in millimeters
        >>> camera_intrinsic = np.array(
        ...     [[600, 0, 320], [0, 600, 240], [0, 0, 1]]
        ... )
        >>> range_image = depth_to_range_image(
        ...     depth, camera_intrinsic, depth_scale=1000.0
        ... )
        >>> print(range_image.shape)
        (480, 640, 3)

    Notes:
        - The function assumes the depth map is aligned with the camera's
          coordinate system and that each pixel's depth value represents
          the distance from the camera plane.
        - The `depth_scale` parameter allows flexibility with depth data
          formats and should be set according to the depth sensor's scale.
    """  # noqa

    if camera_intrinsic.shape != (3, 3):
        raise ValueError(
            "camera_intrinsic should with shape (3, 3), but get shape = {}".format(  # noqa: E501
                camera_intrinsic.shape
            )
        )

    fx = camera_intrinsic[0, 0]
    cx = camera_intrinsic[0, 2]
    fy = camera_intrinsic[1, 1]
    cy = camera_intrinsic[1, 2]

    xmap, ymap = np.meshgrid(
        np.arange(depth.shape[1]), np.arange(depth.shape[0])
    )

    z = depth / depth_scale
    x = (xmap - cx) * z / fx
    y = (ymap - cy) * z / fy

    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud


FLOAT_T = Optional[float]

POINTS_LIMIT_TYPE = Tuple[FLOAT_T, FLOAT_T, FLOAT_T, FLOAT_T, FLOAT_T, FLOAT_T]
POINTS_BORDER_FLAG_TYPE = Tuple[bool, bool, bool, bool, bool, bool]


def mask_points(
    data: np.ndarray,
    workspace_limits: POINTS_LIMIT_TYPE,
    border_flags: POINTS_BORDER_FLAG_TYPE,
) -> np.ndarray:
    """Mask point clouds based on specified boundaries.

    Args:
        data (np.ndarray): The input point cloud with shape (..., 3),
            where the last dimension represents the x, y, z coordinates.
        workspace_limits (Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]): The workspace
            limits in the format (min_x, max_x, min_y, max_y, min_z, max_z).
        border_flags (Tuple[bool, bool, bool, bool, bool, bool]): Boolean flags specifying if each limit is inclusive
            (True) or exclusive (False).

    Returns:
        np.ndarray: A mask array with the same spatial shape as the input
            data's preceding dimensions (e.g., (H, W) if input is (H, W, 3)),
            where True indicates the points within the workspace limits.

    """  # noqa

    # Ensure the input data has the shape (..., 3)
    if data.shape[-1] != 3:
        raise ValueError("Input data must have shape (..., 3)")

    # Initialize mask with True values
    mask = np.ones(data.shape[:-1], dtype=bool)

    # Iterate over each dimension and apply limits if they are
    # specified (not None)
    for i, (min_limit, max_limit, include_min, include_max) in enumerate(
        zip(
            workspace_limits[::2],
            workspace_limits[1::2],
            border_flags[::2],
            border_flags[1::2],
            strict=False,
        )
    ):
        # Determine which axis to apply, i.e., x, y, or z (0, 1, or 2)
        axis = i

        # Apply minimum limit if specified
        if min_limit is not None:
            if include_min:
                mask &= data[..., axis] >= min_limit
            else:
                mask &= data[..., axis] > min_limit

        # Apply maximum limit if specified
        if max_limit is not None:
            if include_max:
                mask &= data[..., axis] <= max_limit
            else:
                mask &= data[..., axis] < max_limit

    return mask
