"""Rigid-pose operations using scalar-first quaternions."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from math import sqrt
import numpy as np

def quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        values = [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
        elif axis == 1:
            scale = sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = [(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = [(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
    norm = sqrt(sum((value * value for value in values)))
    return [float(value / norm) for value in values]

def rotation_from_quaternion(quaternion: Sequence[float], *, error_type=ValueError) -> np.ndarray:
    w, x, y, z = quaternion
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise error_type('quaternion must be nonzero')
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)

def grasp_pose_to_end_effector(robot: Any, pose: Sequence[float], *, rotation_fn=rotation_from_quaternion, quaternion_fn=quaternion_from_matrix) -> list[float]:
    rotation = rotation_fn(pose[3:])
    position = np.asarray(pose[:3], dtype=float)
    position += rotation @ np.asarray([0.12 - float(robot.gripper_bias), 0.0, 0.0])
    rotation = rotation @ np.asarray(robot.inv_delta_matrix, dtype=float).reshape(3, 3)
    return position.tolist() + quaternion_fn(rotation)

def compose_pose(parent: Sequence[float], local: Sequence[float], *, rotation_fn=rotation_from_quaternion, quaternion_fn=quaternion_from_matrix) -> list[float]:
    parent_rotation = rotation_fn(parent[3:])
    position = np.asarray(parent[:3], dtype=float) + parent_rotation @ np.asarray(local[:3], dtype=float)
    rotation = parent_rotation @ rotation_fn(local[3:])
    return position.tolist() + quaternion_fn(rotation)
