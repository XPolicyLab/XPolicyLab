"""WorldScape-owned rigid-rotation conversions."""

from __future__ import annotations

import numpy as np


def quaternion_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to the first two matrix columns, row-flattened."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError(f"quaternion must end in dimension 4, got {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-8, value / np.maximum(norm, 1e-8), 0.0)
    safe = np.where(
        np.broadcast_to(norm > 1e-8, safe.shape),
        safe,
        np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    x, y, z, w = np.moveaxis(safe, -1, 0)
    matrix = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(value.shape[:-1] + (3, 3))
    return matrix[..., :, :2].reshape(value.shape[:-1] + (6,)).astype(
        np.float32, copy=False
    )


def quaternion_pose_to_rotation6d(pose: np.ndarray) -> np.ndarray:
    """Convert [..., xyz, xyzw] poses to [..., xyz, rotation6d]."""

    value = np.asarray(pose)
    if value.shape[-1] != 7:
        raise ValueError(f"pose must end in dimension 7, got {value.shape}")
    position = np.where(np.isclose(value[..., :3], -1.0), 0.0, value[..., :3])
    return np.concatenate(
        (position.astype(np.float32, copy=False), quaternion_to_rotation6d(value[..., 3:7])),
        axis=-1,
    )


__all__ = ["quaternion_pose_to_rotation6d", "quaternion_to_rotation6d"]
