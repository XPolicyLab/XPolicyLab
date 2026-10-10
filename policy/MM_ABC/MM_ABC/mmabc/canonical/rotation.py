"""Minimal vectorised rotation utilities.

Storage keeps absolute orientation as the 6D continuous representation (the
first two columns of the rotation matrix). Regression targets are increments,
which sit near identity where axis-angle is compact and singularity-free.
Everything here is numpy and batched over a leading chunk axis so it can run
inside a dataloader worker without per-step Python overhead.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on the first two columns."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS, None)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.clip(np.linalg.norm(a2_proj, axis=-1, keepdims=True), _EPS, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6), the inverse of rot6d_to_matrix up to gauge."""
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) rotation vector (axis * angle).

    The angle comes from ``atan2(|skew|, trace - 1)`` rather than
    ``arccos((trace - 1) / 2)``. Both are algebraically the same, but arccos has
    unbounded derivative as its argument approaches -1, so it loses roughly half
    the available precision for rotations near half a turn. atan2 stays well
    conditioned everywhere except exactly at pi, where the skew part vanishes
    and the axis has to come from the symmetric part instead.
    """
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    skew = np.stack(
        [
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ],
        axis=-1,
    )
    skew_norm = np.linalg.norm(skew, axis=-1)  # = 2 sin(angle)
    angle = np.arctan2(skew_norm, trace - 1.0)  # trace - 1 = 2 cos(angle)

    # Generic case: the axis is the normalised skew vector.
    out = skew * (angle / np.clip(skew_norm, _EPS, None))[..., None]

    # Vanishing skew is ambiguous between "no rotation" and "half a turn", which
    # the sign of cos(angle) separates.
    degenerate = skew_norm < 1e-7
    if np.any(degenerate):
        near_zero = degenerate & (trace - 1.0 > 0.0)
        # rotvec -> skew / 2 as the angle vanishes.
        out = np.where(near_zero[..., None], skew * 0.5, out)

        near_pi = degenerate & (trace - 1.0 <= 0.0)
        if np.any(near_pi):
            idx = np.nonzero(near_pi)
            # At a half turn, (R + I) / 2 is the outer product of the unit axis
            # with itself, so its largest column is parallel to the axis.
            sym = (mat[idx] + np.eye(3)) * 0.5
            k = np.argmax(np.einsum("...ii->...i", sym), axis=-1)
            axis = np.take_along_axis(sym, k[..., None, None], axis=-1)[..., 0]
            axis = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), _EPS, None)
            # Either sign describes the same rotation at exactly pi; pick the
            # one consistent with whatever skew remains.
            sign = np.sign(np.sum(axis * skew[idx], axis=-1))
            sign = np.where(sign == 0.0, 1.0, sign)
            out[idx] = axis * (angle[idx] * sign)[..., None]
    return out


def axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) by Rodrigues' formula."""
    angle = np.linalg.norm(vec, axis=-1, keepdims=True)
    axis = vec / np.clip(angle, _EPS, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack(
        [
            np.stack([zeros, -z, y], axis=-1),
            np.stack([z, zeros, -x], axis=-1),
            np.stack([-y, x, zeros], axis=-1),
        ],
        axis=-2,
    )
    a = angle[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    return eye + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def relative_rotation(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """inv(r_from) @ r_to for (..., 3, 3) inputs."""
    return np.einsum("...ji,...jk->...ik", r_from, r_to)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first unit quaternion -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _EPS, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) scalar-first unit quaternion with w >= 0.

    Shepperd's method: pick the largest of (trace, diagonal) as the pivot so the
    square root never sees a near-zero argument.
    """
    m = np.asarray(mat, dtype=np.float64)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    cand = np.stack([trace, m00, m11, m22], axis=-1)
    pivot = np.argmax(cand, axis=-1)
    q = np.zeros(m.shape[:-2] + (4,), dtype=np.float64)

    def _set(sel, w, x, y, z):
        q[sel, 0], q[sel, 1], q[sel, 2], q[sel, 3] = w[sel], x[sel], y[sel], z[sel]

    s0 = np.sqrt(np.clip(1.0 + trace, _EPS, None)) * 2.0
    _set(pivot == 0, 0.25 * s0, (m[..., 2, 1] - m[..., 1, 2]) / s0,
         (m[..., 0, 2] - m[..., 2, 0]) / s0, (m[..., 1, 0] - m[..., 0, 1]) / s0)
    s1 = np.sqrt(np.clip(1.0 + m00 - m11 - m22, _EPS, None)) * 2.0
    _set(pivot == 1, (m[..., 2, 1] - m[..., 1, 2]) / s1, 0.25 * s1,
         (m[..., 0, 1] + m[..., 1, 0]) / s1, (m[..., 0, 2] + m[..., 2, 0]) / s1)
    s2 = np.sqrt(np.clip(1.0 - m00 + m11 - m22, _EPS, None)) * 2.0
    _set(pivot == 2, (m[..., 0, 2] - m[..., 2, 0]) / s2, (m[..., 0, 1] + m[..., 1, 0]) / s2,
         0.25 * s2, (m[..., 1, 2] + m[..., 2, 1]) / s2)
    s3 = np.sqrt(np.clip(1.0 - m00 - m11 + m22, _EPS, None)) * 2.0
    _set(pivot == 3, (m[..., 1, 0] - m[..., 0, 1]) / s3, (m[..., 0, 2] + m[..., 2, 0]) / s3,
         (m[..., 1, 2] + m[..., 2, 1]) / s3, 0.25 * s3)

    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _EPS, None)
    return np.where(q[..., :1] < 0.0, -q, q)
