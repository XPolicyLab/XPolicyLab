"""Shared EEF math for the YAM 32D state/action layout.

Ported verbatim from yam_openpi:
  https://github.com/.../yam_openpi/src/openpi/policies/yam_eef_util.py

Single source of truth for the rotation / SE(3) helpers used by:
- the v2 dataset builder (rd-convert lowdim.action -> 32D parquet)
- the dataloader transforms (FlexPi Yam32DRelativeAction)
- compute_norm_stats post-processing (rot6d identity override)

Numpy-only — no torch / jax. The functions are batched over leading dims.

`rot6d` convention: **first two rows of R, flattened row-major.** Matches
yam_openpi's `_rot9_to_rot6d` and `_fk_to_xyz_rot6d` (which builds rot_6d as
`T[:2, :3].flatten()`). All callers must agree on rows; mixing rows-vs-columns
silently corrupts every downstream phase.

The 32D layout follows the eef_pose_support design doc §0.1:

    [left_pos(3) left_rot6d(6) right_pos(3) right_rot6d(6)
     left_grip(1) right_grip(1) left_joint(6) right_joint(6)]
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np


STATE_LAYOUT = MappingProxyType({
    "left_pos":    slice(0, 3),
    "left_rot6d":  slice(3, 9),
    "right_pos":   slice(9, 12),
    "right_rot6d": slice(12, 18),
    "left_grip":   slice(18, 19),
    "right_grip":  slice(19, 20),
    "left_joint":  slice(20, 26),
    "right_joint": slice(26, 32),
})

# Indices to override to identity in norm_stats (mean=0, std=1, q01=0, q99=1).
# Z-scoring rot6d distorts SO(3); see design doc §0.2.
ROT6D_SLICES = (STATE_LAYOUT["left_rot6d"], STATE_LAYOUT["right_rot6d"])

STATE_DIM = 32


def rot9_to_rot6d(rot9: np.ndarray) -> np.ndarray:
    """Convert flattened 3x3 rotation matrices to the 6D representation.

    Input rot9 is row-major (`R.flatten()` in C order); the rot6d output is
    `R[:2, :].flatten()` — first two rows, flattened.

    Shape: `(..., 9) -> (..., 6)`.
    """
    rot9 = np.asarray(rot9)
    R = rot9.reshape(rot9.shape[:-1] + (3, 3))
    return R[..., :2, :].reshape(rot9.shape[:-1] + (6,))


def rot6d_to_rot_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to a 3x3 rotation matrix.

    Gram-Schmidt orthonormalisation over the first two rows, third row from
    cross product.

    Shape: `(..., 6) -> (..., 3, 3)`.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def pose9d_to_mat(p9: np.ndarray) -> np.ndarray:
    """9D `(xyz, rot6d)` -> 4x4 SE(3) homogeneous transform.

    Shape: `(..., 9) -> (..., 4, 4)`.
    """
    p9 = np.asarray(p9, dtype=np.float64)
    xyz = p9[..., 0:3]
    R = rot6d_to_rot_mat(p9[..., 3:9])
    T = np.zeros(p9.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def mat_to_pose9d(mat: np.ndarray) -> np.ndarray:
    """4x4 SE(3) -> 9D `(xyz, rot6d)`. rot6d is the first two rows of R.

    Shape: `(..., 4, 4) -> (..., 9)`.
    """
    mat = np.asarray(mat, dtype=np.float64)
    xyz = mat[..., :3, 3]
    rot6d = mat[..., :2, :3].reshape(mat.shape[:-2] + (6,))
    return np.concatenate([xyz, rot6d], axis=-1)


def convert_pose_mat_rep(
    pose_mat: np.ndarray,
    base_pose_mat: np.ndarray,
    pose_rep: str = "abs",
    backward: bool = False,
) -> np.ndarray:
    """Port of UMI's pose-representation converter.

    Modes (forward = training transform; backward = inference inverse):
    - `'abs'`     : pass-through.
    - `'relative'`: `base^{-1} @ pose` (forward) / `base @ pose` (backward).
    - `'rel'`     : legacy UMI alias — pos diff + rot left-mul. Kept for
                    compat; prefer `'relative'`.
    - `'delta'`   : per-step diff vs previous pose. Untested in this repo
                    (out-of-scope for Phase 1+2); ported for completeness.

    `pose_mat` shape: `(..., 4, 4)`. `base_pose_mat` shape: `(4, 4)`.
    """
    pose_mat = np.asarray(pose_mat, dtype=np.float64)
    base_pose_mat = np.asarray(base_pose_mat, dtype=np.float64)

    if not backward:
        if pose_rep == "abs":
            return pose_mat
        if pose_rep == "rel":
            pos = pose_mat[..., :3, 3] - base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ np.linalg.inv(base_pose_mat[:3, :3])
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        if pose_rep == "relative":
            return np.linalg.inv(base_pose_mat) @ pose_mat
        if pose_rep == "delta":
            all_pos = np.concatenate(
                [base_pose_mat[None, :3, 3], pose_mat[..., :3, 3]], axis=0
            )
            out_pos = np.diff(all_pos, axis=0)
            all_rot = np.concatenate(
                [base_pose_mat[None, :3, :3], pose_mat[..., :3, :3]], axis=0
            )
            prev_rot = np.linalg.inv(all_rot[:-1])
            out_rot = np.matmul(all_rot[1:], prev_rot)
            out = np.copy(pose_mat)
            out[..., :3, :3] = out_rot
            out[..., :3, 3] = out_pos
            return out
        raise ValueError(f"Unsupported pose_rep: {pose_rep!r}")

    # backward (inference inverse)
    if pose_rep == "abs":
        return pose_mat
    if pose_rep == "rel":
        pos = pose_mat[..., :3, 3] + base_pose_mat[:3, 3]
        rot = pose_mat[..., :3, :3] @ base_pose_mat[:3, :3]
        out = np.copy(pose_mat)
        out[..., :3, :3] = rot
        out[..., :3, 3] = pos
        return out
    if pose_rep == "relative":
        return base_pose_mat @ pose_mat
    if pose_rep == "delta":
        out_pos = np.cumsum(pose_mat[..., :3, 3], axis=0) + base_pose_mat[:3, 3]
        out_rot = np.zeros_like(pose_mat[..., :3, :3])
        curr = base_pose_mat[:3, :3]
        for i in range(len(pose_mat)):
            curr = pose_mat[i, :3, :3] @ curr
            out_rot[i] = curr
        out = np.copy(pose_mat)
        out[..., :3, :3] = out_rot
        out[..., :3, 3] = out_pos
        return out
    raise ValueError(f"Unsupported pose_rep: {pose_rep!r}")
