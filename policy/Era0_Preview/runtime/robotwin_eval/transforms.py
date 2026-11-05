# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal numpy/scipy helpers for Hy-VLA RoboTwin evaluation.

Only the helpers actually consumed by ``policy_wrapper.HyVLAPolicyWrapper``
live here:

* ``get_norm_data``                    -- load action/state mean/std from a pickle.
* ``pos_quat_to_pos_rotation_matrix``  -- 8d (xyz, quat_xyzw, gripper) -> 10d (xyz, 6d rotmat, gripper).
* ``pos_rotation_matrix_to_pos_quat``  -- inverse of the above (used by the abs decoding head).
* ``convert_pose``                     -- 16d dual-arm EE state (robotwin wxyz) -> normalized 20d (1, 20).
* ``relative_to_dual_arm_poses``       -- decode the network's 20d RT-relative output back to 16d dual-arm PosQuat.

The two private helpers ``_cross`` and ``_rotation_6d_to_matrix`` are
implementation details of ``pos_rotation_matrix_to_pos_quat`` and
``relative_to_dual_arm_poses`` respectively.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Norm-stat I/O
# ---------------------------------------------------------------------------
def get_norm_data(pkl_path: str) -> dict[str, np.ndarray]:
    """Load (qpos_mean, qpos_std, act_mean, act_std [, act_mean_abs,
    act_std_abs]) from a unified Hy-VLA norm-stats pickle.

    The pickle is the single artifact produced by
    ``scripts/compute_norm_robotwin.py``:

    Required keys::
        qpos_mean, qpos_std, action_mean, action_std

    Optional keys (present iff the ckpt was trained with
    ``act_type=relative_chunk_ee_RT_with_absolute``):
        action_mean_abs, action_std_abs
    Returned dict mirrors that structure with ``act_*`` aliases for
    ``action_*``; the abs entries are ``None`` when not present in the
    pkl, so downstream code can branch on truthiness.
    """
    with open(pkl_path, "rb") as f:
        info: dict[str, Any] = pickle.load(f)
    out: dict[str, np.ndarray | None] = {
        "qpos_mean": np.array(info["qpos_mean"]),
        "qpos_std": np.array(info["qpos_std"]),
        "act_mean": np.array(info["action_mean"]),
        "act_std": np.array(info["action_std"]),
        "act_mean_abs": None,
        "act_std_abs": None,
    }
    if "action_mean_abs" in info and "action_std_abs" in info:
        out["act_mean_abs"] = np.array(info["action_mean_abs"])
        out["act_std_abs"] = np.array(info["action_std_abs"])
    return out


# ---------------------------------------------------------------------------
# Single-frame PosQuat <-> PosRotMat
# ---------------------------------------------------------------------------
def pos_quat_to_pos_rotation_matrix(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: float) -> np.ndarray:
    """Pack (xyz, quat_xyzw, gripper) into a 10-d (xyz, 6d rotmat, gripper).

    The 6-d rotation is the first two rows of the rotation matrix,
    matching the convention used at training time.
    """
    out = np.ones(10, dtype=pos.dtype)
    matrix = R.from_quat(quat_xyzw).as_matrix()
    out[0:3] = pos.copy()
    out[3:6] = matrix[0, :]
    out[6:9] = matrix[1, :]
    out[9] = gripper
    return out


def _cross_normalized(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Cross product of two normalized vectors, re-normalized.

    Used to recover the third row of a rotation matrix from its first two.
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1n, v2n)
    return v3 / np.linalg.norm(v3)


def pos_rotation_matrix_to_pos_quat(pos_rm: np.ndarray) -> np.ndarray:
    """Inverse of ``pos_quat_to_pos_rotation_matrix``: 10-d -> 8-d PosQuat (xyzw)."""
    out = np.ones(8, dtype=pos_rm.dtype)
    pos = pos_rm[0:3]
    c0 = pos_rm[3:6]
    c1 = pos_rm[6:9]
    c2 = _cross_normalized(c0, c1)
    rotation_matrix = np.stack((c0, c1, c2), axis=0)
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    out[0:3] = pos.copy()
    out[3:7] = quat_xyzw.copy()
    out[7] = pos_rm[9]
    return out


# ---------------------------------------------------------------------------
# Dual-arm EE state encoding (16d -> normalized 20d)
# ---------------------------------------------------------------------------
def convert_pose(eepose16_wxyz: np.ndarray, qpos_mean: np.ndarray, qpos_std: np.ndarray) -> np.ndarray:
    """Encode a 16-d dual-arm EE state for the network.

    Input layout (RoboTwin convention, quaternion is wxyz):
      [left_xyz(3), left_quat_wxyz(4), left_gripper(1),
       right_xyz(3), right_quat_wxyz(4), right_gripper(1)]

    Output: ``(1, 20)`` float, normalized by ``qpos_mean`` / ``qpos_std``.
    """
    e = eepose16_wxyz.copy()
    # wxyz -> xyzw
    e[3:7] = eepose16_wxyz[[4, 5, 6, 3]]
    e[11:15] = eepose16_wxyz[[12, 13, 14, 11]]

    left = pos_quat_to_pos_rotation_matrix(e[:3], e[3:7], e[7])
    right = pos_quat_to_pos_rotation_matrix(e[8:11], e[11:15], e[15])
    ee_prop = np.concatenate([left, right])
    ee_prop = (ee_prop - qpos_mean) / qpos_std
    return ee_prop[None, ...]


# ---------------------------------------------------------------------------
# Network-output decoding: relative 20d -> dual-arm absolute 16d (PosQuat)
# ---------------------------------------------------------------------------
def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Recover a (N, 3, 3) rotation matrix from a (N, 6) Gram-Schmidt 6-d
    representation (rows 0/1 of the matrix flattened)."""
    a1 = d6[:, :3]
    a2 = d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    dot_prod = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot_prod * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)


def _relative_matrices_to_poses(relative_matrices: np.ndarray, start_pose_xyzw: np.ndarray) -> np.ndarray:
    """Per-arm: turn ``(N, 9)`` rel = [trans3, rot6d] into ``(N, 7)`` PosQuat (xyzw)."""
    n = relative_matrices.shape[0]

    pos0 = start_pose_xyzw[:3]
    quat0 = start_pose_xyzw[3:]
    t0 = np.eye(4)
    t0[:3, :3] = R.from_quat(quat0).as_matrix()
    t0[:3, 3] = pos0

    translations = relative_matrices[:, :3]
    rotations_6d = relative_matrices[:, 3:]
    delta_r = _rotation_6d_to_matrix(rotations_6d)

    delta_t = np.eye(4).reshape(1, 4, 4).repeat(n, axis=0)
    delta_t[:, :3, :3] = delta_r
    delta_t[:, :3, 3] = translations

    ti = t0 @ delta_t
    rec_pos = ti[:, :3, 3]
    rec_quat = R.from_matrix(ti[:, :3, :3]).as_quat()
    return np.concatenate([rec_pos, rec_quat], axis=1)


def relative_to_dual_arm_poses(relative_output: np.ndarray, start_dual_pose_xyzw: np.ndarray) -> np.ndarray:
    """Decode the network's 20-d RT-relative output to a 16-d dual-arm PosQuat (xyzw).

    ``relative_output`` is ``(N, 20)`` =
      [left_trans(3), left_rot6d(6), left_gripper(1),
       right_trans(3), right_rot6d(6), right_gripper(1)].

    ``start_dual_pose_xyzw`` is the 16-d initial EE state already converted
    to xyzw quaternions (i.e. the first row of ``encode_obs(...)``'s
    ``initial_ee_pose`` after the wxyz->xyzw flip).
    """
    rel_arm1 = relative_output[:, 0:9]
    grip_arm1 = relative_output[:, 9:10]
    rel_arm2 = relative_output[:, 10:19]
    grip_arm2 = relative_output[:, 19:20]

    start_arm1 = start_dual_pose_xyzw[0:7]
    start_arm2 = start_dual_pose_xyzw[8:15]

    pose_arm1 = _relative_matrices_to_poses(rel_arm1, start_arm1)
    pose_arm2 = _relative_matrices_to_poses(rel_arm2, start_arm2)

    return np.concatenate([pose_arm1, grip_arm1, pose_arm2, grip_arm2], axis=1)



__all__ = [
    "get_norm_data",
    "pos_quat_to_pos_rotation_matrix",
    "pos_rotation_matrix_to_pos_quat",
    "convert_pose",
    "relative_to_dual_arm_poses",
]
