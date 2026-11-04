"""Rigid-body helpers shared by motion and vision (numpy only)."""

from __future__ import annotations

import math

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: rotate about x, then y, then z (all fixed axes)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w) -> np.ndarray:
    """Rotation matrix of the rotation vector ``w`` (axis times angle, radians)."""
    w = np.asarray(w, float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def log_so3(r: np.ndarray) -> np.ndarray:
    """Rotation vector of ``r``; well defined up to and including half turns."""
    cos = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    theta = math.acos(cos)
    if theta < 1e-9:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
    if math.pi - theta > 1e-4:
        v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        return v * (theta / (2.0 * math.sin(theta)))
    # near a half turn: the axis is the dominant column of (r + I) / 2 = a a^T
    sym = (r + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(sym)))
    axis = sym[:, i] / math.sqrt(max(sym[i, i], 1e-12))
    # pick the sign that agrees with the (small) antisymmetric part
    anti = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if float(anti @ axis) < 0:
        axis = -axis
    return axis * theta


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (radians) of the rotation taking ``a`` to ``b``."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos)))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a @ exp_so3(t * log_so3(a.T @ b))


def homogeneous(r: np.ndarray, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(p, float)
    return t


def invert(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def apply(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an (N, 3) point array by the 4x4 ``t``."""
    return pts @ t[:3, :3].T + t[:3, 3]


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_from_xyzwxyz(v) -> np.ndarray:
    """4x4 pose from RoboDojo's [x, y, z, qw, qx, qy, qz]."""
    v = [float(a) for a in v]
    return homogeneous(quat_wxyz_to_matrix(v[3:7]), v[:3])


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a
