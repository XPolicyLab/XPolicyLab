"""Rigid-body helpers. Quaternions are (w, x, y, z); frames are right-handed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def as_vector(value: Iterable[float], size: int, name: str = "vector") -> np.ndarray:
    array = np.asarray(list(value) if not isinstance(value, np.ndarray) else value, dtype=float).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be {size} finite numbers")
    return array


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    q = as_vector(q, 4, "quaternion")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero quaternion")
    q = q / n
    return q if q[0] >= 0 else -q


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return quat_normalize(q)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    x, y, z = unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation_angle(r0: np.ndarray, r1: np.ndarray) -> float:
    """Angle (rad) of the relative rotation r0 -> r1."""
    cos = (float(np.trace(np.asarray(r0).T @ np.asarray(r1))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos)))


def quat_angle(q0: Sequence[float], q1: Sequence[float]) -> float:
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * math.acos(min(1.0, d))


def slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    a, b = quat_normalize(q0), quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta = math.acos(dot)
    return quat_normalize((math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta))


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


@dataclass(frozen=True)
class Pose:
    """Position (m) plus orientation quaternion (w, x, y, z) in the world frame."""

    xyz: tuple
    quat: tuple

    @staticmethod
    def make(xyz: Sequence[float], quat: Sequence[float]) -> "Pose":
        return Pose(tuple(float(v) for v in as_vector(xyz, 3, "xyz")), tuple(float(v) for v in quat_normalize(quat)))

    @staticmethod
    def from_array(values: Sequence[float]) -> "Pose":
        values = as_vector(values, 7, "pose")
        return Pose.make(values[:3], values[3:])

    @staticmethod
    def from_matrix(t: np.ndarray) -> "Pose":
        t = np.asarray(t, dtype=float)
        return Pose.make(t[:3, 3], matrix_to_quat(t[:3, :3]))

    @property
    def position(self) -> np.ndarray:
        return np.array(self.xyz, dtype=float)

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.quat)

    def array(self) -> np.ndarray:
        return np.r_[self.xyz, self.quat].astype(float)

    def matrix(self) -> np.ndarray:
        t = np.eye(4)
        t[:3, :3] = self.rotation
        t[:3, 3] = self.xyz
        return t

    def transform(self, local_point: Sequence[float]) -> np.ndarray:
        return self.rotation @ as_vector(local_point, 3, "point") + self.position

    def with_position(self, xyz: Sequence[float]) -> "Pose":
        return Pose.make(xyz, self.quat)

    def with_rotation(self, r: np.ndarray) -> "Pose":
        return Pose.make(self.xyz, matrix_to_quat(r))

    def errors_to(self, other: "Pose") -> tuple[float, float]:
        """(translation m, rotation rad) between two poses."""
        return float(np.linalg.norm(self.position - other.position)), quat_angle(self.quat, other.quat)

    def rounded(self, digits: int = 4) -> dict:
        return {"xyz": [round(v, digits) for v in self.xyz], "quat_wxyz": [round(v, digits) for v in self.quat]}


def point_with_offset(flange: Pose, local_offset: Sequence[float]) -> np.ndarray:
    """World position of a point fixed in the flange frame (e.g. the grasp point)."""
    return flange.transform(local_offset)


def flange_for_point(target_point: Sequence[float], rotation: np.ndarray, local_offset: Sequence[float]) -> Pose:
    """Flange pose that puts a flange-fixed point at target_point with the given flange rotation."""
    p = as_vector(target_point, 3, "target point") - np.asarray(rotation) @ as_vector(local_offset, 3, "offset")
    return Pose.make(p, matrix_to_quat(rotation))
