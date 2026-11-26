"""Policy-facing state and tool results for the RoboDojo adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

TCP_OFFSET = 0.145


@dataclass(frozen=True)
class ArmState:
    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    step: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Move:
    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    note: str | None
    blocked: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Home:
    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    step_limit: int | None
    truncated: bool


@dataclass(frozen=True)
class ObjectSight:
    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None
    views: int = 1


@dataclass(frozen=True)
class SeenPoint:
    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str
    surface: str = "table"


def _tuple3(value: Any) -> tuple[float, float, float]:
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        values = ()
    if len(values) != 3:
        return (math.nan, math.nan, math.nan)
    return values[0], values[1], values[2]


def _rotation(quat: Sequence[float]) -> list[list[float]]:
    """Convert a [w, x, y, z] quaternion to rotation matrix rows."""
    qw, qx, qy, qz = (float(component) for component in quat)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def gripper_angles(quat: Sequence[float]) -> tuple[float, float, float]:
    """Return pitch, yaw, roll in degrees; within 1° of vertical, roll reads zero."""
    rotation = _rotation(quat)
    point = (rotation[0][0], rotation[1][0], rotation[2][0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        jaw = (rotation[0][1], rotation[1][1])
        return pitch, _wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    azimuth, tilt = math.radians(90.0 + yaw), math.radians(pitch)
    ca, sa, cp, sp = math.cos(azimuth), math.sin(azimuth), math.cos(tilt), math.sin(tilt)
    base = [[ca * cp, -sa, ca * sp], [sa * cp, ca, sa * sp], [-sp, 0.0, cp]]
    r21 = sum(base[index][2] * rotation[index][1] for index in range(3))
    r11 = sum(base[index][1] * rotation[index][1] for index in range(3))
    return pitch, _wrap(yaw), _wrap(math.degrees(math.atan2(r21, r11)))
