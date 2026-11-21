"""Gripper orientation in words a person (or a model) can use: pitch, yaw, roll in degrees.

* ``pitch``: how far the pointing direction (wrist -> fingertips) is below the horizontal;
  0 level, 90 straight down.
* ``yaw``: heading of the pointing direction on the table; 0 away from the robot (world +y),
  +90 toward the robot's left (world -x).  When the gripper points straight down, ``yaw`` instead
  turns the fingers about the vertical: at yaw 0 they close along world x, at yaw 90 along y.
* ``roll``: turn of the fingers about the pointing direction; 0 as at the start.

Rotation of link6: ``Rz(90 + yaw) @ Ry(pitch) @ Rx(roll)``.  Pointing straight down, yaw and roll
turn about the same axis; :func:`angles_of` then reports everything as yaw and roll 0.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import rot_x, rot_y, rot_z, wrap_deg

#: Below this horizontal share of the pointing direction the gripper counts as vertical.
_VERTICAL = math.sin(math.radians(1.0))


def rotation_of(pitch: float, yaw: float, roll: float) -> np.ndarray:
    r = math.radians
    return rot_z(r(90.0 + yaw)) @ rot_y(r(pitch)) @ rot_x(r(roll))


def angles_of(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a link6 rotation."""
    point = rot[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(point[2])))))
    if math.hypot(point[0], point[1]) < _VERTICAL:
        # vertical: the closing axis lies on the table at heading yaw + 180 (roll reported as 0)
        close = rot[:, 1]
        yaw = math.degrees(math.atan2(-close[1], -close[0]))
        return round_angles(pitch, wrap_deg(yaw), 0.0)
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    frame = rot_z(math.radians(90.0 + yaw)) @ rot_y(math.radians(pitch))
    rest = frame.T @ rot
    roll = math.degrees(math.atan2(rest[2, 1], rest[1, 1]))
    return round_angles(pitch, wrap_deg(yaw), wrap_deg(roll))


def round_angles(pitch: float, yaw: float, roll: float) -> tuple[float, float, float]:
    return (round(pitch, 2), round(yaw, 2), round(roll, 2))


def closing_axis_heading(rot: np.ndarray) -> float:
    """Heading (degrees, mod 180) of the line the fingers close along, projected on the table."""
    close = rot[:, 1]
    return math.degrees(math.atan2(close[1], close[0])) % 180.0


def finger_flip(rot: np.ndarray) -> np.ndarray:
    """The same pointing direction with the fingers turned half a turn (an empty gripper's twin)."""
    return rot @ rot_x(math.pi)
