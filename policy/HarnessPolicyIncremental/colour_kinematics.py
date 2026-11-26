# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Adapted for HarnessPolicyIncremental; see NOTICE.
"""Kinematics of RoboDojo's two ARX-X5 arms, computed in the host: full pose, no RPC.

RPent's RoboDojo environment exposes one IK call, ``solve_ik_position``: a target position, a
few candidate orientations (the current one, the starting one, 25 degree tilts of the current
one about world x and y), the one needing the least joint motion.  Nothing can ask for an
orientation, and the tilts accumulate: an arm sent away and back to its starting position by
``move_to`` came back rotated by 47 to 52 degrees.  The simulator's own full-pose action path
(``<arm>_ee_pose``) swings the arm when its IK diverges.

So the host solves IK itself.  The chain is the benchmark's URDF (``Assets/Robots/x5/X5A.urdf``,
joints ``joint1`` .. ``joint6``, ending at ``link6``), and each arm's base pose was fitted to
113 (joint angles, reported end-effector pose) pairs recorded in four episodes: the
reported ``<arm>_ee_pose`` is exactly the pose of ``link6`` in the world (0.00 mm and 0.000
degrees of residual), with the bases on the table edge at x = -0.30 / +0.30, y = -0.45,
z = 0.765, turned 90 degrees about z.  The solution is commanded as joint angles, the action
path RPent's ``move_to`` uses for its own IK solutions.

Frames.  ``link6``'s x axis is the gripper's pointing direction (wrist to fingertips) and the
two fingers close along its y axis.  At the start both grippers point away from the robot
(+y in the world) with the fingers side by side (closing along world x).  :data:`TCP_OFFSET`
places the grasp point, between the fingertips, on the pointing axis.

Orientation for people: ``pitch`` is how far the pointing direction is below the horizontal
(0 level, 90 straight down), ``yaw`` its heading on the table (0 away from the robot, +90 to
the robot's left, i.e. world -x), ``roll`` the turn of the fingers about the pointing direction
(0 as at the start).  The rotation is ``Rz(90 + yaw) @ Ry(pitch) @ Rx(roll)``; pointing straight
down, ``yaw`` and ``roll`` both turn about the vertical, and :func:`angles` reports roll 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .colour_constants import TCP_OFFSET as _TCP_OFFSET

ARMS = ("left", "right")

#: Joint origins (xyz, rpy) and axes of joint1 .. joint6 of X5A.urdf.
_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
#: Joint limits the simulator enforces, measured (each joint alone driven to +-3 rad from the
#: start, examples/robodojo_joint_limits.py), with a small margin; X5A.urdf says +-10 (joint6
#: +-3.14), and an IK solution with joint5 at -pi stopped at -1.55.  joint5 stops at +-1.56,
#: joint3 at -0.04 and joint2 at -0.23 below the start (0); joint1, joint3 (up) and joint6 turned
#: the full 3 rad; joint2 and joint4 were stopped by the table / the arm itself before a limit
#: (joint4 reached -1.68 and +1.78 in other runs).
JOINT_LIMITS = (
    (-3.1, 3.1),
    (-0.2, 3.1),
    (0.0, 3.1),
    (-1.75, 1.75),
    (-1.52, 1.52),
    (-3.1, 3.1),
)
#: Fitted base of each arm: (x, y, z) and yaw about the world z axis.
BASES = {
    "left": ((-0.3, -0.45, 0.765), 1.5710982518),
    "right": ((0.3, -0.45, 0.765), 1.5710982518),
}
#: Table surface height (the arm bases stand on it).
TABLE_Z = 0.765
#: The table top's extent [x], [y] (RoboDojo's ``env_cfg/scene/default.yml``: a 1.4 x 1.1 m slab
#: centred at y = -0.05).  The arms stand on it at y = -0.45; an elbow folded back past its
#: near edge (y = -0.60) may go lower than the top.
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)
_EDGE_MARGIN = 0.03
#: The grasp point on link6's x axis, metres: the middle of the finger pads, which span 0.13 to
#: 0.155 (finger meshes); RoboDojo's own robot config calls it ``gripper_bias``.
TCP_OFFSET = _TCP_OFFSET
#: The fingertips, for the table check.
TIP_OFFSET = 0.158


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _homogeneous(rot: np.ndarray, pos: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


_ORIGINS = tuple(
    _homogeneous(_rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0]), xyz) for xyz, rpy, _ in _CHAIN
)
_AXES = tuple(np.asarray(axis) for _, _, axis in _CHAIN)
_BASE_T = {arm: _homogeneous(_rot_z(yaw), xyz) for arm, (xyz, yaw) in BASES.items()}


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")


def frames(arm: str, q: Sequence[float]) -> list[np.ndarray]:
    """World poses of each joint frame (after its rotation) and of link6, in chain order."""

    _check_arm(arm)
    if len(q) != 6:
        raise ValueError("an X5 arm has 6 joints")
    t = _BASE_T[arm]
    out = []
    for origin, axis, angle in zip(_ORIGINS, _AXES, q, strict=True):
        t = t @ origin @ _homogeneous(_axis_angle(axis, float(angle)), (0.0, 0.0, 0.0))
        out.append(t)
    return out


def fk(arm: str, q: Sequence[float]) -> np.ndarray:
    """World pose (4x4) of link6, the frame RoboDojo reports as ``<arm>_ee_pose``."""

    return frames(arm, q)[-1]


def tcp(pose: np.ndarray, offset: float = TCP_OFFSET) -> np.ndarray:
    """The grasp point of a link6 pose."""

    return pose[:3, 3] + pose[:3, 0] * offset


def quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """A rotation matrix as a unit quaternion [w, x, y, z] (w >= 0)."""

    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    return q if q[0] >= 0 else tuple(-v for v in q)  # type: ignore[return-value]


def rot_from_quat(q: Sequence[float]) -> np.ndarray:
    """[w, x, y, z] to a rotation matrix."""

    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The gripper rotation for (pitch, yaw, roll) in degrees (module docstring)."""

    return (
        _rot_z(math.radians(90.0 + yaw)) @ _rot_y(math.radians(pitch)) @ _rot_x(math.radians(roll))
    )


_VERTICAL = math.sin(math.radians(1.0))


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a gripper rotation; roll 0 when pointing straight down."""

    point = rot[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    horizontal = math.hypot(point[0], point[1])
    # within a degree of straight up or down, heading and roll turn about (almost) the same
    # axis and only their difference is well defined: report it as the heading, roll 0
    if horizontal < _VERTICAL:
        jaw = rot[:, 1]
        # with roll 0 the jaw axis is Rz(90 + yaw) applied to +y, i.e. (-cos yaw, -sin yaw, 0)
        yaw = math.degrees(math.atan2(-jaw[1], -jaw[0]))
        return pitch, _wrap(yaw), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    base = orientation(pitch, yaw, 0.0)
    rest = base.T @ rot  # a rotation about x by roll
    roll = math.degrees(math.atan2(rest[2, 1], rest[1, 1]))
    return pitch, _wrap(yaw), _wrap(roll)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two rotations."""

    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _rotvec(rot: np.ndarray) -> np.ndarray:
    c = (np.trace(rot) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # angle near pi: take the axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) / 2.0)
        axis = axis * np.sign(w + (w == 0))
        return axis / np.linalg.norm(axis) * angle
    return w / (2 * s) * angle


def jacobian(arm: str, q: Sequence[float], offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(6x6 world-frame geometric Jacobian of the point ``offset`` along link6's x, its pose)."""

    fr = frames(arm, q)
    end = fr[-1]
    point = end[:3, 3] + end[:3, 0] * offset
    jac = np.zeros((6, 6))
    for i, (t, axis) in enumerate(zip(fr, _AXES, strict=True)):
        a = t[:3, :3] @ axis
        jac[:3, i] = np.cross(a, point - t[:3, 3])
        jac[3:, i] = a
    return jac, end


@dataclass(frozen=True)
class IKResult:
    q: tuple[float, ...]
    pos_err_m: float
    rot_err_deg: float
    ok: bool
    collides: bool


def clearance_violations(arm: str, q: Sequence[float], offset: float = TCP_OFFSET) -> list[str]:
    """Parts of the arm in the table: below its top, over its extent."""

    def over(p: np.ndarray) -> bool:
        return (
            TABLE_X[0] - _EDGE_MARGIN <= p[0] <= TABLE_X[1] + _EDGE_MARGIN
            and TABLE_Y[0] - _EDGE_MARGIN <= p[1] <= TABLE_Y[1] + _EDGE_MARGIN
        )

    fr = frames(arm, q)
    bad = []
    for i, t in enumerate(fr[1:], start=2):  # joint1's frame is at the base
        if t[2, 3] < TABLE_Z + 0.015 and over(t[:3, 3]):
            bad.append(f"joint{i}")
    tip = fr[-1][:3, 3] + fr[-1][:3, 0] * max(offset, TIP_OFFSET)
    if tip[2] < TABLE_Z + 0.002 and over(tip):
        bad.append("fingertips")
    return bad


def solve(
    arm: str,
    target: np.ndarray,
    seed: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    rot_weight: float = 0.2,
    pos_tol: float = 0.002,
    rot_tol_deg: float = 1.0,
    iterations: int = 150,
) -> IKResult:
    """Damped least squares from ``seed`` for a 4x4 target of the point ``offset`` along link6's x.

    ``rot_weight`` is metres per radian in the error: it only sets how the solver trades the two
    while it converges, and what it settles on when the pose is out of reach.
    """

    q = np.array(seed, dtype=float)
    lo = np.array([lim[0] for lim in JOINT_LIMITS])
    hi = np.array([lim[1] for lim in JOINT_LIMITS])
    damping = 0.02
    pos_err = rot_err = math.inf
    for _ in range(iterations):
        jac, end = jacobian(arm, q, offset)
        point = end[:3, 3] + end[:3, 0] * offset
        dp = target[:3, 3] - point
        dr = _rotvec(target[:3, :3] @ end[:3, :3].T)
        pos_err, rot_err = float(np.linalg.norm(dp)), math.degrees(float(np.linalg.norm(dr)))
        if pos_err < pos_tol * 0.25 and rot_err < rot_tol_deg * 0.25:
            break
        err = np.concatenate([dp, rot_weight * dr])
        j = jac.copy()
        j[3:] *= rot_weight
        step = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), err)
        norm = float(np.linalg.norm(step))
        if norm > 0.25:
            step *= 0.25 / norm
        q = np.clip(q + step, lo, hi)
    end = fk(arm, q)
    point = end[:3, 3] + end[:3, 0] * offset
    pos_err = float(np.linalg.norm(target[:3, 3] - point))
    rot_err = rotation_error(target[:3, :3], end[:3, :3])
    ok = pos_err <= pos_tol and rot_err <= rot_tol_deg
    return IKResult(
        tuple(float(v) for v in q), pos_err, rot_err, ok, bool(clearance_violations(arm, q, offset))
    )


def best_solution(
    arm: str,
    target: np.ndarray,
    current: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    extra_seeds: Sequence[Sequence[float]] = (),
    **kwargs,
) -> IKResult:
    """The solution nearest ``current`` that reaches the target and clears the table.

    Seeds: the current joints, the same with the wrist turned half a turn either way, the extra
    seeds given (the starting joints), and a few fixed perturbations of the current joints.  When
    no seed reaches the target, the closest miss is returned (``ok`` False).
    """

    cur = np.array(current, dtype=float)
    seeds = [cur]
    for d in (math.pi, -math.pi):
        s = cur.copy()
        s[5] = s[5] + d
        if JOINT_LIMITS[5][0] <= s[5] <= JOINT_LIMITS[5][1]:
            seeds.append(s)
    seeds.extend(np.array(s, dtype=float) for s in extra_seeds)
    rng = np.random.default_rng(0)
    seeds.extend(cur + rng.normal(0.0, 0.6, size=6) for _ in range(6))
    good, misses = [], []
    for s in seeds:
        res = solve(arm, target, s, offset=offset, **kwargs)
        (good if res.ok and not res.collides else misses).append(res)
    if good:
        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
        return min(good, key=lambda r: float(np.sum(weights * np.abs(np.array(r.q) - cur))))
    return min(misses, key=lambda r: (r.collides, r.pos_err_m + 0.1 * math.radians(r.rot_err_deg)))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Rotation a fraction ``t`` of the way from ``a`` to ``b``."""

    rv = _rotvec(a.T @ b)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-9:
        return a.copy()
    return a @ _axis_angle(rv / angle, angle * t)


def line_path(
    arm: str,
    start_q: Sequence[float],
    target: np.ndarray,
    *,
    offset: float = TCP_OFFSET,
    step_m: float = 0.02,
    step_deg: float = 8.0,
) -> tuple[list[tuple[float, ...]], str | None]:
    """Joint waypoints that move the grasp point along a straight line to ``target``.

    Every waypoint is solved from the previous one, so the arm cannot jump between IK branches
    on the way; returns the waypoints and, when the line cannot be followed, why not.
    """

    start = fk(arm, start_q)
    p0 = start[:3, 3] + start[:3, 0] * offset
    dist = float(np.linalg.norm(target[:3, 3] - p0))
    turn = rotation_error(start[:3, :3], target[:3, :3])
    n = max(1, math.ceil(dist / step_m), math.ceil(turn / step_deg))
    q = tuple(float(v) for v in start_q)
    out = []
    for k in range(1, n + 1):
        t = k / n
        way = np.eye(4)
        way[:3, :3] = slerp(start[:3, :3], target[:3, :3], t)
        way[:3, 3] = p0 + (target[:3, 3] - p0) * t
        res = solve(arm, way, q, offset=offset)
        if not res.ok:
            return out, f"the line leaves the arm's reach {t * dist:.3f} m along it"
        if res.collides:
            return out, f"the line takes the arm into the table {t * dist:.3f} m along it"
        if max(abs(a - b) for a, b in zip(res.q, q, strict=True)) > 0.6:
            return out, "the line passes a singular pose (the wrist would flip)"
        q = res.q
        out.append(q)
    return out, None


__all__ = [
    "ARMS",
    "BASES",
    "IKResult",
    "JOINT_LIMITS",
    "TABLE_X",
    "TABLE_Y",
    "TABLE_Z",
    "TCP_OFFSET",
    "TIP_OFFSET",
    "angles",
    "best_solution",
    "clearance_violations",
    "fk",
    "frames",
    "jacobian",
    "line_path",
    "orientation",
    "quat_wxyz",
    "rot_from_quat",
    "rotation_error",
    "slerp",
    "solve",
    "tcp",
]
