"""Forward and inverse kinematics of the two ARX-X5 arms, solved in the host process.

Conventions
-----------
* Every pose is a 4x4 homogeneous matrix in the world frame.
* ``link6`` is the flange. Its +x axis points from the wrist towards the fingertips and the fingers
  close along its +y axis. The grasp point sits ``TCP_OFFSET`` metres along +x.
* Gripper angles, in degrees: ``pitch`` is the dip of the pointing axis below horizontal (90 means
  straight down), ``yaw`` its heading on the table (0 away from the robot, +90 towards world -x),
  ``roll`` the turn of the fingers about the pointing axis. The rotation they describe is
  ``Rz(90 + yaw) Ry(pitch) Rx(roll)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ARMS = ("left", "right")


@dataclass(frozen=True)
class UrdfJoint:
    """One revolute joint of X5A.urdf: parent-to-joint origin (xyz, rpy) and rotation axis."""

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


#: joint1 .. joint6 of X5A.urdf, ending at link6.
URDF_JOINTS = (
    UrdfJoint((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    UrdfJoint((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    UrdfJoint((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    UrdfJoint((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    UrdfJoint((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    UrdfJoint((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
#: Sum of the joint offsets: an upper bound on how far link6 can be from the base.
CHAIN_REACH_M = float(sum(math.hypot(*j.xyz) for j in URDF_JOINTS))

#: Joint ranges, radians, as the simulator enforces them (narrower than the URDF's +-10 rad).
JOINT_LIMITS = (
    (-3.1, 3.1),
    (-0.2, 3.1),
    (0.0, 3.1),
    (-1.75, 1.75),
    (-1.52, 1.52),
    (-3.1, 3.1),
)
_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

#: Base of each arm in the world: position and heading about +z (radians).
BASES = {
    "left": ((-0.3, -0.45, 0.765), 1.5710982518),
    "right": ((0.3, -0.45, 0.765), 1.5710982518),
}

#: Table top: height and extent. The bases stand on it.
TABLE_Z = 0.765
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)
#: Points this far outside the table outline still count as over the table.
TABLE_EDGE_MARGIN = 0.03
#: Lowest height above the table top for each joint frame.
JOINT_TABLE_MARGIN = 0.015
#: Lowest height above the table top for the fingertips.
TIP_TABLE_MARGIN = 0.0005

#: Grasp point (middle of the finger pads) and fingertips, metres along link6's +x.
TCP_OFFSET = 0.145
TIP_OFFSET = 0.158


# --- rotations ------------------------------------------------------------------------------

# The IK below iterates up to 150 times from distant restarts, so rounding differences of 1e-16
# grow into different (equally valid) solutions. The rotation formulas are therefore written out
# element by element in a fixed order rather than composed from generic matrix products.

def _rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about the unit vector ``axis`` (expanded Rodrigues formula)."""

    x, y, z = (float(v) for v in axis)
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return np.array([
        [v * x * x + c, v * x * y - s * z, v * x * z + s * y],
        [v * x * y + s * z, v * y * y + c, v * y * z - s * x],
        [v * x * z - s * y, v * y * z + s * x, v * z * z + c],
    ])


def _about_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _about_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _about_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: Rz(yaw) Ry(pitch) Rx(roll)."""

    return _about_z(yaw) @ _about_y(pitch) @ _about_x(roll)


def _pose(rot: np.ndarray, xyz: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def _log(rot: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle, radians) of a rotation matrix."""

    cos_angle = min(1.0, max(-1.0, (float(np.trace(rot)) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    skew = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    sin_angle = math.sin(angle)
    if sin_angle >= 1e-6:
        return skew / (2.0 * sin_angle) * angle
    # Near a half turn the skew part vanishes; recover the axis from the diagonal, signed by skew.
    axis = np.sqrt(np.clip((np.diag(rot) + 1.0) / 2.0, 0.0, None))
    axis = np.where(skew < 0.0, -axis, axis)
    return axis / np.linalg.norm(axis) * angle


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees of the rotation taking ``a`` to ``b``."""

    cos_angle = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """The rotation a fraction ``t`` of the way from ``a`` to ``b`` along the shortest arc."""

    delta = _log(a.T @ b)
    angle = float(np.linalg.norm(delta))
    if angle < 1e-9:
        return a.copy()
    return a @ _rotation(delta / angle, angle * t)


def quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0."""

    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    # Shepperd: divide by the largest of the four candidate components for stability.
    if trace > 0.0:
        r = 2.0 * math.sqrt(1.0 + trace)
        q = (r / 4.0, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        r = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / r, r / 4.0, (m[0, 1] + m[1, 0]) / r, (m[0, 2] + m[2, 0]) / r)
    elif m[1, 1] > m[2, 2]:
        r = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / r, (m[0, 1] + m[1, 0]) / r, r / 4.0, (m[1, 2] + m[2, 1]) / r)
    else:
        r = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / r, (m[0, 2] + m[2, 0]) / r, (m[1, 2] + m[2, 1]) / r, r / 4.0)
    norm = math.sqrt(sum(c * c for c in q))
    sign = 1.0 if q[0] >= 0.0 else -1.0
    return tuple(sign * c / norm for c in q)  # type: ignore[return-value]


def rot_from_quat(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion; the quaternion need not be normalised."""

    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Gripper rotation for (pitch, yaw, roll) in degrees; see the module docstring."""

    return _about_z(math.radians(90.0 + yaw)) @ _about_y(math.radians(pitch)) @ _about_x(math.radians(roll))


#: Below this horizontal component the pointing axis is treated as vertical (about one degree).
_NEAR_VERTICAL = math.sin(math.radians(1.0))


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a gripper rotation.

    Pointing (nearly) straight up or down, yaw and roll turn about the same axis; the combined turn
    is reported as yaw and roll as 0.
    """

    pointing, fingers = rot[:, 0], rot[:, 1]
    pitch = math.degrees(math.asin(min(1.0, max(-1.0, -pointing[2]))))
    if math.hypot(pointing[0], pointing[1]) < _NEAR_VERTICAL:
        # With roll 0 the finger axis is (-cos yaw, -sin yaw, 0).
        return pitch, _wrap_deg(math.degrees(math.atan2(-fingers[1], -fingers[0]))), 0.0
    yaw = math.degrees(math.atan2(pointing[1], pointing[0])) - 90.0
    about_x = orientation(pitch, yaw, 0.0).T @ rot
    roll = math.degrees(math.atan2(about_x[2, 1], about_x[1, 1]))
    return pitch, _wrap_deg(yaw), _wrap_deg(roll)


# --- forward kinematics ---------------------------------------------------------------------

_JOINT_ORIGINS = tuple(_pose(_rpy(*j.rpy), j.xyz) for j in URDF_JOINTS)
_JOINT_AXES = tuple(np.array(j.axis) for j in URDF_JOINTS)
_BASE_POSES = {arm: _pose(_about_z(heading), xyz) for arm, (xyz, heading) in BASES.items()}


def _require_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")


def frames(arm: str, q: Sequence[float]) -> list[np.ndarray]:
    """World pose of every joint frame after its rotation, joint1 first; the last is link6."""

    _require_arm(arm)
    if len(q) != len(URDF_JOINTS):
        raise ValueError("an X5 arm has 6 joints")
    pose = _BASE_POSES[arm]
    out = []
    for origin, axis, angle in zip(_JOINT_ORIGINS, _JOINT_AXES, q):
        pose = pose @ origin @ _pose(_rotation(axis, float(angle)))
        out.append(pose)
    return out


def fk(arm: str, q: Sequence[float]) -> np.ndarray:
    """World pose of link6, the pose the simulator reports as ``<arm>_ee_pose``."""

    return frames(arm, q)[-1]


def _point(pose: np.ndarray, offset: float) -> np.ndarray:
    return pose[:3, 3] + offset * pose[:3, 0]


def tcp(pose: np.ndarray, offset: float = TCP_OFFSET) -> np.ndarray:
    """Grasp point of a link6 pose."""

    return _point(pose, offset)


def jacobian(arm: str, q: Sequence[float], offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Geometric Jacobian (6x6, world frame, linear rows first) of the point ``offset`` along
    link6's +x, and the link6 pose."""

    joint_frames = frames(arm, q)
    end = joint_frames[-1]
    point = _point(end, offset)
    jac = np.empty((6, 6))
    for col, (frame, axis) in enumerate(zip(joint_frames, _JOINT_AXES)):
        world_axis = frame[:3, :3] @ axis
        jac[:3, col] = np.cross(world_axis, point - frame[:3, 3])
        jac[3:, col] = world_axis
    return jac, end


def clearance_violations(arm: str, q: Sequence[float], offset: float = TCP_OFFSET) -> list[str]:
    """Names of the parts (``jointN``, ``fingertips``) that are over the table and below its top."""

    def over_table(p: np.ndarray) -> bool:
        return (TABLE_X[0] - TABLE_EDGE_MARGIN <= p[0] <= TABLE_X[1] + TABLE_EDGE_MARGIN
                and TABLE_Y[0] - TABLE_EDGE_MARGIN <= p[1] <= TABLE_Y[1] + TABLE_EDGE_MARGIN)

    joint_frames = frames(arm, q)
    hits = [f"joint{n}" for n, frame in enumerate(joint_frames[1:], start=2)  # joint1 is at the base
            if frame[2, 3] < TABLE_Z + JOINT_TABLE_MARGIN and over_table(frame[:3, 3])]
    tip = _point(joint_frames[-1], max(offset, TIP_OFFSET))
    if tip[2] < TABLE_Z + TIP_TABLE_MARGIN and over_table(tip):
        hits.append("fingertips")
    return hits


# --- inverse kinematics ---------------------------------------------------------------------

@dataclass(frozen=True)
class IKResult:
    q: tuple[float, ...]
    pos_err_m: float
    rot_err_deg: float
    ok: bool
    collides: bool


_DAMPING = 0.02
_MAX_STEP_RAD = 0.25


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
    """Joint angles that put the point ``offset`` along link6's +x at the 4x4 ``target``.

    Damped least squares started from ``seed`` (clipped into the joint limits). Rotation errors are
    scaled by ``rot_weight`` metres per radian, which only shapes the descent and the compromise
    when the target is out of reach. The iteration stops early once both errors are a quarter of
    their tolerance; ``ok`` compares the final errors with the tolerances themselves.
    """

    q = np.clip(np.asarray(seed, dtype=float), _LOWER, _UPPER)
    weights = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    regulariser = _DAMPING ** 2 * np.eye(6)
    for _ in range(iterations):
        jac, end = jacobian(arm, q, offset)
        dp = target[:3, 3] - _point(end, offset)
        dr = _log(target[:3, :3] @ end[:3, :3].T)
        if np.linalg.norm(dp) < 0.25 * pos_tol and math.degrees(np.linalg.norm(dr)) < 0.25 * rot_tol_deg:
            break
        jw = jac * weights[:, None]
        dq = jw.T @ np.linalg.solve(jw @ jw.T + regulariser, np.concatenate([dp, rot_weight * dr]))
        size = float(np.linalg.norm(dq))
        if size > _MAX_STEP_RAD:
            dq *= _MAX_STEP_RAD / size
        q = np.clip(q + dq, _LOWER, _UPPER)
    end = fk(arm, q)
    pos_err = float(np.linalg.norm(target[:3, 3] - _point(end, offset)))
    rot_err = rotation_error(target[:3, :3], end[:3, :3])
    return IKResult(
        q=tuple(float(v) for v in q),
        pos_err_m=pos_err,
        rot_err_deg=rot_err,
        ok=pos_err <= pos_tol and rot_err <= rot_tol_deg,
        collides=bool(clearance_violations(arm, q, offset)),
    )


#: Joint-travel weights when choosing among solutions (shoulder moves cost most).
_TRAVEL_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
#: Random restarts around the current joints: count, spread (radians) and fixed generator seed.
_RESTARTS, _RESTART_SPREAD, _RESTART_SEED = 6, 0.6, 0


def _seeds(current: np.ndarray, extra: Sequence[Sequence[float]]) -> list[np.ndarray]:
    seeds = [current]
    for half_turn in (math.pi, -math.pi):  # same pose, wrist flipped
        flipped = current.copy()
        flipped[5] += half_turn
        if JOINT_LIMITS[5][0] <= flipped[5] <= JOINT_LIMITS[5][1]:
            seeds.append(flipped)
    seeds += [np.asarray(s, dtype=float) for s in extra]
    rng = np.random.default_rng(_RESTART_SEED)
    seeds += [current + rng.normal(0.0, _RESTART_SPREAD, size=6) for _ in range(_RESTARTS)]
    return seeds


def best_solution(
    arm: str,
    target: np.ndarray,
    current: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    extra_seeds: Sequence[Sequence[float]] = (),
    use_table_filter: bool = True,
    **kwargs,
) -> IKResult:
    """Of the solutions that reach ``target`` (and clear the table), the one with the least
    weighted joint travel from ``current``; otherwise the closest miss, with ``ok`` False.

    Starts are the current joints, the same with the wrist flipped by half a turn, ``extra_seeds``,
    and a fixed set of random restarts around the current joints.
    """

    cur = np.asarray(current, dtype=float)
    results = [solve(arm, target, s, offset=offset, **kwargs) for s in _seeds(cur, extra_seeds)]

    def usable(r: IKResult) -> bool:
        return r.ok and not (use_table_filter and r.collides)

    reached = [r for r in results if usable(r)]
    if reached:
        return min(reached, key=lambda r: float(np.sum(_TRAVEL_WEIGHTS * np.abs(np.asarray(r.q) - cur))))
    return min(
        results,
        key=lambda r: (use_table_filter and r.collides, r.pos_err_m + 0.1 * math.radians(r.rot_err_deg)),
    )


def line_path(
    arm: str,
    start_q: Sequence[float],
    target: np.ndarray,
    *,
    offset: float = TCP_OFFSET,
    step_m: float = 0.02,
    step_deg: float = 8.0,
) -> tuple[list[tuple[float, ...]], str | None]:
    """Joint waypoints moving the point ``offset`` along a straight line (and the rotation along
    the shortest arc) from the pose at ``start_q`` to ``target``.

    Each waypoint is solved from the previous one so the arm stays on one IK branch. Returns the
    waypoints reached and, if the line cannot be followed to the end, the reason.
    """

    start = fk(arm, start_q)
    p0 = _point(start, offset)
    span = target[:3, 3] - p0
    length = float(np.linalg.norm(span))
    steps = max(1, math.ceil(length / step_m), math.ceil(rotation_error(start[:3, :3], target[:3, :3]) / step_deg))
    q = tuple(float(v) for v in start_q)
    waypoints: list[tuple[float, ...]] = []
    for k in range(1, steps + 1):
        frac = k / steps
        goal = _pose(slerp(start[:3, :3], target[:3, :3], frac), p0 + span * frac)
        res = solve(arm, goal, q, offset=offset)
        along = f"{frac * length:.3f} m along it"
        if not res.ok:
            return waypoints, f"the line leaves the arm's reach {along}"
        if res.collides:
            return waypoints, f"the line takes the arm into the table {along}"
        if max(abs(a - b) for a, b in zip(res.q, q)) > 0.6:
            return waypoints, "the line passes a singular pose (the wrist would flip)"
        q = res.q
        waypoints.append(q)
    return waypoints, None


__all__ = [
    "ARMS",
    "BASES",
    "CHAIN_REACH_M",
    "IKResult",
    "JOINT_LIMITS",
    "TABLE_X",
    "TABLE_Y",
    "TABLE_Z",
    "TCP_OFFSET",
    "TIP_OFFSET",
    "URDF_JOINTS",
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
