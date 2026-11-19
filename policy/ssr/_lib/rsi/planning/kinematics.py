from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

ARMS = "left", "right"
BASE_POS = {"left": (-0.3, -0.45, 0.765), "right": (0.3, -0.45, 0.765)}
BASE_YAW = math.pi / 2
TABLE_Z = 0.765
_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
JOINT_LIMITS = np.array([(-3.1, 3.1), (-0.2, 3.1), (0.0, 3.1), (-1.75, 1.75), (-1.52, 1.52), (-3.1, 3.1)])
URDF_JOINT6_LIMIT = 3.14
TCP_OFFSET = 0.145
FINGERTIP_OFFSET = 0.158


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle(axis, angle: float) -> np.ndarray:
    x, y, z = (float(v) for v in axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotvec(r: np.ndarray) -> np.ndarray:
    c = (np.trace(r) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-09:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = math.sin(angle)
    if s < 1e-06:
        axis = np.sqrt(np.maximum(np.diag(r) + 1.0, 0.0) / 2.0)
        if axis[0] > 1e-06:
            axis[1] = math.copysign(axis[1], r[0, 1])
            axis[2] = math.copysign(axis[2], r[0, 2])
        elif axis[1] > 1e-06:
            axis[2] = math.copysign(axis[2], r[1, 2])
        return axis / np.linalg.norm(axis) * angle
    return w / (2 * s) * angle


def rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def slerp_mat(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    rv = rotvec(a.T @ b)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-09:
        return a.copy()
    return a @ axis_angle(rv / ang, ang * t)


def pose_to_mat(pose) -> np.ndarray:
    p = np.asarray(pose, dtype=float).reshape(-1)
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(p[3:7])
    t[:3, 3] = p[:3]
    return t


def top_down(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[0.0, c, s], [0.0, s, -c], [-1.0, 0.0, 0.0]])


def _origin(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    t[:3, 3] = xyz
    return t


_ORIGINS = tuple(_origin(xyz, rpy) for xyz, rpy, _ in _CHAIN)
_AXES = tuple(np.asarray(ax, dtype=float) for _, _, ax in _CHAIN)
_AXIS_ID = tuple(int(np.argmax(np.abs(ax))) for ax in _AXES)


def _joint_rot(axis_id: int, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    t = np.eye(4)
    if axis_id == 0:
        t[1, 1], t[1, 2], t[2, 1], t[2, 2] = c, -s, s, c
    elif axis_id == 1:
        t[0, 0], t[0, 2], t[2, 0], t[2, 2] = c, s, -s, c
    else:
        t[0, 0], t[0, 1], t[1, 0], t[1, 1] = c, -s, s, c
    return t


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def base_mat(arm: str) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot_z(BASE_YAW)
    t[:3, 3] = BASE_POS[arm]
    return t


_BASE = {a: base_mat(a) for a in ARMS}


def frames(arm: str, q: Sequence[float]) -> list[np.ndarray]:
    t = _BASE[arm]
    out = []
    for origin, aid, angle in zip(_ORIGINS, _AXIS_ID, q):
        t = t @ origin @ _joint_rot(aid, float(angle))
        out.append(t)
    return out


def fk(arm: str, q: Sequence[float]) -> np.ndarray:
    return frames(arm, q)[-1]


def point_of(t: np.ndarray, offset: float) -> np.ndarray:
    return t[:3, 3] + t[:3, 0] * offset


def jacobian(arm: str, q: Sequence[float], offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    fr = frames(arm, q)
    end = fr[-1]
    p = point_of(end, offset)
    j = np.zeros((6, 6))
    for i, (t, aid) in enumerate(zip(fr, _AXIS_ID)):
        a = t[:3, aid]
        j[:3, i] = _cross(a, p - t[:3, 3])
        j[3:, i] = a
    return j, end


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    ok: bool


def solve(
    arm: str,
    target: np.ndarray,
    seed: Sequence[float],
    offset: float = 0.0,
    *,
    rot_weight: float = 0.3,
    pos_tol: float = 0.0005,
    rot_tol: float = math.radians(0.5),
    iters: int = 100,
    limits: np.ndarray | None = None,
    stall: int = 0,
) -> IKResult:
    lim = JOINT_LIMITS if limits is None else np.asarray(limits, dtype=float)
    q = np.clip(np.array(seed, dtype=float), lim[:, 0], lim[:, 1])
    pe = re = math.inf
    best, since = math.inf, 0
    for _ in range(iters):
        j, end = jacobian(arm, q, offset)
        dp = target[:3, 3] - point_of(end, offset)
        dr = rotvec(target[:3, :3] @ end[:3, :3].T)
        pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        if pe < pos_tol * 0.2 and re < rot_tol * 0.2:
            break
        if stall > 0:
            e = pe + 0.1 * re
            if e < best * 0.99:
                best, since = e, 0
            else:
                since += 1
                if since >= stall and e > 0.01:
                    break
        err = np.concatenate([dp, rot_weight * dr])
        jw = j.copy()
        jw[3:] *= rot_weight
        dq = jw.T @ np.linalg.solve(jw @ jw.T + 0.01 * 0.01 * np.eye(6), err)
        n = float(np.max(np.abs(dq)))
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, lim[:, 0], lim[:, 1])
    end = fk(arm, q)
    pe = float(np.linalg.norm(target[:3, 3] - point_of(end, offset)))
    re = rot_angle(target[:3, :3], end[:3, :3])
    return IKResult(q, pe, re, pe <= pos_tol and re <= rot_tol)


JOINT_COST_W = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])


def solve_near(
    arm: str,
    target: np.ndarray,
    current: Sequence[float],
    offset: float = 0.0,
    *,
    extra_seeds: Sequence[Sequence[float]] = (),
    n_random: int = 6,
    **kw,
) -> IKResult:
    cur = np.asarray(current, dtype=float)
    seeds = [cur]
    lim = JOINT_LIMITS if kw.get("limits") is None else np.asarray(kw["limits"], dtype=float)
    for d in (math.pi, -math.pi):
        s = cur.copy()
        s[5] += d
        if lim[5, 0] <= s[5] <= lim[5, 1]:
            seeds.append(s)
    seeds.extend(np.asarray(s, dtype=float) for s in extra_seeds)
    rng = np.random.default_rng(0)
    seeds.extend(cur + rng.normal(0.0, 0.6, size=6) for _ in range(n_random))
    good, miss = [], []
    for s in seeds:
        r = solve(arm, target, s, offset, **kw)
        (good if r.ok else miss).append(r)
        if r.ok and np.max(np.abs(r.q - cur)) < 0.3:
            return r
    if good:
        return min(good, key=lambda r: float(np.sum(JOINT_COST_W * np.abs(r.q - cur))))
    return min(miss, key=lambda r: r.pos_err + 0.1 * r.rot_err)
