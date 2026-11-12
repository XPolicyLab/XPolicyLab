from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..planning import kinematics as K

FINGER_DEPTH = 0.013
FINGER_HALF_WIDTH = 0.011
TIP_CLEARANCE = 0.003
TILTS = 0.0, math.radians(15), math.radians(30), math.radians(40)
MAX_REACH_XY = 0.6


def grasp_rotation(yaw: float, tilt: float = 0.0) -> np.ndarray:
    r = K.top_down(yaw)
    if tilt:
        r = K.axis_angle(r[:, 1], tilt) @ r
    return r


def base_seed(arm: str, xy) -> np.ndarray:
    bx, by = K.BASE_POS[arm][:2]
    j1 = math.atan2(-(xy[0] - bx), xy[1] - by)
    return np.array([j1, 1.6, 1.1, -1.0, 0.0, 0.0])


def pose(rot: np.ndarray, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = p
    return t


def tcp_min_z(z_support: float, tilt: float) -> float:
    return z_support + FINGER_DEPTH * math.cos(tilt) + FINGER_HALF_WIDTH * math.sin(abs(tilt)) + TIP_CLEARANCE


@dataclass
class Grasp:
    arm: str
    rot: np.ndarray
    yaw: float
    tilt: float
    tcp: np.ndarray
    q: np.ndarray
    hover_z: float
    q_hover: np.ndarray
    cost: float


def _ik(arm, t, seeds, offset=K.TCP_OFFSET):
    for s in seeds:
        r = K.solve(arm, t, s, offset, iters=80)
        if r.ok:
            return r


def grasp_candidates(
    arm: str,
    center,
    size: float,
    yaw_box: float,
    q_cur,
    *,
    z_support: float = K.TABLE_Z,
    hover_dz: float = 0.08,
    tilts=TILTS,
    blocked_yaws=(),
    prefer_z=None,
    pref_weight: float = 1.5,
    other=None,
    min_clear: float = 0.05,
    boxes=(),
    box_margin: float = 0.008,
    time_w: float = 0.0,
) -> list[Grasp]:
    from ..planning import collision as CO

    out = []
    q_cur = np.asarray(q_cur, dtype=float)
    seed0 = base_seed(arm, center[:2])
    bx, by = K.BASE_POS[arm][:2]
    if math.hypot(center[0] - bx, center[1] - by) > MAX_REACH_XY:
        return []
    to_base = np.array([bx - center[0], by - center[1]])
    to_base /= max(1e-09, np.linalg.norm(to_base))

    def clear(q) -> bool:
        if other is not None:
            qo, go = other
            d = CO.arm_distance(q, qo, 1.0, go) if arm == "left" else CO.arm_distance(qo, q, go, 1.0)
            if d < min_clear:
                return False
        if boxes:
            open_g = min(1.0, ((size + 0.024) / 2 - 0.0004 + 0.01) / 0.054)
            if CO.box_clearance(arm, q, boxes, open_g) < box_margin:
                return False
        return True

    for tilt in tilts:
        for k in range(4):
            yaw = yaw_box + k * math.pi / 2
            if any(abs(math.remainder(yaw - b, math.pi)) < math.radians(20) for b in blocked_yaws):
                continue
            for sgn in (1.0,) if tilt == 0 else (1.0, -1.0):
                rot = grasp_rotation(yaw, sgn * tilt)
                if tilt and float(np.dot(-rot[:2, 0], to_base)) < 0.2:
                    continue
                z = max(center[2], tcp_min_z(z_support, tilt))
                tcp = np.array([center[0], center[1], z])
                seeds = [q_cur, seed0, seed0 + np.array([0, 0, 0, 0, 0, yaw])]
                seeds += [seed0 + np.array([0, 0, 0, 0, 0, yaw + d]) for d in (math.pi, -math.pi)]
                r = _ik(arm, pose(rot, tcp), seeds)
                if r is None or not clear(r.q):
                    continue
                hz = z + hover_dz
                min_hover = size / 2 + 0.02
                rh = _ik(arm, pose(rot, tcp + [0, 0, hover_dz]), [r.q])
                while rh is None and hz > z + min_hover + 1e-06:
                    hz = max(z + min_hover, hz - 0.01)
                    rh = _ik(arm, pose(rot, [tcp[0], tcp[1], hz]), [r.q])
                if rh is None or not clear(rh.q):
                    continue
                cost = float(np.sum(K.JOINT_COST_W * np.abs(rh.q - q_cur))) + 3.0 * tilt + 0.2 * (z + hover_dz - hz)
                cost += time_w * float(np.max(np.abs(rh.q - q_cur)))
                if prefer_z is not None:
                    zh = rot[:2, 2]
                    nz = float(np.linalg.norm(zh))
                    if nz > 1e-06:
                        cost += pref_weight * (1.0 - float(np.dot(zh / nz, prefer_z)))
                out.append(Grasp(arm, rot, yaw, sgn * tilt, tcp, r.q, hz, rh.q, cost))
        if out:
            break
    out.sort(key=lambda g: g.cost)
    return out


def place_rotation(
    arm: str,
    rot_grasp: np.ndarray,
    p_tcp,
    q_cur,
    *,
    extra_z=(0.0,),
    yaw_steps=12,
    prefer_z=None,
    pref_weight: float = 1.5,
    time_w: float = 0.0,
):
    q_cur = np.asarray(q_cur, dtype=float)
    seed0 = base_seed(arm, p_tcp[:2])
    best = None
    for i in range(yaw_steps):
        d = (i + 1) // 2 * (2 * math.pi / yaw_steps) * (1 if i % 2 else -1) if i else 0.0
        rot = K.rot_z(d) @ rot_grasp
        qs = []
        prev = [q_cur, seed0]
        for dz in extra_z:
            r = _ik(
                arm,
                pose(rot, np.asarray(p_tcp) + [0, 0, dz]),
                prev + [seed0 + np.array([0, 0, 0, 0, 0, a]) for a in (math.pi, -math.pi)],
            )
            if r is None:
                break
            qs.append(r.q)
            prev = [r.q]
        if len(qs) != len(extra_z):
            continue
        cost = float(np.sum(K.JOINT_COST_W * np.abs(qs[0] - q_cur))) + 0.3 * abs(d)
        cost += time_w * float(np.max(np.abs(qs[0] - q_cur)))
        if prefer_z is not None:
            zh = rot[:2, 2]
            nz = float(np.linalg.norm(zh))
            if nz > 1e-06:
                cost += pref_weight * (1.0 - float(np.dot(zh / nz, prefer_z)))
        if best is None or cost < best[0]:
            best = cost, rot, qs
    if best is None:
        return None, None
    return best[1], best[2]
