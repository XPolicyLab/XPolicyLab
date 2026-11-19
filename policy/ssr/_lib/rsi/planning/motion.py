from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import kinematics as K

MAX_JOINT_STEP = 0.05
MAX_LIN_STEP = 0.012
MAX_ANG_STEP = math.radians(3.0)


def _profile(n: int) -> np.ndarray:
    t = np.arange(1, n + 1) / n
    return 0.5 - 0.5 * np.cos(np.pi * t)


@dataclass
class Plan:
    q: list[np.ndarray] = field(default_factory=list)
    ok: bool = True
    reason: str = ""

    def __len__(self) -> int:
        return len(self.q)

    def last(self, default=None):
        return self.q[-1] if self.q else default


def n_steps_for(dist: float, ang: float, lin_step: float = MAX_LIN_STEP, ang_step: float = MAX_ANG_STEP) -> int:
    n_lin = math.ceil(dist * math.pi / 2 / max(lin_step, 1e-06))
    n_ang = math.ceil(ang * math.pi / 2 / max(ang_step, 1e-06))
    return max(1, n_lin, n_ang)


def line(
    arm: str,
    q0: Sequence[float],
    goal: np.ndarray,
    offset: float = 0.0,
    *,
    lin_step: float = MAX_LIN_STEP,
    ang_step: float = MAX_ANG_STEP,
    joint_step: float = MAX_JOINT_STEP,
    min_steps: int = 1,
    start: np.ndarray | None = None,
    limits: np.ndarray | None = None,
) -> Plan:
    q0 = np.asarray(q0, dtype=float)
    t0 = K.fk(arm, q0) if start is None else np.asarray(start)
    p0 = K.point_of(t0, offset)
    p1 = goal[:3, 3]
    dist = float(np.linalg.norm(p1 - p0))
    ang = K.rot_angle(t0[:3, :3], goal[:3, :3])
    n = max(min_steps, n_steps_for(dist, ang, lin_step, ang_step))
    for _attempt in range(4):
        plan = _line_n(arm, q0, t0, p0, goal, offset, n, limits)
        if not plan.ok:
            return plan
        qs = [q0] + plan.q
        if np.ndim(joint_step) == 0:
            worst = max(float(np.max(np.abs(b - a))) for a, b in zip(qs[:-1], qs[1:]))
            if worst <= joint_step * 1.05:
                return plan
            n = math.ceil(n * worst / joint_step)
        else:
            js = np.asarray(joint_step, dtype=float)
            worst = max(float(np.max(np.abs(b - a) / js)) for a, b in zip(qs[:-1], qs[1:]))
            if worst <= 1.05:
                return plan
            n = math.ceil(n * worst)
    return plan


def _line_n(arm, q0, t0, p0, goal, offset, n, limits=None) -> Plan:
    s = _profile(n)
    q = q0.copy()
    out = Plan()
    for k, sk in enumerate(s):
        way = np.eye(4)
        way[:3, :3] = K.slerp_mat(t0[:3, :3], goal[:3, :3], float(sk))
        way[:3, 3] = p0 + (goal[:3, 3] - p0) * float(sk)
        r = K.solve(arm, way, q, offset, iters=60, limits=limits)
        if not r.ok:
            if not (k < n - 1 and r.pos_err < 0.004 and (r.rot_err < math.radians(2.0))):
                out.ok = False
                out.reason = (
                    f"ik failed at {k + 1}/{n} (pos {r.pos_err * 1000:.1f} mm, rot {math.degrees(r.rot_err):.1f} deg)"
                )
                return out
        if float(np.max(np.abs(r.q - q))) > 0.5:
            out.ok = False
            out.reason = f"branch jump at {k + 1}/{n}"
            return out
        q = r.q
        out.q.append(q.copy())
    return out


def joint_move(
    q0: Sequence[float], q1: Sequence[float], *, joint_step: float = MAX_JOINT_STEP, min_steps: int = 1
) -> Plan:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if np.ndim(joint_step) == 0:
        d = float(np.max(np.abs(q1 - q0)))
        n = max(min_steps, math.ceil(d * math.pi / 2 / joint_step))
    else:
        d = float(np.max(np.abs(q1 - q0) / np.asarray(joint_step, dtype=float)))
        n = max(min_steps, math.ceil(d * math.pi / 2))
    s = _profile(n)
    return Plan([q0 + (q1 - q0) * float(sk) for sk in s])


def hold(q: Sequence[float], n: int) -> Plan:
    q = np.asarray(q, dtype=float)
    return Plan([q.copy() for _ in range(max(0, n))])


def concat(*plans: Plan) -> Plan:
    out = Plan()
    for p in plans:
        out.q.extend(p.q)
        if not p.ok:
            out.ok = False
            out.reason = p.reason
            break
    return out
