
from __future__ import annotations

import math

import numpy as np

from ...lib.kinematics import chain as kin

L_UP = 0.264
L_FORE = math.hypot(0.245, 0.056)
DELTA = math.atan2(0.056, 0.245)
SHOULDER = (0.02, 0.0605 + 0.04)
OFF_X4 = 0.06775
OFF_Y = 0.0005
OFF_A = 0.02895
LIM = kin.JOINT_LIMITS
_BASE_INV = {arm: np.linalg.inv(kin.base_T(arm)) for arm in kin.BASES}

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _q6(arm, q5s, R6):
    T5 = kin.fk_chain(arm, np.r_[q5s, 0.0])[4]

    M = (T5[:3, :3] @ _rx(-3.1416)).T @ R6
    return math.atan2(M[2, 1], M[1, 1])

def raw_solutions(arm, T6):
    Tb = _BASE_INV[arm] @ T6
    p, R = Tb[:3, 3], Tb[:3, :3]
    a = R[:, 0]
    w = p - OFF_A * a
    w = w - np.array([0.0, 0.0, 0.0605])
    rho = math.hypot(w[0], w[1])
    if rho < 1e-6:
        return []
    alpha = math.atan2(w[1], w[0])
    s = OFF_Y / rho
    if abs(s) > 1:
        return []
    out = []
    for q1 in (alpha + math.asin(s), alpha - math.pi - math.asin(s)):
        c1, s1 = math.cos(q1), math.sin(q1)
        x1 = np.array([c1, s1, 0.0])
        y1 = np.array([-s1, c1, 0.0])
        ax, ay, az = a @ x1, a @ y1, a[2]
        if abs(ay) > 1:
            continue
        for q5 in (-math.asin(ay), math.pi + math.asin(ay)):
            c5 = math.cos(q5)
            if abs(c5) < 1e-6:
                continue
            x4x, x4z = ax / c5, az / c5
            phi = math.atan2(-x4z, x4x)

            px = (p @ x1) - OFF_X4 * x4x - OFF_A * ax
            pz = p[2] - OFF_X4 * x4z - OFF_A * az
            dx, dz = px - SHOULDER[0], pz - SHOULDER[1]
            d2 = dx * dx + dz * dz
            cg = (d2 - L_UP ** 2 - L_FORE ** 2) / (2 * L_UP * L_FORE)
            if abs(cg) > 1:
                continue
            for g in (math.acos(cg), -math.acos(cg)):
                bu = math.atan2(dz, dx) - math.atan2(L_FORE * math.sin(g), L_UP + L_FORE * math.cos(g))
                bf = bu + g
                q2 = math.pi - bu
                psi = DELTA - bf
                q3 = q2 - psi
                q4 = psi - phi
                q = np.array([q1, q2, q3, q4, q5, 0.0])
                q[:5] = [_fit(v, LIM[i]) for i, v in enumerate(q[:5])]
                q[5] = _fit(_q6(arm, q[:5], T6[:3, :3]), LIM[5])
                out.append(q)
    return out

def _fit(v, lim):
    for k in (0, -1, 1, -2, 2):
        x = v + 2 * math.pi * k
        if lim[0] - 1e-9 <= x <= lim[1] + 1e-9:
            return x
    return v

def _within(q, margin=0.0):
    return bool(np.all(q >= LIM[:, 0] - 1e-6 + margin) and np.all(q <= LIM[:, 1] + 1e-6 - margin))

def _polish(arm, T6, q, iters=3):
    q, ep, er = kin.ik(arm, T6, q, iters=iters, tol_p=2e-4, tol_r=2e-3)
    return q, ep, er

def solutions(arm, T6, margin=0.0):
    out = []
    for q in raw_solutions(arm, T6):
        if not _within(q, margin):
            continue
        T = kin.fk(arm, q)
        ep = float(np.linalg.norm(T[:3, 3] - T6[:3, 3]))
        if ep > 2e-3:
            q, ep, _ = _polish(arm, T6, q)
            if ep > 2e-3 or not _within(q):
                continue
        out.append(q)
    return out

def solve_tcp(arm, Ttcp, q_ref, margin=0.0, offset=kin.TCP_X):
    sols = solutions(arm, kin.link6_from_tcp(Ttcp, offset), margin)
    if not sols:
        return None
    q_ref = np.asarray(q_ref, float)
    return min(sols, key=lambda q: float(np.max(np.abs(q - q_ref) * _W)))

_W = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.5])
