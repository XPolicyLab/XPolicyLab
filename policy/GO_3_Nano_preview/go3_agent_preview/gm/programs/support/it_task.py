from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.motion import plan as m
from ...lib.geometry.transforms import TABLE_Z, axis_angle, make_T, rot_z, wrap
from . import it_see as S

FINGERTIP = kin.FINGER_TIP_X - kin.TCP_X
GRIP_FROM_TIP = 0.085

PINCH_TCP_Z = 0.79645
AXIS_Z_GRASP = S.TUBE_TIP_Z + GRIP_FROM_TIP * np.sin(S.TUBE_TILT)
PINCH_DZ = PINCH_TCP_Z - AXIS_Z_GRASP
OPEN_APPROACH = 0.62
HOLD_CMD = 0.0
HOLD_MIN = 0.12
RELEASE = 0.88

LIFT = 0.12
LIFT1 = 0.035
STAND_REACH = 0.36

SEATED_TIP_MAX = 0.77992
DEEP_ENOUGH = SEATED_TIP_MAX

ABOVE = 0.06
CARRY_TIP_Z = TABLE_Z + 0.150
ENTRY_TIP_Z = S.RACK_TOP + 0.008
SEAT_TIP_Z = SEATED_TIP_MAX - 0.0038
DESCENT_STEP = 0.003
BLOCK_LAG = 0.0025

SLIDE_MAX = 0.005

CAP_SHIFT_MAX_PX = 30
REINSERT_TRIES = 1
REALIGN_GATE = np.radians(1.0)
REINSERT_STEPS = 32
FINISH_STEPS = 36
SEATED_SCORE_MIN = 0.25

LEAN_TX_REF = np.radians(-0.53)
LEAN_TY_REF = np.radians(1.22)
LEAN_SLIDE_REF = 0.0005
LEAN_GATE = np.radians(4.0)
LEAN_RES_MAX = 8.0
CAP_DROP_MAX = 0.008
SLIDE_COMP_MAX = 0.0056
RISE = 0.07
BACK = 0.10

SEATED_CAP_TOP = S.RACK_TOP + 0.063

FRONT_ROW = -0.01797
COLUMNS = (-0.07198, 0.0, 0.07198)

SURVEY_Q = np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0])

def tilted_axis(a_xy):
    a = np.asarray(a_xy, float)[:2]
    a = a / np.linalg.norm(a)
    return np.array([a[0] * np.cos(S.TUBE_TILT), a[1] * np.cos(S.TUBE_TILT), np.sin(S.TUBE_TILT)])

def pinch_T(g_xy, a_xy, flip=False):
    a = np.asarray(a_xy, float)[:2]
    a = a / np.linalg.norm(a)
    yaw = float(np.arctan2(a[0], -a[1]))
    if flip:
        yaw += np.pi
    return m.down_pose([g_xy[0], g_xy[1], PINCH_TCP_Z], yaw)

def stand_R(R_pinch, a3, psi):
    up = np.array([0.0, 0.0, 1.0])
    k = np.cross(a3, up)
    th = float(np.arccos(np.clip(a3 @ up, -1.0, 1.0)))
    R1 = axis_angle(k / np.linalg.norm(k), th) @ np.asarray(R_pinch, float)[:3, :3]
    return rot_z(psi) @ R1

def hand_psi(R_pinch, a3, rack_yaw):
    x = stand_R(R_pinch, a3, 0.0)[:, 0]
    want = np.array([-np.sin(rack_yaw), np.cos(rack_yaw)])
    return float(wrap(np.arctan2(want[1], want[0]) - np.arctan2(x[1], x[0])))

def tcp_T(R, contact):
    return make_T(R, np.asarray(contact, float) - R[:, 0] * PINCH_DZ)

def contact_at(T):
    return T[:3, 3] + T[:3, 0] * PINCH_DZ

def tip_z_of(T):
    return float(contact_at(T)[2] - GRIP_FROM_TIP)

def poses(tube, bore, rack_yaw, flip, base_xy):
    a = tube["axis"]
    g = tube["tip"] + GRIP_FROM_TIP * a
    a3 = tilted_axis(a)
    Tp = pinch_T(g, a, flip)
    Ta = Tp.copy()
    Ta[2, 3] += ABOVE
    Tl1 = Tp.copy()
    Tl1[2, 3] += LIFT1
    fwd = g - np.asarray(base_xy, float)
    dist = float(np.linalg.norm(fwd))
    gs = g - fwd / dist * max(0.0, dist - STAND_REACH)
    Tl = Tp.copy()
    Tl[:2, 3] += gs - g
    Tl[2, 3] += LIFT
    R2 = stand_R(Tp, a3, hand_psi(Tp, a3, rack_yaw))
    Ts = tcp_T(R2, [gs[0], gs[1], AXIS_Z_GRASP + LIFT])
    Tc = tcp_T(R2, [bore[0], bore[1], CARRY_TIP_Z + GRIP_FROM_TIP])
    Te = tcp_T(R2, [bore[0], bore[1], ENTRY_TIP_Z + GRIP_FROM_TIP])
    Tse = tcp_T(R2, [bore[0], bore[1], SEAT_TIP_Z + GRIP_FROM_TIP])
    Tr = Tse.copy()
    Tr[2, 3] += RISE
    back = R2[:2, 0] / np.linalg.norm(R2[:2, 0])
    Tb = Tr.copy()
    Tb[:2, 3] -= back * BACK
    return {"above": Ta, "pinch": Tp, "lift1": Tl1, "lift": Tl, "stand": Ts, "carry": Tc, "entry": Te,
            "seat": Tse, "rise": Tr, "back": Tb, "grasp": g, "a3": a3}

CHAIN = ("above", "pinch", "lift1", "lift", "stand", "carry", "entry", "seat", "rise", "back")
LINE_STEPS = ("pinch", "lift1", "entry", "seat", "rise")

def plan(arm, tube, bore, rack_yaw, q0):
    best = None
    base = np.array(kin.BASES[arm][0][:2], float)
    for flip in (False, True):
        P = poses(tube, bore, rack_yaw, flip, base)
        q = np.asarray(q0, float)
        qs = {}
        cost = 0.0
        ok = True
        for name in CHAIN:
            try:
                qn = m.solve(arm, P[name], q)
            except m.PlanError:
                ok = False
                break
            dq = float(np.max(np.abs(wrap(qn - q))))
            if name in LINE_STEPS and dq > 0.6:
                ok = False
                break
            cost += dq
            qs[name] = qn
            q = qn
        if ok and (best is None or cost < best[0]):
            best = (cost, flip, P, qs)
    if best is None:
        return None
    return {"cost": best[0], "flip": best[1], "T": best[2], "q": best[3]}

HAND_LINKS = ("link5", "link6", "link7", "link8")
SEATED_CLEAR_MIN = 0.010

def _box_grid(lo, hi, n=7):
    return np.array([[x, y, z] for x in np.linspace(lo[0], hi[0], n) for y in np.linspace(lo[1], hi[1], n)
                     for z in np.linspace(lo[2], hi[2], n)])

_HAND_GRIDS = {k: _box_grid(*map(np.array, kin.LINK_BOXES[k])) for k in HAND_LINKS}

def hand_points(arm, q):
    fr = kin.link_frames(arm, q)
    return np.vstack([_HAND_GRIDS[k] @ fr[k][:3, :3].T + fr[k][:3, 3] for k in HAND_LINKS])

def seated_clearance(P, bores):
    c = np.inf
    for b in bores:
        d = np.hypot(P[:, 0] - b[0], P[:, 1] - b[1]) - S.TUBE_R_CAP
        c = min(c, float(np.where(P[:, 2] <= SEATED_CAP_TOP, d, np.inf).min()))
    return c

def assign_bores(rk, tubes):
    order = sorted(range(len(tubes)), key=lambda i: tubes[i]["tip"][0] + 0.0 * tubes[i]["axis"][0])
    cols = COLUMNS if len(tubes) == 3 else COLUMNS[: len(tubes)]
    out = {}
    for i, lx in zip(order, cols):
        out[i] = (lx, S.bore_xy(rk, lx, FRONT_ROW))
    return out
