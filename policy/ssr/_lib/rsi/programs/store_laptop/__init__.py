from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np

from ...core import actions
from ...perception import robot_mask
from . import scene as SS
from ...perception.cameras import head
from ...planning import kinematics as K
from ...planning import motion
from ...skills import dual
from ...skills.runner import Arm, link_arms, run_arms
from .. import Program, register

Z_UP = np.array([0.0, 0.0, 1.0])
STACK_MID_Z = 0.003
PINCH_Y = 0.002
HINGE_EDGE_Y = 0.1248
HINGE_Y, HINGE_Z = SS.HINGE_Y, SS.HINGE_Z
PUSH_R = 0.1
PUSH_X = -0.07
PUSH_ENTRY_GAP = 0.03
PUSH_GAP = 0.019
PUSH_DYAW = 30.0
PUSH_PHIS = tuple(range(115, -5, -5))
PUSH_RATE_HI = 1.0
PUSH_RATE_LO = 1.6
PUSH_PHI_FAST = 20.0
PUSH_RATE_START = 1.0
PUSH_PHI_SLOW = 80.0
PUSH_ROLL = 0.0
PUSH_LIN = 0.0018
PUSH_ENTRY = 0.11
LID_SETTLE = 0
UP_X, UP_Y = 0.05, -0.15
TILT = 65.0
HINGE_IN = 0.027
FAR_L = HINGE_EDGE_Y - HINGE_IN
TIP_DOWN = 0.004
TILT_CLEAR = 0.006
TILT_HALF_T = 0.0065


def tilt_raise() -> float:
    if TILT_CLEAR is None:
        return 0.0
    return float(TILT_CLEAR + TIP_DOWN + TILT_HALF_T * math.cos(math.radians(TILT)))


PRE = 0.04
TURN_STEP = 3.0
RACK_SLOT_Z = 0.77
RACK_LIN = 0.0065
RACK_REST = 6
RACK_WRIST_SPAN = 0.04
RACK_WRIST_DYAW = 20.0
RACK_WRIST_IOU = 0.8
RACK_WRIST_PX = 3000
RELEASE_GAP = 0.02
DROP_H = 0.018
DROP_SETTLE = 4
DROP_FOLLOW = 0.015
RISER_TOP = 0.819
SLID_FAR_L = 0.126
RISER_CLEAR = 0.008
SLOT_SHIFTS = (
    0.0,
    0.015,
    0.03,
    0.045,
)
RIGHT_BASE_XY = np.array([0.3, -0.45])
STAND_CLEAR = 0.035
PLATE_HALF_W = 0.0897 + 0.01
FREE_JOINT_STEP = 0.1
EMPTY_JOINT_STEP = 0.14
R_TILT_DEG = 1.6
CARRY_TURN_STEP = 2.0
CARRY_Z_SAFE = 0.15
CARRY_WIDE_DZ = (
    0.022,
    0.034,
)
CARRY_WIDE_CHECK = 0.008
PRESENT_Z = 0.96


def _is_store(instruction: str) -> bool:
    s = instruction.lower()
    return "laptop" in s and ("headphone" in s or "headset" in s) and ("stand" in s)


def tcp_T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def gripper_rot(d, c) -> np.ndarray:
    x = np.asarray(d, float) / np.linalg.norm(d)
    y = np.asarray(c, float) - x * (x @ np.asarray(c, float))
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def choose_rot(arm: Arm, R: np.ndarray, p, q) -> np.ndarray | None:
    best, cost = None, math.inf
    for Rr in (R, R @ K.rot_x(math.pi)):
        r = K.solve_near(arm.name, tcp_T(Rr, p), q, K.TCP_OFFSET, n_random=4)
        if r.ok:
            c = float(np.sum(np.abs(np.asarray(r.q) - np.asarray(q))))
            if c < cost:
                best, cost = Rr, c
    return best


def obs_tcp(arm: Arm) -> np.ndarray:
    q = arm.ctx.obs.state[f"{arm.name}_arm_joint_state"]
    return K.point_of(K.fk(arm.name, q), K.TCP_OFFSET)


def obs_grip(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


class Shared:
    def __init__(self):
        self.flags: set[str] = set()
        self.TL: np.ndarray | None = None
        self.rack = None
        self.TLp: np.ndarray | None = None
        self.lstand = None
        self.hstand = None
        self.right_push = False
        self.hs = None
        self.hp_zone = False
        self.TL0: np.ndarray | None = None
        self.hp_side = None


def rigid(Rax: np.ndarray, pivot, shift, T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rax @ T[:3, :3]
    out[:3, 3] = pivot + Rax @ (T[:3, 3] - pivot) + shift
    return out


def minjerk(n: int) -> np.ndarray:
    t = np.arange(1, n + 1) / max(1, n)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def path_plan(arm: Arm, poses, offset: float, max_joint_step: float = 0.15) -> motion.Plan:
    q = np.asarray(arm.q, dtype=float).copy()
    out = motion.Plan()
    n = len(poses)
    for k, T in enumerate(poses):
        r = K.solve(arm.name, T, q, offset, iters=60)
        if not r.ok and (not (k < n - 1 and r.pos_err < 0.004 and (r.rot_err < math.radians(2.0)))):
            out.ok = False
            out.reason = (
                f"ik failed at {k + 1}/{n} (pos {r.pos_err * 1000:.1f} mm, rot {math.degrees(r.rot_err):.1f} deg)"
            )
            return out
        if float(np.max(np.abs(r.q - q))) > max_joint_step:
            out.ok = False
            out.reason = f"joint step {float(np.max(np.abs(r.q - q))):.2f} rad at {k + 1}/{n}"
            return out
        q = r.q
        out.q.append(q.copy())
    return out


def follow_poses(arm: Arm, poses, offset: float = K.TCP_OFFSET, grip=None):
    plan = path_plan(arm, poses, offset)
    if not plan.ok:
        arm.status = "nopath"
        return
    ok = yield from arm.follow(plan, grip)
    arm.status = "ok" if ok else "blocked"


def turn_about(arm: Arm, axis, pivot, G: np.ndarray, deg: float, n: int):
    for m in (n, int(math.ceil(1.7 * n))):
        poses = [rigid(K.axis_angle(axis, math.radians(deg) * s), pivot, np.zeros(3), G) for s in minjerk(m)]
        yield from follow_poses(arm, poses, offset=0.0)
        if arm.status != "nopath":
            return
    steps = max(1, int(math.ceil(abs(deg) / TURN_STEP)))
    for k in range(1, steps + 1):
        T = rigid(K.axis_angle(axis, math.radians(deg * k / steps)), pivot, np.zeros(3), G)
        yield from arm.line_to(T, offset=0.0, lin_step=0.01, ang_step=math.radians(TURN_STEP + 0.1), joint_step=0.1)
        if arm.status != "ok":
            return


def follow_retimed(
    arm: Arm,
    pose_fn,
    n_fine: int,
    offset: float = 0.0,
    max_joint: float = 0.08,
    max_lin: float = 0.012,
    max_ang: float = math.radians(3.0),
    ramp: int = 6,
):
    us = np.linspace(0.0, 1.0, n_fine + 1)
    poses = [pose_fn(float(u)) for u in us]
    fine = path_plan(arm, poses[1:], offset, max_joint_step=0.3)
    if not fine.ok:
        arm.status = "nopath"
        return
    qs = [np.asarray(arm.q, dtype=float)] + fine.q
    cost = [0.0]
    for i in range(1, len(qs)):
        dq = float(np.max(np.abs(qs[i] - qs[i - 1]))) / max_joint
        dp = float(np.linalg.norm(K.point_of(poses[i], offset) - K.point_of(poses[i - 1], offset))) / max_lin
        da = K.rot_angle(poses[i - 1][:3, :3], poses[i][:3, :3]) / max_ang
        cost.append(cost[-1] + max(dq, dp, da))
    total = cost[-1]
    r = max(1, min(ramp, int(total // 3) + 1))
    n = int(math.ceil(total + r)) + 1
    v = np.ones(n)
    for k in range(r):
        f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / (r + 1))
        v[k] = min(v[k], f)
        v[n - 1 - k] = min(v[n - 1 - k], f)
    c = np.cumsum(v)
    c *= total / c[-1]
    plan = motion.Plan()
    for ck in c:
        i = int(np.searchsorted(cost, ck))
        i = min(max(i, 1), len(qs) - 1)
        w = (ck - cost[i - 1]) / max(1e-09, cost[i] - cost[i - 1])
        plan.q.append(qs[i - 1] + (qs[i] - qs[i - 1]) * float(np.clip(w, 0.0, 1.0)))
    plan.q[-1] = qs[-1].copy()
    ok = yield from arm.follow(plan)
    arm.status = "ok" if ok else "blocked"


def push_phis() -> list[float]:
    out = []
    phi = float(PUSH_PHIS[0])
    k = 0
    while phi > 0.0:
        rate = PUSH_RATE_START if phi > PUSH_PHI_SLOW else PUSH_RATE_HI if phi > PUSH_PHI_FAST else PUSH_RATE_LO
        rate *= min(1.0, (k + 1) / 4.0)
        phi = max(0.0, phi - rate)
        out.append(phi)
        k += 1
    return out


def camera_side(R: np.ndarray, up) -> np.ndarray:
    return R if R[:, 2] @ np.asarray(up, float) >= 0 else R @ K.rot_x(math.pi)


def obs_T(arm: Arm) -> np.ndarray:
    return K.fk(arm.name, np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], float))


def tcp1() -> np.ndarray:
    t = math.radians(TILT)
    return np.array([UP_X, UP_Y - FAR_L * math.cos(t), K.TABLE_Z - TIP_DOWN + FAR_L * math.sin(t)])


LAP_C = np.array([0.0, 0.0623, 0.003])
CARRY_LIN = 0.0065
CARRY_DEG = 1.15


def carry_moves(TL0: np.ndarray, TLp: np.ndarray):
    c0 = TL0[:3, :3] @ LAP_C + TL0[:3, 3]
    c1 = TLp[:3, :3] @ LAP_C + TLp[:3, 3]
    yb = TL0[:3, 1].copy()
    yb[2] = 0.0
    yb /= np.linalg.norm(yb)
    t = (c1[1] - c0[1]) / yb[1]
    v1 = t * yb
    n1 = max(10, int(math.ceil(abs(t) / CARRY_LIN)))
    P1 = [rigid(np.eye(3), c0, s * v1, TL0) for s in minjerk(n1)]
    TL1 = P1[-1]
    cs = c0 + v1
    psi0 = math.atan2(TL0[1, 1], TL0[0, 1])
    psi1 = math.atan2(TLp[1, 1], TLp[0, 1])
    d = (psi1 - psi0 + math.pi) % (2 * math.pi) - math.pi
    n2 = max(20, int(math.ceil(abs(math.degrees(d)) / CARRY_DEG)))
    P2 = [rigid(K.rot_z(d * s), cs, np.zeros(3), TL1) for s in minjerk(n2)]
    TL2 = P2[-1]
    dv = TLp[:3, 3] - TL2[:3, 3]
    n3 = max(8, int(math.ceil(float(np.linalg.norm(dv)) / CARRY_LIN)))
    P3 = []
    for s in minjerk(n3):
        T = TL2.copy()
        T[:3, 3] = TL2[:3, 3] + s * dv
        P3.append(T)
    P3[-1][:3, :3] = TLp[:3, :3]
    return [("carry_pull", P1), ("carry_turn", P2), ("carry_slide", P3)]


def left_carry_task(arm: Arm, sh: Shared, TL: np.ndarray, home: bool = True, early: bool = False):
    if not early and "pushed" not in sh.flags:
        sh.flags.add("carry_failed")
        return
    yield from arm.hold(LID_SETTLE)
    yl, o = TL[:3, 1], TL[:3, 3]
    target = o + PINCH_Y * yl + STACK_MID_Z * Z_UP
    R = choose_rot(arm, gripper_rot(yl, Z_UP), target, arm.q)
    if R is None:
        sh.flags.add("carry_failed")
        return
    yield from arm.transit(
        tcp_T(R, target - 0.065 * yl),
        z_safe=o[2] + CARRY_Z_SAFE,
        grip=dual.grip_for_gap(0.05),
        joint_step=FREE_JOINT_STEP,
    )
    if arm.status != "ok":
        sh.flags.add("carry_failed")
        return
    if early:
        yield from arm.wait_until(lambda: sh.flags & {"push_clear", "push_failed"}, RIGHT_LID_WAIT)
        if "push_clear" not in sh.flags:
            sh.flags.add("carry_failed")
            return
    wide = False
    dz = float(obs_tcp(arm)[2] - target[2])
    if dz >= CARRY_WIDE_CHECK:
        yield from arm.hold(2)
        dz = float(obs_tcp(arm)[2] - target[2])
    if CARRY_WIDE_DZ[0] <= dz <= CARRY_WIDE_DZ[1]:
        wide = True
        yield from arm.set_grip(1.0, n=6)
    yield from arm.line_to(tcp_T(R, target), lin_step=0.008)
    yield from arm.hold(3)
    yield from arm.set_grip(0.0, n=14 if wide else 10)
    yield from arm.hold(4)
    TLm = TL.copy()
    TLm[2, 3] = obs_tcp(arm)[2] - STACK_MID_Z
    H = np.linalg.inv(obs_T(arm)) @ TLm
    yield from arm.line_to(tcp_T(arm.T[:3, :3], arm.tcp() + max(0.03, PRESENT_Z - arm.tcp()[2]) * Z_UP), lin_step=0.008)
    TLp = np.eye(4)
    TLp[:3, :3] = K.rot_z(-math.pi / 2)
    TLp[:3, 3] = tcp1() + np.array([HINGE_IN - HINGE_EDGE_Y, 0.0, 0.0])
    TLp[2, 3] = PRESENT_Z - STACK_MID_Z
    Hi = np.linalg.inv(H)
    sh.TLp = TLp
    for what, TLs in carry_moves(arm.T @ H, TLp):
        if what == "carry_slide":
            sh.flags.add("carry_sliding")
        yield from follow_poses(arm, [T @ Hi for T in TLs], offset=0.0)
        if arm.status != "ok":
            break
    if arm.status == "nopath":
        yield from arm.line_to(TLp @ Hi, offset=0.0, lin_step=0.005, ang_step=math.radians(CARRY_TURN_STEP))
    if arm.status != "ok":
        sh.flags.add("carry_failed")
        return
    yield from arm.hold(6)
    sh.TL = obs_T(arm) @ H
    sh.flags.add("presented")
    yield from arm.wait_until(lambda: sh.flags & {"right_holds", "hinge_failed"}, 300)
    yield from arm.set_grip(dual.grip_for_gap(0.03), n=6)
    yield from arm.hold(3)
    sh.flags.add("left_open")
    p = arm.tcp()
    yield from arm.line_to(tcp_T(arm.T[:3, :3], p - 0.04 * arm.T[:3, 0]), lin_step=0.008)
    sh.flags.add("left_free")
    if home:
        yield from arm.line_to(tcp_T(arm.T[:3, :3], arm.tcp() + 0.05 * Z_UP), lin_step=0.01)
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)


HP_PUSH_Z = 0.015
HP_PUSH_CLEAR = 0.03
HP_PUSH_LIN = 0.006
HP_PUSH_START = 0.035
HP_PUSH_TILT = 30.0
HP_PUSH_BACK = 0.12
HP_CARRY_CLEAR = 0.045
HP_PUSH_EXTRA = tuple(0.01 * i for i in range(9))
HP_REFIT_YAW = -45.0, 110.0
HP_REFIT_GATE = 0.15


def carry_approach_qs(TL: np.ndarray, q_seed) -> list:
    yl, o = TL[:3, 1], TL[:3, 3]
    target = o + PINCH_Y * yl + STACK_MID_Z * Z_UP
    q = np.asarray(q_seed, float)
    R, cost = None, math.inf
    for Rr in (gripper_rot(yl, Z_UP), gripper_rot(yl, Z_UP) @ K.rot_x(math.pi)):
        r = K.solve_near("left", tcp_T(Rr, target), q, K.TCP_OFFSET, n_random=4)
        if r.ok and float(np.sum(np.abs(np.asarray(r.q) - q))) < cost:
            R, cost = Rr, float(np.sum(np.abs(np.asarray(r.q) - q)))
    if R is None:
        return []
    pre = target - 0.065 * yl
    qs = []
    for s in np.linspace(0.0, 1.0, 5):
        r = K.solve_near("left", tcp_T(R, pre + s * (target - pre)), q, K.TCP_OFFSET, n_random=4)
        if r.ok:
            q = np.asarray(r.q)
            qs.append(q)
    return qs


def push_target_x(hs, TL0, q_left) -> tuple[float, float]:
    x_t = tilt_zone()[0] - HP_PUSH_CLEAR
    if TL0 is None or q_left is None:
        return x_t, math.nan
    qs = carry_approach_qs(TL0, q_left)
    if not qs:
        return x_t, math.nan
    W = headset_points(hs)
    x_r = float(W[:, 0].max())
    best = -math.inf, x_t
    for extra in HP_PUSH_EXTRA:
        Ws = W - np.array([x_r - (x_t - extra), 0.0, 0.0])
        c = arm_clearance("left", qs, Ws, step=1)
        if c >= HP_CARRY_CLEAR:
            return x_t - extra, c
        best = max(best, (c, x_t - extra))
    return best[1], best[0]


def push_plan(arm: Arm, hs, TL0=None, q_left=None, extra: float = 0.0):
    W = headset_points(hs)
    x_far, clr = push_target_x(hs, TL0, q_left)
    x_far -= extra
    x_r = float(W[:, 0].max())
    Th = hs.T()
    y_p = float((Th[:3, :3] @ np.array([0.0, 0.0, -0.035]) + Th[:3, 3])[1])
    z = K.TABLE_Z + HP_PUSH_Z
    off = K.FINGERTIP_OFFSET
    t = math.radians(HP_PUSH_TILT)
    R0 = gripper_rot(np.array([-math.sin(t), 0.0, -math.cos(t)]), np.array([0.0, 1.0, 0.0]))
    aligns = [(0.0, 0.0)]
    a, sh_x = push_align(W, y_p, z)
    aligns = [(a, sh_x), (0.0, 0.0)] if abs(a) > 1e-09 else aligns
    x_ts = [x_far] + ([tilt_zone()[0] - HP_PUSH_CLEAR] if x_far < tilt_zone()[0] - HP_PUSH_CLEAR else [])
    for x_t, (a, sh_x) in [(x_t, al) for x_t in x_ts for al in aligns]:
        D = x_r - x_t
        if D <= 0.0:
            return None
        end = np.array([x_t + 0.01 + sh_x, y_p, z])
        start = np.array([x_r + HP_PUSH_START + sh_x, y_p, z])
        Ra = K.rot_z(a) @ R0
        for R in (Ra, Ra @ K.rot_x(math.pi)):
            q, ok = np.asarray(arm.q, float), True
            for pt in (start + 0.08 * Z_UP, start, end, end + 0.08 * Z_UP):
                r = K.solve_near(arm.name, tcp_T(R, pt), q, off, n_random=4)
                if not r.ok:
                    ok = False
                    break
                q = np.asarray(r.q)
            if ok:
                return dict(
                    R=R,
                    start=start,
                    end=end,
                    D=D,
                    carry_clear=clr if x_t == x_far else math.nan,
                    align_deg=math.degrees(a),
                )


HP_PUSH_ALIGN_MAX = 25.0
HP_FINGER_SPAN = 0.0444, 0.0644


def push_align(W: np.ndarray, y_p: float, z_tip: float) -> tuple[float, float]:
    t = math.tan(math.radians(HP_PUSH_TILT))
    band = W[(W[:, 2] >= z_tip) & (W[:, 2] <= z_tip + 0.06)]
    if len(band) == 0:
        return 0.0, 0.0
    key = band[:, 0] - (band[:, 2] - z_tip) * t

    def contacts(a):
        out = []
        for s in (-1.0, 1.0):
            lo, hi = sorted((s * HP_FINGER_SPAN[0], s * HP_FINGER_SPAN[1]))
            dy0, dx_per = math.cos(a), -math.sin(a)
            off = (band[:, 1] - y_p) / max(dy0, 1e-06)
            m = (off >= lo) & (off <= hi)
            if not np.any(m):
                return None
            out.append(float(np.max(key[m] - off[m] * dx_per)))
        return out

    c0 = contacts(0.0)
    if c0 is None:
        return 0.0, 0.0
    best = abs(c0[0] - c0[1]), 0.0, max(c0)
    for a_deg in np.arange(-HP_PUSH_ALIGN_MAX, HP_PUSH_ALIGN_MAX + 1e-09, 2.5):
        c = contacts(math.radians(a_deg))
        if c is not None and abs(c[0] - c[1]) < best[0] - 0.0001:
            best = abs(c[0] - c[1]), math.radians(a_deg), max(c)
    return best[1], best[2] - max(c0)


def _home_of(arm):
    return None if arm is None else np.asarray(arm.q_home, float)


def right_push_task(arm: Arm, sh: Shared, hs):
    for attempt in range(HP_PUSH_TRIES):
        if attempt:
            plan = push_plan(arm, hs, sh.TL0, _home_of(arm.other))
            end = arm.ctx.obs.step + HP_PUSH_FIXED + (plan["D"] / HP_PUSH_LIN if plan is not None else 0.0)
            if end > HP_PUSH_END_BY:
                return
        hs, measured = yield from _push_once(arm, sh, hs)
        if hs is None or not measured or headset_case(hs) != "zone":
            return


HP_PUSH_TRIES = 2
HP_PUSH_END_BY = 260
HP_PUSH_FIXED = 90


def _push_once(arm: Arm, sh: Shared, hs):
    plan = push_plan(arm, hs, sh.TL0, _home_of(arm.other), extra=0.0)
    if plan is None:
        return None, False
    off, R = K.FINGERTIP_OFFSET, plan["R"]
    yield from arm.transit(
        tcp_T(R, plan["start"]), z_safe=plan["start"][2] + 0.08, offset=off, grip=1.0, joint_step=FREE_JOINT_STEP
    )
    if arm.status != "ok":
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
        return None, False
    yield from arm.line_to(tcp_T(R, plan["end"]), offset=off, lin_step=HP_PUSH_LIN)
    tip = K.point_of(arm.T, off)
    moved = float(plan["start"][0] - tip[0]) - HP_PUSH_START
    back = min(HP_PUSH_BACK, max(0.0, float(plan["start"][0] - tip[0])))
    if back > 0.0:
        yield from arm.line_to(tcp_T(R, tip + np.array([back, 0.0, 0.005])), offset=off, lin_step=0.012)
    yield from arm.line_to(tcp_T(R, K.point_of(arm.T, off) + 0.08 * Z_UP), offset=off, lin_step=0.01)
    yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
    hs2 = SS.Fit(hs.category, hs.x - moved, hs.y, hs.yaw, hs.z, hs.iou, hs.index)
    return _refit_after_move(
        arm, sh, hs, hs2, (hs.yaw + math.radians(HP_REFIT_YAW[0]), hs.yaw + math.radians(HP_REFIT_YAW[1]))
    )


def _refit_after_move(arm: Arm, sh: Shared, hs, hs2, yaw_rng):
    o = arm.ctx.obs
    try:
        ign = robot_mask.mask(
            head(),
            o.state["left_arm_joint_state"],
            o.state["right_arm_joint_state"],
            float(o.state["left_ee_joint_state"][0]),
            float(o.state["right_ee_joint_state"][0]),
        )
        f = SS.perceive(
            o.images["cam_head"], ignore=ign, only=("headset",), headset_index=hs.index, headset_yaw=yaw_rng
        ).headset
    except Exception:
        f = None
    if f is not None and abs(f.x - hs2.x) < HP_REFIT_GATE and (abs(f.y - hs2.y) < HP_REFIT_GATE):
        hs2 = f
    sh.hs = hs2
    sh.hp_zone = headset_case(hs2) == "zone"
    return hs2, hs2 is f


def left_after_task(arm: Arm, sh: Shared, hs, st, case: str):
    if getattr(sh, "hs", None) is not None:
        hs = sh.hs
        case = headset_case(hs)
    if hs is not None and st is not None:
        if case == "right":
            yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
            yield from arm.wait_until(
                lambda: (
                    sh.flags
                    & {
                        "over_rack",
                        "pivot_failed",
                        "hinge_failed",
                        "probe_done",
                        "carry_failed",
                        "push_failed",
                        "right_gone",
                    }
                ),
                400,
            )
            yield from left_headset_task(arm, sh, hs, st)
        elif case == "zone":
            yield from left_headset_task(
                arm,
                sh,
                hs,
                st,
                x_max=laptop_far_x() - HP_LAPTOP_CLEAR,
                lifts=(HP_ZONE_LIFT,),
                clear_x=laptop_far_x() - ZONE_MARGIN,
                direct=True,
            )
        else:
            x_max = min(
                laptop_far_x() - HP_LAPTOP_CLEAR, pivot_x() - RAIL_HALF - RAIL_GAP - RAIL_HALF * abs(math.cos(hs.yaw))
            )
            yield from left_headset_task(
                arm,
                sh,
                hs,
                st,
                x_max=x_max,
                direct=True,
                wait_flags=("zone_free", "pivot_failed", "hinge_failed", "probe_done"),
            )
    sh.flags.add("hp_clear")
    yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)


PRESENT_FIT_IOU = 0.94
PRESENT_FIT_DYAW = 15.0
PRESENT_FIT_MAX_DYAW = 18.0
PRESENT_FIT_MAX = 0.04
PRESENT_FIT_WIDE_FRACS = 1.0, 0.75, 0.5, 0.25, 0.0
PRESENT_FIT_OVERLAP = (
    0.6
)
CENTRE_MIN_W = (
    0.0115
)
REGRASP_TRIES = 2
REGRASP_OPEN = 0.016
REGRASP_GAIN = 2.0
REGRASP_MAX = 0.011
REGRASP_MAX_W = 0.021


def regrasp_pinch(arm: Arm):
    w = dual.held_width(obs_grip(arm))
    if w <= CENTRE_MIN_W:
        return
    if w > REGRASP_MAX_W:
        return
    c = arm.T[:3, 1] if arm.T[2, 1] >= 0.0 else -arm.T[:3, 1]
    p = arm.tcp()
    sgn = 1.0
    for k in range(REGRASP_TRIES):
        d = min(REGRASP_MAX, REGRASP_GAIN * 0.5 * (w - 0.009))
        yield from arm.set_grip(dual.grip_for_gap(w + REGRASP_OPEN), n=1)
        p = p + sgn * d * c
        yield from arm.line_to(tcp_T(arm.T[:3, :3], p), lin_step=0.005)
        yield from arm.set_grip(0.0, n=2)
        yield from arm.hold(2)
        w2 = dual.held_width(obs_grip(arm))
        if w2 <= CENTRE_MIN_W:
            return
        if w2 > w - 0.001:
            sgn = -sgn
        w = w2


def presented_refit(arm: Arm, TL: np.ndarray) -> np.ndarray:
    o = arm.ctx.obs
    yaw0 = math.atan2(TL[1, 0], TL[0, 0])
    try:
        from ...perception import seg

        ign = robot_mask.mask(
            head(),
            o.state["left_arm_joint_state"],
            o.state["right_arm_joint_state"],
            float(o.state["left_ee_joint_state"][0]),
            float(o.state["right_ee_joint_state"][0]),
        )
        ins = seg.segment(o.images["cam_head"], camera="head")
        if not any(i.category == "laptop" for i in ins):
            P = SS.laptop_points(0.0) @ TL[:3, :3].T + TL[:3, 3]
            sil = SS.render(head(), P, 1.0) & ~ign
            n = max(1, int(sil.sum()))
            ov = [(float((i.mask.astype(bool) & sil).sum()) / n, i) for i in ins]
            ov = [(v, i) for v, i in ov if v >= PRESENT_FIT_OVERLAP]
            ins = [dataclasses.replace(max(ov, key=lambda t: t[0])[1], category="laptop")] if ov else []
        f = SS.fit_laptop(
            o.images["cam_head"],
            xy0=TL[:2, 3],
            yaw0=yaw0,
            z=float(TL[2, 3]),
            angle_deg=0.0,
            dyaw=math.radians(PRESENT_FIT_DYAW),
            ignore=ign,
            instances=ins,
        )
    except Exception:
        return TL
    if f is None:
        return TL
    TLf = TL.copy()
    TLf[:3, :3] = K.rot_z(f.yaw - yaw0) @ TL[:3, :3]
    TLf[:2, 3] = f.x, f.y
    dh = TLf[:3, 3] + HINGE_EDGE_Y * TLf[:3, 1] - (TL[:3, 3] + HINGE_EDGE_Y * TL[:3, 1])
    dyaw = math.degrees(f.yaw - yaw0)
    ok = (
        f.iou >= PRESENT_FIT_IOU
        and float(np.linalg.norm(dh[:2])) < PRESENT_FIT_MAX
        and (abs(dyaw) <= PRESENT_FIT_MAX_DYAW)
    )
    return TLf if ok else TL


ROOM_AFTER = 3
ROOM_MOVES = (0.06, 0.0, 0.04), (0.0, -0.06, 0.04), (0.05, -0.05, 0.0), (0.0, 0.0, 0.08)
ROOM_MIN_GAIN = 0.01
ROOM_MAX = 2


def wait_making_room(arm: Arm, cond, n_max: int):
    o = arm.other
    n, run, moves = 0, 0, 0
    last = o.waited if o is not None else 0
    while not cond() and n < n_max:
        n += 1
        if o is not None:
            run = run + 1 if o.waited > last else 0
            last = o.waited
        if run >= ROOM_AFTER and moves < ROOM_MAX:
            moves += 1
            run = 0
            d0 = arm._dist(arm.q)
            best = None
            for dp in ROOM_MOVES:
                r = K.solve_near(
                    arm.name, tcp_T(arm.T[:3, :3], arm.tcp() + np.asarray(dp)), arm.q, K.TCP_OFFSET, n_random=2
                )
                if r.ok:
                    d = arm._dist(r.q)
                    if best is None or d > best[0]:
                        best = d, dp
            if best is not None and best[0] > d0 + ROOM_MIN_GAIN:
                arm.waiting = False
                yield from arm.line_to(tcp_T(arm.T[:3, :3], arm.tcp() + np.asarray(best[1])), lin_step=0.012)
                continue
        arm.waiting = True
        yield
    arm.waiting = False


def right_task(arm: Arm, sh: Shared):
    first = lambda: sh.flags & {"carry_sliding", "presented", "carry_failed", "push_failed"}
    yield from wait_making_room(arm, first, 700)
    if not sh.flags & {"carry_sliding", "presented"}:
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
        return

    def hinge_target(TL):
        return TL[:3, 3] + (HINGE_EDGE_Y - HINGE_IN) * TL[:3, 1] + STACK_MID_Z * TL[:3, 2]

    TLpl = sh.TLp if sh.TLp is not None else sh.TL
    d = -TLpl[:3, 1]
    target = hinge_target(TLpl)
    R = choose_rot(arm, gripper_rot(d, TLpl[:3, 2]), target, arm.q)
    if R is None:
        sh.flags.add("hinge_failed")
        return
    yield from arm.move_to(
        tcp_T(R, target - PRE * d), min_z=target[2] - 0.04, grip=dual.grip_for_gap(0.035), joint_step=FREE_JOINT_STEP
    )
    yield from arm.wait_until(lambda: sh.flags & {"presented", "carry_failed"}, 300)
    if "presented" not in sh.flags:
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
        return
    TL = presented_refit(arm, sh.TL)
    if TL is not sh.TL:
        R2 = choose_rot(arm, gripper_rot(-TL[:3, 1], TL[:3, 2]), hinge_target(TL), arm.q)
        if "present_wide" in sh.flags:
            tgt = hinge_target(TL)
            dyaw_m = math.atan2(TL[1, 0], TL[0, 0]) - math.atan2(sh.TL[1, 0], sh.TL[0, 0])
            frac = 1.0
            for frac in PRESENT_FIT_WIDE_FRACS:
                R2 = choose_rot(arm, gripper_rot(K.rot_z(frac * dyaw_m) @ -sh.TL[:3, 1], TL[:3, 2]), tgt, arm.q)
                if R2 is not None:
                    break
            if R2 is None:
                TL = sh.TL
            else:
                yield from arm.line_to(tcp_T(R2, tgt - PRE * R2[:3, 0]), lin_step=0.007, ang_step=math.radians(3.0))
        if R2 is not None:
            R = R2
        sh.TL = TL
    target = hinge_target(TL)
    yield from arm.line_to(tcp_T(R, target), lin_step=0.007)
    if arm.status != "ok":
        sh.flags.add("hinge_failed")
        return
    yield from arm.hold(3)
    yield from arm.set_grip(0.0, n=10)
    yield from arm.hold(4)
    yield from regrasp_pinch(arm)
    sh.flags.add("right_holds")
    piv = arm.tcp()
    yield from arm.wait_until(lambda: "left_free" in sh.flags, 150)
    if getattr(sh, "hp_zone", False):
        yield from arm.wait_until(lambda: sh.flags & {"hp_clear", "hp_failed"}, 150)
    u = -sh.TL[:3, 1].copy()
    u[2] = 0.0
    u /= np.linalg.norm(u)
    ex = np.cross(Z_UP, u)
    ex /= np.linalg.norm(ex)
    G1 = arm.T.copy()
    p = arm.tcp()
    lift = tilt_raise()
    dz = max(0.0, p[2] - (tcp1()[2] + lift))
    s_t = np.clip((p[2] - dz - (K.TABLE_Z - TIP_DOWN + lift)) / FAR_L, 0.0, 1.0)
    tilt = math.degrees(math.asin(s_t))

    def tilt_pose(u):
        return rigid(K.axis_angle(ex, math.radians(tilt) * u), p, np.array([0.0, 0.0, -dz * u]), G1)

    yield from follow_retimed(arm, tilt_pose, max(20, int(tilt * 2)), offset=0.0, max_ang=math.radians(2.4))
    if arm.status == "nopath":
        yield from follow_poses(
            arm, [tilt_pose(u) for u in minjerk(max(8, int(math.ceil(1.7 * tilt / R_TILT_DEG))))], offset=0.0
        )
    if arm.status != "ok":
        sh.flags.add("pivot_failed")
        return
    yield from arm.hold(2)
    G2 = arm.T.copy()
    d2 = G2[:3, 0]
    P = arm.tcp() + FAR_L * d2
    rest = 90.0 - tilt
    yield from turn_about(arm, ex, P, G2, rest, max(6, int(math.ceil(rest / R_TILT_DEG))))
    if arm.status != "ok":
        sh.flags.add("pivot_failed")
        return
    sh.TL = rigid(K.axis_angle(ex, math.radians(90.0)), piv, np.zeros(3), sh.TL)
    yield from arm.hold(2)
    yield from arm.line_to(tcp_T(arm.T[:3, :3], arm.tcp() + 0.025 * Z_UP), lin_step=0.004)
    yield from arm.hold(3)
    sh.flags.add("zone_free")
    yield from right_to_rack(arm, sh)
    sh.flags.add("probe_done")


def right_to_rack(arm: Arm, sh: Shared):
    rack = sh.rack
    if rack is None:
        return
    T0 = arm.T.copy()
    p = arm.tcp()
    xl = sh.TL[:3, 0]
    a_now = math.atan2(xl[1], xl[0])
    da = (rack.yaw - a_now + math.pi / 2) % math.pi - math.pi / 2
    R = K.rot_z(da) @ T0[:3, :3]
    goal = None
    over_slot = False
    u = np.array([math.cos(rack.yaw), math.sin(rack.yaw), 0.0])
    if u[:2] @ (RIGHT_BASE_XY - np.array([rack.x, rack.y])) < 0.0:
        u = -u
    z_req = RISER_TOP + SLID_FAR_L + RISER_CLEAR
    for s in SLOT_SHIFTS:
        for z in (0.96, 0.955, 0.95):
            if z < z_req - 1e-06:
                continue
            g = np.array([rack.x, rack.y, z]) + s * u
            if K.solve_near(arm.name, tcp_T(R, g), arm.q, K.TCP_OFFSET, n_random=4).ok:
                goal, over_slot = g, True
                break
        if goal is not None:
            break
    if goal is None:
        for dy in (0.0, -0.12):
            for z in (0.96, 0.95, 0.94, 0.93):
                if dy == 0.0 and z < 0.94:
                    continue
                g = np.array([rack.x, rack.y + dy, z])
                if K.solve_near(arm.name, tcp_T(R, g), arm.q, K.TCP_OFFSET, n_random=4).ok:
                    goal, over_slot = g, dy == 0.0
                    break
            if goal is not None:
                break
    if goal is None:
        return
    z_up = 0.912
    for zc in (goal[2], 0.95, 0.94, 0.93):
        if (
            zc <= goal[2] + 1e-06
            and K.solve_near(arm.name, tcp_T(T0[:3, :3], (p[0], p[1], zc)), arm.q, K.TCP_OFFSET, n_random=4).ok
        ):
            z_up = max(z_up, zc)
            break
    yield from arm.line_to(tcp_T(T0[:3, :3], (p[0], p[1], z_up)), lin_step=0.006)
    legs = []
    ls = sh.lstand
    if ls is not None:
        Tls = ls.T()
        W = SS.asset_points("laptop_stand", int(ls.index or 1)) @ Tls[:3, :3].T + Tls[:3, 3]
        x1 = float(W[:, 0].max()) + STAND_CLEAR
        if x1 > p[0] + 0.02 and goal[1] + PLATE_HALF_W > float(W[:, 1].min()) - 0.01:
            legs.append(("to_rack_x", tcp_T(T0[:3, :3], (min(x1, goal[0]), p[1], z_up))))
    legs.append(("to_rack", tcp_T(R, goal)))
    for what, Tb in legs:
        Ta = tcp_T(arm.T[:3, :3], arm.tcp())
        dist = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
        ang = math.degrees(K.rot_angle(Ta[:3, :3], Tb[:3, :3]))
        n = int(max(20, math.ceil(dist / RACK_LIN), math.ceil(ang / 2.0)))
        yield from follow_poses(
            arm,
            [
                tcp_T(K.slerp_mat(Ta[:3, :3], Tb[:3, :3], float(u)), Ta[:3, 3] + float(u) * (Tb[:3, 3] - Ta[:3, 3]))
                for u in minjerk(n)
            ],
        )
        if arm.status == "nopath":
            yield from arm.line_to(Tb, lin_step=0.008, ang_step=math.radians(4.0), joint_step=0.08)
        if arm.status != "ok":
            break
    yield from arm.hold(RACK_REST)
    sh.flags.add("over_rack")
    if arm.status != "ok" or not over_slot:
        return
    z_ins = RACK_SLOT_Z + FAR_L + DROP_H
    yield from arm.line_to(tcp_T(R, (goal[0], goal[1], z_ins)), lin_step=0.004)
    if arm.status != "ok":
        return
    yield from arm.hold(2)
    yield from arm.set_grip(dual.grip_for_gap(RELEASE_GAP), n=4)
    yield from arm.hold(DROP_SETTLE)
    yield from arm.line_to(tcp_T(R, (goal[0], goal[1], z_ins - DROP_FOLLOW)), lin_step=0.004)
    yield from arm.hold(2)
    yield from arm.set_grip(1.0, n=4)
    yield from arm.hold(1)
    z_up = z_ins + 0.05
    z_up = max(z_up, float(goal[2]))
    yield from arm.line_to(tcp_T(R, (goal[0], goal[1], z_up)), lin_step=0.012)
    yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=EMPTY_JOINT_STEP)


RIGHT_LID_HF_CASES = ("zone",)
RIGHT_LID_WAIT = 400
PUSH_BACK_HOLD = 14
PUSH_BACK_GAP = 0.04
PUSH_DYAW_ALT = (45.0,)
PUSH_STAND_MIN = 0.005


def transit_path(arm: Arm, T_goal: np.ndarray, z_safe: float, offset: float):
    q = np.asarray(arm.q, float)
    qs = [q]
    T = K.fk(arm.name, q)
    p0 = K.point_of(T, offset)
    pg = T_goal[:3, 3]
    z_cross = max(z_safe, pg[2])
    if p0[2] < z_safe - 0.003:
        up = T.copy()
        up[:3, 3] = [p0[0], p0[1], z_safe]
        plan = motion.line(arm.name, q, up, offset, lin_step=0.014)
        if not plan.ok:
            return None
        qs += list(plan.q)
        q = np.asarray(plan.q[-1], float)
    Tc = T_goal.copy()
    Tc[:3, 3] = [pg[0], pg[1], z_cross]
    plan = None
    r = K.solve_near(arm.name, Tc, q, offset, n_random=3)
    if r.ok:
        plan = motion.joint_move(q, r.q, joint_step=FREE_JOINT_STEP)
        zs = [K.point_of(K.fk(arm.name, qq), offset)[2] for qq in plan.q[:: max(1, len(plan.q) // 8)]]
        if min(zs + [10.0]) < z_safe - 0.005:
            plan = None
    if plan is None:
        plan = motion.line(arm.name, q, Tc, offset, lin_step=0.016, ang_step=math.radians(6))
        if not plan.ok:
            return None
    qs += list(plan.q)
    q = np.asarray(plan.q[-1], float)
    if z_cross > pg[2] + 0.002:
        plan = motion.line(arm.name, q, T_goal, offset, lin_step=0.012)
        if not plan.ok:
            return None
        qs += list(plan.q)
    return qs


def stand_points(st) -> np.ndarray:
    T = st.T()
    return SS.asset_points("headset_stand", 0) @ T[:3, :3].T + T[:3, 3]


def left_push_task(arm: Arm, sh: Shared, TL: np.ndarray):
    xl0, yl, zl, o = TL[:3, 0], TL[:3, 1], TL[:3, 2], TL[:3, 3]
    sgn = 1.0 if arm.name == "left" else -1.0
    xl = sgn * xl0
    hinge = o + HINGE_Y * yl + HINGE_Z * zl
    d = K.rot_z(math.radians(sgn * PUSH_DYAW)) @ xl
    R = camera_side(gripper_rot(d, np.cross(d, zl)), Z_UP) @ K.rot_x(math.radians(PUSH_ROLL))

    def tip(phi_deg, dx=0.0, gap=0.0):
        f = math.radians(phi_deg)
        s = -yl * math.cos(f) + zl * math.sin(f)
        n = yl * math.sin(f) + zl * math.cos(f)
        return hinge + PUSH_R * s + (PUSH_GAP + gap) * n + (PUSH_X + dx) * xl

    off = K.FINGERTIP_OFFSET
    entry = tip(PUSH_PHIS[0], -PUSH_ENTRY, PUSH_ENTRY_GAP)
    side = True
    back = tip(PUSH_PHIS[0], 0.0, PUSH_BACK_GAP)
    qs = transit_path(arm, tcp_T(R, back), o[2] + 0.15, off)
    W = None if sh.hstand is None else stand_points(sh.hstand)
    dyaw, tried = PUSH_DYAW, []
    if qs is not None:
        entry, side = back, False
    clear = None if qs is None or W is None else arm_clearance(arm.name, qs, W, step=3)
    if clear is not None and clear < PUSH_STAND_MIN:
        best = clear, dyaw, R
        for a in PUSH_DYAW_ALT:
            if a == dyaw:
                continue
            da = K.rot_z(math.radians(sgn * a)) @ xl
            Ra = camera_side(gripper_rot(da, np.cross(da, zl)), Z_UP) @ K.rot_x(math.radians(PUSH_ROLL))
            qa = transit_path(arm, tcp_T(Ra, back), o[2] + 0.15, off)
            ca = None if qa is None else arm_clearance(arm.name, qa, W, step=3)
            tried.append((a, ca))
            if ca is not None and ca > best[0]:
                best = ca, a, Ra
            if ca is not None and ca >= PUSH_STAND_MIN:
                break
        clear, dyaw, R = best
    yield from arm.transit(tcp_T(R, entry), z_safe=o[2] + 0.15, offset=off, grip=0.0, joint_step=FREE_JOINT_STEP)
    if arm.status != "ok":
        sh.flags.add("push_failed")
        return
    if side:
        yield from arm.line_to(tcp_T(R, tip(PUSH_PHIS[0], 0.0, PUSH_ENTRY_GAP)), offset=off, lin_step=0.009)
    elif sh.right_push:
        yield from arm.hold(PUSH_BACK_HOLD)
    yield from arm.line_to(tcp_T(R, tip(PUSH_PHIS[0])), offset=off, lin_step=0.005)
    sh.flags.add("push_started")
    yield from follow_poses(arm, [tcp_T(R, tip(phi)) for phi in push_phis()], offset=off)
    if arm.status == "nopath":
        for phi in PUSH_PHIS[1:]:
            yield from arm.line_to(tcp_T(R, tip(phi)), offset=off, lin_step=PUSH_LIN)
            if arm.status != "ok":
                break
    if arm.status != "ok":
        sh.flags.add("push_failed")
        return
    yield from arm.hold(6)
    p = K.point_of(arm.T, off)
    yield from arm.line_to(tcp_T(R, p + 0.04 * Z_UP), offset=off, lin_step=0.008)
    sh.flags.add("pushed")
    yield from arm.line_to(tcp_T(R, p + 0.04 * Z_UP - PUSH_ENTRY * xl), offset=off, lin_step=0.01)
    sh.flags.add("push_clear")


EAR_PINCH_Z = -0.005
EAR_PINCH_Y = 0.0
HANG_TOP_MARGIN = 0.012
HANG_REL_ABOVE_LO = 0.01
HP_PRE = (0.0, 0.0, 0.06), (0.0, 0.0, 0.035), (0.0, -0.05, 0.01)
HP_Z_SAFE = 0.88, 0.85, 0.83
HP_DIRECT_Z = 0.93, 0.9, 0.88
HP_CLEAR = 0.015
HP_RISES = 0.0, 0.03, 0.06, 0.09
HP_LIFTS = (0.0, 0.0, 0.1), (-0.06, 0.0, 0.08), (-0.1, 0.0, 0.05)
HP_JOINT_STEP = 0.06
HP_TURN_LIN = 0.0085
HP_TURN_DEG = 2.3
HP_SWING_E = 0.001
HP_SWING_N = 10
HP_HOLD = 5
HANG_Y = 0.005
HANG_IN = 0.1
ZONE_TOP = 0.87
ZONE_MARGIN = 0.015
HP_LAPTOP_CLEAR = 0.045
HP_ZONE_LIFT = -0.1, 0.0, 0.05
RAIL_HALF = 0.08
RAIL_GAP = 0.04


@lru_cache(maxsize=16)
def headset_geom(index: int) -> tuple[float, float, float]:
    H = SS.asset_points("headset", index)
    S = SS.asset_points("headset_stand", 0)
    Q = H[(np.abs(H[:, 2] - EAR_PINCH_Z) < 0.006) & (np.abs(H[:, 1]) < 0.015)]
    xs = np.abs(Q[:, 0])
    xs = xs[xs > 0.02]
    ear_x = 0.5 * float(xs.min() + xs.max())
    C = S[(S[:, 2] > 0.1) & (np.abs(S[:, 0]) < 0.06)]
    prof = []
    for a in np.arange(-0.055, 0.056, 0.005):
        c = C[(C[:, 0] >= a) & (C[:, 0] < a + 0.005)]
        if len(c):
            prof.append((a + 0.0025, float(c[:, 2].min()), float(c[:, 2].max())))
    B = H[np.abs(H[:, 1]) < 0.03]
    ok = []
    for oz in np.arange(0.03, 0.13, 0.002):
        if all(
            not np.any((np.abs(B[:, 0] - x) < 0.004) & (B[:, 2] > z0 - oz) & (B[:, 2] < z1 - oz)) for x, z0, z1 in prof
        ):
            ok.append(float(oz))
    if not ok:
        return ear_x, 0.066, 0.1
    return ear_x, ok[0], ok[-1]


HP_PINCH_MAX_W = 0.02
HP_PAD_HALF = 0.011
HP_PINCH_ZS = tuple(EAR_PINCH_Z + 0.0025 * i for i in range(17))


@lru_cache(maxsize=None)
def hang_pinch(index: int) -> tuple[float, float]:
    H = SS.asset_points("headset", index)
    sel = (np.abs(H[:, 1]) < 0.013) & (H[:, 0] > 0.02)

    def width(z, half, below=0.0):
        Q = H[sel & (H[:, 2] >= z - half - below) & (H[:, 2] <= z + half)]
        return (float(np.ptp(Q[:, 0])), 0.5 * float(Q[:, 0].min() + Q[:, 0].max())) if len(Q) > 3 else (math.inf, 0.0)

    if width(EAR_PINCH_Z, 0.003)[0] <= HP_PINCH_MAX_W:
        return headset_geom(index)[0], EAR_PINCH_Z
    for z in HP_PINCH_ZS:
        if width(z, HP_PAD_HALF)[0] <= HP_PINCH_MAX_W:
            return width(z, 0.003)[1], float(z)
    return headset_geom(index)[0], EAR_PINCH_Z


HP_WIDE_READ = 0.3
HP_BLOCK_READ = 0.6
HP_REPINCH_Z = 0.015


def section_width(index: int, z: float, half: float = 0.003) -> float:
    H = SS.asset_points("headset", index)
    Q = H[(np.abs(H[:, 1]) < 0.013) & (H[:, 0] > 0.02) & (np.abs(H[:, 2] - z) <= half)]
    return float(np.ptp(Q[:, 0])) if len(Q) > 3 else math.inf


def repinch_model(hs, z: float, cands=None):
    for _, j in list(cands or ()) + list((hs.extra or {}).get("cands") or ()):
        if section_width(int(j), z) > HP_PINCH_MAX_W:
            return int(j)


def slider_x(index: int, z: float) -> float:
    H = SS.asset_points("headset", index)
    Q = H[(np.abs(H[:, 1]) < 0.013) & (H[:, 0] > 0.02) & (np.abs(H[:, 2] - z) <= 0.003)]
    return 0.5 * float(Q[:, 0].min() + Q[:, 0].max()) if len(Q) > 3 else headset_geom(index)[0]


def hang_params(index: int, alts: tuple = ()) -> tuple[float, float, float, float]:
    ear_x, lo, hi = headset_geom(index)
    pinch_z = EAR_PINCH_Z
    ear_x, pinch_z = hang_pinch(index)
    for j in alts:
        zj = hang_pinch(j)[1]
        if zj > pinch_z + 1e-09:
            pinch_z = zj
            ear_x = slider_x(index, pinch_z)
        _, loj, hij = headset_geom(j)
        if max(lo, loj) + HP_WINDOW_MIN <= min(hi, hij):
            lo, hi = max(lo, loj), min(hi, hij)
    return ear_x, pinch_z, lo, hi


HP_ID_MARGIN = 0.02
HP_WINDOW_MIN = 0.012


def tilt_zone() -> tuple[float, float, float, float]:
    t1 = tcp1()
    p = np.array([t1[0], t1[1], PRESENT_Z])
    dz = PRESENT_Z - t1[2]
    tilt = math.radians(TILT)
    xs = [t1[0] - FAR_L * math.cos(tilt)]
    for u in np.linspace(0.0, 1.0, 41):
        a = tilt * u
        for s in np.linspace(0.0, 1.0, 21):
            if p[2] - dz * u - s * FAR_L * math.sin(a) - STACK_MID_Z < ZONE_TOP:
                xs.append(p[0] - s * FAR_L * math.cos(a))
    xP = t1[0] - FAR_L * math.cos(tilt)
    for b in np.linspace(tilt, math.pi / 2, 21):
        for r in np.linspace(0.0, FAR_L, 21):
            if K.TABLE_Z + r * math.sin(b) < ZONE_TOP:
                xs.append(xP + r * math.cos(b))
    w = 0.0897 + ZONE_MARGIN
    return min(xs) - ZONE_MARGIN, max(xs) + ZONE_MARGIN, t1[1] - w, t1[1] + w


def arm_clearance(name: str, qs, W: np.ndarray, step: int = 2) -> float:
    sp = robot_mask.spheres()
    Wd = W[:: max(1, len(W) // 600)]
    best = math.inf
    for q in list(qs)[::step]:
        for link, T in robot_mask.link_poses(name, q, 1.0).items():
            S = sp.get(link)
            if S is None or not len(S):
                continue
            C = S[:, :3] @ T[:3, :3].T + T[:3, 3]
            d = np.sqrt(((C[:, None, :] - Wd[None, :, :]) ** 2).sum(-1)).min(axis=1) - S[:, 3]
            best = min(best, float(d.min()))
    return best


def headset_points(hs) -> np.ndarray:
    T = hs.T()
    return SS.asset_points("headset", hs.index) @ T[:3, :3].T + T[:3, 3]


def headset_case(hs) -> str:
    W = headset_points(hs)
    x0, x1, y0, y1 = tilt_zone()
    if np.any((W[:, 0] > x0) & (W[:, 0] < x1) & (W[:, 1] > y0) & (W[:, 1] < y1)):
        return "zone"
    return "right" if float(np.median(W[:, 0])) > x1 else "left"


def laptop_far_x() -> float:
    return float(tcp1()[0] - FAR_L)


def pivot_x() -> float:
    return float(tcp1()[0] - FAR_L * math.cos(math.radians(TILT)))


def headset_plan(
    arm: Arm,
    Th: np.ndarray,
    Ts: np.ndarray,
    index: int,
    x_max: float = math.inf,
    lifts=HP_LIFTS,
    q_seed=None,
    alts=(),
    sides=(-1, 1),
    pinch_min=None,
):
    best, cost = None, math.inf
    Rs = Ts[:3, :3]
    ear_x, pinch_z, lo, hi = hang_params(int(index), tuple(int(j) for j in alts))
    if pinch_min is not None and pinch_z < pinch_min:
        pinch_z = float(pinch_min)
        ear_x = slider_x(int(index), pinch_z)
    za, zr = hi - HANG_TOP_MARGIN, lo + HANG_REL_ABOVE_LO

    def hung(y_off, z_off):
        H = np.eye(4)
        H[:3, :3] = Rs
        H[:3, 3] = Ts[:3, 3] + Rs @ np.array([0.0, y_off, z_off])
        return H

    def ik_chain(poses, keys, q):
        c = 0.0
        for k in keys:
            r = K.solve_near(arm.name, poses[k], q, K.TCP_OFFSET, n_random=4)
            if not r.ok:
                return None, q
            c += float(np.sum(np.abs(np.asarray(r.q) - q)))
            q = np.asarray(r.q)
        return c, q

    q0 = np.asarray(arm.q if q_seed is None else q_seed, float)
    for s in sides:
        p = Th[:3, :3] @ np.array([s * ear_x, EAR_PINCH_Y, pinch_z]) + Th[:3, 3]
        if p[0] >= x_max:
            continue
        d0 = Th[:3, 1]
        for R in (gripper_rot(d0, Th[:3, 0]), gripper_rot(d0, Th[:3, 0]) @ K.rot_x(math.pi)):
            pick = tcp_T(R, p)
            G = np.linalg.inv(Th) @ pick
            hang = dict(
                slide0=hung(HANG_Y - HANG_IN, za) @ G, slide1=hung(HANG_Y, za) @ G, release=hung(HANG_Y, zr) @ G
            )
            for pre in HP_PRE:
                if p[0] + pre[0] >= x_max:
                    continue
                poses = dict(hang, pre=tcp_T(R, p + np.asarray(pre)), pick=pick)
                c1, q1 = ik_chain(poses, ("pre", "pick"), q0)
                if c1 is None:
                    continue
                for lift in lifts:
                    poses["lift"] = tcp_T(R, p + np.asarray(lift))
                    c2, _ = ik_chain(poses, ("lift", "slide0", "slide1", "release"), q1)
                    if c2 is not None:
                        if c1 + c2 < cost:
                            best, cost = (
                                dict(poses, s=s, pre_d=pre, lift_d=lift, ear_x=ear_x, pinch_z=pinch_z, za=za, zr=zr),
                                c1 + c2,
                            )
                        break
                break
    return best


HP_SWING_FIX_IOU = 0.8
HP_SWING_FIX_MIN = 6.0
HP_SWING_FIX_MAX = 30.0
HP_SWING_FIX_FRACS = 1.0, 0.75, 0.5
HP_SWING_FIX_JOINT = 0.1


def swing_fix(arm: Arm, plan: dict, Th: np.ndarray, Ts: np.ndarray, index: int) -> dict | None:
    o = arm.ctx.obs
    q = np.asarray(o.state[f"{arm.name}_arm_joint_state"], float)
    Tt = K.fk(arm.name, q).copy()
    Tt[:3, 3] = K.point_of(Tt, K.TCP_OFFSET)
    G0 = np.linalg.inv(plan["pick"]) @ Th
    ign = robot_mask.mask(
        head(),
        o.state["left_arm_joint_state"],
        o.state["right_arm_joint_state"],
        float(o.state["left_ee_joint_state"][0]),
        float(o.state["right_ee_joint_state"][0]),
    )
    G, iou, _ = SS.fit_in_hand(
        o.images["cam_head"], head(), Tt, G0, SS.asset_points("headset", index), ignore=ign, T_stand=Ts
    )
    w = K.rotvec(G[:3, :3] @ G0[:3, :3].T)
    wy = math.degrees(float(w[1]))
    moved = None
    if iou >= HP_SWING_FIX_IOU and abs(wy) >= HP_SWING_FIX_MIN:
        a = -math.radians(float(np.clip(wy, -HP_SWING_FIX_MAX, HP_SWING_FIX_MAX)))
        for f in HP_SWING_FIX_FRACS:
            C = K.rot_y(f * a)
            cand = {k: tcp_T(plan[k][:3, :3] @ C, plan[k][:3, 3]) for k in ("slide0", "slide1", "release")}
            qq, ok = np.asarray(arm.q, float), True
            for k in ("slide0", "slide1", "release"):
                r = K.solve_near(arm.name, cand[k], qq, K.TCP_OFFSET, n_random=4)
                if not r.ok:
                    ok = False
                    break
                qq = np.asarray(r.q)
            if ok:
                moved = cand
                break
    return moved


def left_headset_task(
    arm: Arm,
    sh: Shared,
    hs,
    st,
    x_max: float = math.inf,
    lifts=HP_LIFTS,
    clear_x=None,
    direct: bool = False,
    wait_flags=None,
):
    Th, Ts = hs.T(), st.T()
    q_seed = arm.q_home if direct else None
    alts = tuple(j for v, j in (hs.extra or {}).get("cands", ()) if j != hs.index and hs.iou - v < HP_ID_MARGIN)
    plan = None
    if getattr(sh, "hp_side", None) is not None:
        plan = headset_plan(
            arm, Th, Ts, hs.index, x_max=x_max, lifts=lifts, q_seed=q_seed, alts=alts, sides=(sh.hp_side,)
        )
    if plan is None:
        plan = headset_plan(arm, Th, Ts, hs.index, x_max=x_max, lifts=lifts, q_seed=q_seed, alts=alts)
    if plan is None and wait_flags is not None:
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)
        yield from arm.wait_until(lambda: sh.flags & set(wait_flags), 300)
        q_seed = None
        plan = headset_plan(arm, Th, Ts, hs.index, lifts=lifts, q_seed=q_seed, alts=alts)
    if plan is None:
        sh.flags.add("hp_failed")
        return
    Tp = plan["pre"]
    z_safe = None
    if (
        direct
    ):
        for zc in HP_DIRECT_Z:
            Tc = tcp_T(Tp[:3, :3], [Tp[0, 3], Tp[1, 3], max(zc, Tp[2, 3])])
            r = K.solve_near(arm.name, Tc, arm.q_home, K.TCP_OFFSET, n_random=4)
            if not r.ok:
                continue
            Wh = headset_points(hs)
            for rise in HP_RISES:
                if rise > 0.0:
                    yield from arm.line_to(tcp_T(arm.T[:3, :3], arm.tcp() + (rise - done) * Z_UP), lin_step=0.01)
                done = rise
                jp = motion.joint_move(arm.q, r.q, joint_step=EMPTY_JOINT_STEP)
                low = arm_clearance(arm.name, jp.q[:-3], Wh)
                if low >= HP_CLEAR or rise == HP_RISES[-1]:
                    break
            if low >= HP_CLEAR:
                ok = yield from arm.follow(jp, 1.0)
                arm.status = "ok" if ok else "blocked"
                if ok:
                    yield from arm.line_to(Tp, lin_step=0.012)
            break
    for z in () if direct else HP_Z_SAFE:
        Tc = tcp_T(Tp[:3, :3], [Tp[0, 3], Tp[1, 3], max(z, Tp[2, 3])])
        if K.solve_near(arm.name, Tc, arm.q, K.TCP_OFFSET, n_random=4).ok:
            z_safe = z
            break
    if z_safe is not None:
        yield from arm.transit(Tp, z_safe=z_safe, grip=1.0, joint_step=FREE_JOINT_STEP)
    if (
        z_safe is None
        and (not direct)
        or arm.status != "ok"
        or (direct and float(np.linalg.norm(arm.tcp() - Tp[:3, 3])) > 0.01)
    ):
        r = K.solve_near(arm.name, Tp, arm.q, K.TCP_OFFSET, n_random=4)
        if not r.ok:
            sh.flags.add("hp_failed")
            return
        yield from arm.joints_to(r.q, grip=1.0, joint_step=FREE_JOINT_STEP)
    yield from arm.line_to(plan["pick"], lin_step=0.006)
    yield from arm.set_grip(0.0, n=10)
    yield from arm.hold(4)
    if HP_WIDE_READ <= obs_grip(arm) < HP_BLOCK_READ and plan["pinch_z"] < HP_REPINCH_Z - 1e-09:
        idx2 = repinch_model(hs, plan["pinch_z"], getattr(sh, "hp_cands", None))
        plan2 = (
            None
            if idx2 is None
            else headset_plan(
                arm, Th, Ts, idx2, x_max=x_max, lifts=lifts, alts=(), sides=(plan["s"],), pinch_min=HP_REPINCH_Z
            )
        )
        if plan2 is not None:
            yield from arm.set_grip(1.0, n=4)
            yield from arm.line_to(plan2["pick"], lin_step=0.006)
            yield from arm.set_grip(0.0, n=8)
            yield from arm.hold(3)
            plan = plan2
    yield from arm.line_to(tcp_T(arm.T[:3, :3], plan["lift"][:3, 3]), lin_step=0.008)
    T0, T1 = tcp_T(arm.T[:3, :3], arm.tcp()), plan["slide0"]
    dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    ang = math.degrees(K.rot_angle(T0[:3, :3], T1[:3, :3]))
    n = int(max(30, math.ceil(dist / HP_TURN_LIN), math.ceil(ang / HP_TURN_DEG)))
    poses = [
        tcp_T(K.slerp_mat(T0[:3, :3], T1[:3, :3], float(u)), T0[:3, 3] + float(u) * (T1[:3, 3] - T0[:3, 3]))
        for u in minjerk(n)
    ]
    k = 0
    if clear_x is not None:
        Ph = SS.asset_points("headset", hs.index)
        Gh = np.linalg.inv(plan["pick"]) @ Th
        k = len(poses)
        for i, T in enumerate([tcp_T(arm.T[:3, :3], arm.tcp())] + poses):
            M = T @ Gh
            if float(np.max(Ph @ M[0, :3] + M[0, 3])) < clear_x:
                k = i
                break
    if k == 0:
        sh.flags.add("hp_clear")
        yield from follow_poses(arm, poses)
    else:
        yield from follow_poses(arm, poses[:k])
        if arm.status == "ok" and k < len(poses):
            sh.flags.add("hp_clear")
            yield from follow_poses(arm, poses[k - 1 :])
    sh.flags.add("hp_clear")
    if arm.status == "nopath":
        r = K.solve_near(arm.name, plan["slide0"], arm.q, K.TCP_OFFSET, n_random=4)
        if not r.ok:
            sh.flags.add("hp_failed")
            return
        yield from arm.joints_to(r.q, joint_step=HP_JOINT_STEP)
    if obs_grip(arm) > 0.02:
        g0 = obs_grip(arm)
        rng = dual.GRIP_SCALE[1] - dual.GRIP_SCALE[0]
        g_loose = float(min(1.0, g0 + 0.2 - HP_SWING_E / rng))
        yield from arm.set_grip(g_loose, n=2)
        yield from arm.hold(HP_SWING_N)
        yield from arm.set_grip(0.0, n=4)
    else:
        yield from arm.hold(HP_HOLD)
    slide_kw = {}
    if obs_grip(arm) > 0.02:
        try:
            moved = swing_fix(arm, plan, Th, Ts, hs.index)
        except Exception:
            moved = None
        if moved is not None:
            plan = dict(plan, **moved)
            slide_kw = dict(ang_step=math.radians(HP_TURN_DEG), joint_step=HP_SWING_FIX_JOINT)
    yield from arm.line_to(plan["slide1"], lin_step=0.007, **slide_kw)
    yield from arm.line_to(plan["release"], lin_step=0.004)
    yield from arm.set_grip(1.0, n=6)
    yield from arm.hold(2)
    T = arm.T
    yield from arm.line_to(tcp_T(T[:3, :3], arm.tcp() - 0.07 * Ts[:3, 1]), lin_step=0.008)
    sh.flags.add("hp_done")
    yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=EMPTY_JOINT_STEP)


def left_headset_first(arm: Arm, sh: Shared, hs, st):
    yield from left_headset_task(arm, sh, hs, st)
    sh.flags.add("hp_clear")


def left_hp_check(arm: Arm):
    yield from arm.hold(0)


def right_lid_home(arm: Arm, sh: Shared):
    if "push_clear" in sh.flags:
        yield from arm.joints_to(arm.q_home, grip=1.0, joint_step=FREE_JOINT_STEP)


def rack_wrist(o, rk):
    from ...perception import cameras, seg

    img = o.images["cam_right_wrist"]
    i = SS._best(seg.segment(img, camera="right_wrist"), "vertical_laptop_storage_rack")
    px = 0 if i is None else int(np.count_nonzero(i.mask))
    if i is None or px < RACK_WRIST_PX:
        return rk
    d = math.radians(RACK_WRIST_DYAW)
    s, x, y, a = SS.fit_pose(
        cameras.wrist(o.state["right_ee_pose"]),
        i.mask,
        SS.object_points("vertical_laptop_storage_rack", 0),
        rk.z,
        (rk.x, rk.y),
        (rk.yaw - d, rk.yaw + d),
        span=RACK_WRIST_SPAN,
        centred=True,
    )
    used = s >= RACK_WRIST_IOU
    return SS.Fit(rk.category, x, y, a, rk.z, s, rk.index) if used else rk


def _chain(*tasks):
    for t in tasks:
        yield from t


@register
class StoreLaptop(Program):
    name = "store_laptop"
    background = True

    def match(self, instruction: str, first_obs) -> float:
        return 0.9 if _is_store(instruction) else 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        rgb = o.images["cam_head"]
        ign = robot_mask.mask(
            head(),
            o.state["left_arm_joint_state"],
            o.state["right_arm_joint_state"],
            float(o.state["left_ee_joint_state"][0]),
            float(o.state["right_ee_joint_state"][0]),
        )
        scene = SS.perceive(rgb, ignore=ign, only=("laptop", "rack", "stand", "headset", "laptop_stand"))
        lp = scene.laptop
        rk = scene.rack
        if rk is not None:
            rk = rack_wrist(o, rk)
        hs, st = scene.headset, scene.stand
        case = headset_case(hs) if hs is not None else "none"
        if lp is not None:
            sh = Shared()
            sh.rack = rk
            sh.lstand = scene.laptop_stand
            sh.hstand = st
            sh.hp_zone = case == "zone"
            sh.hp_cands = None if hs is None else (hs.extra or {}).get("cands")
            TL = lp.T()
            sh.TL0 = TL
            right = right_task(arms["right"], sh)
            hf_first = st is not None and case in RIGHT_LID_HF_CASES
            if hs is not None and case in ("zone", "right") and (not hf_first):
                right = _chain(right_push_task(arms["right"], sh, hs), right)
                sh.right_push = True
            hf = hs is not None and st is not None and (case in RIGHT_LID_HF_CASES)
            if hf:
                tasks = {
                    "right": _chain(
                        left_push_task(arms["right"], sh, TL),
                        right_lid_home(arms["right"], sh),
                        right_task(arms["right"], sh),
                    ),
                    "left": _chain(
                        left_headset_first(arms["left"], sh, hs, st),
                        left_hp_check(arms["left"]),
                        left_carry_task(arms["left"], sh, TL, home=True, early=True),
                        left_hp_check(arms["left"]),
                    ),
                }
            elif not sh.right_push:
                tasks = {
                    "right": _chain(left_push_task(arms["right"], sh, TL), right_lid_home(arms["right"], sh), right),
                    "left": _chain(
                        left_carry_task(arms["left"], sh, TL, home=False, early=True),
                        left_after_task(arms["left"], sh, hs, st, case),
                    ),
                }
            else:
                tasks = {
                    "right": right,
                    "left": _chain(
                        left_push_task(arms["left"], sh, TL),
                        left_carry_task(arms["left"], sh, TL, home=False),
                        left_after_task(arms["left"], sh, hs, st, case),
                    ),
                }
            yield from run_arms(arms, tasks)
        while True:
            yield [actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)]
