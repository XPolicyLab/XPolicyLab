from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields, replace

import numpy as np

from . import table_bottles as BT
from ...perception import cameras as C
from ...perception import detector as DT
from . import dustbin as DB
from ...perception import robot_mask as RM
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills import dual as D
from ...skills.runner import Arm, link_arms, run_arms
from .. import Program, register

LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
STEP_BUDGET = 700
HOME_RESERVE = 45
JOB_STEPS = {"bin": 95, "stage": 105}
MAX_ROUNDS = 4
CLEAR_Z = 0.87
LIFT_OFF = 0.04
PATH_MARGIN = 0.04
HELD_STEP = 0.085
OPEN = 1.0
MIN_HELD = 0.015
SETTLE_TOL = 0.01
FREE_STEP = 0.11
SIDE_MARGIN = 0.25
UNSURE_IOU = 0.1
OVERRULE_MAX_GAP = 0.08
CONTEST_PX = 10
SIDE_TURN_W = 0.8
HINT_PX = 300
HIDDEN_FRAC = 0.25
MATCH_GAP = 0.02
UNHIDDEN_IOU = 0.75
GRASP_ABOVE_COM = 0.03
SIDE_AXIS_X = 0.138
PAD_HALF = 0.011
MAX_TAPER = 0.002
ARM_MARGIN = 0.015
MAX_GRASP_FRAC = 0.75
EXIT_CLEAR = 0.04
EXIT_MARGIN = 0.015
RETRY_PAUSE = 8
RETRIES = 4
BOTTLE_AREA = (-0.4, 0.56), (-0.33, 0.15)
NEAR_Y = -0.42
NEAR_IOU = 0.75
DROP_MARGIN = 0.04
FINGER_OUT = 0.054
GRIP_HALF = 0.087
CORRIDOR_BEYOND = 0.06
CORRIDOR_MARGIN = 0.02
MAX_DEFERS = 2
TOUCH_PX = 3
TOUCH_FRAC = 0.05
TOUCH_LOS = 0.9
UNHIDE_OVERLAP = 0.5
UNHIDE_HINT_PX = 300
MAX_VIA_TRIES = 4
VIA_RISE = 0.08, 0.15
VIA_COST = 0.3
CARRY_RISE = 0.0, 0.05, 0.1, 0.15
CARRY_LIN = 0.02
CARRY_ANG = math.radians(6)
TOP_ROLLS = 0.0, 0.15, -0.15
FINGER_FLOOR = 0.004
MARGIN_X = 0.05
BODY_X = 0.05
PARK_SWING = 0.8
IDLE_GUARD = 0.012
CLAIM_WAIT = 300
YIELD_WAIT = 40
REFINE_MAX = 0.03
REFINE_MIN_IOU = 0.7
REFINE_MIN_GAIN = 0.02
REFINE_MIN_SHIFT = 0.003
REFINE_MIN_IOU_SIDE = 0.85
REFINE_ROUNDS_SIDE = 2
REFINE_SIDE_MODEL_DR = 0.01
REFINE_ROUNDS_TOP = 2
REFINE_MOVE_FIRST = 0.015
REFINE_ABSENT_IOU0 = 0.3
TOP_ABSENT_IOU0 = 0.1
REAIM_IOU0 = 0.3
REAIM_IOU = 0.5
REAIM_BACK = 0.04
REAIM_CONFIRM_IOU0 = 0.5
SIDE_ABSENT_IOU0 = 0.1
REAIM_MAX = 2
RETRY_NEAR = 0.15
RETRY_CLEAR = 0.12
RETRY_MIN_IOU = 0.75
RETRY_WAIT = 120
RETRY_EXPLAINED = 0.06
REACH_TOL = 0.02
REACH_LAT = 0.012
REACH_ALONG = 0.025
LEFT_TOP_REACH = 0.6
NO_REACH_DIST = 0.05
LONG_LINE_MIN = 30
LONG_LINE_RATIO = 4
LOS_MATCH_LAT = 0.025
LOS_MATCH_ALONG = 0.2
LOS_AMBIG = 0.5
STAGE_GAP = 0.11
MAX_CLEAN_CHECKS = 3
MAX_STAGE_EVALS = 30
_BIN_WORDS = re.compile(r"\b(dust ?bin|trash ?(can|bin)|waste ?(bin|basket)|garbage ?(can|bin))\b")


def line_plan(
    name: str,
    q0,
    T_goal: np.ndarray,
    offset: float,
    lin_step: float,
    ang_step: float,
    joint_step: float,
    fast_step: float,
):
    plan = motion.line(name, q0, T_goal, offset, lin_step=lin_step, ang_step=ang_step, joint_step=joint_step)
    if not (plan.ok and joint_step < fast_step and (len(plan.q) > LONG_LINE_MIN)):
        return plan, None
    t0 = K.fk(name, q0)
    n0 = motion.n_steps_for(
        float(np.linalg.norm(T_goal[:3, 3] - K.point_of(t0, offset))),
        K.rot_angle(t0[:3, :3], T_goal[:3, :3]),
        lin_step,
        ang_step,
    )
    if len(plan.q) <= LONG_LINE_RATIO * n0:
        return plan, None
    fast = motion.line(name, q0, T_goal, offset, lin_step=lin_step, ang_step=ang_step, joint_step=fast_step)
    info = len(plan.q), len(fast.q) if fast.ok else None
    return fast if fast.ok and len(fast.q) < len(plan.q) else plan, info


class BinArm(Arm):
    def line_to(
        self,
        T_goal: np.ndarray,
        offset: float = K.TCP_OFFSET,
        *,
        lin_step: float = motion.MAX_LIN_STEP,
        ang_step: float = motion.MAX_ANG_STEP,
        joint_step: float = motion.MAX_JOINT_STEP,
        grip=None,
        fallback_joint: bool = True,
    ):
        closed = (self.g if grip is None else min(self.g, float(grip))) < 0.5
        plan, _ = line_plan(
            self.name, self.q, T_goal, offset, lin_step, ang_step, joint_step, HELD_STEP if closed else FREE_STEP
        )
        if not plan.ok:
            if not fallback_joint:
                self.status = "fail"
                return
            r = K.solve_near(self.name, T_goal, self.q, offset)
            if not r.ok:
                self.status = "fail"
                return
            plan = motion.joint_move(self.q, r.q, joint_step=joint_step)
        ok = yield from self.follow(plan, grip)
        self.status = "ok" if ok else "blocked"


def frame(x, y) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)
    y = np.asarray(y, dtype=float)
    y = y - x * float(x @ y)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def pose(rot, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = p
    return t


def rot_about_z(rot, a: float) -> np.ndarray:
    return K.rot_z(a) @ rot


@dataclass(eq=False)
class Grasp:
    rot: np.ndarray
    tcp: np.ndarray
    pre: np.ndarray
    q: np.ndarray
    q_pre: np.ndarray
    below: float
    cost: float
    kind: str
    via: tuple = ()


@dataclass(eq=False)
class Job:
    bottle: BT.Bottle
    arm: str
    kind: str
    state: str = "todo"
    spot: np.ndarray | None = None
    staged: BT.Bottle | None = None


@dataclass(eq=False)
class Shared:
    jobs: list
    bin: DB.Bin
    scene: list = field(default_factory=list)
    staged: list = field(default_factory=list)
    front: dict = field(default_factory=dict)
    want: dict = field(default_factory=dict)
    gone: set = field(default_factory=set)
    version: int = 0
    q_drop: np.ndarray | None = None
    parked: set = field(default_factory=set)

    def present(self, bottles) -> list:
        return [b for b in bottles if id(b) not in self.gone]

    def others(self, b) -> list:
        return [o for o in self.scene if o is not b and id(o) not in self.gone]

    def remove(self, b) -> None:
        self.scene = [o for o in self.scene if o is not b]
        self.version += 1

    def add(self, b) -> None:
        if all(o is not b for o in self.scene):
            self.scene.append(b)
        self.version += 1


def steps_left(ctx) -> int:
    return STEP_BUDGET - ctx.obs.step


def _front(name: str, x: float) -> float:
    return x + BODY_X if name == "left" else x - BODY_X


def _other(name: str) -> str:
    return "right" if name == "left" else "left"


def _compatible(name: str, f: float, of) -> bool:
    if of is None:
        return True
    return f + MARGIN_X <= of if name == "left" else of + MARGIN_X <= f


def _claim_ok(sh: Shared, name: str, f: float, since: int) -> bool:
    o = _other(name)
    if o in sh.parked:
        return True
    own = sh.front.get(name)
    if own is not None and (f <= own if name == "left" else f >= own):
        return True
    if not _compatible(name, f, sh.front.get(o)):
        return False
    w = sh.want.get(o)
    if w is not None and w[1] < since and (not _compatible(name, f, w[0])) and _compatible(o, w[0], own):
        return False
    return True


def _blocks_want(sh: Shared, name: str) -> bool:
    w = sh.want.get(_other(name))
    return w is not None and (not _compatible(_other(name), w[0], sh.front.get(name)))


def claim(arm: Arm, sh: Shared, x: float):
    f = _front(arm.name, x)
    since = int(arm.ctx.obs.step)
    if not _claim_ok(sh, arm.name, f, since):
        sh.want[arm.name] = f, since
        n, yielded = 0, False
        while not _claim_ok(sh, arm.name, f, since) and n < CLAIM_WAIT:
            if not yielded and arm.g >= 0.5 and _blocks_want(sh, arm.name):
                yielded = True
                arm.waiting = False
                yield from park_clear(arm, sh)
                continue
            n += 1
            arm.waiting = True
            yield
        arm.waiting = False
        sh.want.pop(arm.name, None)
        sh.front[arm.name] = f
        return True
    sh.front[arm.name] = f
    return False


def should_yield(sh: Shared, name: str, f_next: float) -> bool:
    if name != "right":
        return False
    w = sh.want.get("left")
    return w is not None and (not _compatible("left", w[0], f_next))


def yield_to_left(arm: Arm, sh: Shared, f_next: float):
    if arm.other is None or not arm.other.active or (not should_yield(sh, arm.name, f_next)):
        return False
    yield from park(arm, sh, LOOK_Q)
    sh.parked.add(arm.name)
    yield from arm.wait_until(lambda: "left" not in sh.want or not arm.other.active, max_steps=YIELD_WAIT)
    sh.parked.discard(arm.name)
    return True


def settle_front(arm: Arm, sh: Shared, extent: float = 0.0):
    x = float(arm.tcp()[0])
    sh.front[arm.name] = _front(arm.name, x + extent if arm.name == "left" else x - extent)


def track_front(arm: Arm, sh: Shared, extent: float = 0.0) -> None:
    def after():
        x = float(arm.tcp()[0])
        new = _front(arm.name, x + extent if arm.name == "left" else x - extent)
        old = sh.front.get(arm.name)
        if old is None or (new < old if arm.name == "left" else new > old):
            sh.front[arm.name] = new

    arm._after_step = after


def untrack(arm: Arm) -> None:
    arm.__dict__.pop("_after_step", None)


def park_q(arm: Arm, q_base=None) -> np.ndarray:
    q = np.array(arm.q_home if q_base is None else q_base, dtype=float)
    q[0] += PARK_SWING if arm.name == "left" else -PARK_SWING
    return q


def park(arm: Arm, sh: Shared, q_base=None):
    q = park_q(arm, q_base)
    track_front(arm, sh)
    yield from arm.joints_to(q, grip=OPEN, joint_step=FREE_STEP)
    untrack(arm)
    settle_front(arm, sh)


def _obstacles_ok(points, bottles, skip, clearance: float = 0.012) -> bool:
    for b in bottles:
        if b is skip:
            continue
        for p in points:
            if b.standing:
                if p[2] < K.TABLE_Z + b.height + 0.01 and np.linalg.norm(p[:2] - b.xy) < b.radius + clearance:
                    return False
            else:
                a = np.array([math.cos(b.yaw), math.sin(b.yaw)])
                s = float(np.clip((p[:2] - b.xy) @ a, -b.height / 2, b.height / 2))
                q = b.xy + a * s
                if p[2] < K.TABLE_Z + 2 * b.radius + 0.01 and np.linalg.norm(p[:2] - q) < b.radius + clearance:
                    return False
    return True


def _capsule(b: BT.Bottle):
    r = b.radius
    if b.standing:
        return np.array([b.xy[0], b.xy[1], K.TABLE_Z + r]), np.array([b.xy[0], b.xy[1], K.TABLE_Z + b.height - r]), r
    a = np.array([math.cos(b.yaw), math.sin(b.yaw), 0.0])
    c = np.array([b.xy[0], b.xy[1], K.TABLE_Z + b.extra.get("rest_r", r)])
    half = max(0.0, b.height / 2 - r)
    return c - a * half, c + a * half, r


def _seg_dist(P, a, b) -> np.ndarray:
    d = b - a
    L2 = float(d @ d)
    t = np.zeros(len(P)) if L2 < 1e-12 else np.clip((P - a) @ d / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def _segseg(p1, q1, p2, q2) -> float:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def held_local(arm: str, g, b: BT.Bottle):
    T = K.fk(arm, g.q)
    a, e, r = _capsule(b)
    R, t = T[:3, :3], T[:3, 3]
    return (a - t) @ R, (e - t) @ R, r


def _gripper_grid() -> tuple[np.ndarray, np.ndarray]:
    body = np.array(
        [
            [x, y, z]
            for x in np.linspace(0.0, 0.084, 5)
            for y in np.linspace(-0.087, 0.087, 9)
            for z in (-0.03, 0.0, 0.03)
        ]
        + [[x, y, z] for x in (0.06, 0.085, 0.11) for y in (-0.03, 0.0, 0.03) for z in (0.05, 0.08)]
    )
    finger = np.array(
        [[x, dy, z] for x in np.linspace(0.087, 0.158, 5) for dy in (0.0, 0.018, 0.037) for z in (-0.03, 0.0, 0.03)]
    )
    return body, finger


_GRIP_BODY, _GRIP_FINGER = _gripper_grid()


def gripper_points(arm: str, q, grip: float = 1.0) -> np.ndarray:
    T = K.fk(arm, q)
    inner = 0.0004 + max(0.0, RM.finger_q(grip))
    f1 = _GRIP_FINGER.copy()
    f1[:, 1] = inner + f1[:, 1]
    f2 = f1.copy()
    f2[:, 1] = -f2[:, 1]
    P = np.vstack([_GRIP_BODY, f1, f2])
    return P @ T[:3, :3].T + T[:3, 3]


def arm_clear(arm: str, q, bottles, grip: float = 1.0, margin: float = ARM_MARGIN, held=None) -> bool:
    if not bottles:
        return True
    sp = CO.arm_spheres(arm, q, grip)[3:]
    gp = gripper_points(arm, q, grip)
    hw = None
    if held is not None:
        T = K.fk(arm, q)
        hw = T[:3, :3] @ held[0] + T[:3, 3], T[:3, :3] @ held[1] + T[:3, 3], held[2]
    for b in bottles:
        a, e, r = _capsule(b)
        if np.any(_seg_dist(sp[:, :3], a, e) - sp[:, 3] - r < margin):
            return False
        if np.any(_seg_dist(gp, a, e) - r < margin):
            return False
        if hw is not None and _segseg(hw[0], hw[1], a, e) - hw[2] - r < margin:
            return False
    return True


def path_clear(arm: str, q0, q1, bottles, grip: float = 0.0, n: int = 8, margin: float = ARM_MARGIN, held=None) -> bool:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    return all(arm_clear(arm, q0 + t * (q1 - q0), bottles, grip, margin, held) for t in np.linspace(0.0, 1.0, n))


def side_grasp_z(b: BT.Bottle) -> float:
    h = b.height
    m = BT.BY_NAME.get(b.model)
    if m is None:
        return float(min(h - 0.035, max(0.47 * h + 0.04, 0.5 * h)))
    lo = m.com + GRASP_ABOVE_COM
    best = None
    for z in np.arange(lo, min(h - PAD_HALF - 0.004, max(lo, MAX_GRASP_FRAC * h)) + 1e-09, 0.004):
        rs = np.interp([z - PAD_HALF, z, z + PAD_HALF], m.z, m.r)
        if rs.max() - rs.min() > MAX_TAPER:
            continue
        cost = float(rs.max()) + 0.02 * float(z - lo)
        if best is None or cost < best[0]:
            best = cost, float(z)
    if best is None:
        return float(min(h - 0.035, max(m.com + 0.04, 0.5 * h)))
    return best[1]


def _finger_points(rot, tcp) -> list:
    y = rot[:, 1]
    x = rot[:, 0]
    return [tcp + s * y * FINGER_OUT + x * dx for s in (-1, 1) for dx in (0.0, 0.012)]


def approach_via(arm: str, q_cur, rot, q_first, others):
    p_first = K.point_of(K.fk(arm, q_first), K.TCP_OFFSET)
    for dz in VIA_RISE:
        rv = K.solve_near(arm, pose(rot, p_first + np.array([0.0, 0.0, dz])), q_first, K.TCP_OFFSET, n_random=1)
        if (
            rv.ok
            and path_clear(arm, q_cur, rv.q, others, grip=1.0)
            and path_clear(arm, rv.q, q_first, others, grip=1.0, n=4)
        ):
            return (rv.q,)
    if path_clear(arm, q_cur, LOOK_Q, others, grip=1.0) and path_clear(arm, LOOK_Q, q_first, others, grip=1.0):
        return (LOOK_Q.copy(),)


def bin_carry(arm: str, g, b: BT.Bottle, bin_, others, zb: float):
    if not others:
        return zb, "free", None, None
    held = held_local(arm, g, b)
    tight = [o for o in others if not arm_clear(arm, g.q, [o], grip=0.0, held=held)]
    rest = [o for o in others if all(o is not t for t in tight)]

    def ok(q, n_ok=True):
        return arm_clear(arm, q, rest, grip=0.0, held=held) and arm_clear(
            arm, q, tight, grip=0.0, held=held, margin=0.0
        )

    def clear(q0, q1, n=8):
        q0 = np.asarray(q0, float)
        q1 = np.asarray(q1, float)
        return all(ok(q0 + t * (q1 - q0)) for t in np.linspace(0.0, 1.0, n))

    tops = [K.TABLE_Z + (o.height if o.standing else 2 * o.radius) + 0.03 for o in others]
    zs = sorted({round(z, 3) for z in [zb + dz for dz in CARRY_RISE] + tops if z >= zb - 1e-06})
    step = HELD_STEP if g.kind == "side" else FREE_STEP
    tried_line = 0
    for z in zs:
        z_goal = max(g.tcp[2] + 0.05, z + g.below)
        r = K.solve_near(arm, pose(g.rot, [g.tcp[0], g.tcp[1], z_goal]), g.q, K.TCP_OFFSET, n_random=1)
        if not r.ok or not clear(g.q, r.q, n=4):
            continue
        dp = drop_pose(arm, g, bin_, r.q, z_goal)
        if dp is None or dp[1] is None:
            continue
        if clear(r.q, dp[1]):
            return z_goal - g.below, "joint", dp[1], dp[0]
        if tried_line < 2:
            tried_line += 1
            pl = motion.line(arm, r.q, dp[0], K.TCP_OFFSET, lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=step)
            if pl.ok and all(ok(q) for q in list(pl.q[:: max(1, len(pl.q) // 12)]) + [pl.q[-1]]):
                return z_goal - g.below, "line", np.asarray(pl.q[-1]), dp[0]


def grasp_options(
    arm: str, b: BT.Bottle, q_cur, others=(), *, max_opts: int = 1, check_arm: bool = True
) -> list[Grasp]:
    bx, by = K.BASE_POS[arm][:2]
    cands = []
    if b.standing:
        z_rel = side_grasp_z(b)
        c = np.array([b.xy[0], b.xy[1], K.TABLE_Z + z_rel])
        phi0 = math.atan2(b.xy[1] - by, b.xy[0] - bx)
        phi_c = math.atan2(b.xy[1] - C.HEAD_POS[1], b.xy[0] - C.HEAD_POS[0])
        turn_w = SIDE_TURN_W
        dphis = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
        for dphi in dphis:
            for pitch in (0.0, 0.3, 0.55):
                phi = phi0 + dphi
                x = np.array([math.cos(phi) * math.cos(pitch), math.sin(phi) * math.cos(pitch), -math.sin(pitch)])
                y = np.array([-math.sin(phi), math.cos(phi), 0.0])
                rot = frame(x, y)
                tcp = c + x * (K.TCP_OFFSET - SIDE_AXIS_X)
                pre = tcp - x * (b.radius + 0.07)
                mis = abs(math.remainder(phi - phi_c, 2 * math.pi))
                cost = abs(dphi) * turn_w + mis * 0.0 + pitch * 0.5
                cands.append((cost, rot, tcp, pre, z_rel, "side"))
    else:
        a = np.array([math.cos(b.yaw), math.sin(b.yaw), 0.0])
        rest = b.extra.get("rest_r", b.radius)
        tcp = np.array([b.xy[0], b.xy[1], K.TABLE_Z + rest - 0.002])
        for sign in (1.0, -1.0):
            y0 = sign * np.array([-a[1], a[0], 0.0])
            for roll in TOP_ROLLS:
                for tilt in (0.0, 0.3, -0.3, 0.55, -0.55, 0.75, -0.75):
                    rot = K.axis_angle(a, roll) @ frame(K.axis_angle(y0, tilt) @ np.array([0.0, 0.0, -1.0]), y0)
                    x, y = rot[:, 0], rot[:, 1]
                    if (
                        roll
                        and min(
                            (tcp + s * y * FINGER_OUT + x * (K.FINGERTIP_OFFSET - K.TCP_OFFSET))[2] for s in (-1.0, 1.0)
                        )
                        < K.TABLE_Z + FINGER_FLOOR
                    ):
                        continue
                    side = 1.0 if rot[0, 2] * (1 if arm == "right" else -1) > 0 else 0.0
                    pre = tcp - x * 0.1
                    cands.append((abs(tilt) * 0.6 + abs(roll) * 1.5 + 0.1 * (1 - side), rot, tcp, pre, rest, "top"))
    cands.sort(key=lambda t: t[0])
    out = []
    blocked = []
    transit = list(others) + [b]
    for cost, rot, tcp, pre, below, kind in cands:
        if not _obstacles_ok(_finger_points(rot, tcp), others, b):
            continue
        T = pose(rot, tcp)
        r = K.solve_near(arm, T, q_cur, K.TCP_OFFSET, n_random=2)
        if not r.ok:
            continue
        rp = K.solve_near(arm, pose(rot, pre), r.q, K.TCP_OFFSET, n_random=1)
        if not rp.ok:
            continue
        if check_arm:
            qs = [r.q, rp.q, 0.5 * (r.q + rp.q)]
            q_first = rp.q
            if kind == "side":
                rh = K.solve_near(arm, pose(rot, pre + np.array([0.0, 0.0, 0.07])), rp.q, K.TCP_OFFSET, n_random=1)
                if rh.ok:
                    qs.append(rh.q)
                    q_first = rh.q
            if others and (not all(arm_clear(arm, qq, others) for qq in qs)):
                continue
            if not path_clear(arm, q_cur, q_first, transit, grip=1.0):
                if len(blocked) < MAX_VIA_TRIES:
                    blocked.append((cost, rot, tcp, pre, below, kind, r.q, rp.q, q_first))
                continue
        jc = float(np.max(np.abs(rp.q - np.asarray(q_cur))))
        out.append(Grasp(rot, tcp, pre, r.q, rp.q, below, cost + 0.15 * jc, kind))
        if len(out) >= max_opts:
            break
    if not out:
        for cost, rot, tcp, pre, below, kind, q_g, q_p, q_first in blocked:
            via = approach_via(arm, q_cur, rot, q_first, transit)
            if via is None:
                continue
            jc = float(np.max(np.abs(q_p - np.asarray(q_cur))))
            out.append(Grasp(rot, tcp, pre, q_g, q_p, below, cost + 0.15 * jc + VIA_COST, kind, via=via))
            if len(out) >= max_opts:
                break
    out.sort(key=lambda g: g.cost)
    return out


def drop_pose(arm: str, g: Grasp, bin_: DB.Bin, q_cur, z_carry: float) -> tuple[np.ndarray, np.ndarray] | None:
    z_min = DB.RIM_Z + g.below + DROP_MARGIN
    best = None
    for dx, z in (
        (0.06, z_carry),
        (0.1, z_carry),
        (0.06, z_carry - 0.04),
        (0.1, max(z_min, z_carry - 0.08)),
        (0.02, max(z_min, z_carry - 0.08)),
    ):
        p = np.array([bin_.centre[0] + dx, bin_.centre[1], z])
        for k in range(12):
            a = (k + 1) // 2 * (math.pi / 6) * (1 if k % 2 else -1)
            rot = rot_about_z(g.rot, a)
            r = K.solve_near(arm, pose(rot, p), q_cur, K.TCP_OFFSET, n_random=1)
            if r.ok:
                c = float(np.sum(K.JOINT_COST_W * np.abs(r.q - q_cur)))
                if best is None or c < best[0]:
                    best = c, pose(rot, p), r.q
        if best is not None:
            return best[1], best[2]


def settle(arm: Arm, tol: float = SETTLE_TOL, max_steps: int = 8):
    for _ in range(max_steps):
        q_obs = arm.ctx.obs.state[f"{arm.name}_arm_joint_state"]
        if float(np.max(np.abs(q_obs - arm.q))) < tol:
            return
        yield


def retry(arm: Arm, make):
    for i in range(RETRIES):
        yield from make()
        if arm.status != "blocked":
            return
        if arm.other is not None and (not arm.other.active):
            arm.idle_blocked = True
            return
        yield from arm.hold(RETRY_PAUSE)


def held(arm: Arm) -> float:
    return D.held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]), 0.0)


def tcp_obs(arm: Arm) -> np.ndarray:
    return K.point_of(K.fk(arm.name, arm.ctx.obs.state[f"{arm.name}_arm_joint_state"]), K.TCP_OFFSET)


def wrist_refine(arm: Arm, g: Grasp, b: BT.Bottle, others=()) -> bool:
    ctx = arm.ctx
    o = ctx.obs
    arm.refine_res = "none", None
    img = o.images.get(f"cam_{arm.name}_wrist")
    if img is None:
        return False
    cam = C.wrist(o.state[f"{arm.name}_ee_pose"])
    tref = getattr(arm, "table_ref", None)
    if b.standing:
        sim = [n for n in sorted(BT.BY_NAME) if abs(float(BT.BY_NAME[n].r.max()) - b.radius) < REFINE_SIDE_MODEL_DR]
        r = BT.refine_standing_wide(
            img,
            cam,
            b,
            bad=_robot_bad(cam, o),
            others=others,
            max_shift=REFINE_MAX,
            rounds=REFINE_ROUNDS_SIDE,
            models=sim,
            good_iou=REFINE_MIN_IOU_SIDE,
            table_ref=tref,
        )
    else:
        r = BT.refine_lying_wide(
            img,
            cam,
            b,
            bad=_robot_bad(cam, o),
            others=others,
            max_shift=REFINE_MAX,
            rounds=REFINE_ROUNDS_TOP,
            table_ref=tref,
        )
    arm.refine_res = "measured", r
    applied = False
    min_iou = REFINE_MIN_IOU_SIDE if b.standing else REFINE_MIN_IOU
    if r is not None and r["iou"] >= min_iou and (r["iou"] - r["iou0"] >= REFINE_MIN_GAIN):
        if b.standing:
            shift = np.array([r["dx"], r["dy"]])
        else:
            n = np.array([-math.sin(b.yaw), math.cos(b.yaw)])
            shift = r["lat"] * n
        s3 = np.array([shift[0], shift[1], 0.0])
        tcp2 = g.tcp + s3
        new_xy = np.asarray(b.xy, float) + shift
        free = all(BT._footprint_dist(new_xy, ob) >= b.radius - 0.01 for ob in others if ob is not b)
        if (
            float(np.linalg.norm(shift)) >= REFINE_MIN_SHIFT
            and free
            and _obstacles_ok(_finger_points(g.rot, tcp2), others, b)
        ):
            rr = K.solve_near(arm.name, pose(g.rot, tcp2), g.q, K.TCP_OFFSET, n_random=1)
            if rr.ok:
                g.tcp, g.pre, g.q = tcp2, g.pre + s3, rr.q
                b.xy = np.asarray(b.xy, float) + shift
                applied = True
    return applied


def refine_absent(res) -> bool:
    if res is None:
        return True
    if float(res["iou0"]) < TOP_ABSENT_IOU0 and float(res["iou"]) < REFINE_MIN_IOU:
        return True
    return abs(res["lat"]) >= REFINE_MAX - 1e-06 and res["iou"] < REFINE_MIN_IOU and (res["iou0"] < REFINE_ABSENT_IOU0)


def side_absent(res) -> bool:
    if res is None:
        return True
    return float(res["iou0"]) < SIDE_ABSENT_IOU0 and float(res["iou"]) < REFINE_MIN_IOU_SIDE


def side_elsewhere(res, b: BT.Bottle, others, iou0_max: float = REAIM_IOU0) -> np.ndarray | None:
    if res is None or res["iou0"] >= iou0_max or res["iou"] < REAIM_IOU:
        return None
    new_xy = np.asarray(b.xy, float) + np.array([float(res["dx"]), float(res["dy"])])
    if not all(BT._footprint_dist(new_xy, ob) >= b.radius - 0.01 for ob in others if ob is not b):
        return None
    return new_xy


def reach_ok(ev, rot) -> bool:
    ev = np.asarray(ev, float)
    return (
        abs(float(ev @ rot[:, 0])) <= REACH_TOL
        and abs(float(ev @ rot[:, 1])) <= REACH_LAT
        and (abs(float(ev @ rot[:, 2])) <= REACH_ALONG)
    )


def grasp(arm: Arm, g: Grasp, b: BT.Bottle | None = None, others=(), may_reaim: bool = False, confirm: bool = False):
    arm.reaim_to = None
    for qv in g.via:
        yield from retry(arm, lambda: arm.joints_to(qv, grip=OPEN, joint_step=FREE_STEP))
        if arm.status != "ok":
            return False
    if g.kind == "side":
        hover = pose(g.rot, g.pre + np.array([0.0, 0.0, 0.07]))
        r = K.solve_near(arm.name, hover, g.q_pre, K.TCP_OFFSET, n_random=1)
        if r.ok:
            yield from retry(arm, lambda: arm.joints_to(r.q, grip=OPEN, joint_step=FREE_STEP))
            if arm.status != "ok":
                return False
            yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.pre), lin_step=0.014, grip=OPEN))
        else:
            yield from retry(arm, lambda: arm.joints_to(g.q_pre, grip=OPEN, joint_step=FREE_STEP))
    else:
        yield from retry(arm, lambda: arm.joints_to(g.q_pre, grip=OPEN, joint_step=FREE_STEP))
    if arm.status != "ok":
        return False
    if g.kind == "top" and b is not None and (not b.standing):
        pre0 = g.pre.copy()
        if wrist_refine(arm, g, b, others) and float(np.linalg.norm(g.pre - pre0)) > REFINE_MOVE_FIRST:
            yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.pre), lin_step=0.01, grip=OPEN))
            if arm.status != "ok":
                return False
        kind, res = getattr(arm, "refine_res", ("none", None))
        if kind == "measured" and refine_absent(res):
            return False
    elif g.kind == "side" and b is not None and b.standing:
        if wrist_refine(arm, g, b, others):
            yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.pre), lin_step=0.01, grip=OPEN))
            if arm.status != "ok":
                return False
        else:
            kind, res = getattr(arm, "refine_res", ("none", None))
            iou0_max = REAIM_CONFIRM_IOU0 if confirm else REAIM_IOU0
            new_xy = side_elsewhere(res, b, others, iou0_max) if kind == "measured" and may_reaim else None
            if new_xy is not None:
                for d in (REAIM_BACK, REAIM_BACK / 2):
                    p = g.pre - g.rot[:, 0] * d
                    if K.solve_near(arm.name, pose(g.rot, p), arm.q, K.TCP_OFFSET, n_random=1).ok:
                        yield from retry(arm, lambda p=p: arm.line_to(pose(g.rot, p), lin_step=0.012, grip=OPEN))
                        if arm.status == "ok":
                            g.pre = p
                        break
                arm.reaim_to = new_xy
                return False
            if confirm and kind == "measured" and (res is None or res["iou0"] < REAIM_CONFIRM_IOU0):
                return False
            if kind == "measured" and side_absent(res):
                return False
    near = g.tcp - g.rot[:, 0] * 0.03
    if (
        float(np.linalg.norm(near - arm.tcp())) > 0.02
        and K.solve_near(arm.name, pose(g.rot, near), arm.q, K.TCP_OFFSET, n_random=1).ok
    ):
        yield from retry(arm, lambda: arm.line_to(pose(g.rot, near), lin_step=0.02, grip=OPEN))
        if arm.status != "ok":
            return False
    yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.tcp), lin_step=0.009, grip=OPEN))
    if arm.status != "ok":
        return False
    yield from settle(arm, max_steps=6)
    ev = tcp_obs(arm) - g.tcp
    ok = reach_ok(ev, g.rot)
    ev = [round(float(x), 4) for x in ev]
    if not ok and b is not None:
        no_reach = getattr(arm, "no_reach", None)
        if no_reach is not None:
            no_reach.append(np.asarray(b.xy, float).copy())
        return False
    yield from arm.set_grip(0.0, n=6)
    yield from arm.hold(2)
    w = held(arm)
    return w >= MIN_HELD


def held_extent(b: BT.Bottle, g: Grasp) -> float:
    if g.kind == "side":
        return max(b.radius, 0.09)
    return b.height / 2


def held_segment(b: BT.Bottle, g: Grasp):
    if g.kind == "side":
        return None
    return np.array([math.cos(b.yaw), math.sin(b.yaw)]) * max(0.0, b.height / 2 - b.radius)


def _seg2_dist(p0, p1, q0, q1) -> float:
    ts = np.linspace(0.0, 1.0, 9)
    P = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    d = q1 - q0
    L2 = float(d @ d)
    t = np.zeros(len(P)) if L2 < 1e-12 else np.clip((P - q0) @ d / L2, 0.0, 1.0)
    d1 = float(np.min(np.linalg.norm(P - (q0 + t[:, None] * d), axis=1)))
    Q = q0[None, :] + ts[:, None] * d[None, :]
    e = p1 - p0
    M2 = float(e @ e)
    u = np.zeros(len(Q)) if M2 < 1e-12 else np.clip((Q - p0) @ e / M2, 0.0, 1.0)
    return min(d1, float(np.min(np.linalg.norm(Q - (p0 + u[:, None] * e), axis=1))))


def carry_bottom_z(p_from, p_to, others, extent: float = 0.0, half=None, turn: float = 0.0) -> float:
    a = np.asarray(p_from, float)[:2]
    d = np.asarray(p_to, float)[:2] - a
    z = K.TABLE_Z + LIFT_OFF
    n = max(2, int(math.ceil(float(np.linalg.norm(d)) / 0.03)) + 1)
    for o in others:
        if o.standing:
            o0 = o1 = np.asarray(o.xy, float)
        else:
            ax = np.array([math.cos(o.yaw), math.sin(o.yaw)]) * max(0.0, o.height / 2 - o.radius)
            o0, o1 = o.xy - ax, o.xy + ax
        near = False
        for t in np.linspace(0.0, 1.0, n):
            c = a + t * d
            if half is None:
                h0 = h1 = c
            else:
                hv = K.rot_z(turn * t)[:2, :2] @ half
                h0, h1 = c - hv, c + hv
            if _seg2_dist(h0, h1, o0, o1) < extent + PATH_MARGIN + o.radius:
                near = True
                break
        if near:
            z = max(z, K.TABLE_Z + (o.height if o.standing else 2 * o.radius) + 0.03)
    return z


def lift(arm: Arm, g: Grasp, z_bottom: float = CLEAR_Z):
    z_goal = max(g.tcp[2] + 0.05, z_bottom + g.below)
    for z in (z_goal, z_goal - 0.03, z_goal - 0.06, g.tcp[2] + 0.08, g.tcp[2] + 0.05):
        T = pose(g.rot, [g.tcp[0], g.tcp[1], z])
        if K.solve_near(arm.name, T, arm.q, K.TCP_OFFSET, n_random=1).ok:
            yield from retry(arm, lambda: arm.line_to(T, lin_step=0.014))
            return arm.status == "ok"
    return False


def release(arm: Arm, n_hold: int = 2):
    yield from settle(arm, max_steps=4)
    yield from arm.set_grip(OPEN, n=4)
    yield from arm.hold(n_hold)


def retreat(arm: Arm, rot, p):
    for back in (0.08, 0.05, 0.03):
        T = pose(rot, p - rot[:, 0] * back)
        if K.solve_near(arm.name, T, arm.q, K.TCP_OFFSET, n_random=1).ok:
            yield from retry(arm, lambda: arm.line_to(T, lin_step=0.012))
            if arm.status == "ok":
                break
    q = arm.tcp()
    for up in (0.06, 0.04, 0.02):
        T = pose(rot, q + np.array([0.0, 0.0, up]))
        if K.solve_near(arm.name, T, arm.q, K.TCP_OFFSET, n_random=1).ok:
            yield from retry(arm, lambda: arm.line_to(T, lin_step=0.012))
            return


def exit_standing(arm: Arm, sh: Shared, rot, p, nb: BT.Bottle):
    yield from arm.hold(3)
    x = rot[:, 0]
    back = K.FINGERTIP_OFFSET - SIDE_AXIS_X + nb.radius + EXIT_CLEAR
    for d in (back, back - 0.02, 0.06):
        T = pose(rot, p - x * d)
        if K.solve_near(arm.name, T, arm.q, K.TCP_OFFSET, n_random=1).ok:
            yield from retry(arm, lambda: arm.line_to(T, lin_step=0.01))
            if arm.status == "ok":
                break
    obstacles = sh.others(None)
    targets = park_q(arm), park_q(arm, LOOK_Q), LOOK_Q
    p0 = arm.tcp()
    for up in (0.0, 0.03, 0.06, 0.09, 0.12):
        T = pose(rot, p0 + np.array([0.0, 0.0, up]))
        if up > 0:
            r = K.solve_near(arm.name, T, arm.q, K.TCP_OFFSET, n_random=1)
            if not r.ok:
                break
            if not path_clear(arm.name, arm.q, r.q, obstacles, grip=1.0, margin=0.0):
                break
            yield from retry(arm, lambda: arm.line_to(T, lin_step=0.012))
        if any(path_clear(arm.name, arm.q, qt, obstacles, grip=1.0, margin=EXIT_MARGIN) for qt in targets):
            return True
    return False


def back_off(arm: Arm, g: Grasp):
    yield from arm.set_grip(OPEN, n=6)
    yield from arm.line_to(pose(g.rot, g.pre), lin_step=0.012)


def _options(arm: Arm, b: BT.Bottle, others) -> list:
    opts = grasp_options(arm.name, b, arm.q, others)
    if opts:
        return opts
    for level, obst in ((1, others), (2, [o for o in others if o.standing])):
        opts = grasp_options(arm.name, b, arm.q, obst, check_arm=False)
        if opts:
            return opts
    return []


def clean_grasp(arm: Arm, sh: Shared, b: BT.Bottle) -> bool:
    return bool(grasp_options(arm.name, b, arm.q, sh.others(b)))


def _claim_x(arm: Arm, g: Grasp, b: BT.Bottle) -> float:
    if arm.name == "left":
        return float(max(g.tcp[0], g.pre[0])) + b.radius
    return float(min(g.tcp[0], g.pre[0])) - b.radius


def aimed_grasp(arm: Arm, sh: Shared, g: Grasp, b: BT.Bottle):
    ok = yield from grasp(arm, g, b, sh.others(b), may_reaim=True)
    for k in range(REAIM_MAX):
        new_xy = getattr(arm, "reaim_to", None)
        if ok or new_xy is None:
            return ok
        arm.reaim_to = None
        b.xy = np.asarray(new_xy, float)
        sh.version += 1
        opts = grasp_options(arm.name, b, arm.q, sh.others(b))
        if opts and (yield from claim(arm, sh, _claim_x(arm, opts[0], b))):
            opts = grasp_options(arm.name, b, arm.q, sh.others(b))
            if opts:
                yield from claim(arm, sh, _claim_x(arm, opts[0], b))
        if not opts:
            return False
        for f in fields(Grasp):
            setattr(g, f.name, getattr(opts[0], f.name))
        ok = yield from grasp(arm, g, b, sh.others(b), may_reaim=k + 1 < REAIM_MAX, confirm=True)
    return ok


def clean_pick(arm: Arm, sh: Shared, b: BT.Bottle) -> bool:
    others = sh.others(b)
    opts = grasp_options(arm.name, b, arm.q, others)
    if not opts:
        return False
    if not others:
        return True
    g = opts[0]
    hs = held_segment(b, g)
    zb = carry_bottom_z(b.xy, sh.bin.centre, others, b.radius if hs is not None else held_extent(b, g), hs)
    return bin_carry(arm.name, g, b, sh.bin, others, zb) is not None


def _carry_plan(arm: Arm, sh: Shared, b: BT.Bottle, g):
    others = sh.others(b)
    hs = held_segment(b, g)
    zb = carry_bottom_z(b.xy, sh.bin.centre, others, b.radius if hs is not None else held_extent(b, g), hs)
    return bin_carry(arm.name, g, b, sh.bin, others, zb)


def to_bin(arm: Arm, sh: Shared, b: BT.Bottle, may_defer: bool = False, defer_if=None):
    def may() -> bool:
        return may_defer or (defer_if is not None and bool(defer_if()))

    others = sh.others(b)
    clean = grasp_options(arm.name, b, arm.q, others)
    if not clean and others and may():
        return "defer"
    opts = clean or _options(arm, b, others)
    if not opts:
        return "unreachable"
    g = opts[0]
    if clean and others and may() and (_carry_plan(arm, sh, b, g) is None):
        return "defer"
    waited = yield from claim(arm, sh, float(max(g.tcp[0], g.pre[0])) + b.radius)
    if waited:
        clean = grasp_options(arm.name, b, arm.q, sh.others(b))
        if sh.others(b) and may() and (not clean or _carry_plan(arm, sh, b, clean[0]) is None):
            settle_front(arm, sh)
            return "defer"
        opts = clean or _options(arm, b, sh.others(b))
        if not opts:
            return "unreachable"
        g = opts[0]
        yield from claim(arm, sh, float(max(g.tcp[0], g.pre[0])) + b.radius)
    arm.idle_blocked = False
    ok = yield from aimed_grasp(arm, sh, g, b)
    if not ok:
        if arm.idle_blocked and getattr(arm, "no_reach", None) is not None:
            arm.no_reach.append(np.asarray(b.xy, float).copy())
        yield from back_off(arm, g)
        settle_front(arm, sh)
        return "failed"
    cplan = _carry_plan(arm, sh, b, g)
    sh.remove(b)
    hs = held_segment(b, g)
    zb = carry_bottom_z(b.xy, sh.bin.centre, sh.others(b), b.radius if hs is not None else held_extent(b, g), hs)
    if cplan is not None:
        zb = cplan[0]
    ok = yield from lift(arm, g, zb)
    z_carry = float(arm.tcp()[2])
    ext = max(0.0, held_extent(b, g) - BODY_X)
    if arm.name == "left":
        settle_front(arm, sh, ext)
        track_front(arm, sh, ext)
    dp = drop_pose(arm.name, g, sh.bin, arm.q, z_carry)
    if dp is None:
        dp = pose(g.rot, [sh.bin.centre[0] + 0.06, sh.bin.centre[1], z_carry]), None
    step = HELD_STEP if g.kind == "side" else FREE_STEP
    seeds = () if dp[1] is None else (dp[1],)
    planned = cplan is not None and cplan[2] is not None and (abs(z_carry - (cplan[0] + g.below)) < 0.01)
    if planned and cplan[1] == "line":
        yield from retry(
            arm,
            lambda: arm.line_to(
                cplan[3], lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=step, fallback_joint=False
            ),
        )
        if arm.status == "fail":
            yield from retry(arm, lambda: arm.joints_to(cplan[2], joint_step=step))
    elif planned:
        yield from retry(arm, lambda: arm.joints_to(cplan[2], joint_step=step))
    else:
        yield from retry(
            arm, lambda: arm.move_to(dp[0], min_z=min(z_carry, dp[0][2, 3]) - 0.03, joint_step=step, seeds=seeds)
        )
        if arm.status == "fail" and dp[1] is not None:
            yield from retry(arm, lambda: arm.joints_to(dp[1], joint_step=step))
    untrack(arm)
    yield from arm.hold(2 if g.kind == "side" else 0)
    w = held(arm)
    yield from release(arm)
    sh.q_drop = arm.q.copy()
    settle_front(arm, sh)
    return "done" if w >= MIN_HELD else "lost"


def stage_spots(taken, standing: bool = False) -> list:
    out = []
    base = K.base_mat("left")[:2, 3]
    for y in (-0.2, -0.12, -0.05, -0.28) if standing else (-0.2, -0.12, -0.28, -0.05):
        for x in (0.15, 0.1, 0.05, 0.0, 0.2, -0.05):
            p = np.array([x, y])
            if not standing and float(np.linalg.norm(p - base)) > LEFT_TOP_REACH:
                continue
            if all(np.linalg.norm(p - t) >= STAGE_GAP for t in taken):
                out.append(p)
    return out


def footprint_clear(nb: BT.Bottle, others, margin: float = 0.03) -> bool:
    if nb.standing:
        pts = [nb.xy]
    else:
        a = np.array([math.cos(nb.yaw), math.sin(nb.yaw)])
        pts = [nb.xy + a * s for s in np.linspace(-nb.height / 2, nb.height / 2, 7)]
    for o in others:
        if o is nb:
            continue
        for p in pts:
            if _dist_to_bottle(p, o) < nb.radius + margin:
                return False
    return True


def _dist_to_bottle(p, o: BT.Bottle) -> float:
    if o.standing:
        return float(np.linalg.norm(p - o.xy)) - o.radius
    a = np.array([math.cos(o.yaw), math.sin(o.yaw)])
    s = float(np.clip((p - o.xy) @ a, -o.height / 2, o.height / 2))
    return float(np.linalg.norm(p - (o.xy + a * s))) - o.radius


STAGE_QUALITY = "clean", "unclean", "exit", "low"
WAY_CLEAR = GRIP_HALF + 0.02


def _rank(quality) -> int:
    return STAGE_QUALITY.index(quality) if quality in STAGE_QUALITY else len(STAGE_QUALITY)


def _without_left(sh: Shared) -> Shared:
    left = [j for j in sh.jobs if j.arm == "left" and j.kind == "bin" and (j.state == "todo")]
    jobs = [replace(j, state="done") if any(j is lj for lj in left) else j for j in sh.jobs]
    scene = [o for o in sh.scene if not any(o is lj.bottle for lj in left)]
    return Shared(
        jobs=jobs,
        bin=sh.bin,
        scene=scene,
        staged=list(sh.staged),
        front=dict(sh.front),
        want=dict(sh.want),
        gone=set(sh.gone),
        version=sh.version,
        q_drop=sh.q_drop,
        parked=set(sh.parked),
    )


def way_clear(p_from, p_to, bottles, clear: float = WAY_CLEAR) -> bool:
    a, c = np.r_[np.asarray(p_from, float)[:2], 0.0], np.r_[np.asarray(p_to, float)[:2], 0.0]
    for o in bottles:
        p, q, r = _capsule(o)
        p, q = np.r_[p[:2], 0.0], np.r_[q[:2], 0.0]
        if _segseg(a, c, p, q) - r < clear:
            return False
    return True


def _left_busy(arm: Arm, sh: Shared) -> bool:
    o = arm.other
    if o is None or not o.active:
        return False
    return any(j.arm == "left" and j.kind == "bin" and (j.state == "todo") for j in sh.jobs)


def _exit_q(arm: str, rot, p, nb: BT.Bottle, q_place, others, targets):
    x = rot[:, 0]
    obst = list(others) + [nb]
    if nb.standing:
        back = K.FINGERTIP_OFFSET - SIDE_AXIS_X + nb.radius + EXIT_CLEAR
        base = None
        for d in (back, back - 0.02, 0.06):
            r = K.solve_near(arm, pose(rot, p - x * d), q_place, K.TCP_OFFSET, n_random=1)
            if r.ok:
                base = p - x * d, r.q
                break
        if base is None:
            return None
        p0, q = base
        for up in (0.0, 0.03, 0.06, 0.09, 0.12):
            if up > 0:
                r = K.solve_near(arm, pose(rot, p0 + np.array([0.0, 0.0, up])), q, K.TCP_OFFSET, n_random=1)
                if not r.ok or not path_clear(arm, q, r.q, obst, grip=1.0, margin=0.0, n=4):
                    return None
                q = r.q
            for qt in targets:
                if path_clear(arm, q, qt, obst, grip=1.0, margin=EXIT_MARGIN):
                    return q, qt
        return None
    for back in (0.08, 0.05, 0.03):
        r = K.solve_near(arm, pose(rot, p - x * back), q_place, K.TCP_OFFSET, n_random=1)
        if r.ok:
            break
    else:
        return None
    q = r.q
    pt = p - x * back
    for up in (0.06, 0.04, 0.02):
        ru = K.solve_near(arm, pose(rot, pt + np.array([0.0, 0.0, up])), q, K.TCP_OFFSET, n_random=1)
        if ru.ok:
            q = ru.q
            break
    for qt in targets:
        if path_clear(arm, q, qt, obst, grip=1.0, margin=EXIT_MARGIN):
            return q, qt


def left_corridor(t: BT.Bottle):
    base = np.asarray(K.BASE_POS["left"][:2], float)
    d = t.xy - base
    d = d / max(1e-09, float(np.linalg.norm(d)))
    return base, t.xy + d * CORRIDOR_BEYOND


def in_corridor(nb: BT.Bottle, c) -> bool:
    if not nb.standing:
        return False
    a, e = c
    return _seg2_dist(a, e, nb.xy, nb.xy) < GRIP_HALF + nb.radius + CORRIDOR_MARGIN


def by_travel(cands, q0, step: float) -> list:
    q0 = np.asarray(q0, dtype=float)
    return sorted(
        cands, key=lambda c: math.ceil(float(np.max(np.abs(np.asarray(c[4].q, dtype=float) - q0))) / step - 1e-09)
    )


def _stage_plan(arm: Arm, sh: Shared, job: Job, g: Grasp):
    b = job.bottle
    others = sh.others(b)
    taken = [j.spot for j in sh.jobs if j.spot is not None and j is not job and (j.state in ("staged", "todo"))]
    off = g.tcp[:2] - b.xy
    z_place = g.tcp[2] + 0.003
    hs = held_segment(b, g)
    ext = held_extent(b, g)
    held = held_local(arm.name, g, b)
    q_l = arm.other.q if arm.other is not None else LOOK_Q
    pend = [j.bottle for j in sh.jobs if j.kind == "bin" and j.state == "todo"]
    pend += [j.staged for j in sh.staged if j.state == "staged" and j.staged is not None]
    pend = [pb for pb in pend if any(o is pb for o in sh.scene) and pb is not b]
    corridors = [left_corridor(pb) for pb in pend]
    starts = [q_l] + ([sh.q_drop] if sh.q_drop is not None else [])

    def left_clean(t, obst) -> bool:
        return all(grasp_options("left", t, q0, obst) for q0 in starts)

    pend = [pb for pb in pend if left_clean(pb, [o for o in sh.others(pb) if o is not b])]
    targets = [park_q(arm), park_q(arm, LOOK_Q), LOOK_Q]
    bx, by = K.BASE_POS[arm.name][:2]
    phi_g = math.atan2(g.rot[1, 0], g.rot[0, 0])
    why = {"footprint": 0, "ik": 0, "left": 0}
    found = {}
    left_ok = {}
    n_checks = n_evals = 0
    step = HELD_STEP if g.kind == "side" else FREE_STEP
    z_lo = K.TABLE_Z + LIFT_OFF + g.below
    lift_ik = {}

    def lifts(z_need):
        out = []
        for zc in sorted(
            {round(z_need, 3), round(max(z_lo, min(z_need - 0.06, z_lo + 0.06)), 3), round(z_lo, 3)}, reverse=True
        ):
            if zc > z_need + 1e-06:
                continue
            if zc not in lift_ik:
                lift_ik[zc] = K.solve_near(
                    arm.name, pose(g.rot, [g.tcp[0], g.tcp[1], max(zc, g.tcp[2] + 0.05)]), g.q, K.TCP_OFFSET, n_random=1
                )
            if lift_ik[zc].ok:
                out.append((zc, lift_ik[zc]))
        return out

    tight_cache = {}

    def split_tight(q1):
        key = tuple(np.round(q1, 4))
        if key not in tight_cache:
            tight = [o for o in others if not arm_clear(arm.name, q1, [o], grip=0.0, held=held)]
            tight_cache[key] = tight, [o for o in others if all(o is not t for t in tight)]
        return tight_cache[key]

    def bump(k):
        why[k] = why.get(k, 0) + 1

    def screened(spot) -> bool:
        for o in others:
            if (
                o.standing
                and float(_seg_dist(np.r_[o.xy, 0.0][None], np.r_[b.xy, 0.0], np.r_[spot, 0.0])[0])
                < 0.087 + o.radius + 0.01
            ):
                return True
        return False

    spots = stage_spots(taken, b.standing)
    spots = [sp for _, sp in sorted(enumerate(spots), key=lambda t: (screened(t[1]), t[0]))]
    for spot in spots:
        if "clean" in found or (n_evals >= MAX_STAGE_EVALS and found):
            break
        if n_evals >= 2 * MAX_STAGE_EVALS:
            break
        if b.standing:
            a0 = math.remainder(math.atan2(spot[1] - by, spot[0] - bx) - phi_g, 2 * math.pi)
            turns = [a0 + d for d in (0.0, 0.3, -0.3, 0.6, -0.6)]
        else:
            turns = [(k + 1) // 2 * (math.pi / 6) * (1 if k % 2 else -1) for k in range(12)]
        cands = []
        for a in turns:
            rot = rot_about_z(g.rot, a)
            nb = BT.Bottle(
                spot.copy(),
                b.standing,
                math.remainder(b.yaw + a, 2 * math.pi),
                b.model,
                b.height,
                b.radius,
                b.iou,
                extra=dict(b.extra),
            )
            if not footprint_clear(nb, others):
                why["footprint"] += 1
                continue
            p = np.r_[spot + K.rot_z(a)[:2, :2] @ off, z_place]
            r = K.solve_near(arm.name, pose(rot, p), g.q, K.TCP_OFFSET, n_random=1)
            if not r.ok:
                why["ik"] += 1
                continue
            cands.append((a, rot, nb, p, r))
        if not b.standing:
            cands = by_travel(cands, g.q, step)
        for a, rot, nb, p, r in cands:
            if arm.other is not None and _reach_failed(arm.other, nb):
                bump("no_reach")
                continue
            n_evals += 1
            key = float(spot[0]), float(spot[1]), None if b.standing else round(nb.yaw % math.pi, 1)
            if key not in left_ok:
                left_ok[key] = bool(grasp_options("left", nb, q_l, others=(), check_arm=False))
            if not left_ok[key]:
                why["left"] += 1
                continue
            if not arm_clear(arm.name, r.q, others, grip=0.0):
                bump("place_arm")
                continue
            z_need = carry_bottom_z(b.xy, spot, others, b.radius if hs is not None else ext, hs, a) + g.below
            carry, z_any, tried_line = None, None, False
            for zc, r1 in lifts(z_need):
                r2 = K.solve_near(
                    arm.name, pose(rot, [p[0], p[1], max(zc, p[2] + 0.03)]), r.q, K.TCP_OFFSET, n_random=1
                )
                if not r2.ok:
                    continue
                if z_any is None:
                    z_any = zc, r1.q, r2.q
                tight, rest = split_tight(r1.q)
                if path_clear(arm.name, r1.q, r2.q, rest, grip=0.0, held=held) and path_clear(
                    arm.name, r1.q, r2.q, tight, grip=0.0, held=held, margin=0.0
                ):
                    carry = zc, r1.q, r2.q, "joint"
                    break
                if not tried_line:
                    tried_line = True
                    pl = motion.line(
                        arm.name,
                        r1.q,
                        pose(rot, [p[0], p[1], max(zc, p[2] + 0.03)]),
                        K.TCP_OFFSET,
                        lin_step=CARRY_LIN,
                        ang_step=CARRY_ANG,
                        joint_step=step,
                    )
                    if pl.ok:
                        qs = list(pl.q[:: max(1, len(pl.q) // 10)]) + [pl.q[-1]]
                        if all(arm_clear(arm.name, qq, rest, grip=0.0, held=held) for qq in qs) and all(
                            arm_clear(arm.name, qq, tight, grip=0.0, held=held, margin=0.0) for qq in qs
                        ):
                            carry = zc, r1.q, np.asarray(pl.q[-1]), "line"
                            break
            entry = dict(spot=spot, rot=rot, p=p, q_place=r.q, nb=nb)
            if carry is None:
                bump("path" if z_any is not None else "carry_ik")
                if "low" not in found:
                    zc, q1, q2 = z_any if z_any is not None else (z_need, None, None)
                    found["low"] = dict(entry, z=zc, q1=q1, q2=q2)
                continue
            entry.update(z=carry[0], q1=carry[1], q2=carry[2], how=carry[3])
            if _exit_q(arm.name, rot, p, nb, r.q, others, targets) is None:
                bump("exit")
                found.setdefault("exit", entry)
                continue
            if n_checks >= MAX_CLEAN_CHECKS or any(in_corridor(nb, c) for c in corridors):
                if n_checks < MAX_CLEAN_CHECKS:
                    bump("corridor")
                found.setdefault("unclean", entry)
                continue
            n_checks += 1
            clean = left_clean(nb, others)
            if clean:
                for pb in pend:
                    if not left_clean(pb, [o for o in sh.others(pb) if o is not b] + [nb]):
                        clean = False
                        bump("blocks_left")
                        break
            else:
                bump("unclean")
            if clean:
                found["clean"] = entry
                break
            found.setdefault("unclean", entry)
    for quality in ("clean", "unclean", "exit", "low"):
        if quality in found:
            return quality, found[quality], why
    return None, None, why


def to_stage(arm: Arm, sh: Shared, job: Job):
    b = job.bottle
    ctx = arm.ctx
    need = JOB_STEPS["stage"] + JOB_STEPS["bin"] + HOME_RESERVE
    hope = None
    while True:
        opts = _options(arm, b, sh.others(b))
        if not opts:
            return "unreachable"
        g = opts[0]
        quality, choice, _ = _stage_plan(arm, sh, job, g)
        if quality in ("clean", "unclean") or not _left_busy(arm, sh) or steps_left(ctx) < need + 40:
            break
        if hope is None:
            hope, _, _ = _stage_plan(arm, _without_left(sh), job, g)
            hope = hope or "none"
            leaving = [j.bottle for j in sh.jobs if j.arm == "left" and j.kind == "bin" and (j.state == "todo")]
            if (
                quality == "low"
                and _rank(hope) >= _rank(quality)
                and (choice is not None)
                and way_clear(b.xy, choice["spot"], [o for o in leaving if any(o is s for s in sh.scene)])
            ):
                break
        v = sh.version
        yield from park(arm, sh, LOOK_Q)
        sh.parked.add(arm.name)
        yield from arm.wait_until(lambda: sh.version != v or not _left_busy(arm, sh), max_steps=150)
        sh.parked.discard(arm.name)
    if steps_left(ctx) < need:
        return "failed"
    if (yield from yield_to_left(arm, sh, _front(arm.name, float(min(g.tcp[0], g.pre[0])) - b.radius))):
        if steps_left(ctx) < need:
            return "failed"
        opts = _options(arm, b, sh.others(b))
        if not opts:
            return "unreachable"
        g = opts[0]
        quality, choice, _ = _stage_plan(arm, sh, job, g)
    waited = yield from claim(arm, sh, float(min(g.tcp[0], g.pre[0])) - b.radius)
    if waited:
        opts = _options(arm, b, sh.others(b))
        if not opts:
            return "unreachable"
        g = opts[0]
        quality, choice, _ = _stage_plan(arm, sh, job, g)
        yield from claim(arm, sh, float(min(g.tcp[0], g.pre[0])) - b.radius)
    off0, rot0 = g.tcp[:2] - b.xy, g.rot.copy()
    ok = yield from aimed_grasp(arm, sh, g, b)
    if not ok:
        yield from back_off(arm, g)
        settle_front(arm, sh)
        arm.grasp_failed = True
        return "failed"
    if float(np.abs(g.rot - rot0).max()) > 1e-06 or float(np.linalg.norm(g.tcp[:2] - b.xy - off0)) > 0.001:
        quality, choice, _ = _stage_plan(arm, sh, job, g)
    sh.remove(b)
    step = HELD_STEP if g.kind == "side" else FREE_STEP
    if choice is not None:
        spot, rot, p, q_place, nb = choice["spot"], choice["rot"], choice["p"], choice["q_place"], choice["nb"]
        yield from lift(arm, g, choice["z"] - g.below)
        if arm.status == "ok" and choice["q2"] is not None:
            yield from claim(arm, sh, float(p[0]) - b.radius - 0.03)
            if choice.get("how") == "line":
                T_above = pose(rot, [p[0], p[1], max(choice["z"], p[2] + 0.03)])
                yield from retry(
                    arm,
                    lambda: arm.line_to(
                        T_above, lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=step, fallback_joint=False
                    ),
                )
                if arm.status == "fail":
                    yield from retry(arm, lambda: arm.joints_to(choice["q2"], joint_step=step))
            else:
                yield from retry(arm, lambda: arm.joints_to(choice["q2"], joint_step=step))
        else:
            above = np.array([p[0], p[1], max(choice["z"], p[2] + 0.03)])
            yield from claim(arm, sh, float(p[0]) - b.radius - 0.03)
            yield from retry(
                arm,
                lambda: arm.move_to(
                    pose(rot, above), min_z=min(float(arm.tcp()[2]), above[2]) - 0.03, joint_step=step, seeds=(q_place,)
                ),
            )
        if arm.status == "ok":
            low = p + np.array([0.0, 0.0, 0.035])
            if (
                float(arm.tcp()[2]) - low[2] > 0.02
                and K.solve_near(arm.name, pose(rot, low), arm.q, K.TCP_OFFSET, n_random=1).ok
            ):
                yield from retry(arm, lambda: arm.line_to(pose(rot, low), lin_step=0.02))
            if g.kind == "side":
                yield from arm.hold(3)
            if arm.status == "ok":
                yield from retry(arm, lambda: arm.line_to(pose(rot, p), lin_step=0.008))
        if arm.status == "ok":
            yield from release(arm, n_hold=2)
            sh.add(nb)
            if b.standing:
                yield from exit_standing(arm, sh, rot, p, nb)
            else:
                yield from retreat(arm, rot, p)
            settle_front(arm, sh)
            job.spot = spot
            job.staged = nb
            yield from park_clear(arm, sh)
            sh.staged.append(job)
            return "staged"
    yield from retry(arm, lambda: arm.move_to(pose(g.rot, g.tcp + np.array([0.0, 0.0, 0.04])), min_z=g.tcp[2]))
    yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.tcp + np.array([0.0, 0.0, 0.003])), lin_step=0.008))
    yield from release(arm)
    yield from retry(arm, lambda: arm.line_to(pose(g.rot, g.pre), lin_step=0.012))
    sh.add(b)
    settle_front(arm, sh)
    return "failed"


def park_clear(arm: Arm, sh: Shared):
    obst = sh.others(None)
    for qt in (park_q(arm), park_q(arm, LOOK_Q), LOOK_Q):
        if path_clear(arm.name, arm.q, qt, obst, grip=1.0, margin=EXIT_MARGIN):
            track_front(arm, sh)
            yield from arm.joints_to(qt, grip=OPEN, joint_step=FREE_STEP)
            untrack(arm)
            settle_front(arm, sh)
            return
    yield from park(arm, sh)


def left_task(arm: Arm, sh: Shared):
    ctx = arm.ctx
    pending = [j for j in sh.jobs if j.arm == "left"]
    defers = {}
    while pending:
        if steps_left(ctx) < JOB_STEPS["bin"] + HOME_RESERVE:
            break
        job = next((j for j in pending if clean_pick(arm, sh, j.bottle)), None)
        if job is None:
            st = clean_staged(arm, sh)
            if st is not None:
                yield from take_staged(arm, sh, st, defers)
                continue
        job = job or pending[0]
        pending = [j for j in pending if j is not job]
        put_back = defers.get(id(job), 0) < 1
        res = yield from to_bin(
            arm, sh, job.bottle, defer_if=(lambda: clean_staged(arm, sh) is not None) if put_back else None
        )
        if res == "defer":
            defers[id(job)] = defers.get(id(job), 0) + 1
            pending = [job] + pending
            continue
        job.state = res
        if job.state in ("done", "lost"):
            sh.gone.add(id(job.bottle))
    while any(j.state in ("todo", "staged") for j in sh.jobs if j.kind == "stage"):
        ready = [j for j in sh.staged if j.state == "staged"]
        if not ready:
            if arm.other is not None and (not arm.other.active):
                break
            yield from arm.joints_to(LOOK_Q, joint_step=FREE_STEP)
            settle_front(arm, sh)
            sh.parked.add(arm.name)
            yield from arm.wait_until(
                lambda: any(j.state == "staged" for j in sh.staged) or not arm.other.active, max_steps=200
            )
            sh.parked.discard(arm.name)
            continue
        job = next((j for j in ready if clean_pick(arm, sh, j.staged)), ready[0])
        if steps_left(ctx) < JOB_STEPS["bin"] + HOME_RESERVE:
            break
        yield from take_staged(arm, sh, job, defers)
    track_front(arm, sh)
    yield from arm.joints_to(arm.q_home, joint_step=FREE_STEP)
    untrack(arm)
    settle_front(arm, sh)
    sh.parked.add(arm.name)


def clean_staged(arm: Arm, sh: Shared):
    return next((j for j in sh.staged if j.state == "staged" and clean_pick(arm, sh, j.staged)), None)


def take_staged(arm: Arm, sh: Shared, job: Job, defers: dict):
    nb = redetect(arm.ctx, job.staged)
    if nb is None:
        job.state = "failed"
        return
    sh.remove(job.staged)
    sh.add(nb)
    job.staged = nb
    defer = arm.other is not None and arm.other.active and (defers.get(id(job), 0) < MAX_DEFERS)
    res = yield from to_bin(arm, sh, nb, may_defer=defer)
    if res == "defer":
        defers[id(job)] = defers.get(id(job), 0) + 1
        v = sh.version
        n_ready = sum(j.state == "staged" for j in sh.staged)
        yield from defer_wait(
            arm,
            sh,
            lambda: sh.version != v or not arm.other.active or sum(j.state == "staged" for j in sh.staged) != n_ready,
            80,
        )
        return
    job.state = "done" if res == "done" else "failed"


def defer_wait(arm: Arm, sh: Shared, cond, max_steps: int):
    ctx = arm.ctx
    n0 = ctx.obs.step
    yield from arm.wait_until(lambda: cond() or _blocks_want(sh, arm.name), max_steps=max_steps)
    if cond() or not _blocks_want(sh, arm.name):
        return
    yield from arm.joints_to(LOOK_Q, joint_step=FREE_STEP)
    settle_front(arm, sh)
    sh.parked.add(arm.name)
    yield from arm.wait_until(cond, max_steps=max(0, max_steps - (ctx.obs.step - n0)))
    sh.parked.discard(arm.name)


def want_folded_look(hint: int, bottles, touched=()) -> bool:
    if hint >= HINT_PX or not bottles:
        return True
    return any(t and (b.standing or los_angle(b) < TOUCH_LOS) for b, t in zip(bottles, touched))


def los_angle(b: BT.Bottle, cam=None) -> float:
    p = (cam or C.head()).pos
    a = math.atan2(float(b.xy[1]) - float(p[1]), float(b.xy[0]) - float(p[0]))
    return abs((float(b.yaw) - a + math.pi / 2) % math.pi - math.pi / 2)


def touch_frac(cam, b: BT.Bottle, bad_near) -> float:
    hm = BT.hypothesis_mask(cam, b.model, b.standing, float(b.xy[0]), float(b.xy[1]), b.yaw, bad_near.shape)
    n = int(hm.sum())
    return float((hm & bad_near).sum()) / n if n else 0.0


def touching(cam, bottles, bad) -> list[bool]:
    if bad is None:
        return [False] * len(bottles)
    import cv2

    k = 2 * TOUCH_PX + 1
    near = cv2.dilate(bad.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return [touch_frac(cam, b, near) >= TOUCH_FRAC for b in bottles]


def right_task(arm: Arm, sh: Shared):
    ctx = arm.ctx
    retry = []
    pending = [j for j in sh.jobs if j.arm == "right"]
    while pending:
        if steps_left(ctx) < JOB_STEPS["stage"] + JOB_STEPS["bin"] + HOME_RESERVE:
            for j in pending:
                j.state = "failed"
            break
        job = pending[0]
        if len(pending) > 1 and (not clean_grasp(arm, sh, job.bottle)):
            alt = next((j for j in pending[1:] if clean_grasp(arm, sh, j.bottle)), None)
            if alt is not None:
                job = alt
        pending = [j for j in pending if j is not job]
        arm.grasp_failed = False
        res = yield from to_stage(arm, sh, job)
        job.state = res
        if res == "failed" and arm.grasp_failed:
            job.state = "todo"
            retry.append(job)
    yield from park(arm, sh, LOOK_Q)
    sh.parked.add(arm.name)
    for job in retry:
        nb = None
        if steps_left(ctx) >= JOB_STEPS["stage"] + JOB_STEPS["bin"] + HOME_RESERVE:
            nb = yield from look_again(arm, sh, job.bottle)
        if nb is None:
            job.state = "failed"
            continue
        sh.remove(job.bottle)
        sh.add(nb)
        job.bottle = nb
        sh.parked.discard(arm.name)
        arm.grasp_failed = False
        job.state = yield from to_stage(arm, sh, job)
        yield from park(arm, sh, LOOK_Q)
        sh.parked.add(arm.name)


def _covers(cam, mask: np.ndarray, xy, radius: float, height: float = 0.26) -> bool:
    import cv2

    ang = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)
    ring = np.stack([xy[0] + radius * np.cos(ang), xy[1] + radius * np.sin(ang)], axis=1)
    pts = np.concatenate([np.column_stack([ring, np.full(len(ring), K.TABLE_Z + z)]) for z in (0.0, height)])
    uv, d = cam.project(pts)
    if (d <= 0.02).any():
        return True
    poly = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(poly, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return bool((mask & (poly > 0)).any())


def retry_pick(found, scene, old: BT.Bottle) -> BT.Bottle | None:
    best = None
    for b in found:
        if b.iou < RETRY_MIN_IOU:
            continue
        d = float(np.linalg.norm(np.asarray(b.xy, float) - np.asarray(old.xy, float)))
        if d > RETRY_NEAR:
            continue
        if any(
            o is not old and float(np.linalg.norm(np.asarray(b.xy, float) - np.asarray(o.xy, float))) < RETRY_EXPLAINED
            for o in scene
        ):
            continue
        if best is None or d < best[0]:
            best = d, b
    return None if best is None else best[1]


def look_again(arm: Arm, sh: Shared, old: BT.Bottle):
    ctx = arm.ctx
    cam = C.head()

    def clear() -> bool:
        o = ctx.obs
        m = RM.mask(cam, o.state["left_arm_joint_state"], None, float(o.state["left_ee_joint_state"][0]))
        return not _covers(cam, m, old.xy, RETRY_NEAR + RETRY_CLEAR)

    yield from arm.wait_until(clear, max_steps=RETRY_WAIT)
    if not clear():
        return None
    o = ctx.obs
    img = o.images.get("cam_head")
    if img is None:
        return None
    found = [b for b in _detect(img, cam, o) if _in_area(b)]
    found = [_side_decide(b, found, [])[0] for b in found]
    nb = retry_pick(found, sh.others(None), old)
    return nb


def _robot_bad(cam, o):
    return RM.mask(
        cam,
        o.state["left_arm_joint_state"],
        o.state["right_arm_joint_state"],
        float(o.state["left_ee_joint_state"][0]),
        float(o.state["right_ee_joint_state"][0]),
    )


def _detect(img, cam, o) -> list:
    inst = DT.detect(img, min_score=0.1)
    bottles, _ = BT.detect_with_instances(img, cam, inst, bad=_robot_bad(cam, o))
    return bottles


def _side_decide(b: BT.Bottle, bottles, views):
    def tref(v):
        return v[3] if len(v) > 3 else None

    evs = [
        BT.side_evidence(v[0], v[1], b, bottles, bad=v[2], contest_margin_px=CONTEST_PX, table_ref=tref(v))
        for v in views
    ]
    standing, score = BT.decide_standing(b, evs, margin=SIDE_MARGIN)
    confirmed = False
    if abs(score) <= SIDE_MARGIN:
        evs0 = [BT.side_evidence(v[0], v[1], b, bottles, bad=v[2], table_ref=tref(v)) for v in views]
        st0, sc0 = BT.decide_standing(b, evs0, margin=SIDE_MARGIN)
        confirmed = abs(sc0) > SIDE_MARGIN and st0 == b.standing
    alt = b.extra.get("alt")
    kept = (
        standing != b.standing
        and abs(score) > SIDE_MARGIN
        and (alt is not None)
        and (b.iou - alt["iou"] > OVERRULE_MAX_GAP)
    )
    if kept:
        standing = b.standing
    if (
        abs(score) <= SIDE_MARGIN
        and (not confirmed)
        and (not standing)
        and (alt is not None)
        and alt["standing"]
        and (alt["iou"] >= b.iou - UNSURE_IOU)
    ):
        standing = True
    nb = BT.swap_to_alt(b) if standing != b.standing and alt is not None else b
    return nb, (round(score, 3), [None if e is None else (e.get("stand"), e.get("lie")) for e in evs], confirmed, kept)


def _in_area(b: BT.Bottle) -> bool:
    (x0, x1), (y0, y1) = BOTTLE_AREA
    if not x0 <= b.xy[0] <= x1 or b.xy[1] > y1:
        return False
    if b.xy[1] >= y0:
        return b.iou >= 0.45
    return b.xy[1] >= NEAR_Y and b.iou >= NEAR_IOU


def perceive(ctx, arms, side: bool = True):
    o = ctx.obs
    img = o.images.get("cam_head")
    if img is None:
        return [], None, [], None
    cam = C.head()
    bad = _robot_bad(cam, o)
    inst = DT.detect(img, min_score=0.1)
    bottles, _ = BT.detect_with_instances(img, cam, inst, bad=bad)
    bottles = [b for b in bottles if _in_area(b)]
    _, tref = BT.foreground(img, cam, bad=bad)
    for a in arms.values():
        a.table_ref = tref
    sides = {}
    views = []
    if side:
        for wname, arm in C.CAM_ARM.items():
            wimg = o.images.get(wname)
            if wimg is not None:
                wcam = C.wrist(o.state[f"{arm}_ee_pose"])
                views.append((wimg, wcam, _robot_bad(wcam, o), tref))
        out = []
        for b in bottles:
            nb, sides[id(b)] = _side_decide(b, bottles, views)
            sides[id(nb)] = sides[id(b)]
            out.append(nb)
        bottles = out
    tm = BT.table_polygon_mask(cam, img.shape[:2], region=((-0.41, 0.59), (-0.6, 0.5)))
    bin_ = DB.detect(img, cam, tm)
    return bottles, bin_, views, bad


def hint_mask(img, cam, bottles, bad) -> np.ndarray:
    import cv2

    fg, _ = BT.foreground(img, cam, bad=bad)
    fg = fg > 0
    expl = np.zeros(fg.shape, bool)
    for b in bottles:
        expl |= BT.hypothesis_mask(cam, b.model, b.standing, float(b.xy[0]), float(b.xy[1]), b.yaw, fg.shape)
    expl = cv2.dilate(expl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near = (cv2.dilate(bad.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~bad
    return fg & ~expl & near


def _explains_hint(hm: np.ndarray, hint) -> bool:
    if hint is None:
        return False
    import cv2

    hd = cv2.dilate(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return int((hd & hint).sum()) >= UNHIDE_HINT_PX


def _footprint_gap(a: BT.Bottle, b: BT.Bottle) -> float:
    def seg(o):
        if o.standing:
            return o.xy, o.xy
        ax = np.array([math.cos(o.yaw), math.sin(o.yaw)]) * max(0.0, o.height / 2 - o.radius)
        return o.xy - ax, o.xy + ax

    (a0, a1), (b0, b1) = seg(a), seg(b)
    return _seg2_dist(a0, a1, b0, b1) - a.radius - b.radius


def los_ambiguous(d: BT.Bottle, cam=None) -> bool:
    alt = d.extra.get("alt")
    return not d.standing and alt is not None and bool(alt.get("standing")) and (los_angle(d, cam) < LOS_AMBIG)


def behind_on_los(d: BT.Bottle, b: BT.Bottle, cam=None) -> bool:
    if not (d.standing and b.standing):
        return False
    p = np.asarray((cam or C.head()).pos[:2], float)
    u = np.asarray(d.xy, float) - p
    u = u / max(1e-09, float(np.linalg.norm(u)))
    diff = np.asarray(b.xy, float) - np.asarray(d.xy, float)
    along = float(diff @ u)
    lat = float(diff @ np.array([-u[1], u[0]]))
    return abs(lat) <= LOS_MATCH_LAT and 0.0 < along <= LOS_MATCH_ALONG


def unhide_merge(cam, bottles, dets, bad0, hint=None) -> tuple[list, list]:
    import cv2

    shape = bad0.shape
    k = 2 * TOUCH_PX + 1
    near = cv2.dilate(bad0.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    info = []
    for b in bottles:
        hb = BT.hypothesis_mask(cam, b.model, b.standing, float(b.xy[0]), float(b.xy[1]), b.yaw, shape)
        n = max(1, int(hb.sum()))
        weak = float((hb & bad0).sum()) / n >= HIDDEN_FRAC
        weak = weak or float((hb & near).sum()) / n >= TOUCH_FRAC
        info.append((b, hb & ~bad0, weak))
    drop: list = []
    added = []

    def match(d, hm):
        out = []
        for b, vis, weak in info:
            nv = int(vis.sum())
            if (
                _footprint_gap(d, b) < MATCH_GAP
                or (nv and float((hm & vis).sum()) / nv >= UNHIDE_OVERLAP)
                or (weak and behind_on_los(d, b, cam))
            ):
                out.append((b, weak))
        return out

    for d in dets:
        if los_ambiguous(d, cam):
            m0 = match(d, BT.hypothesis_mask(cam, d.model, d.standing, float(d.xy[0]), float(d.xy[1]), d.yaw, shape))
            if not (m0 and all(not b.standing for b, _ in m0)):
                d = BT.swap_to_alt(d)
                d.extra["los_swap"] = True
        if d.iou < UNHIDDEN_IOU:
            continue
        hm = BT.hypothesis_mask(cam, d.model, d.standing, float(d.xy[0]), float(d.xy[1]), d.yaw, shape)
        n = int(hm.sum())
        if n == 0:
            continue
        matched = match(d, hm)
        if any(not weak for _, weak in matched):
            continue
        if not matched and float((hm & bad0).sum()) / n < HIDDEN_FRAC and (not _explains_hint(hm, hint)):
            continue
        reps = [b for b, _ in matched]
        drop += [b for b in reps if all(b is not x for x in drop)]
        added.append((d, reps))
    kept = [b for b in bottles if all(b is not x for x in drop)]
    return kept, added


def look_unhidden(ctx, bottles, views, bad0, hint=None):
    o = ctx.obs
    img = o.images.get("cam_head")
    if img is None or bad0 is None:
        return bottles
    cam = C.head()
    dets = [d for d in _detect(img, cam, o) if _in_area(d)]
    out, added = unhide_merge(cam, bottles, dets, bad0, hint=hint)
    for d, reps in added:
        if d.extra.get("los_swap"):
            nb = d
        else:
            nb, _ = _side_decide(d, dets, views) if views else (d, None)
        out.append(nb)
    return out


def redetect(ctx, expected: BT.Bottle, max_dist: float = 0.08, fell_dist: float = 0.05) -> BT.Bottle | None:
    o = ctx.obs
    img = o.images.get("cam_head")
    if img is None:
        return None
    cam = C.head()
    dets = _detect(img, cam, o)
    best = None
    for d in dets:
        if d.iou < 0.45:
            continue
        cand = d
        if d.standing == expected.standing:
            dist = float(np.linalg.norm(d.xy - expected.xy))
            ok = dist <= max_dist
        elif expected.standing:
            dist = _dist_to_bottle(expected.xy, d) + d.radius
            ok = dist <= fell_dist
        else:
            if d.extra.get("alt") is None or d.extra["alt"]["standing"] != expected.standing:
                continue
            cand = BT.swap_to_alt(d)
            dist = float(np.linalg.norm(cand.xy - expected.xy))
            ok = dist <= max_dist
        if ok and (best is None or dist < best[0]):
            best = dist, cand
    return None if best is None else best[1]


def _reach_failed(arm: Arm, b: BT.Bottle) -> bool:
    return any(float(np.linalg.norm(p - np.asarray(b.xy, float))) < NO_REACH_DIST for p in getattr(arm, "no_reach", ()))


def _left_top_marginal(b: BT.Bottle) -> bool:
    if b.standing:
        return False
    base = K.base_mat("left")[:2, 3]
    return float(np.linalg.norm(np.asarray(b.xy, float) - base)) > LEFT_TOP_REACH


def plan(bottles, arms) -> list[Job]:
    jobs = []
    q_left = arms["left"].q
    for b in sorted(bottles, key=lambda b: b.xy[0]):
        others = [o for o in bottles if o is not b]
        left_ok = bool(
            grasp_options("left", b, q_left, others, check_arm=False)
            or grasp_options("left", b, q_left, (), check_arm=False)
        )

        def right_ok():
            return bool(
                grasp_options("right", b, arms["right"].q, others, check_arm=False)
                or grasp_options("right", b, arms["right"].q, (), check_arm=False)
            )

        if left_ok and (not _reach_failed(arms["left"], b)) and (not _left_top_marginal(b)):
            jobs.append(Job(b, "left", "bin"))
        elif right_ok():
            jobs.append(Job(b, "right", "stage"))
        elif left_ok:
            jobs.append(Job(b, "left", "bin"))
    return jobs


@register
class BinDisposal(Program):
    name = "bin_disposal"
    background = True
    call_budget_s = 20.0

    def match(self, instruction: str, first_obs) -> float:
        return 0.9 if _BIN_WORDS.search(instruction.lower()) else 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: BinArm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        for a in K.ARMS:
            arms[a].idle_guard = IDLE_GUARD
            arms[a].no_reach = []
        home = {a: arms[a].q.copy() for a in K.ARMS}
        bin_ = None
        for rnd in range(MAX_ROUNDS):
            if rnd > 0:
                yield from run_arms(
                    arms, {a: arms[a].joints_to(home[a], grip=OPEN, force=True, joint_step=FREE_STEP) for a in K.ARMS}
                )
                yield from run_arms(arms, {a: arms[a].hold(3) for a in K.ARMS})
            bottles, b2, views, bad0 = perceive(ctx, arms)
            bin_ = bin_ or b2
            hmask = hint_mask(ctx.obs.images["cam_head"], C.head(), bottles, bad0) if bad0 is not None else None
            hint = 0 if hmask is None else int(hmask.sum())
            touched = touching(C.head(), bottles, bad0)
            if (
                bin_ is not None
                and want_folded_look(hint, bottles, touched)
                and (steps_left(ctx) >= JOB_STEPS["bin"] + HOME_RESERVE + 10)
            ):
                yield from run_arms(
                    arms, {a: arms[a].joints_to(LOOK_Q, grip=OPEN, force=True, joint_step=FREE_STEP) for a in K.ARMS}
                )
                yield from run_arms(arms, {a: arms[a].hold(2) for a in K.ARMS})
                bottles = look_unhidden(ctx, bottles, views, bad0, hint=hmask)
            if bin_ is None:
                break
            if not bottles or steps_left(ctx) < JOB_STEPS["bin"] + HOME_RESERVE:
                break
            jobs = plan(bottles, arms)
            if not jobs:
                break
            sh = Shared(jobs, bin_, scene=list(bottles))
            for a in K.ARMS:
                settle_front(arms[a], sh)
            yield from run_arms(arms, {"left": left_task(arms["left"], sh), "right": right_task(arms["right"], sh)})
        yield from run_arms(
            arms, {a: arms[a].joints_to(home[a], grip=OPEN, force=True, joint_step=FREE_STEP) for a in K.ARMS}
        )
        while True:
            yield [_hold(arms)]


def _hold(arms) -> dict:
    from ...core import actions

    return actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)
