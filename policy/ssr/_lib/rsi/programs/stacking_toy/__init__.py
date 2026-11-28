from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from ...perception import cameras as C
from ...perception import robot_mask as RM
from . import scene as ST
from ...planning import collision as CO
from ...planning import kinematics as K
from ...skills import dual as D
from ...skills import grasp as G
from ...skills.runner import MIN_ARM_DIST, Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
PAD_HALF = 0.011
GRASP_DZ = 0.016
Z_SAFE = (
    K.TABLE_Z + 0.096
)
HOVER_DZ = 0.08
OPEN_MARGIN = 0.02
WIDTH_TOL = 0.006
REMEASURE_MAX = 0.015
ALIGN_DZ = 0.005
RELEASE_DZ = 0.0015
DRAG = 0.003
DRAG_STEP = 0.005
DRAG_HOLD = 1
OPEN_STEPS = 4
WIDTH_READ_BIAS = 0.0003
INSERT_STEP = 0.005
THREAD_DZ = 0.008
SLIDE_STEP = 0.008
ALIGN_STEP = 0.01
FREE_STEP = 0.016
APPROACH_STEP = 0.007
JOINT_STEP = 0.11
MAX_REACH = 0.58
YAW_COST = 1.0
XS = 0.21
SIDE_GAP = 0.1
COORD = "slab"
SITE_CLEAR = 0.03
GAP_X = 0.12
FAR_POLE = 0.065
PARK_X, PARK_Y = 0.32, -0.2
SPOTS = (-0.1, -0.36), (0.0, -0.36), (0.1, -0.36)
PUT_DZ = 0.003
CYCLE = {"direct": 75.0, "give": 60.0, "take": 75.0}
STEPS_PER_M = 120.0
CROSS_STEPS = 18.0
FAR_PENALTY = 40.0
FAR_LOCK_WAIT = 300
TAKE_WAIT = 450
SPOT_WAIT = 300
SLAB = {"place": 45.0, "pick": 45.0, "put": 35.0}
REACH_SOFT = 0.45
REACH_STEPS = 60.0
TAMP_WAIT = 60
TAMP_RISK = {ST.SQUARE: 0.09, ST.RECT: 0.085, ST.DISC: 0.02}
TAMP_SQUEEZE = 0.004
TAMP_CLOSE, TAMP_HOLD, TAMP_OPEN = 5, 3, 4
TAMP_ROUNDS = 2
STEP_LIMIT = 1200
TAMP_END = STEP_LIMIT - 5
TAMP_POLE_STEPS = 32
TAMP_TRIP_STEPS = 56


def _dense_outline(cat: int, step: float = 0.0005) -> np.ndarray:
    o = ST.OUTLINES[cat]
    pts = []
    for a, b in zip(o, np.roll(o, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.arange(n)[:, None] / n
        pts.append(a + t * (b - a))
    return np.vstack(pts)


_DENSE = {c: _dense_outline(c) for c in ST.OUTLINES}


def closing_geometry(cat: int, ang: float):
    u = np.array([math.cos(ang), math.sin(ang)])
    v = np.array([-u[1], u[0]])
    p = _DENSE[cat]
    a, b = p @ u, p @ v
    sel = np.abs(b) <= PAD_HALF
    hp, hm = float(a[sel].max()), float((-a[sel]).max())
    return hp + hm, (hp - hm) / 2.0


def closing_angles(cat: int):
    if cat == ST.STAR:
        return [(math.pi / 2 + k * 2 * math.pi / 5, True) for k in range(5)]
    if cat == ST.RECT:
        return [(0.0, True)]
    if cat == ST.SQUARE:
        return [(0.0, False)]
    return [(0.0, False), (math.pi / 4, False)]


class PieceTask:
    def __init__(self, piece: ST.Piece, pole: int, level: int, kind: str = "direct"):
        self.p = piece
        self.pole = pole
        self.level = level
        self.kind = kind
        self.grasp = None
        self.width = 0.0
        self.c = 0.0
        self.tip_sign = 1.0
        self.head = 0.0
        self.peer: PieceTask | None = None
        self.state = "pending"
        self.spot: Spot | None = None


class Spot:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)
        self.owner: PieceTask | None = None


def plan_grasp(arm: Arm, t: PieceTask, place_head: float | None = None):
    best = None
    for ang, block in closing_angles(t.p.cat):
        w, c = closing_geometry(t.p.cat, ang)
        yaw_u = t.p.yaw + ang
        cx = t.p.x + c * math.cos(yaw_u)
        cy = t.p.y + c * math.sin(yaw_u)
        blocked = (yaw_u + math.pi / 2,) if block else ()
        cands = G.grasp_candidates(
            arm.name,
            (cx, cy, K.TABLE_Z + GRASP_DZ),
            0.02,
            yaw_u,
            arm.q,
            tilts=(0.0,),
            blocked_yaws=blocked,
            hover_dz=HOVER_DZ,
        )
        for g in cands:
            if place_head is not None:
                g.cost += YAW_COST * abs(math.remainder(g.yaw - place_head, math.pi))
            if best is None or g.cost < best[0].cost:
                sign = 1.0 if abs(math.remainder(g.yaw - yaw_u, 2 * math.pi)) < 0.001 else -1.0
                best = g, w, c, sign
    return best


def observed_grip(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


def robot_bad(ctx, cam):
    o = ctx.obs
    return RM.mask(
        cam,
        o.state["left_arm_joint_state"],
        o.state["right_arm_joint_state"],
        float(o.state["left_ee_joint_state"][0]),
        float(o.state["right_ee_joint_state"][0]),
    )


def wrist_remeasure(arm: Arm, t: PieceTask):
    o = arm.ctx.obs
    img = o.images.get(f"cam_{arm.name}_wrist")
    if img is None:
        return None
    cam = C.wrist(o.state[f"{arm.name}_ee_pose"])
    sc = ST.detect_head(img, cam, bad=robot_bad(arm.ctx, cam), min_iou=0.7, base_candidates=0)
    best = None
    for pc in sc.pieces:
        if pc.cat != t.p.cat:
            continue
        d = math.hypot(pc.x - t.p.x, pc.y - t.p.y)
        if d < REMEASURE_MAX and (best is None or d < best[0]):
            best = d, pc
    return None if best is None else best[1]


def reached(arm: Arm, p) -> float:
    o = arm.ctx.obs
    q = np.asarray(o.state[f"{arm.name}_arm_joint_state"], dtype=float)
    return float(np.linalg.norm(K.point_of(K.fk(arm.name, q), K.TCP_OFFSET) - np.asarray(p)))


class Coord:
    def __init__(self, base: ST.Base):
        self.base = base
        self.token: str | None = None
        self.prio = "left"
        self.site = {"left": [LOOK_Q.copy()], "right": [LOOK_Q.copy()]}
        self.pending = {"left": None, "right": None}
        self.guest = {"left": False, "right": False}
        self.wants = {"left": False, "right": False}
        self.want_since = {"left": None, "right": None}
        self.placed = {k: 0 for k in range(4)}
        self.done = {"left": False, "right": False}
        self.far = None
        self.spots = [Spot(s) for s in SPOTS]
        self.takes = {"left": [], "right": []}
        self.can_place = {"left": [True] * 4, "right": [True] * 4}

    def pole_xy(self, k: int) -> np.ndarray:
        return ST.pole_world(self.base, k)[:2]

    def in_slab(self, p) -> bool:
        return abs(float(p[0]) - self.base.x) < XS

    def side_ok(self, arm: str, x: float) -> bool:
        s = 1.0 if arm == "left" else -1.0
        return s * (float(x) - self.base.x) <= XS - SIDE_GAP

    def free_spots(self, *arms):
        return [s for s in self.spots if s.owner is None and all(self.side_ok(a, s.xy[0]) for a in arms)]


class SlabArm(Arm):
    co: "Coord | None" = None
    keep = False

    def _may_step(self, q_next, guard: float = MIN_ARM_DIST) -> bool:
        co = self.co
        if co is not None and co.token != self.name:
            if co.in_slab(K.point_of(K.fk(self.name, q_next), K.TCP_OFFSET)):
                if co.token is not None or co.wants[other_arm(self.name)]:
                    return False
                co.token = self.name
        return super()._may_step(q_next, guard)

    def _after_step(self) -> None:
        co = self.co
        if co is None or co.token != self.name:
            return
        x = abs(float(self.tcp()[0]) - co.base.x)
        if x < XS:
            self.keep = False
        elif not self.keep and x >= XS + 0.01:
            hand_token(self, co)

    def owns_zone(self) -> bool:
        return self.co is not None and self.co.token == self.name


class GuestArm(SlabArm):
    def _may_step(self, q_next, guard: float = MIN_ARM_DIST) -> bool:
        co = self.co
        if co is not None and co.token != self.name and (not co.guest[self.name]):
            if co.in_slab(K.point_of(K.fk(self.name, q_next), K.TCP_OFFSET)) and (not co.in_slab(self.tcp())):
                if co.token is not None or co.wants[other_arm(self.name)]:
                    return False
                co.token = self.name
        return Arm._may_step(self, q_next, guard)


class FrontArm(Arm):
    co: "Coord | None" = None
    keep = False

    def _may_step(self, q_next, guard: float = MIN_ARM_DIST) -> bool:
        o = self.other
        if self.co is not None and o is not None:
            s = 1.0 if self.name == "left" else -1.0
            xo = float(o.tcp()[0])
            d_next = s * (xo - float(K.point_of(K.fk(self.name, q_next), K.TCP_OFFSET)[0]))
            if d_next < GAP_X and d_next < s * (xo - float(self.tcp()[0])) - 0.0001:
                return False
        return super()._may_step(q_next, guard)

    def owns_zone(self) -> bool:
        return self.co is not None and self.co.prio == self.name


def clear_between(arm: str, mine, theirs) -> float:
    if not mine or not theirs:
        return 9.0
    if arm == "left":
        return min(CO.arm_distance(a, b) for a in mine for b in theirs)
    return min(CO.arm_distance(b, a) for a in mine for b in theirs)


def arrived(arm: Arm, co: Coord):
    if COORD in ("site", "guest") and len(co.site[arm.name]) > 1:
        co.site[arm.name] = co.site[arm.name][:-1] + [arm.q.copy()]


def retreat(arm: Arm, co: Coord):
    o = other_arm(arm.name)
    tgt = park_q(arm)
    if tgt is not None and clear_between(arm.name, [tgt], co.site[o] + (co.pending[o] or [])) < SITE_CLEAR:
        tgt = None
    co.site[arm.name] = [tgt if tgt is not None else LOOK_Q.copy(), arm.q.copy()]
    if tgt is None:
        p = arm.tcp()
        if p[2] < Z_SAFE - 0.003:
            yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], Z_SAFE]), lin_step=0.014)
        yield from arm.joints_to(LOOK_Q)
        co.site[arm.name] = [LOOK_Q.copy()]
    else:
        yield from arm.transit(K.fk(arm.name, tgt) @ _tcp_shift(), Z_SAFE, seeds=[tgt])
        co.site[arm.name] = [arm.q.copy()]


def _tcp_shift() -> np.ndarray:
    t = np.eye(4)
    t[0, 3] = K.TCP_OFFSET
    return t


def park_goal(arm: Arm):
    sx = -1.0 if arm.name == "left" else 1.0
    return np.array([sx * PARK_X, PARK_Y, max(Z_SAFE, float(arm.tcp()[2]))])


def park_q(arm: Arm):
    goal = park_goal(arm)
    for d in (0.0, math.pi / 2, -math.pi / 2):
        r = K.axis_angle([0.0, 0.0, 1.0], d) @ arm.T[:3, :3]
        res = K.solve_near(arm.name, tcp_pose(r, goal), arm.q, K.TCP_OFFSET, n_random=3)
        if res.ok:
            return res.q


def park(arm: Arm):
    sx = -1.0 if arm.name == "left" else 1.0
    p = arm.tcp()
    goal = np.array([sx * PARK_X, PARK_Y, max(Z_SAFE, p[2])])
    if np.linalg.norm(p[:2] - goal[:2]) < 0.03:
        arm.status = "ok"
        return
    rot, qp = arm.T[:3, :3], None
    for d in (0.0, math.pi / 2, -math.pi / 2):
        r = K.axis_angle([0.0, 0.0, 1.0], d) @ arm.T[:3, :3]
        res = K.solve_near(arm.name, tcp_pose(r, goal), arm.q, K.TCP_OFFSET, n_random=3)
        if res.ok:
            rot, qp = r, res.q
            break
    yield from arm.transit(tcp_pose(rot, goal), Z_SAFE, seeds=[qp] if qp is not None else ())
    arrived(arm, arm.co_ref)


def is_far(arm: str, k: int, co: Coord) -> bool:
    return (co.pole_xy(k)[0] - co.base.x) * (1 if arm == "left" else -1) >= FAR_POLE


def parked(arm: Arm) -> bool:
    return bool(np.linalg.norm(arm.tcp()[:2] - park_goal(arm)[:2]) < 0.04)


def yield_to_far(arm: Arm, co: Coord):
    if co.far is None or co.far == arm.name:
        return
    release(arm, co)
    yield from park(arm)
    yield from arm.wait_until(lambda: co.far is None, max_steps=FAR_LOCK_WAIT + 100)


def far_lock(arm: Arm, co: Coord):
    co.far = arm.name
    o = arm.other
    if o is None:
        return
    yield from arm.wait_until(lambda: co.done[o.name] or parked(o) or (not o.active), max_steps=FAR_LOCK_WAIT)


def hand_token(arm: Arm, co: Coord):
    o = other_arm(arm.name)
    co.token = o if co.wants[o] else None


def acquire(arm: Arm, co: Coord, max_steps: int = 600):
    me = arm.name
    if co.token != me:
        co.wants[me] = True
        if co.want_since[me] is None:
            co.want_since[me] = int(arm.ctx.obs.step)
        n = 0
        while co.token not in (None, me) and n < max_steps:
            yield from arm.wait_until(lambda: co.token in (None, me), max_steps=20)
            n += 20
            o = arm.other
            if co.token not in (None, me) and o is not None and (o.waiting or co.far == o.name):
                yield from park(arm)
        co.wants[me] = False
        co.want_since[me] = None
        if co.token not in (None, me):
            return False
        co.token = me
    arm.keep = True
    return True


def release(arm: Arm, co: Coord):
    arm.keep = False
    if COORD in ("slab", "guest") and co.token == arm.name and (abs(float(arm.tcp()[0]) - co.base.x) >= XS):
        hand_token(arm, co)


def transit_to(arm: Arm, T_goal: np.ndarray, z_safe: float, seed, grip=None):
    yield from arm.transit(T_goal, z_safe, grip=grip, seeds=[seed] if seed is not None else (), joint_step=JOINT_STEP)
    if arm.status != "fail" or seed is None:
        return
    r = K.solve(arm.name, T_goal, seed, K.TCP_OFFSET)
    if not r.ok:
        return
    yield from arm.joints_to(r.q, grip=grip, joint_step=JOINT_STEP)
    if arm.status == "ok" and reached(arm, T_goal[:3, 3]) > 0.01:
        arm.status = "fail"


def retry_blocked(gen_fn, arm: Arm, co: Coord, tries: int = 4):
    for _ in range(tries):
        yield from gen_fn()
        if arm.status != "blocked":
            return
        if COORD in ("slab", "guest") and co.token != arm.name:
            yield from park(arm)
        elif COORD in ("front", "site") and co.prio != arm.name:
            co.prio = arm.name
            yield from park(arm)
        yield from arm.hold(6)
        arm.status = "blocked"


def pick(arm: Arm, t: PieceTask, co: Coord):
    best = plan_grasp(arm, t, co.base.yaw + math.pi / 2)
    if best is None:
        return False
    g, w, c, sign = best
    yield from yield_to_far(arm, co)
    t.width, t.c, t.tip_sign = w, c, sign
    g_open = min(1.0, D.grip_for_gap(w + OPEN_MARGIN))
    hover = tcp_pose(g.rot, [g.tcp[0], g.tcp[1], g.hover_z])
    if co.in_slab(g.tcp):
        ok = yield from acquire(arm, co)
        if not ok:
            return False
    yield from retry_blocked(lambda: transit_to(arm, hover, Z_SAFE, g.q_hover, grip=g_open), arm, co)
    if arm.status != "ok":
        return False
    arrived(arm, co)
    q_hover = np.array(arm.q, dtype=float)
    yield from arm.hold(1)
    m = wrist_remeasure(arm, t)
    if m is not None:
        t.p = m
    yaw_u = g.yaw if sign > 0 else g.yaw + math.pi
    tcp = np.array([t.p.x + c * math.cos(yaw_u), t.p.y + c * math.sin(yaw_u), K.TABLE_Z + GRASP_DZ])
    yield from arm.line_to(tcp_pose(g.rot, [tcp[0], tcp[1], tcp[2] + 0.03]), lin_step=FREE_STEP)
    yield from arm.line_to(tcp_pose(g.rot, tcp), lin_step=APPROACH_STEP)
    yield from arm.hold(2)
    err = reached(arm, tcp)
    if err > 0.008:
        yield from arm.line_to(hover, lin_step=0.012)
        return False
    yield from arm.set_grip(0.0, n=8)
    yield from arm.hold(2)
    held = D.held_width(observed_grip(arm), 0.0)
    lift = tcp_pose(g.rot, [tcp[0], tcp[1], Z_SAFE])
    yield from arm.line_to(lift, lin_step=FREE_STEP)
    for _ in range(3):
        if arm.status == "ok":
            break
        if arm.status == "blocked":
            yield from arm.hold(6)
            yield from arm.line_to(lift, lin_step=FREE_STEP)
            continue
        yield from arm.joints_to(q_hover, joint_step=JOINT_STEP)
    if arm.status != "ok":
        yield from arm.set_grip(g_open, n=6)
        return False
    if abs(held - w) > WIDTH_TOL:
        yield from arm.set_grip(g_open, n=6)
        return False
    held_up = D.held_width(observed_grip(arm), 0.0)
    if abs(held_up - w) > WIDTH_TOL:
        yield from arm.set_grip(g_open, n=6)
        return False
    t.grasp = g
    t.head = g.yaw
    if t.p.cat == ST.STAR:
        t.c = c + w / 2.0 - (held + WIDTH_READ_BIAS) / 2.0
    return True


def place_plan(arm: str, q, t: PieceTask, co: Coord):
    base = co.base
    pole = ST.pole_world(base, t.pole)
    z_align = pole[2] + ALIGN_DZ + GRASP_DZ
    best = None
    for head in (base.yaw + math.pi / 2, base.yaw - math.pi / 2):
        rot = G.grasp_rotation(head)
        yaw_u = head if t.tip_sign > 0 else head + math.pi
        xy = pole[:2] + t.c * np.array([math.cos(yaw_u), math.sin(yaw_u)])
        r = K.solve_near(arm, tcp_pose(rot, [xy[0], xy[1], z_align]), q, K.TCP_OFFSET)
        if not r.ok:
            continue
        z_cross = None
        for zc in (Z_SAFE,):
            if K.solve(arm, tcp_pose(rot, [xy[0], xy[1], zc]), r.q, K.TCP_OFFSET).ok:
                z_cross = zc
                break
        if z_cross is None:
            continue
        cost = float(np.sum(K.JOINT_COST_W * np.abs(r.q - q)))
        if best is None or cost < best[0]:
            best = cost, rot, xy, z_cross, head, r.q
    return None if best is None else best[1:]


def place(arm: Arm, t: PieceTask, co: Coord):
    base = co.base
    pole = ST.pole_world(base, t.pole)
    z_align = pole[2] + ALIGN_DZ + GRASP_DZ
    best = place_plan(arm.name, arm.q, t, co)
    if best is None:
        return "no_place"
    rot, xy, z_cross, head, q_al = best
    yield from yield_to_far(arm, co)
    far = is_far(arm.name, t.pole, co)
    ok = yield from acquire(arm, co)
    if not ok:
        return "fail"
    if far:
        yield from far_lock(arm, co)
    t.level = co.placed[t.pole]
    z_rel = K.TABLE_Z + ST.PLATE_TOP + t.level * ST.THICK + RELEASE_DZ + GRASP_DZ
    z_pre = min(z_align + 0.02, z_cross)
    yield from retry_blocked(lambda: transit_to(arm, tcp_pose(rot, [xy[0], xy[1], z_pre]), z_cross, q_al), arm, co)
    if arm.status != "ok":
        if co.far == arm.name:
            co.far = None
        return "fail"
    arrived(arm, co)
    t.head = head
    yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], z_align]), lin_step=ALIGN_STEP)
    z_thr = max(z_rel, pole[2] - THREAD_DZ + GRASP_DZ)
    yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], z_thr]), lin_step=INSERT_STEP)
    if z_thr > z_rel + 0.0001:
        yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], z_rel]), lin_step=SLIDE_STEP)
    dxy = DRAG * np.array([math.cos(base.yaw), math.sin(base.yaw)])
    xd, yd = xy[0] + dxy[0], xy[1] + dxy[1]
    yield from arm.line_to(tcp_pose(rot, [xd, yd, z_rel]), lin_step=DRAG_STEP)
    yield from arm.hold(DRAG_HOLD)
    yield from arm.set_grip(min(1.0, D.grip_for_gap(t.width + OPEN_MARGIN)), n=OPEN_STEPS)
    yield from arm.hold(1)
    co.placed[t.pole] += 1
    yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], Z_SAFE]), lin_step=FREE_STEP)
    release(arm, co)
    if co.far == arm.name:
        co.far = None
    return "ok"


def other_arm(name: str) -> str:
    return "right" if name == "left" else "left"


def put_down(arm: Arm, t: PieceTask, co: Coord, receiver: str | None):
    z_put = K.TABLE_Z + PUT_DZ + GRASP_DZ
    pole_xy = co.pole_xy(t.pole)
    users = (arm.name,) if receiver is None else (arm.name, receiver)
    yield from yield_to_far(arm, co)
    for s in sorted(co.free_spots(*users), key=lambda s: float(np.linalg.norm(s.xy - pole_xy))):
        for h in (t.head, t.head + math.pi):
            rot = G.grasp_rotation(h)
            yaw_u = h if t.tip_sign > 0 else h + math.pi
            xy = s.xy + t.c * np.array([math.cos(yaw_u), math.sin(yaw_u)])
            r_put = K.solve_near(arm.name, tcp_pose(rot, [xy[0], xy[1], z_put]), arm.q, K.TCP_OFFSET)
            if not r_put.ok:
                continue
            s.owner = t.peer
            ok = yield from acquire(arm, co)
            if not ok:
                s.owner = None
                return None
            yield from retry_blocked(
                lambda: transit_to(arm, tcp_pose(rot, [xy[0], xy[1], z_put + 0.03]), Z_SAFE, r_put.q), arm, co
            )
            if arm.status != "ok":
                s.owner = None
                release(arm, co)
                return None
            arrived(arm, co)
            yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], z_put]), lin_step=0.008)
            yield from arm.set_grip(min(1.0, D.grip_for_gap(t.width + OPEN_MARGIN)), n=4)
            yield from arm.hold(1)
            yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], Z_SAFE]), lin_step=0.014)
            release(arm, co)
            g_yaw = t.grasp.yaw if t.grasp is not None else t.head
            t.peer.p = ST.Piece(t.p.cat, float(s.xy[0]), float(s.xy[1]), ST.wrap(t.p.yaw + h - g_yaw), t.p.iou)
            t.peer.spot = s
            return s


def hand_over(arm: Arm, t: PieceTask, co: Coord, receiver: str | None):
    if t.peer is None:
        t.peer = PieceTask(t.p, t.pole, 0, kind="take")
        t.peer.peer = t
        if receiver is not None:
            co.takes[receiver].append(t.peer)
    users = (arm.name,) if receiver is None else (arm.name, receiver)
    s = None
    for _ in range(3):
        if not co.free_spots(*users):
            release(arm, co)
            yield from park(arm)
            yield from arm.wait_until(lambda: bool(co.free_spots(*users)), max_steps=SPOT_WAIT)
        s = yield from put_down(arm, t, co, receiver)
        if s is not None:
            break
        yield from arm.hold(6)
    if s is None:
        t.peer.state = "cancelled"
        release(arm, co)
        yield from park(arm)
        yield from arm.set_grip(1.0, n=4)
        return False
    t.peer.state = "ready" if receiver is not None else "cancelled"
    return True


def place_or_hand_over(arm: Arm, t: PieceTask, co: Coord):
    for _ in range(3):
        st = yield from place(arm, t, co)
        if st == "ok":
            return True
        if st == "no_place":
            break
        yield from arm.hold(6)
    release(arm, co)
    o = other_arm(arm.name)
    recv = o if co.can_place[o][t.pole] and (not co.done[o]) else None
    t.peer = None
    yield from hand_over(arm, t, co, recv)
    return False


def take_ready(arm: Arm, co: Coord):
    while True:
        ready = [t for t in co.takes[arm.name] if t.state == "ready"]
        if not ready:
            return
        t = ready[0]
        t.state = "taking"
        ok = False
        for _ in range(2):
            ok = yield from pick(arm, t, co)
            if ok:
                break
            release(arm, co)
        if not ok:
            continue
        if t.spot is not None:
            t.spot.owner = None
        yield from place_or_hand_over(arm, t, co)


def arm_job(arm: Arm, tasks, co: Coord):
    me, o = arm.name, other_arm(arm.name)
    todo = list(tasks)
    for rnd in range(2):
        left = []
        for t in todo:
            if t.kind != "give" or not co.free_spots(me, o):
                yield from take_ready(arm, co)
            ok = yield from pick(arm, t, co)
            if not ok:
                release(arm, co)
                left.append(t)
                continue
            if t.kind == "give":
                yield from hand_over(arm, t, co, o)
            else:
                yield from place_or_hand_over(arm, t, co)
        todo = left
        if not todo:
            break
    for t in todo:
        if t.kind == "give" and t.peer is not None:
            t.peer.state = "cancelled"
    while True:
        yield from take_ready(arm, co)
        pending = [t for t in co.takes[me] if t.state == "pending"]
        if not pending or co.done[o]:
            break
        yield from park(arm)
        yield from arm.wait_until(lambda: co.done[o] or any(t.state != "pending" for t in pending), max_steps=TAKE_WAIT)
        if all(t.state == "pending" for t in pending) and (not co.done[o]):
            break
    release(arm, co)
    yield from yield_to_far(arm, co)
    co.done[me] = True
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=JOINT_STEP)
    co.site[me] = [arm.q_home.copy()]


def reachable(arm: str, xy) -> bool:
    bx, by = K.BASE_POS[arm][:2]
    return math.hypot(xy[0] - bx, xy[1] - by) < MAX_REACH


def can_place_pole(arm: str, k: int, co: Coord) -> bool:
    t = PieceTask(ST.Piece(ST.SQUARE, 0.0, 0.0, 0.0, 1.0), k, 0)
    return place_plan(arm, G.base_seed(arm, co.pole_xy(k)), t, co) is not None


def route(pieces, pole_of_cat, co: Coord):
    head = co.base.yaw + math.pi / 2
    for arm in K.ARMS:
        co.can_place[arm] = [co.side_ok(arm, co.pole_xy(k)[0]) and can_place_pole(arm, k, co) for k in range(4)]
    opts, poles = [], []
    for p in pieces:
        k = pole_of_cat.get(p.cat, 0)
        pxy = co.pole_xy(k)
        pp = np.array([p.x, p.y])
        slab_pick = SLAB["pick"] if co.in_slab(pp) else 0.0
        safe, risky, pickers = [], [], {}
        for arm in K.ARMS:
            bxy = np.asarray(K.BASE_POS[arm][:2])
            if not (reachable(arm, pp) and co.side_ok(arm, p.x)):
                continue
            t = PieceTask(p, k, 0)
            g = plan_grasp(SimpleNamespace(name=arm, q=LOOK_Q), t, head)
            if g is None:
                continue
            reach = REACH_STEPS if float(np.linalg.norm(pp - bxy)) > REACH_SOFT else 0.0
            pickers[arm] = reach
            t.width, t.c, t.tip_sign = g[1], g[2], g[3]
            if not (co.can_place[arm][k] and place_plan(arm, g[0].q, t, co) is not None):
                continue
            side = (pxy[0] - co.base.x) * (1 if arm == "left" else -1)
            c = reach + CYCLE["direct"] + STEPS_PER_M * float(np.linalg.norm(pp - bxy) + np.linalg.norm(pp - pxy))
            cm = {arm: c, "slab": SLAB["place"] + slab_pick}
            if side >= FAR_POLE:
                cm[arm] += FAR_PENALTY
                risky.append((cm, ("direct", arm)))
            else:
                cm[arm] += CROSS_STEPS if side > 0 else 0.0
                safe.append((cm, ("direct", arm)))
        o = list(safe)
        if not safe:
            for giver, reach in pickers.items():
                recv = other_arm(giver)
                spots = [sp.xy for sp in co.spots if co.side_ok(giver, sp.xy[0]) and co.side_ok(recv, sp.xy[0])]
                if not co.can_place[recv][k] or not spots:
                    continue
                sxy = min(spots, key=lambda q: float(np.linalg.norm(q - pxy)))
                bg, br = np.asarray(K.BASE_POS[giver][:2]), np.asarray(K.BASE_POS[recv][:2])
                cg = reach + CYCLE["give"] + STEPS_PER_M * float(np.linalg.norm(pp - bg) + np.linalg.norm(pp - sxy))
                cr = CYCLE["take"] + STEPS_PER_M * float(np.linalg.norm(sxy - br) + np.linalg.norm(sxy - pxy))
                cs = slab_pick + SLAB["put"] + SLAB["pick"] + SLAB["place"]
                o.append(({giver: cg, recv: cr, "slab": cs}, ("give", giver, recv)))
            o += risky
        opts.append(o)
        poles.append(k)
    idx = [i for i, o in enumerate(opts) if o]
    sums = np.zeros((1, 3))
    choice = np.zeros((1, 0), dtype=int)
    for i in idx:
        oc = np.array([[cm.get("left", 0.0), cm.get("right", 0.0), cm.get("slab", 0.0)] for cm, _ in opts[i]])
        m = len(oc)
        sums = (sums[:, None, :] + oc[None, :, :]).reshape(-1, 3)
        choice = np.concatenate([np.repeat(choice, m, axis=0), np.tile(np.arange(m), len(choice))[:, None]], axis=1)
    b = int(np.argmin(sums.max(axis=1) + 0.2 * (sums[:, 0] + sums[:, 1])))
    jobs = {"left": [], "right": []}
    for n_i, i in enumerate(idx):
        _, act = opts[i][int(choice[b, n_i])]
        p, k = pieces[i], poles[i]
        if act[0] == "direct":
            jobs[act[1]].append(PieceTask(p, k, 0))
        else:
            tg, tr = PieceTask(p, k, 0, kind="give"), PieceTask(p, k, 0, kind="take")
            tg.peer, tr.peer = tr, tg
            jobs[act[1]].append(tg)
            co.takes[act[2]].append(tr)
    for arm in K.ARMS:
        bx, by = K.BASE_POS[arm][:2]
        jobs[arm].sort(key=lambda t: (t.kind != "give", is_far(arm, t.pole, co), math.hypot(t.p.x - bx, t.p.y - by)))
    unroutable = [pieces[i].as_dict() for i, o in enumerate(opts) if not o]
    return jobs, unroutable, [float(x) for x in sums[b]]


def tamp_levels(n: int) -> list[float]:
    p = K.TABLE_Z + ST.PLATE_TOP
    if n <= 2:
        return [p + GRASP_DZ]
    if n == 3:
        return [p + GRASP_DZ + ST.THICK / 4]
    return [p + GRASP_DZ, p + GRASP_DZ + 2 * ST.THICK]


def tamp_plan(arm: str, q, co: Coord, k: int):
    base = co.base
    xy = co.pole_xy(k)
    zs = tamp_levels(co.placed[k])
    best = None
    for head in (base.yaw + math.pi / 2, base.yaw - math.pi / 2):
        rot = G.grasp_rotation(head)
        r = K.solve_near(arm, tcp_pose(rot, [xy[0], xy[1], zs[0]]), q, K.TCP_OFFSET)
        if not r.ok:
            continue
        rh = K.solve(arm, tcp_pose(rot, [xy[0], xy[1], Z_SAFE]), r.q, K.TCP_OFFSET)
        if not rh.ok or not all(K.solve(arm, tcp_pose(rot, [xy[0], xy[1], z]), r.q, K.TCP_OFFSET).ok for z in zs[1:]):
            continue
        cost = float(np.sum(K.JOINT_COST_W * np.abs(rh.q - q)))
        if best is None or cost < best[0]:
            best = cost, rot, xy, zs, rh.q
    return None if best is None else best[1:]


def tamp_trips(co: Coord, cat_of_pole: dict, q: dict):
    risk = {k: (co.placed[k] - 1) * TAMP_RISK.get(cat_of_pole.get(k, -1), 0.0) for k in range(4)}
    ks = sorted((k for k in range(4) if risk[k] > 0), key=lambda k: -risk[k])
    trips: dict[str, list[int]] = {}
    for k in ks:
        own = "left" if k <= 1 else "right"
        for a in (own, other_arm(own)):
            if tamp_plan(a, q[a], co, k) is not None:
                trips.setdefault(a, []).append(k)
                break
    return list(trips.items())


def tamp_steps(co: Coord, ks) -> int:
    return TAMP_TRIP_STEPS + sum(TAMP_POLE_STEPS + 12 * (len(tamp_levels(co.placed[k])) - 1) for k in ks)


def tamp(arm: Arm, ks, co: Coord, cat_of_pole: dict):
    ok = yield from acquire(arm, co)
    if not ok:
        return
    for k in ks:
        plan = tamp_plan(arm.name, arm.q, co, k)
        if plan is None:
            continue
        rot, xy, zs, q_hov = plan
        w = closing_geometry(cat_of_pole[k], 0.0)[0]
        yield from transit_to(arm, tcp_pose(rot, [xy[0], xy[1], Z_SAFE]), Z_SAFE, q_hov, grip=1.0)
        if arm.status != "ok":
            break
        for z in zs:
            yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], z]), lin_step=ALIGN_STEP)
            yield from arm.set_grip(D.grip_for_gap(w - TAMP_SQUEEZE), n=TAMP_CLOSE)
            yield from arm.hold(TAMP_HOLD)
            yield from arm.set_grip(1.0, n=TAMP_OPEN)
        yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], Z_SAFE]), lin_step=FREE_STEP)
    release(arm, co)
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=JOINT_STEP)


def tamp_rounds(ctx, arms: dict, co: Coord, cat_of_pole: dict, rounds: int = TAMP_ROUNDS, end: int = TAMP_END):
    for rnd in range(rounds):
        trips = tamp_trips(co, cat_of_pole, {a: arms[a].q for a in K.ARMS})
        if not trips:
            return
        for name, ks in trips:
            ks = list(ks)
            while ks and int(ctx.obs.step) + tamp_steps(co, ks) > end:
                ks.pop()
            if not ks:
                continue
            yield from run_arms(arms, {name: tamp(arms[name], ks, co, cat_of_pole)})


@register
class StackingToy(Program):
    name = "stacking_toy"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "stacking toy" in s:
            return 0.95
        if "peg" in s and "piece" in s:
            return 0.6
        return 0.0

    def run(self, ctx):
        o = ctx.obs
        arm_cls = {"slab": SlabArm, "guest": GuestArm, "front": FrontArm}.get(COORD, Arm)
        arms = {
            a: arm_cls(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0]))
            for a in K.ARMS
        }
        yield from run_arms(arms, {a: arms[a].joints_to(LOOK_Q, force=True, joint_step=JOINT_STEP) for a in K.ARMS})
        yield from run_arms(arms, {a: arms[a].hold(3) for a in K.ARMS})
        img = ctx.obs.images.get("cam_head")
        cam = C.head()
        sc = ST.detect_head(img, cam, bad=robot_bad(ctx, cam))
        if sc.base is None or not sc.pieces:
            yield from run_arms(arms, {a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True) for a in K.ARMS})
            while True:
                yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
        co = Coord(sc.base)
        link_arms(arms)
        for a in K.ARMS:
            arms[a].co = co
            arms[a].co_ref = co
        jobs, _, _ = route(sc.pieces, sc.pole_of_cat, co)
        yield from run_arms(arms, {a: arm_job(arms[a], jobs[a], co) for a in K.ARMS})
        cat_of_pole = {k: c for c, k in sc.pole_of_cat.items()}
        yield from run_arms(arms, {a: arms[a].hold(TAMP_WAIT) for a in K.ARMS})
        yield from tamp_rounds(ctx, arms, co, cat_of_pole)
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
