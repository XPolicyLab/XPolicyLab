from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from ...perception import color_blocks as CB
from ...perception import robot_mask as RM
from ...planning import kinematics as K
from ...planning import motion
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import MIN_ARM_DIST, Arm, Zone, link_arms, run_arms, tcp_pose
from .. import Program, register

CLEAR = 0.02
HOVER_DZ = 0.045
RELEASE_GAP = 0.002
APPROACH_DZ = 0.025
OPEN_MARGIN = 0.024
ZONE_RADIUS = 0.2
EXIT_RADIUS = 0.15
WAIT_DX = 0.21, 0.26, 0.31
WAIT_CLEAR = 0.04
PICK_SEP = 0.22
BLOCKED_RETRIES = 4
BLOCKED_WAIT = 25
TOOL_RADIUS = 0.05
WIDTH_TOL = 0.004
REACH_TOL = 0.008
PARK_GUARD = 0.008
MIN_IOU = 0.85
FREE_RADIUS = 0.075
MOVE_PENALTY = 40.0
TILT_PENALTY = 2.0
CROSS_PENALTY = 60.0
CROSS_MARGIN = 0.03
MAX_ROUNDS = 3
STACK_TOL = 0.0175
VERIFY_IOU = 0.6
LYING_IOU = 0.8
LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
PREFER_Z = {"left": np.array([-1.0, 0.0]), "right": np.array([1.0, 0.0])}
TIME_W = 2.0
FREE_JOINT_STEP = 0.11
REGION = (-0.5, 0.5), (-0.34, 0.06)
_COLOR_WORDS = sorted(set(CB.COLOR_NAMES) | set(CB.SYNONYMS), key=len, reverse=True)
_COLOR_RE = re.compile(r"\b(" + "|".join(_COLOR_WORDS) + r")\b")
_ABOVE_RE = re.compile(r"\b(on top of|onto|on|above|over|atop)\b")
_BELOW_RE = re.compile(r"\b(under|underneath|below|beneath)\b")


def parse_order(instruction: str) -> list[str] | None:
    s = instruction.lower()
    if not re.search(r"\b(stack|stacking|pile|tower)\b", s) and (
        not (re.search(r"\b(put|place|set)\b", s) and _ABOVE_RE.search(s))
    ):
        return None
    if not re.search(r"\b(blocks?|cubes?|bricks?)\b", s):
        return None
    hits = [(m.start(), m.end(), CB.canonical(m.group(1))) for m in _COLOR_RE.finditer(s)]
    colors: list[str] = []
    for _a, _b, c in hits:
        if c is not None and c not in colors:
            colors.append(c)
    if len(colors) < 2:
        return None
    if "top to bottom" in s or "top to the bottom" in s:
        return colors[::-1]
    if "bottom to top" in s or "bottom to the top" in s or "order" in s:
        return colors
    below: dict[str, str] = {}
    for (a0, a1, ca), (b0, _b1, cb) in zip(hits, hits[1:]):
        between = s[a1:b0]
        if ca is None or cb is None or ca == cb or re.search("[.;]", between):
            continue
        if _ABOVE_RE.search(between):
            below[ca] = cb
        elif _BELOW_RE.search(between):
            below[cb] = ca
    if below:
        uppers = set(below)
        lowers = set(below.values())
        bottoms = [c for c in colors if c in lowers and c not in uppers]
        if len(bottoms) == 1:
            chain = [bottoms[0]]
            above = {v: k for k, v in below.items()}
            while chain[-1] in above and above[chain[-1]] not in chain:
                chain.append(above[chain[-1]])
            if len(chain) >= 2 and set(chain) == set(colors):
                return chain
    return colors


@dataclass
class Stack:
    order: list
    cubes: dict
    xy: np.ndarray
    size: float
    n: int = 0
    lying: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)
    place_q: dict = field(default_factory=dict)
    claims: dict = field(default_factory=dict)

    def top_z(self) -> float:
        return K.TABLE_Z + self.n * self.size

    def obstacles(self, exclude=None):
        out = []
        if self.n > 0:
            out.append((self.xy, self.top_z(), 0.75 * self.size))
        for c in self.lying:
            if c != exclude:
                cb = self.cubes[c]
                out.append((cb.xy, cb.top, 0.75 * self.size))
        return out

    def boxes(self, exclude=None, grow: float = 0.003):
        h = self.size / 2 * math.sqrt(2) + grow
        out = []
        if self.n > 0:
            out.append((self.xy[0], self.xy[1], h, K.TABLE_Z, self.top_z() + grow))
        for c in self.lying:
            if c != exclude:
                cb = self.cubes[c]
                out.append((cb.x, cb.y, h, K.TABLE_Z, cb.top + grow))
        return out

    def clear_z(self, p0, p1, below_tcp: float, exclude=None, skip_start: bool = False) -> float:
        a, b = np.asarray(p0, dtype=float)[:2], np.asarray(p1, dtype=float)[:2]
        ab = b - a
        L2 = float(ab @ ab)
        z = K.TABLE_Z
        for xy, top, r in self.obstacles(exclude):
            if skip_start and float(np.linalg.norm(xy - a)) < r + TOOL_RADIUS:
                continue
            t = 0.0 if L2 < 1e-12 else float(np.clip((xy - a) @ ab / L2, 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - xy))
            if d < r + TOOL_RADIUS:
                z = max(z, top)
        return z + CLEAR + below_tcp

    def exit_distance(self, p0, exclude=None) -> float:
        a = np.asarray(p0, dtype=float)[:2]
        d = 0.0
        for xy, _top, r in self.obstacles(exclude):
            dist = float(np.linalg.norm(xy - a))
            if dist < r + TOOL_RADIUS:
                d = max(d, r + TOOL_RADIUS - dist + 0.005)
        return d


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


class StackArm(Arm):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.exit_zone = False
        self.claim_exit = False
        self.stack = None

    def _after_step(self) -> None:
        z = self.zone
        if self.claim_exit and z is not None and (self.stack is not None):
            if _dist(self.tcp(), z.xy) > EXIT_RADIUS:
                self.stack.claims[self.name] = None
                self.claim_exit = False
        if self.exit_zone and z is not None:
            if z.owner != self.name:
                self.exit_zone = False
            elif _dist(self.tcp(), z.xy) > EXIT_RADIUS:
                z.free(self.name)
                self.exit_zone = False

    def leave_zone_later(self) -> None:
        if self.zone is not None and self.zone.owner == self.name:
            self.exit_zone = True

    def joints_to(
        self, q_goal, grip=None, joint_step: float = FREE_JOINT_STEP, force: bool = False, guard: float = MIN_ARM_DIST
    ):
        ok = yield from self.follow(joint_plan(self.q, q_goal, joint_step), grip, force=force, guard=guard)
        self.status = "ok" if ok else "blocked"

    def move_to(
        self,
        T_goal: np.ndarray,
        *,
        min_z: float,
        offset: float = K.TCP_OFFSET,
        grip=None,
        joint_step: float = FREE_JOINT_STEP,
        lin_step: float = 0.016,
        ang_step: float = math.radians(6),
        seeds=(),
    ):
        r = K.solve_near(self.name, T_goal, self.q, offset, extra_seeds=seeds, n_random=3)
        if r.ok:
            plan = joint_plan(self.q, r.q, joint_step)
            zs = [K.point_of(K.fk(self.name, q), offset)[2] for q in plan.q[:: max(1, len(plan.q) // 8)]]
            if min(zs + [10.0]) >= min_z:
                ok = yield from self.follow(plan, grip)
                self.status = "ok" if ok else "blocked"
                return
        yield from self.line_to(T_goal, offset, lin_step=lin_step, ang_step=ang_step, joint_step=joint_step, grip=grip)


JOINT_SPEED = np.array([1.0, 1.0, 1.0, 1.0, 1.3, 1.6])


def joint_plan(q0, q1, joint_step: float = FREE_JOINT_STEP) -> motion.Plan:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.max(np.abs(q1 - q0) / (joint_step * JOINT_SPEED)))
    n = max(1, math.ceil(d * math.pi / 2))
    t = np.arange(1, n + 1) / n
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return motion.Plan([q0 + (q1 - q0) * float(sk) for sk in s])


def reach_z(arm: Arm, rot: np.ndarray, xy, z_lo: float, z_hi: float, seeds=()) -> float | None:
    z = z_hi
    while True:
        if K.solve_near(
            arm.name, tcp_pose(rot, [xy[0], xy[1], z]), arm.q, K.TCP_OFFSET, extra_seeds=seeds, n_random=2
        ).ok:
            return z
        if z <= z_lo + 1e-06:
            return None
        z = max(z_lo, z - 0.01)


def rise(arm: Arm, z_target: float, grip=None):
    p = arm.tcp()
    arm.status = "ok"
    if p[2] >= z_target - 0.003:
        return
    z_up = reach_z(arm, arm.T[:3, :3], p, p[2], z_target)
    if z_up is not None and z_up > p[2] + 0.003:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], z_up]), lin_step=0.014, grip=grip)


def retreat(arm: Arm, st: Stack, below_tcp: float = G.FINGER_DEPTH):
    p = arm.tcp()
    z_need = st.clear_z(p, p, below_tcp)
    if p[2] < z_need - 0.003:
        z_up = reach_z(arm, arm.T[:3, :3], p, p[2], z_need)
        if z_up is not None and z_up > p[2] + 0.003:
            yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], z_up]), lin_step=0.012)
        p = arm.tcp()
    if p[2] < z_need - 0.003:
        d = st.exit_distance(p)
        if d > 0:
            yield from arm.set_grip(1.0, 4)
            zh = arm.T[:2, 2]
            if float(np.linalg.norm(zh)) < 0.3:
                zh = arm.T[:2, 0]
            u = zh / max(1e-09, float(np.linalg.norm(zh)))
            b = np.asarray(K.BASE_POS[arm.name][:2]) - p[:2]
            if float(u @ b) < 0:
                u = -u
            yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0] + u[0] * d, p[1] + u[1] * d, p[2]]), lin_step=0.01)


def travel(arm: Arm, st: Stack, T_goal: np.ndarray, below_tcp: float, *, exclude=None, grip=None, seeds=()):
    p0 = arm.tcp()
    pg = T_goal[:3, 3]
    rot_g = T_goal[:3, :3]
    z_need0 = st.clear_z(p0, p0, below_tcp, exclude=exclude)
    z_mid = st.clear_z(p0, pg, below_tcp, exclude=exclude, skip_start=True)
    d_exit = st.exit_distance(p0, exclude=exclude)
    span = _dist(p0, pg)
    sideways = d_exit > 0 and max(z_mid, pg[2]) < z_need0 - 0.003 and (span > 0.005)
    yield from rise(arm, z_need0 if sideways else max(z_need0, z_mid, pg[2]), grip)
    if arm.status != "ok":
        return
    p0 = arm.tcp()
    if sideways:
        u = (pg[:2] - p0[:2]) / span
        d = d_exit if span > d_exit + 0.01 else span
        T_exit = tcp_pose(arm.T[:3, :3], [p0[0] + u[0] * d, p0[1] + u[1] * d, p0[2]])
        if K.solve_near(arm.name, T_exit, arm.q, K.TCP_OFFSET, extra_seeds=seeds, n_random=2).ok:
            yield from arm.move_to(
                joint_step=FREE_JOINT_STEP, T_goal=T_exit, min_z=p0[2] - 0.005, grip=grip, seeds=seeds
            )
            if arm.status != "ok":
                return
            p0 = arm.tcp()
            z_mid = st.clear_z(p0, pg, below_tcp, exclude=exclude, skip_start=True)
            yield from rise(arm, max(z_mid, pg[2]), grip)
            if arm.status != "ok":
                return
            p0 = arm.tcp()
        else:
            z_mid = max(z_mid, z_need0)
    z_cross = max(z_mid, pg[2])
    d_goal = st.exit_distance(pg, exclude=exclude)
    span = _dist(p0, pg)
    if d_goal > 0 and span > d_goal + 0.01 and (p0[2] < z_cross - 0.003):
        u = (pg[:2] - p0[:2]) / span
        w = pg[:2] - u * d_goal
        z_w = reach_z(arm, rot_g, w, max(p0[2], pg[2]), z_cross, seeds)
        if z_w is not None and z_w > p0[2] + 0.003:
            yield from arm.move_to(
                joint_step=FREE_JOINT_STEP,
                T_goal=tcp_pose(rot_g, [w[0], w[1], z_w]),
                min_z=min(p0[2], z_w) - 0.005,
                grip=grip,
                seeds=seeds,
            )
            if arm.status != "ok":
                return
            p0 = arm.tcp()
    z_c = reach_z(arm, rot_g, pg, pg[2], z_cross, seeds)
    if z_c is None:
        arm.status = "fail"
        return
    yield from arm.move_to(
        joint_step=FREE_JOINT_STEP,
        T_goal=tcp_pose(rot_g, [pg[0], pg[1], z_c]),
        min_z=min(p0[2], z_c) - 0.005,
        grip=grip,
        seeds=seeds,
    )
    if arm.status != "ok":
        return
    if z_c > pg[2] + 0.002:
        yield from arm.line_to(T_goal, lin_step=0.012)


def reached(arm: Arm, p_target) -> float:
    q_obs = np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], dtype=float)
    return float(np.linalg.norm(K.point_of(K.fk(arm.name, q_obs), K.TCP_OFFSET) - np.asarray(p_target)))


def park(arm: Arm, st: Stack):
    below = arm.held["size"] / 2 + arm.held["dz"] if arm.held is not None else G.FINGER_DEPTH
    p = arm.tcp()
    yield from rise(arm, st.clear_z(p, p, below))
    yield from arm.joints_to(arm.q_home, force=True, guard=PARK_GUARD, joint_step=FREE_JOINT_STEP)


def wait_turn(arm: Arm, st: Stack, level: int):
    zone = arm.zone
    o = arm.other
    parked = False
    n = 0
    arm.waiting_zone = True
    while n < 400:
        o_claim = st.claims.get(o.name) if o is not None else None
        if st.n >= level and (zone is None or ((o_claim is None or zone.owner == arm.name) and zone.take(arm.name))):
            break
        if not parked and o is not None and o.waiting and (not o.waiting_zone):
            yield from park(arm, st)
            parked = True
            continue
        if o is None or not o.active:
            if st.n < level:
                break
        n += 1
        arm.waiting = True
        yield
    arm.waiting = arm.waiting_zone = False
    if st.n < level or (zone is not None and zone.owner != arm.name):
        return False
    return True


def free_zone(arm: Arm) -> None:
    if arm.zone is not None:
        arm.zone.free(arm.name)


def pick(arm: Arm, color: str, st: Stack, level: int | None = None):
    cube = st.cubes[color]
    s = st.size
    boxes = st.boxes(exclude=color)
    kw = dict(hover_dz=HOVER_DZ, prefer_z=PREFER_Z[arm.name], time_w=TIME_W)
    gr = G.grasp_candidates(arm.name, cube.center, s, cube.yaw, arm.q, boxes=boxes, **kw)
    if not gr:
        gr = G.grasp_candidates(arm.name, cube.center, s, cube.yaw, arm.q, **kw)
    if not gr:
        return "fail"
    g = gr[0]
    open_g = grip_for_gap(s + OPEN_MARGIN)
    z_hover = max(g.tcp[2] + HOVER_DZ, K.TABLE_Z + s + CLEAR + G.FINGER_DEPTH)
    z_hover = min(z_hover, g.hover_z)
    T_hover = tcp_pose(g.rot, [g.tcp[0], g.tcp[1], z_hover])
    zone = arm.zone
    in_zone = zone is not None and zone.inside(g.tcp)
    if in_zone:
        def other_ready() -> bool:
            o = arm.other
            return o is not None and o.active and (o.held is not None) and (o.held.get("level") == st.n)

        def can_start() -> bool:
            if other_ready() or zone.owner not in (None, arm.name):
                return False
            oc = st.claims.get(arm.other.name) if arm.other is not None else None
            return oc is None or _dist(oc, cube.xy) >= PICK_SEP

        n = 0
        staged = False
        arm.waiting_zone = True
        while n < 300:
            if can_start():
                st.claims[arm.name] = cube.xy.copy()
                arm.claim_exit = False
                break
            if not staged:
                staged = True
                yield from stage(arm, st)
                continue
            n += 1
            arm.waiting = True
            yield
        arm.waiting = arm.waiting_zone = False
        if st.claims.get(arm.name) is None:
            return "fail"
        cube = st.cubes[color]
    yield from travel(arm, st, T_hover, G.FINGER_DEPTH, exclude=color, grip=open_g, seeds=[g.q_hover])
    if arm.status != "ok":
        if in_zone:
            arm.claim_exit = True
        return "fail"
    yield from arm.line_to(tcp_pose(g.rot, g.tcp), lin_step=0.008)
    if arm.status != "ok":
        if in_zone:
            arm.claim_exit = True
        return "fail"
    yield from arm.hold(2)
    err = reached(arm, g.tcp)
    if err > REACH_TOL:
        yield from arm.line_to(T_hover, lin_step=0.012)
        if in_zone:
            arm.claim_exit = True
        return "obstructed"
    yield from arm.set_grip(0.0, 8)
    yield from arm.hold(2)
    g_obs = float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])
    w = held_width(g_obs)
    if abs(w - s) > WIDTH_TOL:
        yield from arm.set_grip(open_g, 5)
        yield from arm.line_to(T_hover, lin_step=0.012)
        if in_zone:
            arm.claim_exit = True
        return "empty"
    if color in st.lying:
        st.lying.remove(color)
    dz = g.tcp[2] - (cube.z0 + s / 2)
    arm.held = {"rot": g.rot, "size": s, "open": open_g, "color": color, "dz": dz, "xy": cube.xy.copy(), "level": level}
    z_up = max(z_hover, st.clear_z(g.tcp, g.tcp, s / 2 + dz))
    yield from arm.line_to(tcp_pose(g.rot, [g.tcp[0], g.tcp[1], z_up]), lin_step=0.012)
    if in_zone:
        arm.claim_exit = True
    return "ok"


def other_place_q(arm: Arm, st: Stack):
    o = arm.other
    if o is None:
        return []
    key = o.name, round(float(st.xy[0]), 3), round(float(st.xy[1]), 3), st.n
    cache = st.place_q
    if key not in cache:
        z = K.TABLE_Z + st.n * st.size + st.size / 2 + RELEASE_GAP + APPROACH_DZ
        qs = []
        for k in range(4):
            rot = G.grasp_rotation(k * math.pi / 4)
            r = K.solve_near(
                o.name,
                tcp_pose(rot, [st.xy[0], st.xy[1], z]),
                o.q_home,
                K.TCP_OFFSET,
                extra_seeds=[G.base_seed(o.name, st.xy)],
                n_random=1,
            )
            if r.ok:
                qs.append(r.q)
        cache[key] = qs
    return cache[key]


def stage(arm: Arm, st: Stack):
    from ...planning import collision as CO

    held = arm.held
    p = arm.tcp()
    side = -1.0 if arm.name == "left" else 1.0
    q_o = other_place_q(arm, st)

    def clear_of_other(q) -> bool:
        for qo in q_o:
            d = CO.arm_distance(q, qo) if arm.name == "left" else CO.arm_distance(qo, q)
            if d < WAIT_CLEAR:
                return False
        return True

    if side * (p[0] - st.xy[0]) >= WAIT_DX[0] - 0.01 and clear_of_other(arm.q):
        return
    below = held["size"] / 2 + held["dz"] if held is not None else G.FINGER_DEPTH
    rot = arm.T[:3, :3]
    for dx in WAIT_DX:
        for dy in (0.0, -0.06):
            w = np.array([st.xy[0] + side * dx, st.xy[1] + dy])
            z = max(p[2], st.clear_z(p, w, below, skip_start=True))
            r = K.solve_near(arm.name, tcp_pose(rot, [w[0], w[1], z]), arm.q, K.TCP_OFFSET, n_random=2)
            if not r.ok or not clear_of_other(r.q):
                continue
            yield from travel(arm, st, tcp_pose(rot, [w[0], w[1], z]), below)
            if arm.status == "ok":
                return
    yield from park(arm, st)
    arm.status = "ok"


def place(arm: Arm, st: Stack, level: int):
    held = arm.held
    s = held["size"]
    if st.n < level:
        yield from stage(arm, st)
    ok = yield from wait_turn(arm, st, level)
    if not ok:
        return False
    z_rel = K.TABLE_Z + level * s + s / 2 + RELEASE_GAP + held["dz"]
    p_rel = np.array([st.xy[0], st.xy[1], z_rel])
    below = s / 2 + held["dz"]
    z_app = z_rel + APPROACH_DZ
    rot, qs = G.place_rotation(
        arm.name, held["rot"], p_rel, arm.q, extra_z=(z_app - z_rel, 0.0), prefer_z=PREFER_Z[arm.name], time_w=TIME_W
    )
    if rot is None:
        free_zone(arm)
        return False
    yield from travel(arm, st, tcp_pose(rot, [p_rel[0], p_rel[1], z_app]), below, seeds=[qs[0]])
    if arm.status != "ok":
        free_zone(arm)
        return False
    w = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]))
    if abs(w - s) > WIDTH_TOL:
        arm.held = None
        yield from arm.set_grip(1.0, 3)
        arm.leave_zone_later()
        return False
    yield from arm.line_to(tcp_pose(rot, p_rel), lin_step=0.006)
    yield from arm.hold(1)
    yield from arm.set_grip(held["open"], 5)
    st.n = level + 1
    c = st.cubes[held["color"]]
    st.cubes[held["color"]] = CB.ColorCube(
        c.color, float(st.xy[0]), float(st.xy[1]), s, c.yaw, K.TABLE_Z + level * s, c.iou, c.area, c.uv
    )
    arm.held = None
    yield from retreat(arm, st)
    arm.leave_zone_later()
    return True


def put_down(arm: Arm, st: Stack):
    held = arm.held
    s = held["size"]
    xy = held["xy"]
    z_rel = K.TABLE_Z + s / 2 + RELEASE_GAP + held["dz"]
    yield from travel(arm, st, tcp_pose(held["rot"], [xy[0], xy[1], z_rel + APPROACH_DZ]), s / 2 + held["dz"])
    if arm.status == "ok":
        yield from arm.line_to(tcp_pose(held["rot"], [xy[0], xy[1], z_rel]), lin_step=0.006)
    yield from arm.set_grip(held["open"], 5)
    c = st.cubes[held["color"]]
    st.cubes[held["color"]] = CB.ColorCube(
        c.color, float(xy[0]), float(xy[1]), s, c.yaw, K.TABLE_Z, c.iou, c.area, c.uv
    )
    if held["color"] not in st.lying:
        st.lying.append(held["color"])
    arm.held = None
    yield from retreat(arm, st)


def go_home(arm: Arm, st: Stack):
    if arm.held is not None:
        yield from put_down(arm, st)
    yield from retreat(arm, st)
    plan = motion.joint_move(arm.q, arm.q_home, joint_step=FREE_JOINT_STEP)
    for q in plan.q[:: max(1, len(plan.q) // 10)]:
        p = K.point_of(K.fk(arm.name, q), K.TCP_OFFSET)
        if p[2] < st.clear_z(p, p, G.FINGER_DEPTH) - 0.005:
            p0 = arm.tcp()
            z = reach_z(arm, arm.T[:3, :3], p0, p0[2], p0[2] + 0.08)
            if z is not None and z > p0[2] + 0.005:
                yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p0[0], p0[1], z]), lin_step=0.014)
            break
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
    yield from arm.set_grip(1.0, 3)


def arm_task(arm: Arm, jobs: list, st: Stack):
    for color, level in jobs:
        res = "fail"
        attempt = blocked = 0
        while attempt < 2:
            res = yield from pick(arm, color, st, level)
            if res == "ok":
                break
            if res == "fail" and arm.status == "blocked" and (blocked < BLOCKED_RETRIES):
                blocked += 1
                yield from park(arm, st)
                target = st.cubes[color].xy
                n = 0
                while (
                    n < 2 * BLOCKED_WAIT
                    and arm.other is not None
                    and arm.other.active
                    and (_dist(arm.other.tcp(), target) < 0.2)
                ):
                    n += 1
                    yield
                yield from arm.hold(5)
                continue
            attempt += 1
            st.failed[color] = st.failed.get(color, 0) + 1
            if res == "empty":
                refresh(arm, st, color)
        if res != "ok":
            break
        ok = yield from place(arm, st, level)
        for _retry in range(2):
            if ok or arm.held is None:
                break
            yield from arm.hold(8)
            ok = yield from place(arm, st, level)
        if not ok:
            if arm.held is not None:
                yield from put_down(arm, st)
            break
    yield from go_home(arm, st)
    free_zone(arm)
    arm.exit_zone = False
    st.claims[arm.name] = None
    arm.claim_exit = False


def refresh(arm: Arm, st: Stack, color: str) -> None:
    o = arm.ctx.obs
    img = o.images.get("cam_head")
    if img is None:
        return
    arms = {arm.name: arm, arm.other.name: arm.other} if arm.other is not None else {arm.name: arm}
    bad = robot_bad(arms)
    got = CB.detect(img, C.head(), [color], region=REGION, bad=bad).get(color)
    if got is not None and got.iou >= 0.6:
        st.cubes[color] = got


def robot_bad(arms) -> np.ndarray:
    ql = arms["left"].q if "left" in arms else None
    qr = arms["right"].q if "right" in arms else None
    gl = arms["left"].g if "left" in arms else 1.0
    gr = arms["right"].g if "right" in arms else 1.0
    return RM.mask(C.head(), ql, qr, gl, gr, dilate=2)


def detect(ctx, arms, colors):
    img = ctx.obs.images.get("cam_head")
    if img is None:
        return {c: None for c in colors}, False
    bad = robot_bad(arms)
    cubes = CB.detect(img, C.head(), colors, region=REGION, bad=bad)
    near = np.zeros_like(bad)
    if bad.any():
        import cv2

        near = cv2.dilate(bad.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    clear = True
    for c, cb in cubes.items():
        if cb is None or cb.iou < MIN_IOU:
            clear = False
            continue
        uv = np.round(cb.uv).astype(int)
        if near[min(479, max(0, uv[1])), min(639, max(0, uv[0]))]:
            clear = False
    return cubes, clear


def can_place(arm: str, xy, zs, other=None, min_clear: float = 0.05) -> bool:
    from ...planning import collision as CO

    for k in range(6):
        rot = G.grasp_rotation(k * math.pi / 6)
        seed = G.base_seed(arm, xy)
        ok = True
        prev = None
        for z in zs:
            seeds = [s for s in (prev, seed, seed + np.array([0, 0, 0, 0, 0, k * math.pi / 6])) if s is not None]
            r = None
            for sd in seeds:
                r = K.solve(arm, tcp_pose(rot, [xy[0], xy[1], z]), sd, K.TCP_OFFSET, iters=80)
                if r.ok:
                    break
            if r is None or not r.ok:
                ok = False
                break
            if other is not None:
                d = CO.arm_distance(r.q, other) if arm == "left" else CO.arm_distance(other, r.q)
                if d < min_clear:
                    ok = False
                    break
            prev = r.q
        if ok:
            return True
    return False


def place_ok(arm: str, xy, level: int, s: float, q_rest: dict, cache: dict) -> bool:
    key = arm, round(float(xy[0]), 3), round(float(xy[1]), 3), level
    if key not in cache:
        zr = K.TABLE_Z + level * s + s / 2 + RELEASE_GAP
        z_ret = K.TABLE_Z + (level + 1) * s + CLEAR + G.FINGER_DEPTH
        o = "right" if arm == "left" else "left"
        cache[key] = can_place(arm, xy, (zr, zr + APPROACH_DZ, max(z_ret, zr + 0.01)), other=q_rest[o])
    return cache[key]


def free_spots(cubes) -> list[np.ndarray]:
    pts = [c.xy for c in cubes]
    grid = [np.array([x, y]) for x in np.arange(-0.12, 0.081, 0.02) for y in np.arange(-0.24, -0.079, 0.02)]
    apart = [p for p in grid if all(_dist(p, q) >= FREE_RADIUS for q in pts)]
    centre = np.array([-0.01, -0.18])
    return sorted(apart, key=lambda p: _dist(p, centre))


def plan_stack(order, cubes: dict, q_cur: dict, q_rest: dict, fixed=None):
    s = CB.CUBE_SIZE
    home_tcp = {a: K.point_of(K.fk(a, q_cur[a]), K.TCP_OFFSET) for a in K.ARMS}
    hb = s / 2 * math.sqrt(2) + 0.003
    tilt = {}

    def grasp_cost(color, a):
        if (color, a) not in tilt:
            c = cubes[color]
            boxes = [(o.x, o.y, hb, K.TABLE_Z, o.top + 0.003) for k, o in cubes.items() if k != color]
            o = "right" if a == "left" else "left"
            tilt[color, a] = None
            for penalty, bx in ((0.0, boxes), (20.0, ())):
                gr = G.grasp_candidates(
                    a,
                    c.center,
                    s,
                    c.yaw,
                    q_cur[a],
                    hover_dz=HOVER_DZ,
                    prefer_z=PREFER_Z[a],
                    other=(q_rest[o], 1.0),
                    min_clear=0.05,
                    boxes=bx,
                )
                if gr:
                    tilt[color, a] = abs(math.degrees(gr[0].tilt)) + penalty
                    break
        return tilt[color, a]

    if fixed is not None:
        options = [(np.asarray(fixed[0], dtype=float), fixed[1], list(fixed[2]))]
    else:
        base = cubes[order[0]]
        options = [(base.xy.copy(), 1, list(order[1:]))]
        for p in free_spots([cubes[c] for c in order])[:6]:
            options.append((p, 0, list(order)))
    t_pick, t_place, t_home, t_clear = 35, 30, 30, 10
    cache: dict = {}
    best = None
    for P, n0, carry in options:
        for assign in itertools.product(K.ARMS, repeat=len(carry)):
            ok = True
            cost_tilt = 0.0
            for k, (color, a) in enumerate(zip(carry, assign)):
                tc = grasp_cost(color, a)
                if tc is None or not place_ok(a, P, n0 + k, s, q_rest, cache):
                    ok = False
                    break
                cost_tilt += tc
            if not ok:
                continue
            t_arm = {a: 0.0 for a in K.ARMS}
            pos = {a: home_tcp[a] for a in K.ARMS}
            t_prev = 0.0
            for color, a in zip(carry, assign):
                c = cubes[color]
                t = t_arm[a] + _dist(pos[a], c.xy) / 0.012 + t_pick + _dist(c.xy, P) / 0.012
                t_done = max(t, t_prev + t_clear) + t_place
                t_prev = t_done
                t_arm[a] = t_done
                pos[a] = P
            used = set(assign)
            makespan = max(t_arm[a] + (t_home if a in used else 0.0) for a in K.ARMS)
            cost = makespan + MOVE_PENALTY * len(carry) + TILT_PENALTY * cost_tilt
            if len(used) == 2:
                for color, a in zip(carry, assign):
                    side = -1.0 if a == "left" else 1.0
                    if side * (cubes[color].x - P[0]) < -CROSS_MARGIN:
                        cost += CROSS_PENALTY
            if best is None or cost < best[0]:
                best = (
                    cost,
                    P,
                    n0,
                    {a: [(c, n0 + k) for k, (c, aa) in enumerate(zip(carry, assign)) if aa == a] for a in K.ARMS},
                    makespan,
                )
    if best is None:
        return None
    return best[1], best[2], best[3]


def look_pose(arms):
    yield from run_arms(arms, {a: arms[a].joints_to(LOOK_Q, force=True, joint_step=FREE_JOINT_STEP) for a in K.ARMS})
    yield from run_arms(arms, {a: arms[a].hold(2) for a in K.ARMS})


def survey(ctx, arms, st: Stack):
    img = ctx.obs.images.get("cam_head")
    if img is None:
        return None
    cam = C.head()
    hsv = CB.hsv_image(img)
    bad = robot_bad(arms)
    n_ok = 0
    prev = np.asarray(st.xy, dtype=float)
    for k, c in enumerate(st.order[: st.n]):
        f = CB.fit_at(img, cam, c, prev, k, others=[o for o in st.order if o != c], bad=bad, hsv=hsv)
        good = f is not None and f.iou >= VERIFY_IOU and (_dist(f.xy, prev) < (0.012 if k == 0 else 0.8 * STACK_TOL))
        if not good:
            break
        st.cubes[c] = f
        n_ok += 1
        prev = f.xy
    carry = list(st.order[n_ok:])
    found = CB.detect(img, cam, carry, region=REGION, bad=bad, hsv=hsv)
    if any(found[c] is None or found[c].iou < LYING_IOU for c in carry):
        return None
    for c in carry:
        st.cubes[c] = found[c]
    st.lying = list(carry)
    q = {a: arms[a].q for a in K.ARMS}
    q_rest = {a: arms[a].q_home for a in K.ARMS}
    if n_ok == 0:
        plan = plan_stack(st.order, st.cubes, q, q_rest)
        if plan is None:
            return None
        P, n0, jobs = plan
        st.xy = np.asarray(P, dtype=float)
        st.n = n0
        st.lying = list(st.order[n0:])
        return P, n0, jobs
    st.xy = prev.copy()
    st.n = n_ok
    return plan_stack(st.order, st.cubes, q, q_rest, fixed=(st.xy, n_ok, carry))


@register
class StackByLanguage(Program):
    name = "stack_in_order"

    def match(self, instruction: str, first_obs) -> float:
        order = parse_order(instruction)
        return 0.95 if order is not None and len(order) >= 2 else 0.0

    def run(self, ctx):
        order = parse_order(ctx.instruction)
        o = ctx.obs
        arms = {
            a: StackArm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0]))
            for a in K.ARMS
        }
        link_arms(arms)
        cubes, clear = detect(ctx, arms, order)
        if not clear:
            yield from look_pose(arms)
            cubes2, _ = detect(ctx, arms, order)
            for c in order:
                if cubes2.get(c) is not None and (cubes.get(c) is None or cubes2[c].iou >= cubes[c].iou - 0.02):
                    cubes[c] = cubes2[c]
        missing = [c for c in order if cubes.get(c) is None]
        if missing:
            yield from run_arms(
                arms,
                {
                    a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
                    for a in K.ARMS
                },
            )
            while True:
                yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
        q_rest = {a: arms[a].q_home for a in K.ARMS}
        plan = plan_stack(order, cubes, {a: arms[a].q for a in K.ARMS}, q_rest)
        st = None
        for rnd in range(MAX_ROUNDS):
            if rnd == 0:
                if plan is None:
                    break
                P, n0, jobs = plan
                st = Stack(
                    order=list(order),
                    cubes=dict(cubes),
                    xy=np.asarray(P, dtype=float),
                    size=CB.CUBE_SIZE,
                    n=n0,
                    lying=list(order[n0:]),
                )
            else:
                yield from run_arms(arms, {a: arms[a].hold(3) for a in K.ARMS})
                got = survey(ctx, arms, st)
                if got is None:
                    break
                P, n0, jobs = got
            link_arms(arms, Zone(st.xy, ZONE_RADIUS))
            st.claims = {}
            for a in K.ARMS:
                arms[a].stack = st
                arms[a].claim_exit = arms[a].exit_zone = False
            tasks = {a: arm_task(arms[a], jobs[a], st) for a in K.ARMS if jobs[a]}
            for a in K.ARMS:
                if a not in tasks:
                    tasks[a] = arms[a].joints_to(arms[a].q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
            yield from run_arms(arms, tasks)
        if st is None:
            yield from run_arms(
                arms,
                {
                    a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
                    for a in K.ARMS
                },
            )
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
