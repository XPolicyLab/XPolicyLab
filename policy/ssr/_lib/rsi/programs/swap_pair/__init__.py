from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from ...perception import prism_fit as PF
from ...perception import robot_mask as RM
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

MODEL = PF.T_BLOCK
GRASP_X = 0.005
GRASP_XS = 0.005, 0.012, -0.001, 0.018
FINGER_W = 0.022
FINGER_T = 0.016
FINGER_MARGIN = 0.004
STEM_W = 0.02
OPEN_GAP = 0.036
G_OPEN = grip_for_gap(OPEN_GAP)
Z_TOP = K.TABLE_Z + MODEL.height
Z_GRASP = K.TABLE_Z + G.FINGER_DEPTH + 0.003
Z_PLACE = Z_GRASP + 0.0005
Z_HOVER = Z_TOP + G.FINGER_DEPTH + 0.025
Z_CARRY = Z_GRASP + 0.045
Z_WAIT = Z_HOVER + 0.04
REACH_TOL = 0.008
WIDTH_TOL = 0.004
ARM_CLEAR = 0.035
JOINT_SPEED = np.array([0.1, 0.1, 0.1, 0.12, 0.14, 0.16])
HOME_SPEED = np.array([0.13, 0.13, 0.13, 0.15, 0.17, 0.19])
CARRY_JOINT_STEP = 0.13
PERCEIVE_TRIES = 4
WRIST_YAW_SPAN = math.radians(12.0)
WRIST_MAX_SHIFT = 0.012
WRIST_MAX_TURN = math.radians(6.0)
WRIST_MIN_IOU = 0.9


@dataclass
class Block:
    colour: str
    x: float
    y: float
    yaw: float
    iou: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def point(self, mx: float, my: float = 0.0) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.x + c * mx - s * my, self.y + s * mx + c * my])

    def as_dict(self) -> dict:
        return {
            "colour": self.colour,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw_deg": round(math.degrees(self.yaw), 2),
            "iou": round(self.iou, 3),
        }


@dataclass
class ArmPlan:
    arm: str
    block: Block
    target: Block
    k: int
    gx: float
    p_grasp: np.ndarray
    rot_grasp: np.ndarray
    q_grasp: np.ndarray
    q_hover: np.ndarray
    p_place: np.ndarray
    rot_place: np.ndarray
    q_place: np.ndarray
    q_place_hover: np.ndarray
    cost: float

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "block": self.block.as_dict(),
            "target": self.target.as_dict(),
            "k": self.k,
            "p_grasp": np.round(self.p_grasp, 4),
            "p_place": np.round(self.p_place, 4),
            "q_grasp": np.round(self.q_grasp, 3),
            "q_place": np.round(self.q_place, 3),
            "cost": round(self.cost, 2),
        }


@dataclass
class Shared:
    plans: dict
    first: str
    second: str
    q_park: np.ndarray | None = None
    p_park: np.ndarray | None = None
    q_wait: np.ndarray | None = None
    state: dict = field(default_factory=dict)


def _sat_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def finger_boxes(block: Block, gx: float, margin: float = FINGER_MARGIN) -> list[np.ndarray]:
    x0, x1 = gx - FINGER_W / 2 - margin, gx + FINGER_W / 2 + margin
    out = []
    for sgn in (1.0, -1.0):
        y0, y1 = sgn * (OPEN_GAP / 2 - 0.001), sgn * (OPEN_GAP / 2 + FINGER_T + margin)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        out.append(np.array([block.point(px, py) for px, py in pts]))
    return out


def fingers_clear(block: Block, other: Block, gx: float) -> bool:
    boxes = MODEL.footprint(other.x, other.y, other.yaw)
    return not any(_sat_overlap(f, b) for f in finger_boxes(block, gx) for b in boxes)


def choose_grasp_x(a: Block, b: Block) -> float | None:
    for gx in GRASP_XS:
        if fingers_clear(a, b, gx) and fingers_clear(b, a, gx):
            return gx


def closing_yaw(yaw: float, k: int) -> float:
    return yaw + math.pi / 2 + k * math.pi


def wrist_for(q1: float, yaw: float) -> float:
    return math.remainder(q1 - yaw - math.pi, 2 * math.pi)


def joint_seeds(arm: str, xy, yaw: float, extra=()) -> list[np.ndarray]:
    s0 = G.base_seed(arm, xy)
    seeds = [np.asarray(e, dtype=float) for e in extra]
    w = wrist_for(s0[0], yaw)
    for d in (0.0, 2 * math.pi, -2 * math.pi):
        s = s0.copy()
        s[5] = w + d
        if K.JOINT_LIMITS[5, 0] <= s[5] <= K.JOINT_LIMITS[5, 1]:
            seeds.append(s)
    seeds.append(s0)
    return seeds


def ik_all(arm: str, T: np.ndarray, seeds) -> list[np.ndarray]:
    out = []
    for s in seeds:
        r = K.solve(arm, T, s, K.TCP_OFFSET, iters=100)
        if r.ok and (not any(float(np.max(np.abs(r.q - o))) < 0.05 for o in out)):
            out.append(r.q)
    return out


def ik_near(arm: str, T: np.ndarray, q_ref) -> np.ndarray | None:
    r = K.solve(arm, T, q_ref, K.TCP_OFFSET, iters=100)
    return r.q if r.ok and float(np.max(np.abs(r.q - q_ref))) < 1.0 else None


def move_time(q0, q1) -> float:
    d = np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))
    return float(np.max(d / JOINT_SPEED)) * math.pi / 2


def plan_arm(arm: str, block: Block, target: Block, q_home, gx: float = GRASP_X) -> list[ArmPlan]:
    out = []
    for k in (0, 1):
        gy = closing_yaw(block.yaw, k)
        rot_g = K.top_down(gy)
        pg_xy = block.point(gx)
        p_grasp = np.array([pg_xy[0], pg_xy[1], Z_GRASP])
        qs_g = ik_all(arm, tcp_pose(rot_g, p_grasp), joint_seeds(arm, pg_xy, gy, [q_home]))
        ty = closing_yaw(target.yaw, k)
        rot_p = K.top_down(ty)
        pp_xy = target.point(gx)
        p_place = np.array([pp_xy[0], pp_xy[1], Z_PLACE])
        for qg in qs_g:
            qh = ik_near(arm, tcp_pose(rot_g, [pg_xy[0], pg_xy[1], Z_HOVER]), qg)
            qc = ik_near(arm, tcp_pose(rot_g, [pg_xy[0], pg_xy[1], Z_CARRY]), qg)
            if qh is None or qc is None:
                continue
            dyaw = math.remainder(target.yaw - block.yaw, 2 * math.pi)
            seeds = []
            for d in (dyaw, dyaw - 2 * math.pi, dyaw + 2 * math.pi):
                s = qg.copy()
                s[0] = G.base_seed(arm, pp_xy)[0]
                s[5] = qg[5] + (s[0] - qg[0]) - d
                if K.JOINT_LIMITS[5, 0] <= s[5] <= K.JOINT_LIMITS[5, 1]:
                    seeds.append(s)
            for qp in ik_all(arm, tcp_pose(rot_p, p_place), seeds):
                qph = ik_near(arm, tcp_pose(rot_p, [pp_xy[0], pp_xy[1], Z_HOVER]), qp)
                qpc = ik_near(arm, tcp_pose(rot_p, [pp_xy[0], pp_xy[1], Z_CARRY]), qp)
                if qph is None or qpc is None:
                    continue
                cost = move_time(q_home, qh) + move_time(qc, qpc) + move_time(qph, q_home)
                out.append(ArmPlan(arm, block, target, k, gx, p_grasp, rot_g, qg, qh, p_place, rot_p, qp, qph, cost))
    out.sort(key=lambda p: p.cost)
    return out


def regrasp(plan: ArmPlan, block: Block) -> bool:
    gy = closing_yaw(block.yaw, plan.k)
    rot_g = K.top_down(gy)
    pg_xy = block.point(plan.gx)
    p_grasp = np.array([pg_xy[0], pg_xy[1], Z_GRASP])
    qg = ik_near(plan.arm, tcp_pose(rot_g, p_grasp), plan.q_grasp)
    if qg is None:
        return False
    plan.block, plan.p_grasp, plan.rot_grasp, plan.q_grasp = block, p_grasp, rot_g, qg
    return True


def retarget(plan: ArmPlan, target: Block) -> bool:
    ty = closing_yaw(target.yaw, plan.k)
    rot_p = K.top_down(ty)
    pp_xy = target.point(plan.gx)
    p_place = np.array([pp_xy[0], pp_xy[1], Z_PLACE])
    qp = ik_near(plan.arm, tcp_pose(rot_p, p_place), plan.q_place)
    if qp is None:
        return False
    qph = ik_near(plan.arm, tcp_pose(rot_p, [pp_xy[0], pp_xy[1], Z_HOVER]), qp)
    if qph is None:
        return False
    plan.target, plan.p_place, plan.rot_place, plan.q_place, plan.q_place_hover = target, p_place, rot_p, qp, qph
    return True


def arm_dist(arm: str, q, q_other) -> float:
    if arm == "left":
        return CO.arm_distance(q, q_other, G_OPEN, G_OPEN)
    return CO.arm_distance(q_other, q, G_OPEN, G_OPEN)


def choose_park(sh: Shared) -> bool:
    f, s = sh.first, sh.second
    pf, ps = sh.plans[f], sh.plans[s]
    side = 1.0 if f == "right" else -1.0
    rot = pf.rot_place
    best = None
    for dx in (0.16, 0.19, 0.22, 0.25):
        for dy in (0.0, -0.05, -0.1):
            for z in (Z_CARRY + 0.02, Z_CARRY + 0.06):
                p = np.array([pf.p_grasp[0] + side * dx, pf.p_grasp[1] + dy, z])
                for r in (rot, pf.rot_grasp):
                    seeds = [pf.q_place, pf.q_grasp, G.base_seed(f, p[:2])]
                    q = None
                    for sd in seeds:
                        rr = K.solve(f, tcp_pose(r, p), sd, K.TCP_OFFSET, iters=100)
                        if rr.ok:
                            q = rr.q
                            break
                    if q is None:
                        continue
                    clear = min(arm_dist(f, q, ps.q_place), arm_dist(f, q, ps.q_place_hover))
                    score = clear - 0.3 * dx - (0.05 if r is not rot else 0.0)
                    if clear >= ARM_CLEAR and (best is None or score > best[0]):
                        best = score, q, p, clear
                    break
    if best is None:
        return False
    sh.q_park, sh.p_park = best[1], best[2]
    return True


def choose_wait(sh: Shared) -> None:
    f, s = sh.first, sh.second
    pf, ps = sh.plans[f], sh.plans[s]
    side = -1.0 if f == "right" else 1.0
    for dx in (0.1, 0.13, 0.16):
        p = np.array([ps.p_grasp[0] + side * dx, ps.p_grasp[1] - 0.02, Z_WAIT])
        q = ik_near(s, tcp_pose(ps.rot_grasp, p), ps.q_hover)
        if q is None:
            continue
        if min(arm_dist(s, q, pf.q_grasp), arm_dist(s, q, pf.q_hover)) >= ARM_CLEAR:
            sh.q_wait = q
            return
    sh.q_wait = None


def g_obs(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


def q_obs(arm: Arm) -> np.ndarray:
    return np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], dtype=float)


def reached(arm: Arm, p) -> float:
    return float(np.linalg.norm(K.point_of(K.fk(arm.name, q_obs(arm)), K.TCP_OFFSET) - np.asarray(p)))


def joints_to(arm: Arm, q_goal, grip=None, force: bool = True, speed=JOINT_SPEED):
    q0 = arm.q.copy()
    q1 = np.asarray(q_goal, dtype=float)
    n = max(1, math.ceil(float(np.max(np.abs(q1 - q0) / speed)) * math.pi / 2))
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
    plan = motion.Plan([q0 + (q1 - q0) * float(v) for v in s])
    ok = yield from arm.follow(plan, grip, force=force)
    arm.status = "ok" if ok else "blocked"
    return ok


def line(
    arm: Arm,
    rot,
    p,
    lin_step: float = 0.012,
    ang_step: float = math.radians(8.0),
    joint_step: float = CARRY_JOINT_STEP,
    grip=None,
):
    yield from arm.line_to(tcp_pose(rot, p), lin_step=lin_step, ang_step=ang_step, joint_step=joint_step, grip=grip)
    return arm.status == "ok"


def wrist_measure(
    arm: Arm,
    block: Block,
    max_shift: float = WRIST_MAX_SHIFT,
    max_turn: float = WRIST_MAX_TURN,
    yaw_span: float = WRIST_YAW_SPAN,
    from_centroid: bool = False,
) -> Block | None:
    o = arm.ctx.obs
    img = o.images.get(f"cam_{arm.name}_wrist")
    if img is None:
        return None
    cam = C.wrist(o.state[f"{arm.name}_ee_pose"])
    qs = {a: o.state[f"{a}_arm_joint_state"] for a in K.ARMS}
    gs = {a: float(o.state[f"{a}_ee_joint_state"][0]) for a in K.ARMS}
    robot = RM.mask(cam, qs["left"], qs["right"], gs["left"], gs["right"], dilate=3)
    best = None
    for col, reg in PF.colour_regions(img, cam, robot, region=np.ones(img.shape[:2], bool), colours=[block.colour]):
        f = PF.fit(
            cam,
            reg,
            MODEL,
            init=None if from_centroid else block.xy,
            yaw_init=block.yaw,
            yaw_span=yaw_span,
            pad=20,
            ss_fine=2,
            ignore=robot,
        )
        if f is not None and (best is None or f.iou > best.iou):
            best = f
    if best is None:
        return None
    shift = float(np.linalg.norm(best.xy - block.xy))
    turn = abs(math.remainder(best.yaw - block.yaw, 2 * math.pi))
    ok = best.iou >= WRIST_MIN_IOU and shift <= max_shift and (turn <= max_turn)
    if not ok:
        return None
    return Block(block.colour, best.x, best.y, best.yaw, best.iou)


def look_and_update(arm: Arm, plan: ArmPlan, other: ArmPlan | None, retry: bool = False):
    yield from arm.hold(2)
    if retry:
        b = wrist_measure(
            arm,
            plan.block,
            max_shift=0.03,
            max_turn=math.radians(25.0),
            yaw_span=math.radians(30.0),
            from_centroid=True,
        )
    else:
        b = wrist_measure(arm, plan.block)
    if b is None:
        return
    regrasp(plan, b)
    if other is not None:
        retarget(other, b)


GRASP_TRIES = 2


def fetch(arm: Arm, plan: ArmPlan, other: ArmPlan | None):
    for attempt in range(GRASP_TRIES):
        yield from look_and_update(arm, plan, other if attempt == 0 else None, retry=attempt > 0)
        held = yield from grasp(arm, plan)
        if held:
            return True
    return False


def grasp(arm: Arm, plan: ArmPlan):
    pg = plan.p_grasp
    p = arm.tcp()
    if float(np.linalg.norm(p[:2] - pg[:2])) > 0.003 and p[2] > Z_HOVER - 0.01:
        yield from line(arm, plan.rot_grasp, [pg[0], pg[1], max(p[2], Z_HOVER)], lin_step=0.01, grip=G_OPEN)
    ok = yield from line(arm, plan.rot_grasp, pg, lin_step=0.01, grip=G_OPEN)
    yield from arm.hold(1)
    err = reached(arm, pg)
    if not ok or err > REACH_TOL:
        yield from line(arm, plan.rot_grasp, [pg[0], pg[1], Z_HOVER])
        return False
    yield from arm.set_grip(0.0, 8)
    yield from arm.hold(2)
    w = held_width(g_obs(arm))
    if abs(w - STEM_W) > WIDTH_TOL:
        yield from arm.set_grip(G_OPEN, 6)
        yield from line(arm, plan.rot_grasp, [pg[0], pg[1], Z_HOVER])
        return False
    yield from line(arm, plan.rot_grasp, [pg[0], pg[1], Z_CARRY])
    return True


def put_down(arm: Arm, plan: ArmPlan):
    pp = plan.p_place
    yield from line(arm, plan.rot_place, [pp[0], pp[1], Z_CARRY], lin_step=0.014)
    yield from line(arm, plan.rot_place, pp, lin_step=0.008)
    yield from arm.hold(2)
    yield from arm.set_grip(G_OPEN, 6)
    yield from arm.hold(1)
    yield from line(arm, plan.rot_place, [pp[0], pp[1], Z_HOVER])
    return True


def go_home(arm: Arm):
    p = arm.tcp()
    if p[2] < Z_HOVER - 0.003:
        yield from line(arm, arm.T[:3, :3], [p[0], p[1], Z_HOVER])
    yield from joints_to(arm, arm.q_home, grip=1.0, speed=HOME_SPEED)
    yield from arm.set_grip(1.0, 2)


FIX_ROUNDS = 2


def fix_task(arm: Arm, plan: ArmPlan):
    yield from joints_to(arm, plan.q_hover, grip=G_OPEN)
    held = yield from fetch(arm, plan, None)
    if held:
        yield from put_down(arm, plan)
    yield from go_home(arm)


def set_aside(arm: Arm, plan: ArmPlan, sh: Shared):
    rot = arm.T[:3, :3]
    p = arm.tcp()
    q = ik_near(arm.name, tcp_pose(rot, [p[0], p[1], Z_PLACE]), arm.q)
    if q is None:
        yield from line(arm, plan.rot_grasp, [plan.p_grasp[0], plan.p_grasp[1], Z_CARRY], lin_step=0.02)
        p = plan.p_grasp
        rot = plan.rot_grasp
    yield from line(arm, rot, [p[0], p[1], Z_PLACE], lin_step=0.01)
    yield from arm.hold(2)
    yield from arm.set_grip(G_OPEN, 6)
    yield from arm.hold(1)
    sh.state[arm.name] = "leaving"
    yield from go_home(arm)
    sh.state[arm.name] = "home"


def first_task(arm: Arm, sh: Shared):
    plan = sh.plans[arm.name]
    other = arm.other
    sh.state[arm.name] = "approach"
    yield from joints_to(arm, plan.q_hover, grip=G_OPEN)
    held = yield from fetch(arm, plan, sh.plans[other.name])
    if not held:
        sh.state[arm.name] = "failed"
        yield from go_home(arm)
        return
    sh.state[arm.name] = "parking"
    yield from joints_to(arm, sh.q_park)
    sh.state[arm.name] = "parked"
    pp = plan.p_place
    goal = tcp_pose(plan.rot_place, [pp[0], pp[1], Z_CARRY])
    path = {}

    def other_clear() -> bool:
        st = sh.state.get(other.name)
        if st in ("home", "failed"):
            return True
        if st != "leaving":
            return False
        if "q" not in path:
            pl = motion.line(
                arm.name,
                arm.q,
                goal,
                K.TCP_OFFSET,
                lin_step=0.02,
                ang_step=math.radians(8.0),
                joint_step=CARRY_JOINT_STEP,
            )
            path["q"] = pl.q[::3] if pl.ok else []
        if not path["q"]:
            return False
        return min(arm_dist(arm.name, q, other.q) for q in path["q"]) >= ARM_CLEAR

    yield from arm.wait_until(other_clear)
    if sh.state.get(other.name) == "failed":
        yield from set_aside(arm, plan, sh)
        return
    ok = yield from line(arm, plan.rot_place, [pp[0], pp[1], Z_CARRY], lin_step=0.02)
    if not ok:
        q = ik_near(arm.name, tcp_pose(plan.rot_place, [pp[0], pp[1], Z_CARRY]), plan.q_place)
        if q is not None:
            yield from joints_to(arm, q)
    yield from put_down(arm, plan)
    sh.state[arm.name] = "leaving"
    yield from go_home(arm)
    sh.state[arm.name] = "home"


def second_task(arm: Arm, sh: Shared):
    plan = sh.plans[arm.name]
    other = arm.other
    sh.state[arm.name] = "waiting"
    if sh.q_wait is not None:
        yield from joints_to(arm, sh.q_wait, grip=G_OPEN)

    def clear() -> bool:
        if sh.state.get(other.name) == "failed":
            return True
        if sh.state.get(other.name) not in ("parking", "parked"):
            return False
        return min(arm_dist(arm.name, plan.q_hover, other.q), arm_dist(arm.name, plan.q_grasp, other.q)) >= ARM_CLEAR

    yield from arm.wait_until(clear)
    if sh.state.get(other.name) == "failed":
        sh.state[arm.name] = "home"
        yield from go_home(arm)
        return
    sh.state[arm.name] = "approach"
    yield from joints_to(arm, plan.q_hover, grip=G_OPEN)
    held = yield from fetch(arm, plan, sh.plans[other.name])
    if not held:
        sh.state[arm.name] = "failed"
        yield from go_home(arm)
        return
    yield from arm.wait_until(lambda: sh.state.get(other.name) in ("parked", "failed", "home"))
    yield from put_down(arm, plan)
    sh.state[arm.name] = "leaving"
    yield from go_home(arm)
    sh.state[arm.name] = "home"


def perceive(ctx, arms):
    o = ctx.obs
    img = o.images.get("cam_head")
    if img is None:
        return []
    cam = C.head()
    robot = RM.mask(cam, arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g, dilate=2)
    dets = PF.detect(img, cam, MODEL, robot=robot, n=2)
    return [Block(c, f.x, f.y, f.yaw, f.iou) for c, f in dets]


def make_plan(arms, blocks) -> Shared | None:
    a, b = sorted(blocks, key=lambda bl: bl.x)
    q_home = {n: arms[n].q_home for n in K.ARMS}
    gx = choose_grasp_x(a, b)
    if gx is None:
        gx = GRASP_X
    cands = {"left": plan_arm("left", a, b, q_home["left"], gx), "right": plan_arm("right", b, a, q_home["right"], gx)}
    if not cands["left"] or not cands["right"]:
        return None
    best = None
    for first in ("right", "left"):
        second = "left" if first == "right" else "right"
        for pf in cands[first][:3]:
            for ps in cands[second][:3]:
                sh = Shared(plans={first: pf, second: ps}, first=first, second=second)
                if not choose_park(sh):
                    continue
                choose_wait(sh)
                cost = pf.cost + ps.cost + (0.0 if sh.q_wait is not None else 20.0)
                if best is None or cost < best[0]:
                    best = cost, sh
    if best is None:
        return None
    sh = best[1]
    return sh


@register
class SwapPair(Program):
    name = "swap_pair"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "swap" in s and ("t-shaped" in s or "t shaped" in s or "t-block" in s):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        sh = None
        for attempt in range(PERCEIVE_TRIES):
            blocks = perceive(ctx, arms)
            if len(blocks) == 2:
                sh = make_plan(arms, blocks)
            if sh is not None or attempt == PERCEIVE_TRIES - 1:
                break
            yield from run_arms(arms, {a: arms[a].hold(5) for a in K.ARMS})
        if sh is not None:
            tasks = {sh.first: first_task(arms[sh.first], sh), sh.second: second_task(arms[sh.second], sh)}
            yield from run_arms(arms, tasks)
        yield from run_arms(arms, {a: joints_to(arms[a], arms[a].q_home, grip=1.0) for a in K.ARMS})
        for _ in range(FIX_ROUNDS):
            if sh is None:
                break
            yield from run_arms(arms, {a: arms[a].hold(3) for a in K.ARMS})
            fix = self.plan_fix(ctx, arms, sh)
            if fix is None:
                break
            yield from run_arms(arms, {fix.arm: fix_task(arms[fix.arm], fix)})
            yield from run_arms(
                arms, {a: joints_to(arms[a], arms[a].q_home, grip=1.0, speed=HOME_SPEED) for a in K.ARMS}
            )
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})

    def plan_fix(self, ctx, arms, sh: Shared) -> ArmPlan | None:
        blocks = perceive(ctx, arms)
        worst = None
        for name, plan in sh.plans.items():
            cur = [b for b in blocks if b.colour == plan.block.colour]
            if not cur:
                continue
            b, t = cur[0], plan.target
            e_pos = float(np.linalg.norm(b.xy - t.xy))
            e_yaw = abs(math.remainder(b.yaw - t.yaw, 2 * math.pi))
            score = e_pos / 0.02 + e_yaw / math.radians(3.0)
            free = all(float(np.linalg.norm(o.xy - t.xy)) > 0.05 for o in blocks if o is not b)
            if free and (worst is None or score > worst[0]):
                worst = score, b, t, plan.gx
        if worst is None:
            return None
        _, b, t, gx = worst
        order = ("left", "right") if b.x < 0 else ("right", "left")
        for name in order:
            cands = plan_arm(name, b, t, arms[name].q_home, gx)
            if cands:
                return cands[0]
