from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from . import lang_sort as LS
from ...perception import seg as SEGM
from . import wrist_obj as WO
from ...planning import collision as CO
from ...planning import kinematics as K
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, Zone, link_arms, run_arms, tcp_pose
from .. import Program, register

SEG_BLOCK_S = 20.0
SEG_WAIT_MAX = 500
_SEG_LOAD: dict = {}
SURVEY_CALL_S = 40.0
SURVEY_REGION = (-0.5, 0.5), (-0.46, 0.03)
SURVEY_DEFER_MAX = 20
_SURVEY_CALL = {"step": -1, "spent": 0.0}


def survey_turn(step: int) -> bool:
    if _SURVEY_CALL["step"] != step:
        _SURVEY_CALL.update(step=step, spent=0.0)
    return _SURVEY_CALL["spent"] < SURVEY_CALL_S


def survey_spent(step: int, seconds: float) -> None:
    if _SURVEY_CALL["step"] == step:
        _SURVEY_CALL["spent"] += float(seconds)


DEDUP_R = 0.015


def dedup_items(items: list) -> tuple[list, list]:
    order = sorted(items, key=lambda it: -float(getattr(it.region, "area", 0) or 0))
    kept, dropped = [], []
    for it in order:
        if any(
            k.category == it.category and float(np.linalg.norm(np.asarray(k.xy) - np.asarray(it.xy))) < DEDUP_R
            for k in kept
        ):
            dropped.append(int(it.id))
        else:
            kept.append(it)
    keep = {id(k) for k in kept}
    return [it for it in items if id(it) in keep], dropped


WRIST_YAW_CATS = "car", "chocolate_bar"
WRIST_MIN_ELONG = 1.4
WRIST_ELONG = {"car": 1.25, "wooden_toy": 1.1}
WRIST_TOY_MIN_L = 0.055
WRIST_MAX_CUT = 0.1
WRIST_WIDE_SHIFT = 0.08
REAIM_MIN_SHIFT = 0.004
REAIM_MIN_TURN = math.radians(8)
LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
RIM_Z = K.TABLE_Z + 0.077
FLOOR_Z = K.TABLE_Z + 0.0095
OVER_RIM = 0.03
RELEASE_DZ = 0.015
EMPTY_Z = K.TABLE_Z + 0.11
CROSS_MIN_Z = K.TABLE_Z + 0.095
CROSS_DZ = 0.01
VIA_BACK = 0.03, 0.06, 0.09, 0.12
OWN_SIDE_X = 0.12
OWN_BASKET = {"left": 0, "right": 2}
MID_WAIT = 150
PLACE_RETREAT = 2
HOLD_XY_MIRROR = {"left": (-0.32, -0.22), "right": (0.32, -0.22)}
HOLD_CLEAR = 0.16
OPEN_MARGIN = 0.024
WALL_NEAR = 0.06
WALL_T = 0.012
WALL_MARGIN = 0.005
RELEASE_OPEN = 0.012
MIN_HELD = 0.004
HOVER_DZ = 0.07
SLOT_DX = 0.07
LONG_AXIS_TOL = math.radians(30)
ZONE_XY, ZONE_R = (0.0, -0.2), 0.13
RELAY_SPOTS = {"left": (0.05, -0.29), "right": (-0.05, -0.29)}
RELAY_SLOT_CENTER = {"left": ((0.06, -0.39), (0.06, -0.325)), "right": ((-0.06, -0.39), (-0.06, -0.325))}
RELAY_LONG_YAW = 0.0
RELAY_ZONE_R = 0.12
SLOT_WAIT = 300
HELD_STEP = 0.07
HELD_STEPS = {"car": 0.05, "wooden_toy": 0.05, "pepper": 0.06, "watch": 0.1, "chocolate_bar": 0.1}
NARROW_CAR = 0.042, 0.049
NARROW_CAR_STEP = 0.04
CAR_STEP = 0.05
DESCENT_STEP_FAST = 0.07
RELEASE_STEPS = 4
RELEASE_HOLD = 1
RELEASE_RISE_DZ = 0.025
RELEASE_RISE_STEP = 0.009
FINGERTIP_DZ = 0.013
RELEASE_CLEAR_MARGIN = 0.005
RELEASE_CLEAR_STEP = 0.012
RELEASE_DZ_CAT: dict = {"pepper": 0.006}
RELEASE_OPEN_STEPS: dict = {"pepper": 8}
RELEASE_SETTLE: dict = {}
SUPPORT_DZ = {"chocolate_bar": -0.002, "car": -0.002}
DIRECT_MAX_TILTS = {"chocolate_bar": math.radians(1), "car": math.radians(1)}
PICK_CANDIDATES = 3
BRANCH_MAX = 0.6
FREE_STEP = 0.15
DIRECT_MAX_TILT = math.radians(31)
MAX_ROUNDS = 4
LAST_ROUND_STEP = 950
LOW_GRASP_Z = {"wooden_toy": 0.018}
BASKET_HALF = 0.126, 0.084
BASKET_MARGIN = 0.02
LIFT_BACK = 0.03, 0.06, 0.09, 0.12, 0.15
LANDING_STEPS = 3.0
FRONT_DYS = 0.04, 0.08, 0.12
BACK_OFF = 0
RECOVER_R = 0.2
RECOVER_STEP = 880
RECOVER_SLOT_R = 0.06
JOB_DEADLINE = 990
MOVED_R = 0.03
MAX_REQUEUE = 2
BLOCK_PAUSE = 12
PUT_DOWN_TRIES = 4
RELOCATE_R = 0.09
RELOCATE_CLEAR = 0.16
OCCLUDED = 0.25
PLACE_TILTS = 0.0, math.radians(15), math.radians(30), math.radians(40), math.radians(50)
GRASP_Z = {"watch": 0.016, "car": 0.019, "wooden_toy": 0.03, "pepper": 0.03, "chocolate_bar": 0.014}
ROUND = "watch", "pepper"
CAR_LOW_Z = 0.014
CAR_ONE_Z = 0.015


def over_rim() -> float:
    return OVER_RIM


def grasp_z(ob) -> float:
    gz = GRASP_Z.get(ob.cat, 0.02)
    if ob.cat == "car" and CAR_ONE_Z is not None:
        gz = CAR_ONE_Z
    elif ob.cat == "car":
        gz = min(gz, max(CAR_LOW_Z, ob.dims[2] / 2 + 0.002))
    if getattr(ob, "low", False) and ob.cat in LOW_GRASP_Z:
        gz = min(gz, LOW_GRASP_Z[ob.cat])
    return gz


def tilt_drop(ob, tilt: float) -> float:
    t = abs(float(tilt))
    if t < 1e-06:
        return 0.0
    return max(0.0, 0.5 * float(ob.dims[0]) * math.sin(t) - grasp_z(ob) * (1.0 - math.cos(t)))


def held_points(ob, rot, tcp) -> np.ndarray:
    L, W, h = (float(v) for v in ob.dims)
    c = np.asarray(ob.xy, float)[:2]
    if ob.cat in ROUND:
        a = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        ring = c + 0.5 * max(L, W) * np.stack([np.cos(a), np.sin(a)], axis=1)
    else:
        u = np.array([math.cos(ob.yaw), math.sin(ob.yaw)])
        v = np.array([-u[1], u[0]])
        ring = np.array([c + sx * 0.5 * L * u + sy * 0.5 * W * v for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
    n = len(ring)
    pts = np.concatenate([np.c_[ring, np.full(n, K.TABLE_Z)], np.c_[ring, np.full(n, K.TABLE_Z + h)]])
    return (pts - np.asarray(tcp, float)[:3]) @ np.asarray(rot, float)


def held_top(arm, ob, rot) -> float:
    pts = (getattr(arm, "held", None) or {}).get("pts")
    if pts is None:
        return float(ob.dims[2]) - grasp_z(ob)
    return float(np.max((np.asarray(rot, float) @ pts.T)[2]))


def release_rise(arm, ob, rot) -> tuple[float, float]:
    rise, lin = RELEASE_RISE_DZ, RELEASE_RISE_STEP
    need = held_top(arm, ob, rot) + FINGERTIP_DZ + RELEASE_CLEAR_MARGIN
    if need > rise:
        rise, lin = need, RELEASE_CLEAR_STEP
    return rise, lin


def release_lift(arm, ob, rot, tilt: float) -> float:
    pts = (getattr(arm, "held", None) or {}).get("pts")
    if pts is None:
        return tilt_drop(ob, tilt)
    low = -float(np.min((np.asarray(rot, float) @ pts.T)[2]))
    return max(0.0, low - grasp_z(ob))


_SLOT_WORDS = {"left": 0, "middle": 1, "center": 1, "centre": 1, "right": 2}
_CLAUSE = re.compile(
    r"(?:put\s+|and\s+|,\s*)?([a-z][a-z _-]*?)\s+objects?\s+(?:into|in|to|onto)\s+the\s+(left|middle|center|centre|right)\s+basket"
)


def canonical_category(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def parse_instruction(instruction: str) -> list[str] | None:
    s = " ".join(str(instruction or "").lower().split())
    found: dict[int, str] = {}
    for m in _CLAUSE.finditer(s):
        cat = m.group(1).strip()
        cat = re.sub(r"^(?:put|and|the|all)\s+", "", cat).strip()
        slot = _SLOT_WORDS[m.group(2)]
        if cat and slot not in found:
            found[slot] = canonical_category(cat)
    if sorted(found) != [0, 1, 2] or len(set(found.values())) != 3:
        return None
    return [found[0], found[1], found[2]]


@dataclass
class Obj:
    id: int
    cat: str
    xy: np.ndarray
    yaw: float
    dims: tuple
    basket: int
    state: str = "table"
    slot: tuple | None = None
    tries: int = 0
    low: bool = False
    touched: bool = False
    recovered: bool = False
    baskets: list | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "cat": self.cat,
            "xy": [round(float(v), 4) for v in self.xy],
            "yaw": round(math.degrees(self.yaw), 1),
            "dims": [round(v, 3) for v in self.dims],
            "basket": self.basket,
            "state": self.state,
            "low": self.low,
        }


@dataclass
class Job:
    obj: Obj
    kind: str
    slot: tuple | None = None
    log: dict = field(default_factory=dict)


@register
class ClassifyByLanguage(Program):
    name = "classify_by_language"

    def match(self, instruction, first_obs) -> float:
        return 0.95 if parse_instruction(instruction) else 0.0

    def run(self, ctx):
        o = ctx.first
        cats = parse_instruction(ctx.instruction)
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        for a in arms.values():
            a.back_off_steps = BACK_OFF
            a.yield_empty_first = True
        zone = Zone(ZONE_XY, ZONE_R)
        zone.relay = {a: Zone(tuple(np.mean(np.asarray(relay_slots(a)), axis=0)), RELAY_ZONE_R) for a in K.ARMS}
        zr = Zone((0.0, -0.36), RELAY_ZONE_R)
        zone.relay = {a: zr for a in K.ARMS}
        zone.relay_occ = {a: [None] * len(relay_slots(a)) for a in K.ARMS}
        zone.cross = Zone((0.0, -0.34), 0.2)
        used: dict[int, list] = {0: [], 1: [], 2: []}
        placed = {c: 0 for c in cats}
        baskets = None
        still: list = []
        seg_preload()
        for rnd in range(MAX_ROUNDS):
            yield from look_pose(arms)
            n_wait = 0
            while n_wait < SEG_WAIT_MAX and seg_loading():
                n_wait += 1
                yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            n_def = 0
            while n_def < SURVEY_DEFER_MAX and (not survey_turn(ctx.obs.step)):
                n_def += 1
                yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            t_survey, step_survey = time.perf_counter(), ctx.obs.step
            img = ctx.obs.images["cam_head"]
            rng = (
                LS.COUNT_RANGE
                if rnd == 0
                else (np.zeros(3), np.array([max(0, LS.COUNT_RANGE[1] - placed[c]) for c in cats], float))
            )
            insts = head_instances(img)
            baskets, items = LS.detect(
                img,
                C.head(),
                cats,
                LOOK_Q,
                LOOK_Q,
                1.0,
                1.0,
                baskets=baskets,
                count_range=rng,
                instances=insts,
                region=SURVEY_REGION,
            )
            items, _ = dedup_items(items)
            objs = []
            for it in items:
                xy, yaw, dims = LS.base_pose(it)
                b = cats.index(it.category)
                ob = Obj(it.id, it.category, xy, yaw, dims, b)
                ob.baskets = baskets
                if rnd and ob.cat in LOW_GRASP_Z:
                    d, low = min(
                        ((float(np.linalg.norm(ob.xy - p)), lw) for p, lw in still),
                        default=(1.0, True),
                        key=lambda t: t[0],
                    )
                    ob.low = low if d <= MOVED_R else True
                objs.append(ob)
            if not objs:
                survey_spent(step_survey, time.perf_counter() - t_survey)
                break
            zone.relay_occ = {
                a: [
                    next((ob.id for ob in objs if float(np.linalg.norm(ob.xy - np.asarray(xy))) < 0.05), None)
                    for xy in relay_slots(a)
                ]
                for a in K.ARMS
            }
            plan = plan_jobs(objs, baskets, arms, used)
            zone.objs = objs
            zone.used = used
            survey_spent(step_survey, time.perf_counter() - t_survey)
            yield from run_arms(arms, {a: arm_task(arms[a], plan[a], zone, baskets, cats) for a in K.ARMS})
            for ob in objs:
                placed[ob.cat] += int(ob.state == "basket")
            still = [(np.asarray(ob.xy, float), ob.low) for ob in objs if not ob.touched and ob.state != "basket"]
            if ctx.obs.step > LAST_ROUND_STEP:
                break
        yield from run_arms(arms, {a: go_home(arms[a]) for a in K.ARMS})
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})


def seg_preload() -> float | None:
    if "thread" in _SEG_LOAD:
        return None
    t0 = time.perf_counter()
    import torch

    dt = time.perf_counter() - t0
    _SEG_LOAD["block"] = SEG_BLOCK_S
    _SEG_LOAD["thread"] = SEGM.preload()
    return dt


def seg_loading() -> bool:
    t = _SEG_LOAD.get("thread")
    if t is None or not t.is_alive():
        return False
    import time

    t0 = time.perf_counter()
    t.join(timeout=max(0.0, _SEG_LOAD["block"]))
    _SEG_LOAD["block"] -= time.perf_counter() - t0
    return t.is_alive()


def head_instances(img):
    insts = SEGM.segment(img, camera="head")
    return insts


def look_pose(arms):
    yield from run_arms(
        arms, {a: arms[a].joints_to(LOOK_Q, grip=1.0, force=True, joint_step=FREE_STEP) for a in K.ARMS}
    )
    yield from run_arms(arms, {a: arms[a].hold(2) for a in K.ARMS})
    for _ in range(30):
        lag = max(float(np.max(np.abs(obs_q(arms[a]) - arms[a].q))) for a in K.ARMS)
        if lag < 0.02:
            break
        yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})


def go_home(arm: Arm):
    yield from rise(arm)
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=FREE_STEP)


def obs_q(arm: Arm) -> np.ndarray:
    q = arm.ctx.obs.state.get(f"{arm.name}_arm_joint_state")
    return arm.q.copy() if q is None else np.asarray(q, dtype=float)


def placing_arm(ob: Obj) -> str:
    if ob.basket == 0:
        return "left"
    if ob.basket == 2:
        return "right"
    return "left" if ob.xy[0] < 0 else "right"


def wrist_heading_ok(cat: str, m) -> bool:
    if cat == "wooden_toy":
        return m.L >= WRIST_TOY_MIN_L and m.L >= WRIST_ELONG[cat] * m.W
    return cat in WRIST_YAW_CATS and m.L >= WRIST_ELONG.get(cat, WRIST_MIN_ELONG) * m.W


def other(a: str) -> str:
    return "right" if a == "left" else "left"


def relay_slots(putter: str) -> tuple:
    return RELAY_SLOT_CENTER[putter]


def hold_xy(name: str) -> tuple:
    return HOLD_XY_MIRROR[name]


def grasp_options(arm: str, ob: Obj, q_cur, xy=None) -> list:
    xy = ob.xy if xy is None else np.asarray(xy, float)
    L, W, _ = ob.dims
    z = K.TABLE_Z + grasp_z(ob)
    yaw_close = ob.yaw + math.pi / 2
    blocked = () if ob.cat in ROUND or L < 1.3 * W else (ob.yaw,)
    kw = dict(
        hover_dz=HOVER_DZ,
        tilts=G.TILTS + (math.radians(50),),
        blocked_yaws=blocked,
        z_support=K.TABLE_Z + SUPPORT_DZ.get(ob.cat, 0.0),
    )
    center = np.array([xy[0], xy[1], z])
    gr = G.grasp_candidates(arm, center, max(W, 0.02), yaw_close, q_cur, **kw)
    walls = basket_walls(xy, ob.baskets)
    if not walls or all(wall_clear(arm, g, walls, max(W, 0.02)) for g in gr[:2]):
        return gr
    size = max(L, W, 0.02) if ob.cat in ROUND else max(W, 0.02)
    gw = G.grasp_candidates(
        arm, center, size, 0.0 if ob.cat in ROUND else yaw_close, q_cur, boxes=walls, box_margin=WALL_MARGIN, **kw
    )
    for g in gw:
        if ob.cat in ROUND:
            g.wall_size = size
    return gw or gr


def basket_walls(xy, baskets) -> tuple:
    if not baskets:
        return ()
    out = []
    hb = WALL_T / 2
    for bk in baskets:
        sx, sy = float(bk.size[0]) / 2, float(bk.size[1]) / 2
        dx, dy = max(0.0, abs(xy[0] - bk.x) - sx), max(0.0, abs(xy[1] - bk.y) - sy)
        if math.hypot(dx, dy) > WALL_NEAR:
            continue
        n = int(math.ceil((2 * sx - WALL_T) / WALL_T))
        for i in range(n + 1):
            out.append((bk.x - sx + hb + i * (2 * sx - WALL_T) / n, bk.y - sy + hb, hb, K.TABLE_Z, RIM_Z))
        for side in (-1.0, 1.0):
            for j in range(1, 5):
                out.append((bk.x + side * (sx - hb), bk.y - sy + hb + j * WALL_T, hb, K.TABLE_Z, RIM_Z))
    return tuple(out)


def wall_clear(arm: str, g, walls, size: float) -> bool:
    open_g = min(1.0, ((size + OPEN_MARGIN) / 2 - 0.0004 + 0.01) / 0.054)
    return all(CO.box_clearance(arm, q, walls, open_g) >= WALL_MARGIN for q in (g.q, g.q_hover))


def held_step(arm: Arm, ob: Obj) -> float:
    w = (arm.held or {}).get("width")
    if ob.cat == "car":
        return CAR_STEP
    if ob.cat == "car" and w is not None and (NARROW_CAR[0] <= w < NARROW_CAR[1]):
        return NARROW_CAR_STEP
    return HELD_STEPS.get(ob.cat, HELD_STEP)


def slots_for(b: int, baskets, used: dict, arm: str) -> tuple:
    bk = baskets[b]
    offs = [-SLOT_DX, 0.0, SLOT_DX]
    if b == 1 and arm == "right":
        offs = offs[::-1]
    for dx in offs + [-SLOT_DX / 2, SLOT_DX / 2]:
        p = bk.x + dx, bk.y - 0.005
        if all(abs(p[0] - q[0]) > 0.03 for q in used[b]):
            used[b].append(p)
            return p
    p = bk.x, bk.y
    used[b].append(p)
    return p


def plan_jobs(objs: list[Obj], baskets, arms, used) -> dict:
    jobs = {"left": [], "right": []}
    relays_in = {"left": [], "right": []}
    load = {"left": 0.0, "right": 0.0}
    side = [ob for ob in objs if ob.basket != 1]
    middle = [ob for ob in objs if ob.basket == 1]
    for ob in sorted(side, key=lambda o: abs(o.xy[0])):
        a = placing_arm(ob)
        slot = slots_for(ob.basket, baskets, used, a)
        own = grasp_options(a, ob, arms[a].q_home)
        alt = grasp_options(other(a), ob, arms[other(a)].q_home)
        lim = DIRECT_MAX_TILTS.get(ob.cat, DIRECT_MAX_TILT)
        direct = bool(own) and (abs(own[0].tilt) <= lim or not alt or abs(own[0].tilt) <= abs(alt[0].tilt))
        if direct:
            jobs[a].append(Job(ob, "place", slot))
            load[a] += 1.0
        elif alt:
            jobs[other(a)].insert(0, Job(ob, "relay"))
            relays_in[a].append(Job(ob, "fetch", slot))
            load[other(a)] += 0.7
            load[a] += 0.8
        else:
            jobs[a].append(Job(ob, "place", slot))
            load[a] += 1.0
    for ob in sorted(middle, key=lambda o: -abs(o.xy[0])):
        a = placing_arm(ob)
        b = other(a)
        if load[b] + 1.0 <= load[a]:
            alt = grasp_options(b, ob, arms[b].q_home)
            if alt and abs(alt[0].tilt) <= DIRECT_MAX_TILTS.get(ob.cat, DIRECT_MAX_TILT):
                a = b
        jobs[a].append(Job(ob, "place", slots_for(1, baskets, used, a)))
        load[a] += 1.0
    for a in K.ARMS:
        jobs[a].sort(key=lambda j: (j.kind != "relay", in_zone(j.obj.xy)))
        jobs[a].extend(relays_in[a])
    return jobs


def in_zone(xy) -> bool:
    return float(np.hypot(xy[0] - ZONE_XY[0], xy[1] - ZONE_XY[1])) < ZONE_R


def own_side(name: str, xy) -> bool:
    return float(xy[0]) * (-1.0 if name == "left" else 1.0) > OWN_SIDE_X


def zone_busy(zone: Zone, name: str) -> bool:
    return zone.owner not in (None, name) or getattr(zone, "queue", None) not in (None, name)


def relay_zone(zone, putter: str):
    rz = getattr(zone, "relay", None)
    return rz[putter] if rz else zone


def job_zone(zone, name: str, job):
    if job.kind == "relay":
        return relay_zone(zone, name)
    if job.kind == "fetch":
        return relay_zone(zone, other(name))


def job_waits(zone, name: str, job) -> bool:
    jz = job_zone(zone, name, job)
    if job.kind == "relay" and getattr(zone, "relay_res", None) not in (None, name):
        return True
    return jz is not None and zone_busy(jz, name)


def release_slot(zone, ob) -> None:
    for occ in (getattr(zone, "relay_occ", None) or {}).values():
        for i, oid in enumerate(occ):
            if oid == ob.id:
                occ[i] = None


def slots_full(zone, name: str) -> bool:
    occ = (getattr(zone, "relay_occ", None) or {}).get(name)
    return bool(occ) and all(o is not None for o in occ)


def take_relay_slot(arm: Arm, zone, ob):
    occ = zone.relay_occ[arm.name]
    n = 0
    while all(o is not None for o in occ) and n < SLOT_WAIT:
        n += 1
        arm.waiting = True
        yield
    arm.waiting = False
    i = next((k for k, o in enumerate(occ) if o is None), 0)
    occ[i] = ob.id
    return relay_slots(arm.name)[i]


def mid_busy(zone: Zone, name: str) -> bool:
    if getattr(zone, "mid_owner", None) not in (None, name):
        return True
    return getattr(zone, "mid_queue", None) not in (None, name)


def gives_way(arm: Arm, zone) -> bool:
    o = other(arm.name)
    rz = getattr(zone, "relay", None) or {}
    owners = [getattr(zone, "owner", None)] + [z.owner for z in rz.values()] + [getattr(zone, "mid_owner", None)]
    for owner in owners:
        if owner == o:
            return True
        if owner == arm.name:
            return False
    return arm.name == "right"


def arm_task(arm: Arm, jobs: list, zone: Zone, baskets, cats=None):
    queue = list(jobs)
    arm.queue = queue
    requeued: dict[int, int] = {}
    while queue:
        if arm.ctx.obs.step > JOB_DEADLINE:
            break
        if queue[0].kind == "fetch" and queue[0].obj.state != "relay":
            k = next((i for i, j in enumerate(queue) if j.kind == "fetch" and j.obj.state == "relay"), None)
            if k:
                queue.insert(0, queue.pop(k))
        if queue[0].kind == "relay" and slots_full(zone, arm.name):
            k = next(
                (i for i, j in enumerate(queue) if j.kind == "place" or (j.kind == "fetch" and j.obj.state == "relay")),
                None,
            )
            if k:
                queue.insert(0, queue.pop(k))
        if job_waits(zone, arm.name, queue[0]):
            ok_baskets = (OWN_BASKET[arm.name], 1) if not mid_busy(zone, arm.name) else (OWN_BASKET[arm.name],)
            k = next(
                (
                    i
                    for i, j in enumerate(queue)
                    if j.kind == "place"
                    and job_zone(zone, arm.name, j) is None
                    and own_side(arm.name, j.obj.xy)
                    and (j.obj.basket in ok_baskets)
                ),
                None,
            )
            if k:
                queue.insert(0, queue.pop(k))
        job = queue.pop(0)
        ob = job.obj
        if job.kind == "fetch":
            if ob.state not in ("relay", "lost", "basket"):
                k = next((i for i, j in enumerate(queue) if j.kind != "fetch"), None)
                if k is not None:
                    queue.insert(k + 1, job)
                    continue
            if ob.state not in ("relay", "lost", "basket"):
                yield from park(arm)

            def other_ready():
                return any(j.kind == "fetch" and j.obj.state == "relay" for j in queue)

            yield from arm.wait_until(lambda: ob.state in ("relay", "lost", "basket") or other_ready(), max_steps=500)
            if ob.state not in ("relay", "lost", "basket") and other_ready():
                queue.insert(0, job)
                continue
            if ob.state != "relay":
                continue
        if job.kind == "place" and ob.basket == 1:
            if mid_busy(zone, arm.name):
                k = next(
                    (
                        i
                        for i, j in enumerate(queue)
                        if j.kind in ("place", "relay") and j.obj.basket != 1 and (not job_waits(zone, arm.name, j))
                    ),
                    None,
                )
                if k is not None:
                    nxt = queue.pop(k)
                    queue.insert(0, job)
                    queue.insert(0, nxt)
                    continue
                yield from park(arm)
                if getattr(zone, "mid_queue", None) is None:
                    zone.mid_queue = arm.name
                n = 0
                while mid_busy(zone, arm.name) and n < MID_WAIT:
                    n += 1
                    arm.waiting = True
                    yield
                arm.waiting = False
            zone.mid_owner = arm.name
            if getattr(zone, "mid_queue", None) == arm.name:
                zone.mid_queue = None
        jz = job_zone(zone, arm.name, job)
        pick_central = job.kind == "fetch" or (jz is zone and in_zone(ob.xy))
        place_central = job.kind == "relay"
        late = place_central and (not pick_central)
        if late and ob.cat == "car":
            late = False
        if job.kind == "relay" and getattr(zone, "relay_res", None) is None:
            zone.relay_res = arm.name
        if jz is not None and (pick_central or place_central) and (not late):
            yield from take_zone(arm, jz)
        res = "fail"
        for attempt in range(2):
            res = yield from pick(arm, ob, attempt, to_basket=job.kind != "relay")
            if res in ("ok", "blocked"):
                break
        if res == "blocked":
            zone.free(arm.name)
            for z in (jz,):
                if z is not None:
                    z.free(arm.name)
            if getattr(zone, "relay_res", None) == arm.name:
                zone.relay_res = None
            if getattr(zone, "mid_owner", None) == arm.name:
                zone.mid_owner = None
            n = requeued.get(id(job), 0)
            if n < MAX_REQUEUE:
                requeued[id(job)] = n + 1
                if job.kind == "relay":
                    queue.insert(min(1, len(queue)), job)
                else:
                    queue.append(job)
                yield from park(arm)
                yield from arm.hold(BLOCK_PAUSE + (10 if arm.name == "right" else 0))
                continue
        ok = res == "ok"
        if ok or job.kind == "fetch":
            release_slot(zone, ob)
        if ok and job.kind == "fetch" and (jz is not None):
            jz.free(arm.name)
        spot = RELAY_SPOTS[arm.name]
        if ok and place_central and getattr(zone, "relay_occ", None):
            spot = yield from take_relay_slot(arm, zone, ob)
        if ok and place_central and (jz.owner != arm.name):
            if late and zone_busy(jz, arm.name):
                yield from hold_clear(arm, ob, always=True)
            yield from take_zone(arm, jz)
        if ok:
            if job.kind == "relay":
                arm.lost_xy = None
                ok = yield from put_down(arm, ob, spot, want=RELAY_LONG_YAW)
                ob.state = "relay" if ok else "recovering" if may_recover(arm, ob, cats) else "lost"
                if ok:
                    ob.xy = np.asarray(spot, float)
                else:
                    release_slot(zone, ob)
            else:
                arm.lost_xy = None
                ok = yield from place(
                    arm,
                    ob,
                    job.slot,
                    baskets,
                    mid=zone,
                    on_cross=lambda: [z.free(arm.name) for z in (zone, jz) if z is not None],
                )
                if getattr(zone, "mid_owner", None) == arm.name:
                    zone.mid_owner = None
                ob.state = "basket" if ok else "lost"
            if arm.held is not None:
                yield from put_down(arm, ob, None)
                arm.held = None
        else:
            ob.state = "lost"
        yield from rise(arm)
        zone.free(arm.name)
        for z in (jz,):
            if z is not None:
                z.free(arm.name)
        if getattr(zone, "relay_res", None) == arm.name:
            zone.relay_res = None
        if getattr(zone, "mid_owner", None) == arm.name:
            zone.mid_owner = None
        if not ok and ob.state in ("lost", "recovering") and may_recover(arm, ob, cats):
            yield from recover(arm, ob, job, queue, zone, baskets, cats)
        elif ob.state == "recovering":
            ob.state = "lost"
        arm.lost_xy = None
    yield from finish(arm)


def gripper_empty(arm: Arm) -> bool:
    return held_width(float(arm.ctx.obs.state.get(f"{arm.name}_ee_joint_state", [0.0])[0])) < MIN_HELD


def watched(arm: Arm, gen):
    while True:
        try:
            next(gen)
        except StopIteration:
            return True
        yield
        if gripper_empty(arm):
            gen.close()
            arm.waiting = False
            if hasattr(arm, "blocked"):
                arm.blocked = False
            arm.status = "ok"
            return False


def held_joints_to(arm: Arm, q_goal, joint_step: float):
    yield from arm.joints_to(q_goal, joint_step=joint_step)


def lost_early(arm: Arm):
    arm.lost_xy = np.asarray(arm.tcp()[:2], float).copy()
    arm.held = None


def may_recover(arm: Arm, ob: Obj, cats) -> bool:
    return bool(
        cats is not None
        and getattr(arm, "lost_xy", None) is not None
        and (not ob.recovered)
        and (arm.ctx.obs.step <= RECOVER_STEP)
    )


def recover(arm: Arm, ob: Obj, job, queue: list, zone, baskets, cats):
    ob.recovered = True
    xy_lost, old = np.asarray(arm.lost_xy, float), np.asarray(ob.xy, float)
    yield from rise(arm)
    yield from park(arm)
    ob.xy = xy_lost.copy()
    others = getattr(zone, "objs", None) or [j.obj for j in queue]
    avoid = [np.asarray(o.xy, float) for o in others if o is not ob and o.state in ("table", "relay")]
    found = yield from relocate(arm, ob, cats, baskets, radius=RECOVER_R, same_cat=True, avoid=avoid)
    if not found:
        ob.xy, ob.state = old, "lost"
        return
    ob.touched = True
    if ob.cat in LOW_GRASP_Z:
        ob.low = True
    if job.kind == "relay":
        slots = relay_slots(arm.name)
        d = [float(np.linalg.norm(ob.xy - np.asarray(sp, float))) for sp in slots]
        i = int(np.argmin(d))
        if d[i] < RECOVER_SLOT_R:
            occ = (getattr(zone, "relay_occ", None) or {}).get(arm.name)
            if occ is not None and occ[i] is None:
                occ[i] = ob.id
            ob.state = "relay"
            return
    own = grasp_options(arm.name, ob, arm.q_home)
    if own and abs(own[0].tilt) <= DIRECT_MAX_TILTS.get(ob.cat, DIRECT_MAX_TILT):
        ob.state = "table"
        queue.insert(0, Job(ob, "relay") if job.kind == "relay" else Job(ob, "place", job.slot))
        return
    ob.state = "lost"


def finish(arm: Arm):
    if arm.held is None and arm.ctx.obs.step > LAST_ROUND_STEP:
        yield from go_home(arm)
        return
    yield from park(arm)
    arm.active = False
    while arm.other is not None and arm.other.active:
        if arm.held is None and arm.ctx.obs.step > LAST_ROUND_STEP:
            arm.active = True
            yield from go_home(arm)
            arm.active = False
            break
        yield


def spot_occlusion(obs, xy, r: float = 0.04) -> float:
    import cv2
    from ...perception import robot_mask as RM

    cam = C.head()
    q = {a: np.asarray(obs.state.get(f"{a}_arm_joint_state", LOOK_Q), float) for a in K.ARMS}
    g = {a: float(obs.state.get(f"{a}_ee_joint_state", [1.0])[0]) for a in K.ARMS}
    t = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    pts = np.stack([xy[0] + r * np.cos(t), xy[1] + r * np.sin(t), np.full_like(t, K.TABLE_Z + 0.02)], axis=1)
    uv, _ = cam.project(pts)
    disc = np.zeros((C.H, C.W), np.uint8)
    cv2.fillConvexPoly(disc, np.round(uv).astype(np.int32), 1)
    n = int(np.count_nonzero(disc))
    if not n:
        return 0.0
    rob = RM.mask(cam, q["left"], q["right"], g["left"], g["right"], dilate=4)
    return float(np.count_nonzero(rob & (disc > 0))) / n


def relocate(arm: Arm, ob: Obj, cats, baskets, radius: float = RELOCATE_R, same_cat: bool = False, avoid=()):
    o = arm.other
    for _ in range(40):
        if o is None or float(np.linalg.norm(o.tcp()[:2] - ob.xy)) > RELOCATE_CLEAR:
            break
        yield from arm.hold(1)
    if spot_occlusion(arm.ctx.obs, ob.xy) > OCCLUDED:
        yield from park(arm)
    yield from arm.hold(2)
    obs = arm.ctx.obs
    q = {a: np.asarray(obs.state.get(f"{a}_arm_joint_state", LOOK_Q), float) for a in K.ARMS}
    g = {a: float(obs.state.get(f"{a}_ee_joint_state", [1.0])[0]) for a in K.ARMS}
    img = obs.images["cam_head"]
    insts = head_instances(img)
    try:
        _, items = LS.detect(
            img,
            C.head(),
            cats,
            q["left"],
            q["right"],
            g["left"],
            g["right"],
            baskets=baskets,
            count_range=(0, 3),
            instances=insts,
        )
    except Exception:
        return None
    near = [
        it
        for it in items
        if float(np.linalg.norm(it.xy - ob.xy)) < radius
        and (not same_cat or it.category == ob.cat)
        and all(float(np.linalg.norm(it.xy - np.asarray(p, float))) > MOVED_R for p in avoid)
    ]
    if not near:
        occ = spot_occlusion(obs, ob.xy)
        if occ > OCCLUDED:
            return None
        return False
    it = min(near, key=lambda i: (i.category != ob.cat, float(np.linalg.norm(i.xy - ob.xy))))
    xy, yaw, dims = LS.base_pose(it, ob.cat)
    ob.xy, ob.yaw, ob.dims = np.asarray(xy, float), float(yaw), tuple(dims)
    return True


def hold_clear(arm: Arm, ob: Obj, always: bool = False):
    p = arm.tcp()
    spots = [np.asarray(xy, float) for a in K.ARMS for xy in relay_slots(a)]
    near = min(float(np.linalg.norm(p[:2] - sp)) for sp in spots)
    if always:
        if float(np.hypot(p[0] - hold_xy(arm.name)[0], p[1] - hold_xy(arm.name)[1])) < 0.05:
            return
    elif near >= HOLD_CLEAR and (not in_zone(p[:2])):
        return
    z = max(float(p[2]), EMPTY_Z, K.TABLE_Z + ob.dims[2] + 0.04)
    hx, hy = hold_xy(arm.name)
    yield from arm.transit(tcp_pose(arm.T[:3, :3], (hx, hy, z)), z, joint_step=held_step(arm, ob))


def take_zone(arm: Arm, zone: Zone):
    if zone.owner == arm.name:
        return
    if zone.owner is not None or getattr(zone, "queue", None) not in (None, arm.name):
        yield from park(arm)
    while True:
        q = getattr(zone, "queue", None)
        if q in (None, arm.name) and zone.take(arm.name):
            break
        if q is None:
            zone.queue = arm.name
        arm.waiting = True
        yield
    if getattr(zone, "queue", None) == arm.name:
        zone.queue = None
    arm.waiting = False


def park(arm: Arm):
    if arm.held is not None:
        yield from rise(arm)
        return
    if float(np.max(np.abs(arm.q - LOOK_Q))) <= 0.05:
        return
    yield from rise(arm)
    yield from arm.joints_to(LOOK_Q, grip=1.0, joint_step=FREE_STEP, force=True)


def rise(arm: Arm):
    p = arm.tcp()
    if p[2] < EMPTY_Z - 0.005:
        T = tcp_pose(arm.T[:3, :3], [p[0], p[1], EMPTY_Z])
        yield from arm.line_to(T, lin_step=0.014)


def grip_reading(arm: Arm, max_steps: int = 10):
    prev = None
    g = 1.0
    for _ in range(max_steps):
        yield from arm.hold(1)
        g = float(arm.ctx.obs.state.get(f"{arm.name}_ee_joint_state", [1.0])[0])
        if prev is not None and abs(g - prev) < 0.003:
            break
        prev = g
    return g


def settle(arm: Arm, tol: float = 0.008, max_steps: int = 8):
    lag = 0.0
    for _ in range(max_steps):
        lag = float(np.max(np.abs(obs_q(arm) - arm.q)))
        if lag < tol:
            break
        yield from arm.hold(1)
    return lag


def cross_height(arm: Arm, g, z_hi: float, z_hover: float):
    z_lo = max(z_hover, CROSS_MIN_Z)
    z = z_hi
    while True:
        q = ik_pose(arm, g.rot, (g.tcp[0], g.tcp[1], z), seeds=[g.q_hover])
        if q is not None:
            return z, q
        if z <= z_lo + 1e-06:
            return z_hi, None
        z = max(z_lo, z - CROSS_DZ)


def hover_height(arm: Arm, g, tcp, h: float) -> float:
    z = max(g.hover_z, K.TABLE_Z + h + 0.03)
    if z > g.hover_z + 0.0001 and ik_pose(arm, g.rot, (tcp[0], tcp[1], z), seeds=[g.q_hover]) is None:
        return g.hover_z
    return z


def lift_clear(arm: Arm, z_c: float):
    p = arm.tcp()
    if p[2] >= z_c - 0.001:
        return True
    rot = arm.T[:3, :3].copy()
    yield from arm.line_to(tcp_pose(rot, (p[0], p[1], z_c)), lin_step=0.012, fallback_joint=False)
    if arm.status == "ok":
        return True
    lo, hi = float(p[2]), float(z_c)
    for _ in range(4):
        m = 0.5 * (lo + hi)
        if ik_pose(arm, rot, (p[0], p[1], m)) is not None:
            lo = m
        else:
            hi = m
    if lo > p[2] + 0.005:
        yield from arm.line_to(tcp_pose(rot, (p[0], p[1], lo)), lin_step=0.012, fallback_joint=False)
        if arm.status == "ok" and lo >= z_c - 0.006:
            return True
    bx, by, _ = K.BASE_POS[arm.name]
    d = np.array([bx - p[0], by - p[1]], dtype=float)
    d /= max(1e-09, float(np.linalg.norm(d)))
    for back in LIFT_BACK:
        xy = np.asarray(p[:2], float) + back * d
        if ik_pose(arm, rot, (xy[0], xy[1], z_c)) is None:
            continue
        yield from arm.line_to(tcp_pose(rot, (xy[0], xy[1], z_c)), lin_step=0.012)
        if arm.status == "ok":
            return True
    return False


def via_point(arm: Arm, g, z_hi: float):
    bx, by, _ = K.BASE_POS[arm.name]
    d = np.array([bx - g.tcp[0], by - g.tcp[1]])
    d /= max(1e-09, float(np.linalg.norm(d)))
    for back in VIA_BACK:
        xy = np.asarray(g.tcp[:2], float) + back * d
        z = z_hi
        while z >= CROSS_MIN_Z - 1e-06:
            q = ik_pose(arm, g.rot, (xy[0], xy[1], z), seeds=[g.q_hover])
            if q is not None:
                return xy, z, q
            z -= CROSS_DZ


def pick(arm: Arm, ob: Obj, attempt: int = 0, to_basket: bool = True):
    gr = grasp_options(arm.name, ob, arm.q)
    if not gr:
        return "fail"
    first = min(attempt, len(gr) - 1) if ob.cat in ROUND else 0
    order = [first] + [i for i in range(len(gr)) if i != first][: PICK_CANDIDATES - 1]
    _, W, h = ob.dims
    open_g = grip_for_gap(min(0.088, max(W, getattr(gr[0], "wall_size", W)) + OPEN_MARGIN))
    for k, gi in enumerate(order):
        g = gr[gi]
        tcp = g.tcp.copy()
        if attempt and (not (ob.low and ob.cat in LOW_GRASP_Z)):
            tcp[2] = max(G.tcp_min_z(K.TABLE_Z + SUPPORT_DZ.get(ob.cat, 0.0), g.tilt), tcp[2] - 0.004)
        z_hover = hover_height(arm, g, tcp, h)
        T_hover = tcp_pose(g.rot, [tcp[0], tcp[1], z_hover])
        z_safe, q_cross = cross_height(arm, g, max(EMPTY_Z, z_hover), z_hover)
        via = via_point(arm, g, max(EMPTY_Z, z_hover)) if q_cross is None else None
        if via is not None:
            vxy, vz, vq = via
            yield from arm.transit(
                tcp_pose(g.rot, [vxy[0], vxy[1], vz]), vz, grip=open_g, seeds=[vq], joint_step=FREE_STEP
            )
            if arm.status == "ok":
                yield from arm.move_to(
                    T_hover, min_z=z_hover - 0.02, grip=open_g, seeds=[g.q_hover], joint_step=FREE_STEP, lin_step=0.012
                )
        else:
            seeds = [g.q_hover] if q_cross is None else [q_cross, g.q_hover]
            yield from arm.transit(T_hover, z_safe, grip=open_g, seeds=seeds, joint_step=FREE_STEP)
        if arm.status == "ok":
            break
        if arm.status == "blocked":
            return "blocked"
    else:
        return "fail"
    yield from arm.set_grip(open_g, 3)
    m = wrist_view(arm, ob)
    if m is not None and m.cut <= WRIST_MAX_CUT:
        new_yaw = ob.yaw
        if wrist_heading_ok(ob.cat, m):
            new_yaw = m.yaw
        dpos = float(np.linalg.norm(m.xy - ob.xy))
        dyaw = abs(math.remainder(new_yaw - ob.yaw, math.pi))
        if dpos > REAIM_MIN_SHIFT or dyaw > REAIM_MIN_TURN:
            old = ob.xy.copy(), ob.yaw
            ob.xy, ob.yaw = np.asarray(m.xy, float), float(new_yaw)
            gr2 = grasp_options(arm.name, ob, arm.q)
            if not gr2:
                ob.xy, ob.yaw = old
            else:
                g2 = gr2[min(attempt, len(gr2) - 1)] if ob.cat in ROUND else gr2[0]
                tcp2 = g2.tcp.copy()
                if attempt and (not (ob.low and ob.cat in LOW_GRASP_Z)):
                    tcp2[2] = max(G.tcp_min_z(K.TABLE_Z + SUPPORT_DZ.get(ob.cat, 0.0), g2.tilt), tcp2[2] - 0.004)
                z2 = hover_height(arm, g2, tcp2, h)
                T2 = tcp_pose(g2.rot, [tcp2[0], tcp2[1], z2])
                yield from arm.move_to(
                    T2, min_z=min(z2, z_hover) - 0.02, grip=open_g, seeds=[g2.q_hover], joint_step=FREE_STEP
                )
                if arm.status == "blocked":
                    return "blocked"
                if arm.status != "ok":
                    return "fail"
                g, tcp, z_hover, T_hover = g2, tcp2, z2, T2
    yield from arm.line_to(tcp_pose(g.rot, tcp), lin_step=0.011)
    if arm.status != "ok":
        st = arm.status
        yield from arm.line_to(T_hover, lin_step=0.012)
        return "blocked" if st == "blocked" else "fail"
    yield from settle(arm)
    yield from arm.set_grip(0.0, 6)
    g_obs = yield from grip_reading(arm)
    w = held_width(g_obs)
    if w < MIN_HELD:
        yield from arm.set_grip(open_g, 4)
        yield from arm.line_to(T_hover, lin_step=0.012)
        return "miss"
    arm.held = {"obj": ob, "rot": g.rot, "width": w, "yaw_close": g.yaw, "pts": held_points(ob, g.rot, tcp)}
    ob.state = "carried"
    ob.touched = True
    lifted = False
    if abs(g.tilt) < math.radians(1):
        z_c = RIM_Z + over_rim() + grasp_z(ob)
        if T_hover[2, 3] < z_c:
            lifted = yield from lift_clear(arm, z_c)
    if not lifted and arm.tcp()[2] < T_hover[2, 3] - 0.003:
        yield from arm.line_to(T_hover, lin_step=0.012, fallback_joint=False)
    g_obs = float(arm.ctx.obs.state.get(f"{arm.name}_ee_joint_state", [0.0])[0])
    if held_width(g_obs) < MIN_HELD:
        arm.held = None
        ob.state = "table"
        if ob.cat in LOW_GRASP_Z:
            ob.low = True
        yield from arm.set_grip(open_g, 4)
        return "miss"
    return "ok"


def wrist_view(arm: Arm, ob: Obj):
    o = arm.ctx.obs
    img = o.images.get(f"cam_{arm.name}_wrist")
    ee = o.state.get(f"{arm.name}_ee_pose")
    if img is None or ee is None:
        return None
    insts = SEGM.segment(img, camera=f"{arm.name}_wrist")
    m = WO.measure(insts, C.wrist(ee), ob.cat, ob.xy, ob.dims[2])
    if m is None:
        m = WO.measure(insts, C.wrist(ee), ob.cat, ob.xy, ob.dims[2], max_shift=WRIST_WIDE_SHIFT)
        if m is not None and in_basket(m.xy, ob.baskets):
            m = None
    return m


def in_basket(xy, baskets, margin: float = 0.01) -> bool:
    for bk in baskets or []:
        if (
            abs(float(xy[0]) - bk.x) < float(bk.size[0]) / 2 + margin
            and abs(float(xy[1]) - bk.y) < float(bk.size[1]) / 2 + margin
        ):
            return True
    return False


def release_rotations(arm: Arm, p_tcp, want_long_yaw: float | None):
    yaw_c = arm.held["yaw_close"]
    if want_long_yaw is None:
        turns = [math.radians(d) for d in range(0, 360, 30)]
    else:
        d0 = math.remainder(want_long_yaw + math.pi / 2 - yaw_c, math.pi)
        d_min = 0.0 if abs(d0) <= LONG_AXIS_TOL else math.copysign(abs(d0) - LONG_AXIS_TOL, d0)
        turns = [d_min] + [
            d0 + e
            for e in (
                0.0,
                math.pi,
                math.radians(20),
                -math.radians(20),
                math.pi + math.radians(20),
                math.pi - math.radians(20),
            )
        ]
    bx, by = K.BASE_POS[arm.name][:2]
    to_base = np.array([bx - p_tcp[0], by - p_tcp[1]])
    to_base /= max(1e-09, np.linalg.norm(to_base))
    out = []
    for tilt in PLACE_TILTS:
        for d in turns:
            yaw = yaw_c + d
            if not tilt:
                out.append((0.0, G.grasp_rotation(yaw, 0.0), d))
                continue
            for sgn in (1.0, -1.0):
                r = G.grasp_rotation(yaw, sgn * tilt)
                if float(np.dot(-r[:2, 0], to_base)) >= 0.2:
                    out.append((tilt, r, d))
    return out


def ik_pose(arm: Arm, rot, p, seeds=()):
    T = tcp_pose(rot, p)
    r = K.solve_near(
        arm.name, T, arm.q, K.TCP_OFFSET, extra_seeds=list(seeds) + [G.base_seed(arm.name, p[:2])], n_random=2
    )
    return r.q if r.ok else None


def over_other_basket(
    arm_name: str, q_a, q_b, target: int, baskets, n: int = 12, joint_step: float | None = None
) -> bool:
    q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
    n_act = max(1, math.ceil(float(np.max(np.abs(q_b - q_a))) * math.pi / 2 / joint_step)) if joint_step else 0
    for t in np.linspace(0.0, 1.0, n):
        pts = [K.point_of(K.fk(arm_name, q_a + (q_b - q_a) * t), K.TCP_OFFSET)]
        if n_act:
            s = (1.0 - math.cos(math.pi * t)) / 2.0
            ds = math.pi / (2 * n_act) * math.sin(math.pi * t)
            p2 = K.point_of(K.fk(arm_name, q_a + (q_b - q_a) * min(1.0, s + 0.001)), K.TCP_OFFSET)
            p1 = K.point_of(K.fk(arm_name, q_a + (q_b - q_a) * max(0.0, s - 0.001)), K.TCP_OFFSET)
            v = (p2 - p1) / max(1e-09, min(1.0, s + 0.001) - max(0.0, s - 0.001)) * ds
            pts = [K.point_of(K.fk(arm_name, q_a + (q_b - q_a) * s), K.TCP_OFFSET)]
            pts += [pts[0] + v * k for k in (1.5, LANDING_STEPS)]
        for p in pts:
            for i, bk in enumerate(baskets or []):
                if (
                    i != target
                    and abs(p[0] - bk.x) < BASKET_HALF[0] + BASKET_MARGIN
                    and (abs(p[1] - bk.y) < BASKET_HALF[1] + BASKET_MARGIN)
                ):
                    return True
    return False


def front_via(arm: Arm, ob: Obj, slot, rot, z, q_goal, baskets, joint_step: float | None = None):
    if not baskets or ob.basket >= len(baskets):
        return None
    js = joint_step
    if not over_other_basket(arm.name, arm.q, q_goal, ob.basket, baskets, joint_step=js):
        return None
    bk = baskets[ob.basket]
    for dy in FRONT_DYS:
        xy = float(slot[0]), float(bk.y) - BASKET_HALF[1] - dy
        r = K.solve_near(
            arm.name,
            tcp_pose(rot, (xy[0], xy[1], z)),
            q_goal,
            K.TCP_OFFSET,
            extra_seeds=[arm.q, G.base_seed(arm.name, xy)],
            n_random=2,
        )
        q = r.q if r.ok else None
        if (
            q is None
            or over_other_basket(arm.name, arm.q, q, ob.basket, baskets, joint_step=js)
            or over_other_basket(arm.name, q, q_goal, ob.basket, baskets, joint_step=js)
        ):
            continue
        return q


def carry_cost(arm: Arm, ob: Obj, slot, rot, z_cross, q_cross, baskets) -> float:
    d = float(np.max(np.abs(np.asarray(q_cross) - arm.q)))
    q_f = front_via(arm, ob, slot, rot, z_cross, q_cross, baskets, joint_step=held_step(arm, ob))
    if q_f is None:
        return d
    return float(np.max(np.abs(np.asarray(q_f) - arm.q))) + float(np.max(np.abs(np.asarray(q_cross) - q_f)))


def release_choice(arm: Arm, ob: Obj, slot, baskets, z_rel: float, z_cross: float, lift: bool = False):
    want = None if ob.cat in ROUND else math.pi / 2
    choice = None
    best = None
    cands = []
    for tilt, rot, _turn in release_rotations(arm, (slot[0], slot[1], z_rel), want):
        if choice is not None and tilt > choice[0] + 1e-06:
            break
        dz = release_lift(arm, ob, rot, tilt) if lift else 0.0
        q_rel = ik_pose(arm, rot, (slot[0], slot[1], z_rel + dz))
        if q_rel is None:
            continue
        q_cross = ik_pose(arm, rot, (slot[0], slot[1], z_cross + dz), seeds=[q_rel])
        if q_cross is None:
            continue
        cost = float(np.max(np.abs(np.asarray(q_cross) - arm.q)))
        cands.append((cost, tilt, rot, q_rel, q_cross, dz))
        if best is None or cost < best - 1e-06:
            choice, best = (tilt, rot, q_rel, q_cross), cost
    if len(cands) > 1:
        best_tot = None
        for cost, tilt, rot, q_rel, q_cross, dz in sorted(cands, key=lambda c: c[0]):
            if best_tot is not None and cost >= best_tot - 1e-06:
                break
            tot = carry_cost(arm, ob, slot, rot, z_cross + dz, q_cross, baskets)
            if best_tot is None or tot < best_tot - 1e-06:
                best_tot, choice = tot, (tilt, rot, q_rel, q_cross)
    return choice


def place(arm: Arm, ob: Obj, slot, baskets, on_cross=None, mid=None):
    gz = grasp_z(ob)
    z_rel = FLOOR_Z + RELEASE_DZ_CAT.get(ob.cat, RELEASE_DZ) + gz
    z_cross = RIM_Z + OVER_RIM + gz
    choice = release_choice(arm, ob, slot, baskets, z_rel, z_cross, lift=True)
    lifted = True
    if choice is None:
        choice, lifted = release_choice(arm, ob, slot, baskets, z_rel, z_cross), False
    if choice is None:
        return (yield from put_down(arm, ob, None))
    tilt, rot, q_rel, q_cross = choice
    dz = release_lift(arm, ob, rot, tilt) if lifted else 0.0
    z_rel, z_cross = z_rel + dz, z_cross + dz
    q_high, z_carry = None, z_cross
    z_hi = RIM_Z + over_rim() + gz + dz
    if z_hi > z_cross + 1e-06:
        q_high = ik_pose(arm, rot, (slot[0], slot[1], z_hi), seeds=[q_cross])
        gap = None if q_high is None else float(np.max(np.abs(np.asarray(q_high) - np.asarray(q_cross))))
        if gap is not None and gap < BRANCH_MAX:
            z_carry = z_hi
        else:
            q_high = None
    q_to = q_cross if q_high is None else q_high
    p = arm.tcp()
    if p[2] < z_carry:
        yield from lift_clear(arm, z_carry)
    step = held_step(arm, ob)
    if mid is not None and abs(float(slot[0])) < 0.13:
        n = 0
        while getattr(mid, "mid_owner", None) not in (None, arm.name) and n < MID_WAIT:
            n += 1
            arm.waiting = True
            yield
        arm.waiting = False
        mid.mid_owner = arm.name
    q_front = front_via(arm, ob, slot, rot, z_carry, q_to, baskets, joint_step=step)
    if q_front is not None:
        held = yield from watched(arm, held_joints_to(arm, q_front, step))
        if not held:
            lost_early(arm)
            return False
    held = yield from watched(arm, held_joints_to(arm, q_to, step))
    if not held:
        lost_early(arm)
        return False
    tries = 0
    while arm.status != "ok" and tries < PLACE_RETREAT and gives_way(arm, mid):
        tries += 1
        yield from hold_clear(arm, ob, always=True)
        yield from arm.hold(BLOCK_PAUSE)
        yield from arm.joints_to(q_to, joint_step=step)
    if arm.status != "ok":
        yield from arm.joints_to(q_to, joint_step=step, force=True)
    if q_high is not None:
        yield from arm.joints_to(q_cross, joint_step=DESCENT_STEP_FAST, force=True)
    if on_cross is not None:
        on_cross()
    g_obs = float(arm.ctx.obs.state.get(f"{arm.name}_ee_joint_state", [0.0])[0])
    if held_width(g_obs) < MIN_HELD:
        arm.lost_xy = np.asarray(arm.tcp()[:2], float).copy()
        arm.held = None
        return False
    same_branch = float(np.max(np.abs(np.asarray(q_cross) - np.asarray(q_rel)))) < BRANCH_MAX
    if same_branch:
        yield from arm.joints_to(q_rel, joint_step=DESCENT_STEP_FAST, force=True)
    else:
        yield from arm.line_to(tcp_pose(rot, (slot[0], slot[1], z_rel)), lin_step=0.01)
    w = arm.held["width"]
    rise, rise_step = release_rise(arm, ob, rot)
    rise = min(rise, max(RELEASE_RISE_DZ, z_cross - z_rel))
    yield from arm.set_grip(grip_for_gap(min(0.088, w + RELEASE_OPEN)), RELEASE_OPEN_STEPS.get(ob.cat, RELEASE_STEPS))
    yield from arm.hold(RELEASE_HOLD)
    if RELEASE_SETTLE.get(ob.cat):
        yield from arm.hold(int(RELEASE_SETTLE[ob.cat]))
    arm.held = None
    yield from arm.line_to(tcp_pose(rot, (slot[0], slot[1], z_rel + rise)), lin_step=rise_step)
    if same_branch:
        yield from arm.joints_to(q_cross, joint_step=FREE_STEP, force=True)
    else:
        yield from arm.line_to(tcp_pose(rot, (slot[0], slot[1], z_cross)), lin_step=0.012)
    if mid is not None and getattr(mid, "mid_owner", None) == arm.name:
        mid.mid_owner = None
    yield from arm.set_grip(1.0, 2)
    return True


def put_down(arm: Arm, ob: Obj, spot, want=None):
    gz = grasp_z(ob)
    p = arm.tcp()
    spot = (p[0], p[1]) if spot is None else spot
    z_put = K.TABLE_Z + gz + 0.004
    rot, turn = arm.T[:3, :3], 0.0
    cands = []
    for _tilt, r, d in release_rotations(arm, (spot[0], spot[1], z_put), None if ob.cat in ROUND else want):
        if cands and _tilt > cands[0][4] + 1e-06:
            break
        zp = z_put + release_lift(arm, ob, r, _tilt)
        q = ik_pose(arm, r, (spot[0], spot[1], zp))
        if q is not None:
            cands.append((r, d, q, zp, _tilt))
    if cands:
        cands.sort(key=lambda c: float(np.max(np.abs(np.asarray(c[2]) - arm.q))))
        cands = cands[:3]
    if not cands:
        return False
    rot, turn, q, z_put = cands[0][:4]
    z_carry = max(EMPTY_Z, K.TABLE_Z + ob.dims[2] + 0.04)
    for k in range(PUT_DOWN_TRIES):
        held = yield from watched(
            arm,
            arm.transit(
                tcp_pose(rot, (spot[0], spot[1], z_put)), z_carry, joint_step=held_step(arm, ob), **{"seeds": [q]}
            ),
        )
        if not held:
            lost_early(arm)
            return False
        if arm.status == "fail" and len(cands) > 1:
            cands.pop(0)
            rot, turn, q, z_put = cands[0][:4]
            yield from arm.transit(
                tcp_pose(rot, (spot[0], spot[1], z_put)), z_carry, joint_step=held_step(arm, ob), seeds=[q]
            )
        if arm.status != "blocked":
            break
        yield from hold_clear(arm, ob)
        yield from arm.hold(BLOCK_PAUSE)
    if arm.status != "ok":
        return False
    g_obs = float(arm.ctx.obs.state.get(f"{arm.name}_ee_joint_state", [0.0])[0])
    if held_width(g_obs) < MIN_HELD:
        arm.lost_xy = np.asarray(arm.tcp()[:2], float).copy()
        arm.held = None
        return False
    yield from arm.set_grip(grip_for_gap(min(0.088, arm.held["width"] + RELEASE_OPEN)), 6)
    arm.held = None
    yield from arm.line_to(tcp_pose(rot, (spot[0], spot[1], z_carry)), lin_step=0.012)
    ob.yaw = float(math.remainder(ob.yaw + turn, math.pi))
    ob.touched = False
    ob.xy = np.asarray(spot[:2], float)
    return True
