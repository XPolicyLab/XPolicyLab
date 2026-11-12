from __future__ import annotations

import math
import re

import numpy as np

from ...core import actions
from ...perception import cameras as C
from ...planning import kinematics as K
from ...planning import motion
from ...skills import dual as D
from ...skills import pinch as P
from ...skills.runner import Arm, link_arms, run_arms
from .. import Program, register

_PAT = re.compile("place all the objects into the box", re.IGNORECASE)
CATEGORIES = "car", "electric_toothbrush", "hammer", "shoe", "box"
LOOK_J1 = 0.7
LOOK_SETTLE = 4
DESCEND_STEP = 0.008
CLOSE_STEPS = 8
SETTLE_CLOSED = 4
RELEASE_MARGIN = 0.012
RELEASE_HOLD = 5
JOINT_SPEED = np.array([0.12, 0.12, 0.12, 0.15, 0.18, 0.2])
CARRY_SPEED = 0.6 * JOINT_SPEED
CARRY_SPEED_SHAPED = 0.8 * JOINT_SPEED
SHAPED_RAMP = 2
SLOW_CLEAR = 0.01
SLOW_FACTOR = 0.5
RELOOK_MOVED = 0.004
RELOOK_IOU = 0.6
FRONT_CATS = ()
FRONT_MARGIN = 0.12
FRONT_RULE = {"car": (0.0, FRONT_MARGIN), "shoe": (FRONT_MARGIN, 0.0)}
HEAD_APP = {"car": 1.0}
HEAD_NET = "car", "shoe"
FRONT_SETTLE = 2
RAMP_STEPS = 4
DT = 0.04
SHAPE_MIN, SHAPE_MAX = 2, 14
LIFT_STEP = 0.012
STEP_LIMIT = 1300
HOME_STEPS = 20
DIRECT_STEPS = 90
HANDOVER_STEPS = 220
SAME_STAGED = 0.04
GRASP_RETRIES = 2
_R15, _R25, _R35 = math.radians(15), math.radians(25), math.radians(35)
TIMED_HANDOVER, TIMED_DIRECT = 320, 230
PLAN_STEPS = ("direct", None), ("stage", ("two", None, None))
RETRY_IOU = 0.6
FLOOR_HALF = 0.1, 0.085
IN_BOX_MARGIN = 0.03


def speed_profile(n: int, ramp: int, shape: int = 0) -> np.ndarray:
    r = min(ramp, n // 2) if n >= 2 else 0
    v = np.ones(n)
    for i in range(r):
        v[i] = v[n - 1 - i] = (i + 1) / (r + 1)
    if shape > 0:
        w = np.zeros(n + shape)
        w[:n] += 0.5 * v
        w[shape:] += 0.5 * v
        v = w
    return v


def joints_fast(arm: Arm, q_goal, grip=None, speed=None, ramp: int = RAMP_STEPS, shape: int = 0):
    speed = JOINT_SPEED if speed is None else speed
    q0 = arm.q.copy()
    q1 = np.asarray(q_goal, dtype=float)
    n = max(1, math.ceil(float(np.max(np.abs(q1 - q0) / speed)) + ramp))
    v = speed_profile(n, ramp, shape)
    sc = np.cumsum(v) / v.sum()
    ok = yield from arm.follow(motion.Plan([q0 + (q1 - q0) * float(x) for x in sc]), grip, force=True)
    arm.status = "ok" if ok else "blocked"


def _hold(arms) -> dict:
    return actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)


def held(ctx, arm: str) -> float:
    return D.held_width(float(ctx.obs.state[f"{arm}_ee_joint_state"][0]), 0.0)


def perceive(ctx, img, tag: str = "look", only=None):
    from . import fit as PF
    from ...perception import seg as SEG

    cam = C.head()
    insts = SEG.segment(img, camera="head")
    fits = {}
    for i in sorted(insts, key=lambda i: -i.score):
        if i.category not in CATEGORIES or i.category in fits or (only is not None and i.category not in only):
            continue
        w = HEAD_APP.get(i.category, 0.0)
        f = PF.fit(cam, i.mask, i.category, img=img if w > 0 else None, w_app=w)
        if f is None:
            continue
        if i.category in HEAD_NET:
            from . import front as FR

            s = FR.front_score(img, cam, i.category, (f.x, f.y, f.z), np.asarray(f.target_dir, dtype=float)[:2])
            if s < 0.0:
                f = turned(f)
            f.front_net = abs(s)
        fits[i.category] = f
    return fits


def think(arms, gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        yield [_hold(arms)]


def joints_poly(arm: Arm, qs, grip=None, speed=None, ramp: int = RAMP_STEPS, shape: int = 0):
    speed = JOINT_SPEED if speed is None else speed
    P_ = np.vstack([arm.q.copy()[None], np.asarray(qs, dtype=float)])
    seg = np.max(np.abs(np.diff(P_, axis=0)) / speed, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    n = max(1, math.ceil(total + ramp))
    v = speed_profile(n, ramp, shape)
    out = []
    for x in np.cumsum(v) / v.sum() * total:
        j = int(min(len(seg) - 1, max(0, np.searchsorted(cum, x, side="right") - 1)))
        t = 1.0 if seg[j] <= 1e-12 else min(1.0, max(0.0, (x - cum[j]) / seg[j]))
        out.append(P_[j] + (P_[j + 1] - P_[j]) * t)
    ok = yield from arm.follow(motion.Plan(out), grip, force=True)
    arm.status = "ok" if ok else "blocked"


def follow_route(arm: Arm, qs, grip=None, speed=None, shape: int = 0, ramp: int = RAMP_STEPS):
    for q in qs:
        if np.ndim(q) == 2:
            yield from joints_poly(arm, q, grip=grip, speed=speed, ramp=ramp, shape=shape)
        else:
            yield from joints_fast(arm, q, grip=grip, speed=speed, ramp=ramp, shape=shape)


def swing_period(points, tcp, closing) -> float:
    c = np.asarray(closing, dtype=float)
    c = c / max(1e-09, float(np.linalg.norm(c)))
    r = np.asarray(points, dtype=float) - np.asarray(tcp, dtype=float)
    perp = r - np.outer(r @ c, c)
    d = float(np.linalg.norm(perp.mean(axis=0)))
    if d < 0.002:
        return math.inf
    return 2.0 * math.pi * math.sqrt(float((perp**2).sum(axis=1).mean()) / (9.81 * d))


def go_route(arm: Arm, obs, goal, grip: float, speed=None):
    from . import plan as PP

    qs, clear, _ = PP.plan_route(obs, arm.name, arm.q, goal, grip)
    if qs is None:
        arm.status = "fail"
        return
    yield from follow_route(arm, qs, grip=grip, speed=slow_if_tight(speed or JOINT_SPEED, clear))


def slow_if_tight(speed, clear: float):
    return speed * SLOW_FACTOR if clear < SLOW_CLEAR else speed


def pick_place(ctx, arm: Arm, plan, q_park, res: dict, obs, park: bool = True, fit=None):
    def fail(why):
        res["fail"] = why

    g_open = P.grip_for_gap(plan.pinch.gap)
    if (
        plan.hover_route is not None
        and plan.q_start is not None
        and (float(np.max(np.abs(arm.q - plan.q_start))) < 0.02)
    ):
        yield from follow_route(arm, plan.hover_route, grip=g_open)
    else:
        yield from go_route(arm, obs, plan.q_hover, g_open)
    if arm.status == "fail":
        fail("hover")
        return
    img = ctx.obs.images.get(f"cam_{arm.name}_wrist")
    if fit is not None and plan.category in FRONT_CATS:
        from . import fit as PF
        from ...perception import robot_mask as RM

        yield from arm.hold(FRONT_SETTLE)
        img = ctx.obs.images.get(f"cam_{arm.name}_wrist")
        if img is not None:
            cam = C.wrist_from_link6(K.fk(arm.name, arm.q))
            qs = {"left": None, "right": None}
            qs[arm.name] = arm.q
            free = ~RM.mask(cam, qs["left"], qs["right"], arm.g, arm.g)
            gen = PF.front_check_iter(cam, img, free, fit, with_fit=True)
            while True:
                try:
                    next(gen)
                except StopIteration as e:
                    s0, s1, f1, o0, o1 = e.value
                    break
                yield from arm.hold(1)
            m_best, m_own = FRONT_RULE.get(plan.category, (FRONT_MARGIN, -9.0))
            flip = s1 - s0 > m_best and o1 - o0 > m_own
            res["front"] = s0, s1
            if flip:
                res["flip"] = f1
                fail("flipped")
                return
    yield from arm.line_to(plan.T_grasp, lin_step=DESCEND_STEP, grip=g_open)
    yield from arm.set_grip(0.0, n=CLOSE_STEPS)
    yield from arm.hold(SETTLE_CLOSED)
    w0 = held(ctx, arm.name)
    yield from arm.line_to(plan.T_up, lin_step=LIFT_STEP)
    yield from arm.hold(2)
    w1 = held(ctx, arm.name)
    res["width"] = w1
    res["closed"] = w0
    res["tcp"] = np.asarray(plan.T_grasp[:3, 3], dtype=float).copy()
    if w1 <= 0.0:
        fail("empty")
        yield from arm.set_grip(1.0, n=6)
        yield from go_route(arm, obs, q_park, 1.0)
        return
    T = math.inf
    if plan.held_l6 is not None and plan.q_grasp is not None:
        T6 = K.fk(arm.name, plan.q_grasp)
        T = swing_period(plan.held_l6 @ T6[:3, :3].T + T6[:3, 3], plan.T_grasp[:3, 3], plan.pinch.closing)
    shape = int(round(T / 2.0 / DT)) if math.isfinite(T) else 0
    shape = shape if SHAPE_MIN <= shape <= SHAPE_MAX else SHAPE_MAX if shape > SHAPE_MAX else 0
    if shape:
        sp, ramp = CARRY_SPEED_SHAPED, SHAPED_RAMP
    else:
        sp, ramp = CARRY_SPEED, RAMP_STEPS
    yield from follow_route(arm, plan.carry, speed=slow_if_tight(sp, plan.carry_clear), shape=shape, ramp=ramp)
    yield from arm.line_to(plan.place.T_rel, lin_step=DESCEND_STEP)
    w2 = held(ctx, arm.name)
    g_rel = max(arm.g, P.grip_for_gap(max(w2, w1) + RELEASE_MARGIN))
    yield from arm.set_grip(g_rel, n=8)
    yield from arm.hold(RELEASE_HOLD)
    if w2 <= 0.0:
        fail("dropped")
    else:
        res["ok"] = True
    yield from arm.line_to(plan.place.T_above, lin_step=0.016)
    if park:
        yield from go_route(arm, obs, q_park, arm.g)


def box_update(fits2, box, obs):
    from . import plan as PP

    b2 = fits2.get("box")
    if b2 is None:
        return box, obs
    moved = float(math.hypot(b2.x - box.xy[0], b2.y - box.xy[1]))
    nb = PP.box_frame(b2)
    turn = math.degrees(math.acos(min(1.0, abs(float(nb.u @ box.u)))))
    use = b2.iou >= RELOOK_IOU and (moved > RELOOK_MOVED or turn > 2.0)
    if use:
        return nb, PP.BoxObstacle(b2)
    return box, obs


def relook(ctx, arms, box, obs, tag: str, cat: str | None = None):
    yield from run_arms(arms, {a: arms[a].hold(LOOK_SETTLE) for a in K.ARMS})
    img = ctx.obs.images.get("cam_head")
    if img is None:
        return (box, obs) if cat is None else (box, obs, None)
    fits2 = perceive(ctx, img, tag=tag, only=("box",) if cat is None else ("box", cat))
    box, obs = box_update(fits2, box, obs)
    if cat is None:
        return box, obs
    return box, obs, fits2.get(cat)


def over_floor(box, fit, margin: float = 0.0) -> bool:
    d = np.array([fit.x, fit.y]) - box.xy
    return abs(float(d @ box.u)) <= FLOOR_HALF[0] + margin and abs(float(d @ box.v)) <= FLOOR_HALF[1] + margin


def turned(fit):
    from . import fit as PF
    from . import plan as PP

    return PF.ObjFit(
        fit.category,
        fit.key,
        fit.x,
        fit.y,
        fit.z,
        math.atan2(math.sin(fit.yaw + math.pi), math.cos(fit.yaw + math.pi)),
        fit.iou,
        fit.alt_iou,
        PP.rot_z(math.pi) @ fit.R,
        -np.asarray(fit.target_dir, dtype=float),
    )


def handover(ctx, arms, look, st, box, obs, placed, last, fit=None):
    from . import plan as PP

    plan_a, sf, plan_b = st
    A, B = plan_a.arm, plan_b.arm
    if last is not None:
        yield from run_arms(arms, {last: go_route(arms[last], obs, look[last], arms[last].g)})
    res = {}
    yield from run_arms(arms, {A: pick_place(ctx, arms[A], plan_a, look[A], res, obs, park=True, fit=fit)})
    if res.get("flip") is not None:
        return res["flip"], box, obs
    if not res.get("ok"):
        return False, box, obs
    a_sure = res.get("front") is not None and res["front"][0] - res["front"][1] > FRONT_MARGIN
    fb = sf
    yield from run_arms(arms, {a: arms[a].hold(LOOK_SETTLE) for a in K.ARMS})
    img = ctx.obs.images.get("cam_head")
    if img is not None:
        fits2 = perceive(ctx, img, tag=f"stage_{sf.category}", only=(sf.category, "box"))
        f2 = fits2.get(sf.category)
        box, obs = box_update(fits2, box, obs)
        if f2 is not None and math.hypot(f2.x - sf.x, f2.y - sf.y) <= SAME_STAGED:
            if float(np.dot(f2.target_dir, sf.target_dir)) < 0.0:
                n1, n2 = getattr(fit, "front_net", None), getattr(f2, "front_net", None)
                if a_sure or (n1 is not None and n2 is not None and (n1 > n2)):
                    f2 = turned(f2)
            p2 = yield from think(arms, PP.plan_object_iter(B, f2, box, arms[B].q, placed, obs=obs))
            if p2 is not None:
                plan_b, fb = p2, f2
    res2 = {}
    check_b = fb if sf.category in FRONT_CATS and (not a_sure) else None
    yield from run_arms(arms, {B: pick_place(ctx, arms[B], plan_b, look[B], res2, obs, park=False, fit=check_b)})
    if res2.get("flip") is not None:
        fb = res2["flip"]
        p3 = yield from think(arms, PP.plan_object_iter(B, fb, box, arms[B].q, placed, obs=obs))
        if p3 is not None:
            plan_b = p3
            res2 = {}
            yield from run_arms(arms, {B: pick_place(ctx, arms[B], plan_b, look[B], res2, obs, park=False)})
    if res2.get("ok"):
        pts = fb.points()
        placed.append(
            (plan_b.footprint[0], plan_b.footprint[1], plan_b.place.support + float(pts[:, 2].max() - pts[:, 2].min()))
        )
        return True, box, obs
    return False, box, obs


@register
class PackBox(Program):
    name = "pack_box"

    def match(self, instruction: str, first_obs) -> float:
        return 0.9 if _PAT.search(instruction or "") else 0.0

    def run(self, ctx):
        from . import fit as PF
        from ...perception import seg as SEG
        from . import plan as PP

        SEG.preload()
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        q_home = {a: arms[a].q.copy() for a in K.ARMS}
        look = {}
        for a in K.ARMS:
            q = q_home[a].copy()
            q[0] += LOOK_J1 if a == "left" else -LOOK_J1
            look[a] = q

        def go_look(arm):
            yield from arm.joints_to(look[arm.name], joint_step=0.08, force=True)
            yield from arm.hold(LOOK_SETTLE)

        yield from run_arms(arms, {a: go_look(arms[a]) for a in K.ARMS})
        img = ctx.obs.images.get("cam_head")
        fits = perceive(ctx, img) if img is not None else {}
        obs = None
        if "box" in fits:
            box = PP.box_frame(fits["box"])
            gate = {a: PP.box_reachable(a, box, arms[a].q) for a in K.ARMS}
            obs = PP.BoxObstacle(fits["box"])
            placed = []
            in_box = set()
            last = None
            for cat in PP.ORDER:
                f = fits.get(cat)
                if f is None:
                    continue
                stop = False
                retries = 0
                avoid = []
                for attempt in range(2 + GRASP_RETRIES):
                    if ctx.obs.step + DIRECT_STEPS > STEP_LIMIT - HOME_STEPS:
                        stop = True
                        break
                    if last is not None:
                        yield from run_arms(arms, {last: go_route(arms[last], obs, look[last], arms[last].g)})
                        if arms[last].status != "fail":
                            last = None
                            box, obs = yield from relook(ctx, arms, box, obs, f"before_{cat}")
                    check = f if attempt == 0 else None
                    pref = "left" if f.x < 0 else "right"
                    plan = None
                    st = None
                    other = "right" if pref == "left" else "left"
                    searched = set()
                    for kind, arg in PLAN_STEPS:
                        if kind == "direct":
                            tiers = arg if arg is not None else (("tight", None), ("loose", None))
                            for (mode, tl), a in ((t, a) for t in tiers for a in (pref, other)):
                                if gate[a]:
                                    plan = yield from think(
                                        arms,
                                        PP.plan_object_iter(
                                            a, f, box, arms[a].q, placed, obs=obs, avoid=avoid, centre=mode, tilts=tl
                                        ),
                                    )
                                    if plan is not None:
                                        break
                        elif ctx.obs.step + HANDOVER_STEPS <= STEP_LIMIT - HOME_STEPS and (
                            not (
                                len(arg) > 3
                                and arg[3]
                                and (
                                    ctx.obs.step
                                    + TIMED_HANDOVER
                                    + TIMED_DIRECT * (len(PP.ORDER) - 1 - PP.ORDER.index(cat))
                                    > STEP_LIMIT - HOME_STEPS
                                )
                            )
                        ):
                            pairs, tl, mt = arg[:3]
                            tl = tuple(t for t in (PP.TILTS if tl is None else tl) if (pairs, t, mt) not in searched)
                            searched.update((pairs, t, mt) for t in tl)
                            others = [
                                fits[c].points() for c in PP.ORDER if c != cat and c in fits and (c not in in_box)
                            ]
                            for a_, b_ in (
                                (((pref, other), (other, pref)) if pairs == "two" else ((pref, pref), (other, other)))
                                if tl
                                else ()
                            ):
                                if not gate[b_]:
                                    continue
                                st = yield from think(
                                    arms,
                                    PP.plan_stage_iter(
                                        a_,
                                        b_,
                                        f,
                                        box,
                                        look[a_],
                                        look[b_],
                                        placed,
                                        obs=obs,
                                        others=others,
                                        tilts=tl,
                                        max_tilt=mt,
                                    ),
                                )
                                if st is not None:
                                    break
                        if plan is not None or st is not None:
                            break
                    if plan is None and st is not None:
                        ok, box, obs = yield from handover(ctx, arms, look, st, box, obs, placed, last, fit=check)
                        if isinstance(ok, PF.ObjFit):
                            f = fits[cat] = ok
                            last = st[0].arm
                            continue
                        last = st[2].arm
                        if ok:
                            in_box.add(cat)
                        break
                    if plan is None:
                        break
                    if ctx.obs.step + DIRECT_STEPS > STEP_LIMIT - HOME_STEPS:
                        stop = True
                        break
                    pl = plan.place
                    if last is not None and last != plan.arm:
                        yield from run_arms(arms, {last: go_route(arms[last], obs, look[last], arms[last].g)})
                    res = {}
                    yield from run_arms(
                        arms,
                        {
                            plan.arm: pick_place(
                                ctx, arms[plan.arm], plan, look[plan.arm], res, obs, park=False, fit=check
                            )
                        },
                    )
                    last = plan.arm
                    if res.get("flip") is not None:
                        f = fits[cat] = res["flip"]
                        continue
                    if res.get("ok"):
                        in_box.add(cat)
                        pts = f.points()
                        placed.append(
                            (
                                plan.footprint[0],
                                plan.footprint[1],
                                pl.support + float(pts[:, 2].max() - pts[:, 2].min()),
                            )
                        )
                    elif res.get("fail") in ("empty", "dropped") and retries < GRASP_RETRIES:
                        retries += 1
                        if res.get("closed", 0.0) > 0.004 and res.get("tcp") is not None:
                            avoid.append(PP.pinch_offset(f, res["tcp"]))
                        ar = arms[plan.arm]
                        if res["fail"] == "dropped":
                            yield from run_arms(arms, {plan.arm: go_route(ar, obs, look[plan.arm], ar.g)})
                        if ar.status == "fail":
                            break
                        last = None
                        box, obs, f2 = yield from relook(ctx, arms, box, obs, f"retry_{cat}", cat=cat)
                        if f2 is None or f2.iou < RETRY_IOU:
                            break
                        if float(np.asarray(f2.target_dir) @ np.asarray(f.target_dir)) < 0.0:
                            f2 = turned(f2)
                        inside = over_floor(box, f2, IN_BOX_MARGIN)
                        f = fits[cat] = f2
                        if inside:
                            in_box.add(cat)
                            pts = f2.points()
                            vv = (pts[:, :2] - box.xy) @ box.v
                            placed.append(
                                (
                                    float(vv.min()),
                                    float(vv.max()),
                                    box.floor_z + float(pts[:, 2].max() - pts[:, 2].min()),
                                )
                            )
                            break
                        continue
                    break
                if stop:
                    break

        def go_home(arm):
            yield from arm.set_grip(1.0, n=4)
            if obs is not None:
                arm.status = ""
                yield from go_route(arm, obs, q_home[arm.name], 1.0)
                if arm.status == "ok":
                    return
                yield from go_route(arm, obs, look[arm.name], 1.0)
            yield from joints_fast(arm, q_home[arm.name])

        yield from run_arms(arms, {a: go_home(arms[a]) for a in K.ARMS})
        while True:
            yield [_hold(arms)]
