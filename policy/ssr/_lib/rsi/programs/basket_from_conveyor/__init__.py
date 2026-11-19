from __future__ import annotations

from ... import ASSETS

import json
import math
import time

import numpy as np

from ...core import actions
from . import basket as BK
from . import belt_objects as BO
from ...perception import cameras, robot_mask, seg
from . import photo_board
from . import conveyor_shapes as CS
from ...skills import grasp as GR
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills.runner import Arm, link_arms, run_arms
from .. import Program, register

V_BELT = 0.004
LEAD = 1
LEAD_OBJ = 0
PRE_BACK = 0.1
PRE_UP = 0.02
Y_IN = 0.002
N_ADV = 14
N_CLOSE = 12
APPROACH_GRIP = 0.6
X_MIN_GRASP = -0.12
LIFT = 0.18
N_LIFT = 30
N_LIFT_FOLLOW = 4
HOLD_TCP = 0.05, -0.05
MAX_DQ = 0.15
SCOOP_TCP_Z = 0.1227
SCOOP_HOLD = 40
SCOOP_X = 0.0
SCOOP_DY = 0.1
SCOOP_STOP_DX = 0.045
DONUT_R = 0.048
SCOOP_LIP_X = 0.075
PUSH_BACK = 0.03
PUSH_STEP = 0.005
PUSH_GUARD = -1.0
PUSH_HOVER = 0.14
PUSH_OPEN_LOW = 0.003
PUSH_CLIMB = 0.025, 0.016
PUSH_OPEN_END = 0.005
PUSH_OPEN_YAW_Y = 0.12
SCOOP_WALL = 0.02, 0.26, 0.23
SPAT_UNDER = 0.06
SPAT_CLOSE = 15
SPAT_PITCH = 30.0
GRIP_LOW_OPEN = np.array(
    [
        [0.062, -0.0855, -0.03, 0.03],
        [0.0795, -0.0855, -0.03, 0.03],
        [0.0717, -0.0818, 0.0104, 0.0244],
        [0.0845, -0.0818, -0.0306, 0.0304],
        [0.087, -0.0745, -0.0306, 0.0304],
        [0.09, -0.073, -0.0298, 0.0279],
        [0.102, -0.0673, -0.0254, 0.0159],
        [0.12, -0.0586, -0.0188, 0.012],
        [0.138, -0.0523, -0.0123, 0.0086],
        [0.15, -0.0484, -0.0079, 0.0063],
        [0.1576, -0.0459, -0.0052, 0.0049],
    ]
)
SPAT_TIP = np.array([0.1576, -0.0459])
SPAT_YAWS = (
    0.0,
    15.0,
    20.0,
    30.0,
)
SPAT_LIFT = 0.045
SPAT_IN = 0.09, 0.07, 0.05
SPAT_SLIDE_OFF = 10
SPAT_BACK = (
    0.12,
    0.03,
)
SCOOP_LOWER_STEPS = 25
SCOOP_FAST_LIN, SCOOP_FAST_ANG, SCOOP_FAST_JOINT = 0.02, 5.0, 0.08


def _rot(x_ee, y_ee) -> np.ndarray:
    x = np.asarray(x_ee, float)
    y = np.asarray(y_ee, float)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def finger_extent(R: np.ndarray, gap: float = 0.0) -> tuple[float, float, float]:
    corners = (
        np.array(
            [[a, k * (gap / 2 + t), b] for a in (F_A0, F_A1) for k in (-1, 1) for t in (0.0, F_T) for b in (-F_B, F_B)]
        )
        @ R.T
    )
    i = int(np.argmin(corners[:, 2]))
    return float(corners[i, 2]), float(corners[i, 0]), float(np.max(corners[:, 0]))


def push_rotation_open(yaw_deg: float, pitch_deg: float = 0.0, level: bool = False) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    a = math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)
    c = math.sin(p) * math.cos(y), math.sin(p) * math.sin(y), math.cos(p)
    if level:
        n = math.hypot(math.sin(p), math.cos(p) * math.cos(y))
        c = math.sin(p) / n, 0.0, math.cos(p) * math.cos(y) / n
    return _rot(a, c)


def open_low_extent(R: np.ndarray) -> tuple[float, float]:
    g = GRIP_LOW_OPEN
    pts = np.vstack([np.column_stack([g[:, 0] - K.TCP_OFFSET, g[:, 1], g[:, k]]) for k in (2, 3)])
    tip = R @ np.array([SPAT_TIP[0] - K.TCP_OFFSET, SPAT_TIP[1], 0.0])
    return float((pts @ R.T)[:, 2].min()), float(tip[0])


def push_path_open(R: np.ndarray, xs: float, x_stop: float, y: float) -> list[np.ndarray]:
    z_lo, _, x_fr = finger_extent(R, GAP_MAX)
    back = x_stop - DONUT_R
    lip = xs + SCOOP_LIP_X

    def tcp(x_front: float, h: float) -> np.ndarray:
        return np.array([x_front - x_fr, y, CS.BELT_Z + h - z_lo])

    h1 = PUSH_OPEN_LOW + PUSH_CLIMB[1]
    x_end = max(lip - PUSH_OPEN_END, back + PUSH_CLIMB[0] + 0.005)
    return [
        tcp(back - PUSH_BACK, PUSH_OPEN_LOW),
        tcp(back, PUSH_OPEN_LOW),
        tcp(back + PUSH_CLIMB[0], h1),
        tcp(x_end, h1 + 0.002),
    ]


def spatula_path(R: np.ndarray, x_stop: float, y: float) -> list[np.ndarray]:
    z_lo, x_tip = open_low_extent(R)
    back = x_stop - DONUT_R
    z = CS.BELT_Z + PUSH_OPEN_LOW - z_lo
    return [np.array([back - PUSH_BACK - x_tip, y, z]), np.array([back + SPAT_UNDER - x_tip, y, z])]


def spatula_yaw(x_stop: float, y: float) -> float:
    for into in (SPAT_IN[0], SPAT_IN[1], None):
        for yaw in SPAT_YAWS:
            R = push_rotation_open(yaw, SPAT_PITCH, True)
            p0, p1 = spatula_path(R, x_stop, y)
            up = p1 + np.array([0.0, 0.0, SPAT_LIFT])
            poses = [p0 + np.array([0.0, 0.0, PUSH_HOVER]), p0, p1, up, up + np.array([-SPAT_BACK[0], 0.0, 0.0])]
            if into is not None:
                poses.append(up + np.array([into, 0.0, 0.0]))
            if all(K.solve_near("left", _pose(R, p), np.zeros(6), offset=K.TCP_OFFSET).ok for p in poses):
                return yaw
    return SPAT_YAWS[-1]


def _pose(R, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = R
    t[:3, 3] = p
    return t


@register
class BasketFromConveyor(Program):
    name = "basket_from_conveyor"
    keep = None

    def match(self, instruction, first_obs) -> float:
        s = (instruction or "").lower()
        return 1.0 if "basket" in s and "conveyor" in s and ("image" in s or "board" in s) else 0.0

    def run(self, ctx):
        o = ctx.first
        img = o.images["cam_head"]
        st = o.state
        board = photo_board.read_board(img)
        seg.get()
        cam = cameras.head()
        bad = robot_mask.mask(
            cam,
            st["left_arm_joint_state"],
            st["right_arm_joint_state"],
            float(st["left_ee_joint_state"][0]),
            float(st["right_ee_joint_state"][0]),
        )
        fit = BK.fit(img, cam, bad=bad)
        if fit is None:
            while True:
                yield actions.hold(ctx.obs, n=10)
        arms = {
            "left": Arm("left", ctx, st["left_arm_joint_state"], float(st["left_ee_joint_state"][0])),
            "right": Arm("right", ctx, st["right_arm_joint_state"], float(st["right_ee_joint_state"][0])),
        }
        link_arms(arms)
        clock = {"t": int(o.step)}
        ctx.memory["clock"] = clock
        shared = {"held": False, "basket": fit}
        scoop = bool(False)
        shared["scoop"] = scoop
        tasks = {"right": self._basket(arms["right"], fit, clock, shared)}
        tasks["left"] = self._target(arms["left"], board, fit, clock, shared)
        trace = HoldTrace(ctx)
        for chunk in run_arms(arms, tasks):
            yield chunk
            clock["t"] += 1
            if trace is not None:
                trace.step(arms["left"], shared.get("phase"))
        if trace is not None:
            trace.flush()
        while True:
            yield [actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)]
            clock["t"] += 1

    def _basket(self, arm: Arm, fit: BK.BasketFit, clock, shared):
        zbar = fit.z + BK.HANDLE_TOP - BK.HANDLE_THICK / 2
        yb = fit.y

        def xb(t):
            return fit.x + V_BELT * (t + 1 + LEAD)

        best = None
        for up in (1.0, -1.0):
            R = _rot((0, 1, 0), (0, 0, up))
            p = np.array([max(xb(clock["t"] + 40), X_MIN_GRASP), yb - PRE_BACK, zbar + PRE_UP])
            r = K.solve_near("right", _pose(R, p), arm.q, offset=K.TCP_OFFSET)
            if r.ok and (best is None or np.sum(np.abs(r.q - arm.q)) < best[0]):
                best = float(np.sum(np.abs(r.q - arm.q))), up, R
            if best is not None and shared.get("scoop"):
                break
        if best is None:
            return
        _, up, R = best
        t = clock["t"]
        n = 30
        for _ in range(3):
            t_arr = t + n
            x_arr = xb(t_arr)
            if x_arr < X_MIN_GRASP:
                t_arr = t + n + int(math.ceil((X_MIN_GRASP - x_arr) / V_BELT))
            p = np.array([xb(t_arr), yb - PRE_BACK, zbar + PRE_UP])
            r = K.solve_near("right", _pose(R, p), arm.q, offset=K.TCP_OFFSET)
            plan = motion.joint_move(arm.q, r.q, joint_step=0.06)
            n = max(len(plan.q), 1)
        wait = max(0, t_arr - (t + n))
        t_lift = t_arr + N_ADV + N_CLOSE
        shared["basket_sched"] = {"lift": int(t_lift), "bar_x": float(xb(t_lift))}
        yield from arm.hold(wait)
        yield from arm.follow(motion.joint_move(arm.q, r.q, joint_step=0.06), grip=APPROACH_GRIP, force=True)

        def track(p, grip=None):
            r = K.solve("right", _pose(R, p), arm.q, offset=K.TCP_OFFSET)
            if r.ok:
                dq = np.clip(r.q - arm.q, -MAX_DQ, MAX_DQ)
                arm.q = arm.q + dq
            if grip is not None:
                arm.g = grip
            return r.ok

        for i in range(N_ADV):
            s = (i + 1) / N_ADV
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            p = np.array([xb(clock["t"]), yb - PRE_BACK + s * (PRE_BACK + Y_IN), zbar + PRE_UP * (1 - s)])
            track(p, APPROACH_GRIP)
            yield
        for i in range(N_CLOSE):
            p = np.array([xb(clock["t"]), yb + Y_IN, zbar])
            track(p, APPROACH_GRIP * max(0.0, 1.0 - (i + 1) / (N_CLOSE * 0.6)))
            yield
        x_end = xb(clock["t"] + N_LIFT_FOLLOW)
        for i in range(N_LIFT):
            s = (i + 1) / N_LIFT
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            x = xb(clock["t"]) if i < N_LIFT_FOLLOW else x_end
            track(np.array([x, yb + Y_IN, zbar + LIFT * s]), 0.0)
            yield
        shared["basket_lifted"] = True
        if shared.get("scoop"):
            yield from self._scoop_probe(arm, R, up, clock, shared)
        R_hold = _rot((-1, 0, 0), (0, 0, up))
        goal = _pose(R_hold, np.array([HOLD_TCP[0], HOLD_TCP[1], zbar + LIFT]))
        plan = motion.line("right", arm.q, goal, offset=K.TCP_OFFSET, lin_step=0.006, ang_step=math.radians(2.0))
        if not plan.ok:
            r = K.solve_near("right", goal, arm.q, offset=K.TCP_OFFSET)
            plan = motion.joint_move(arm.q, r.q, joint_step=0.03) if r.ok else motion.Plan()
        yield from arm.follow(plan, grip=0.0, force=True)
        shared["held"] = True
        shared["hold_tcp"] = arm.tcp().copy()

    def _scoop_probe(self, arm: Arm, R: np.ndarray, up: float, clock=None, shared=None):
        shared = {} if shared is None else shared
        p0 = arm.tcp().copy()
        p0[0] = SCOOP_X
        R_s = _rot((0, 1, 0), (-up, 0, 0))

        def go(R_g, p, kind, fast=False):
            kw = (
                dict(lin_step=SCOOP_FAST_LIN, ang_step=math.radians(SCOOP_FAST_ANG), joint_step=SCOOP_FAST_JOINT)
                if fast
                else dict(lin_step=0.01, ang_step=math.radians(3.0))
            )
            plan = motion.line("right", arm.q, _pose(R_g, p), offset=K.TCP_OFFSET, **kw)
            if not plan.ok:
                return False
            yield from arm.follow(plan, grip=0.0, force=True)
            return True

        push = clock is not None
        if push:
            while shared.get("scoop_xy") is None and (not shared.get("scoop_fail")):
                if clock["t"] > STEP_LIMIT - 300:
                    shared["scoop_fail"] = True
                    return
                yield
            if shared.get("scoop_fail"):
                return
            p0[0], p0[1] = shared["scoop_xy"]
        if not (yield from go(R_s, p0, "scoop_turn", fast=push)):
            shared["scoop_fail"] = True
            return
        p1 = np.array([p0[0], p0[1], CS.BELT_Z + SCOOP_TCP_Z])
        if push:
            tr = shared["scoop_track"]
            tracker = shared.get("scoop_tracker")
            while True:
                t_land = clock["t"] + SCOOP_LOWER_STEPS
                x_front = tr.predict(t_land) + DONUT_R
                if x_front > p0[0] + SCOOP_LIP_X - 0.01:
                    shared["scoop_fail"] = True
                    yield from go(R, p0, "scoop_right")
                    return
                under = (
                    []
                    if tracker is None
                    else [
                        o
                        for o in tracker.tracks
                        if o is not tr
                        and Tracker.moving(o)
                        and (
                            p0[0] + SCOOP_WALL[0] < o.predict(t_land) < p0[0] + SCOOP_WALL[1]
                            and abs(o.y - p0[1]) < SCOOP_WALL[2]
                        )
                    ]
                )
                if not under:
                    break
                yield
        if (yield from go(R_s, p1, "scoop_lower", fast=push)):
            shared["scoop_down"] = int(clock["t"]) if clock is not None else 0
            if push:
                while not shared.get("push_done") and clock["t"] < STEP_LIMIT - 150:
                    yield
            else:
                yield from arm.hold(SCOOP_HOLD)
            yield from go(R_s, p0, "scoop_raise", fast=push)
        else:
            shared["scoop_fail"] = True
        yield from go(R, p0, "scoop_right")

    def _push(self, arm: Arm, tr: "Track", clock, shared):
        y_d = float(tr.y)
        xs = SCOOP_X
        ys = float(np.clip(y_d - SCOOP_DY, -0.12, 0.03))
        x_stop = xs + SCOOP_STOP_DX
        yaw = 0.0 if y_d <= PUSH_OPEN_YAW_Y else 15.0
        R_p = push_rotation_open(yaw)
        path = push_path_open(R_p, xs, x_stop, y_d)
        R_p = push_rotation_open(spatula_yaw(x_stop, y_d), SPAT_PITCH, True)
        path = spatula_path(R_p, x_stop, y_d)
        above = path[0] + np.array([0.0, 0.0, PUSH_HOVER])
        r = None
        for seed in (arm.q, np.zeros(6)):
            r = K.solve_near("left", _pose(R_p, above), seed, offset=K.TCP_OFFSET)
            if r.ok:
                break
        if not r.ok:
            shared["scoop_fail"] = True
            return
        shared["scoop_track"] = tr
        shared["scoop_xy"] = xs, ys
        while shared.get("scoop_down") is None and (not shared.get("scoop_fail")) and (clock["t"] < STEP_LIMIT - 200):
            yield
        if shared.get("scoop_down") is None:
            shared["push_done"] = True
            return
        yield from arm.follow(
            motion.joint_move(arm.q, r.q, joint_step=TRANSIT_STEP), grip=1.0, force=True, guard=PUSH_GUARD
        )
        t_stop = tr.step + (x_stop - tr.x) / V_BELT
        while clock["t"] < t_stop + 8 and clock["t"] < STEP_LIMIT - 160:
            yield
        ok = True
        for k, tcp in enumerate(path):
            plan = motion.line(
                "left",
                arm.q,
                _pose(R_p, tcp),
                offset=K.TCP_OFFSET,
                lin_step=0.01 if k == 0 else PUSH_STEP,
                ang_step=math.radians(3.0),
            )
            if not plan.ok:
                ok = False
                break
            yield from arm.follow(plan, grip=1.0, force=True, guard=PUSH_GUARD)
        yield from self._spatula_take(arm, R_p, ok, clock, shared)

    def _spatula_into(self, arm: Arm, R_p: np.ndarray, held: bool, shared):
        def line(d, grip, kind=None):
            plan = motion.line(
                "left", arm.q, _pose(R_p, arm.tcp() + d), offset=K.TCP_OFFSET, lin_step=0.01, ang_step=math.radians(3.0)
            )
            if not plan.ok:
                return False
            yield from arm.follow(plan, grip=grip, force=True, guard=PUSH_GUARD)
            return True

        if held:
            yield from line(np.array([0.0, 0.0, SPAT_LIFT]), 0.0, "spatula_line_failed")
            for d in SPAT_IN:
                if (yield from line(np.array([d, 0.0, 0.0]), 0.0)):
                    break
        yield from arm.set_grip(1.0, n=6)
        yield from arm.hold(SPAT_SLIDE_OFF)
        if not held:
            yield from line(np.array([0.0, 0.0, SPAT_LIFT]), 1.0, "spatula_line_failed")
        yield from line(np.array([-SPAT_BACK[0], 0.0, 0.0]), 1.0, "spatula_line_failed")
        yield from line(np.array([0.0, 0.0, 0.06]), 1.0)
        shared["push_done"] = True
        R_ready = GR.grasp_rotation(math.radians(READY_PHI), math.radians(READY_TILT))
        r = K.solve_near("left", _pose(R_ready, np.array(READY_TCP)), arm.q, offset=K.TCP_OFFSET)
        if r.ok:
            yield from arm.follow(motion.joint_move(arm.q, r.q, joint_step=TRANSIT_STEP), grip=1.0, force=True)

    def _spatula_take(self, arm: Arm, R_p: np.ndarray, ok: bool, clock, shared):
        if ok:
            yield from arm.set_grip(0.0, n=SPAT_CLOSE)
            yield from arm.hold(4)
        g_obs = float(arm.ctx.obs.state["left_ee_joint_state"][0])
        held = g_obs >= G_EMPTY
        yield from self._spatula_into(arm, R_p, held, shared)

    def _target(self, arm: Arm, board, bfit: BK.BasketFit, clock, shared):
        if board is None or board.asset not in CS.assets():
            return
        tracker = Tracker(cameras.head(), board.asset)
        sched = shared.get("basket_sched")
        self.keep = None if sched is None else lambda t: basket_keepout(t, bfit, sched)
        R_ready = GR.grasp_rotation(math.radians(READY_PHI), math.radians(READY_TILT))
        ik_r = K.solve_near("left", _pose(R_ready, np.array(READY_TCP)), arm.q, offset=K.TCP_OFFSET)
        to_ready = list(motion.joint_move(arm.q, ik_r.q, joint_step=TRANSIT_STEP).q) if ik_r.ok else []
        excluded, regrasp = set(), None
        for n_try in range(1 + N_REGRASP):
            plan = None
            while plan is None:
                if to_ready and arm._may_step(to_ready[0]):
                    arm.q = np.asarray(to_ready.pop(0), dtype=float)
                o = arm.ctx.obs
                st = o.state
                step = int(o.step)
                if regrasp is not None and regrasp[0] not in tracker.tracks:
                    tracker.tracks.append(regrasp[0])
                if tracker.update(step, o.images["cam_head"], st, bfit, shared):
                    if regrasp is None:
                        tr = tracker.choose(step)
                    else:
                        tr = regrasp[0] if regrasp[0].step > regrasp[1] else None
                    if tr is not None and shared.get("scoop"):
                        shared["scoop_tracker"] = tracker
                        yield from self._push(arm, tr, clock, shared)
                        return
                    if tr is not None and getattr(tr, "retry_at", -1) > step:
                        tr = None
                    if tr is not None:
                        t_plan = time.perf_counter()
                        plan = self._plan_grasp(
                            arm,
                            tr,
                            board.asset,
                            clock["t"],
                            tracker.obstacles(tr),
                            exclude=excluded,
                            t_lead=30 if regrasp is None else REGRASP_LEAD,
                        )
                        if plan is None and getattr(tr, "ring", False):
                            tr.retry_at = step + RING_RETRY_STEPS
                        elif plan is None and regrasp is not None:
                            return
                        elif plan is None:
                            tr.rejected = True
                        else:
                            plan["plan_s"] = round(time.perf_counter() - t_plan, 2)
                if regrasp is not None and plan is None and (step - regrasp[1] > REGRASP_WAIT):
                    return
                if clock["t"] > STEP_LIMIT - 120:
                    return
                if plan is None:
                    yield
            g = plan

            def obj_x(t):
                return g["x0"] + V_BELT * (t + 1 + LEAD_OBJ - g["step0"])

            def track(p, grip=None):
                r = K.solve("left", _pose(R, p), arm.q, offset=K.TCP_OFFSET)
                if r.ok:
                    arm.q = arm.q + np.clip(r.q - arm.q, -MAX_DQ, MAX_DQ)
                if grip is not None:
                    arm.g = grip
                return r.ok

            for attempt in range(2):
                R = g["R"]
                a = R[:, 0]
                t_pre = g["t_close"] - N_ADV_OBJ
                wait = max(0, t_pre - g["n_transit"] - clock["t"])
                yield from self._watching(arm.hold(wait), arm, tracker, bfit, shared)
                p_g = np.array([obj_x(t_pre), g["y"], g["z"]])
                p_pre = p_g - PRE_DIST_OBJ * a + np.array([0.0, 0.0, PRE_UP_OBJ])
                r = K.solve_near("left", _pose(R, p_pre), arm.q, offset=K.TCP_OFFSET)
                if not r.ok:
                    return
                g_open = g["grip_open"]
                yield from self._watching(
                    arm.follow(motion.joint_move(arm.q, r.q, joint_step=TRANSIT_STEP), grip=g_open, force=True),
                    arm,
                    tracker,
                    bfit,
                    shared,
                )
                obst = tracker.obstacles(tr)
                if attempt or len(obst) // NB_POINTS <= g["n_obst"]:
                    break
                p_c = np.array([obj_x(g["t_close"]), g["y"], g["z"]])
                ik_c = K.solve("left", _pose(R, p_c), arm.q, offset=K.TCP_OFFSET)
                q_c = ik_c.q if ik_c.ok else arm.q
                clear = self._approach_clear(R, p_c, q_c, g_open, obst + np.array([p_c[0] - g["rel"][0], g["cy"], 0.0]))
                if clear >= 0.0:
                    break
                break
            d_pre = PRE_DIST_OBJ * a - np.array([0.0, 0.0, PRE_UP_OBJ])
            for i in range(N_ADV_OBJ):
                s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / N_ADV_OBJ)
                p = np.array([obj_x(clock["t"]), g["y"], g["z"]]) - (1 - s) * d_pre
                track(p, g_open)
                yield
            for i in range(N_CLOSE_OBJ):
                track(
                    np.array([obj_x(clock["t"]), g["y"], g["z"]]),
                    g_open * max(0.0, 1.0 - (i + 1) / (N_CLOSE_OBJ * 0.7)),
                )
                yield
            shared["phase"] = "lift"
            x_cmd = obj_x(clock["t"] - 1)
            n_lift = lift_steps(board.asset)
            for i in range(n_lift):
                if n_try < N_REGRASP and float(arm.ctx.obs.state["left_ee_joint_state"][0]) < G_EMPTY:
                    break
                s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n_lift)
                if i < N_LIFT_DECEL:
                    x_cmd += V_BELT * (0.5 + 0.5 * math.cos(math.pi * (i + 1) / N_LIFT_DECEL))
                track(np.array([x_cmd, g["y"], g["z"] + LIFT_OBJ * s]), 0.0)
                yield
            g_lift = float(arm.ctx.obs.state["left_ee_joint_state"][0])
            if g_lift >= G_EMPTY or n_try >= N_REGRASP:
                break
            excluded.add((g["phi"], g["tilt"], g["roll"], g["side"]) + ((g["lateral"],) if g.get("ring") else ()))
            p0 = arm.tcp()
            for i in range(N_BACK):
                s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / N_BACK)
                track(p0 + np.array([V_BELT * (i + 1), 0.0, s * BACK_UP]) - s * BACK_DIST * a, g_open)
                yield
            regrasp = tr, int(arm.ctx.obs.step)
            shared["phase"] = None
        x_w = float(np.clip(arm.tcp()[0], -0.45, -0.3))
        p_w = np.array([x_w, Y_CARRY, Z_CARRY])
        heavy = bool(CS.shape(board.asset).mass >= HEAVY_MASS)

        def held_line(q0, goal):
            return held_object_line(q0, goal, heavy)

        R_l = arm.T[:3, :3]
        sup = False
        routes = self._carry_routes(
            arm.q, arm.T[:3, :3], x_w, heavy, N_ROUTES_HEAVY if heavy else 3, support=sup, t0=clock["t"]
        )
        if not routes and sup:
            sup = False
            routes = self._carry_routes(arm.q, arm.T[:3, :3], x_w, heavy, N_ROUTES_HEAVY if heavy else 3, t0=clock["t"])
        if not routes:
            routes, p_w = self._carry_fallback(arm, x_w, heavy)
        if not routes:
            return

        def route_cost(r):
            turn = math.degrees(K.rot_angle(R_l, r[2]))
            return r[0] + (HEAVY_TURN_STEPS * max(0.0, turn - HEAVY_TURN_OK) / 10.0 if heavy else 0.0)

        _, _, R_c, plan, _ = min(routes, key=route_cost)
        shared["phase"] = "carry"
        yield from arm.follow(plan, grip=0.0, force=True)
        yield from self._drop(arm, R_c, g["L"], p_w, heavy, sup, clock, shared)

    def _carry_fallback(self, arm: Arm, x_w: float, heavy: bool):
        p_d0 = np.array([HOLD_TCP[0] - DROP_FROM_BAR, HOLD_TCP[1], Z_DROP])
        p_w = np.array([x_w, Y_CARRY, Z_CARRY])
        routes = []
        tries = [(MAX_JOINT_FALLBACK, x_w)]
        xs = [x_w] + [x for x in FALLBACK_WAIT_X if abs(x - x_w) > 1e-06]
        tries += [(MAX_JOINT_FALLBACK_LAST, x) for x in xs]
        t_fb = time.perf_counter()
        for n_t, (lim, x_f) in enumerate(tries):
            if n_t and time.perf_counter() - t_fb > FALLBACK_BUDGET_S:
                break
            p_f = np.array([x_f, Y_CARRY, Z_CARRY])
            for k, Rc in enumerate(carry_rotations(arm.T[:3, :3])):
                if n_t and k >= FALLBACK_ROTS:
                    break
                ik_w = K.solve_near("left", _pose(Rc, p_f), arm.q, offset=K.TCP_OFFSET, n_random=3 if n_t == 0 else 2)
                if not ik_w.ok or float(np.max(np.abs(ik_w.q - arm.q))) > lim:
                    continue
                if n_t and self.keep is not None and keepout_hit(ik_w.q, self.keep(10**6), 0.0):
                    continue
                p2 = held_object_line(ik_w.q, _pose(Rc, p_d0), heavy)
                if p2.ok:
                    p1 = motion.joint_move(arm.q, ik_w.q, joint_step=HEAVY_JOINT_STEP if heavy else 0.05)
                    routes.append((len(p1.q) + len(p2.q), k, Rc, p1, len(p2.q)))
                    break
            if routes:
                p_w = p_f
                break
        return routes, p_w

    def _drop(self, arm: Arm, R_c: np.ndarray, L: np.ndarray, p_w: np.ndarray, heavy: bool, sup: bool, clock, shared):
        def held_line(q0, goal):
            return held_object_line(q0, goal, heavy)

        shared["phase"] = "wait"
        while not shared.get("held"):
            if clock["t"] > STEP_LIMIT - 40:
                return
            yield
        hb = shared["hold_tcp"]
        if sup:
            drops = sorted(
                [
                    (R_d, p_d, {**info, "roll": s}, K.rot_z(math.radians(info["turn"])) @ R_c)
                    for s in SUPPORT_ROLLS
                    for R_d, p_d, info in drop_poses(R_c @ K.rot_x(-math.radians(s)), L, hb)
                ],
                key=lambda d: (round(d[2]["overflow"], 3), abs(d[2]["turn"])),
            )
        else:
            drops = [(R_d, p_d, info, R_d) for R_d, p_d, info in drop_poses(R_c, L, hb)]
        moved = False
        q_right, g_right = arm.other.q, arm.other.g
        feasible = []
        for R_d, p_d, info, R_m in drops[: N_DROP_TRY * (2 if sup else 1)]:
            plan_d = held_line(arm.q, _pose(R_m, p_d))
            if not plan_d.ok:
                continue
            plan_r = None
            if R_m is not R_d:
                plan_r = held_line(plan_d.q[-1], _pose(R_d, p_d))
                if not plan_r.ok:
                    continue
            qs = plan_d.q[::3] + [plan_d.q[-1]] + (plan_r.q[::3] + [plan_r.q[-1]] if plan_r is not None else [])
            clear = min(CO.arm_distance(q, q_right, 0.0, g_right) for q in qs)
            feasible.append((clear, R_d, p_d, info, plan_d, plan_r))
            if clear >= DROP_MIN_CLEAR:
                break
        if feasible:
            ok_clear = [f for f in feasible if f[0] >= DROP_MIN_CLEAR]
            clear, R_d, p_d, info, plan_d, plan_r = ok_clear[0] if ok_clear else max(feasible, key=lambda f: f[0])
            shared["phase"] = "place"
            yield from arm.follow(plan_d, grip=0.0, force=True, guard=DROP_GUARD)
            if plan_r is not None:
                shared["phase"] = "roll"
                yield from arm.follow(plan_r, grip=0.0, force=True, guard=DROP_GUARD)
            R_c, moved = R_d, True
        if not moved:
            p_d = np.array([hb[0] - DROP_FROM_BAR, hb[1], Z_DROP])
            if not (yield from self._line(arm, R_c, p_d, grip=0.0, heavy=heavy)):
                return
            if sup:
                for s in SUPPORT_ROLLS:
                    R_b = R_c @ K.rot_x(-math.radians(s))
                    if (yield from self._line(arm, R_b, p_d, grip=0.0, heavy=heavy)):
                        R_c = R_b
                        break
        z_rel = release_height(R_c, L, hb[2], s_bar=float(hb[0] - arm.tcp()[0]))
        p_r = np.array([p_d[0], p_d[1], z_rel])
        shared["phase"] = "lower"
        lowered = bool(z_rel < p_d[2] - 0.005) and (yield from self._line(arm, R_c, p_r, grip=0.0, heavy=heavy))
        shared["phase"] = "open"
        yield from arm.set_grip(1.0, n=8)
        yield from arm.hold(10)
        shared["phase"] = None
        if lowered:
            yield from self._line(arm, R_c, p_d, grip=1.0)
        yield from self._line(arm, R_c, p_w, grip=1.0)

    def _watching(self, gen, arm: Arm, tracker: "Tracker", bfit, shared):
        for _ in gen:
            yield
            o = arm.ctx.obs
            tracker.update(int(o.step), o.images["cam_head"], o.state, bfit, shared)

    def _approach_clear(self, Rg, p_g, q_g, grip: float, obst_w: np.ndarray) -> float:
        d_pre = PRE_DIST_OBJ * Rg[:, 0] - np.array([0.0, 0.0, PRE_UP_OBJ])
        worst = approach_clearance(q_g, grip, obst_w)
        q = q_g
        for s in (0.5, 1.0):
            r = K.solve("left", _pose(Rg, p_g - s * d_pre), q, offset=K.TCP_OFFSET)
            if r.ok:
                q = r.q
                worst = min(worst, approach_clearance(q, grip, obst_w))
        return worst

    def _line(self, arm: Arm, R, p, grip, guard: float | None = None, heavy: bool = False):
        goal = _pose(R, p)
        if heavy:
            plan = motion.line(
                arm.name,
                arm.q,
                goal,
                offset=K.TCP_OFFSET,
                lin_step=HEAVY_CARRY_LIN_STEP,
                ang_step=math.radians(HEAVY_ANG_STEP),
                joint_step=HEAVY_JOINT_STEP,
            )
        else:
            plan = motion.line(arm.name, arm.q, goal, offset=K.TCP_OFFSET, lin_step=0.008, ang_step=math.radians(3.0))
        if not plan.ok:
            r = K.solve_near(arm.name, goal, arm.q, offset=K.TCP_OFFSET)
            near = bool(r.ok) and float(np.max(np.abs(r.q - arm.q))) < MAX_JOINT_FALLBACK
            if not near:
                return False
            plan = motion.joint_move(arm.q, r.q, joint_step=min(HEAVY_JOINT_STEP, 0.04) if heavy else 0.04)
        yield from arm.follow(plan, grip=grip, force=True, guard=DROP_GUARD if guard is None else guard)
        return True

    def _route_hits(self, p1, t0) -> bool:
        if t0 is None:
            return any(keepout_hit(q, self.keep(10**6), 0.0) for q in p1.q[::2])
        if keepout_hit(p1.q[-1], self.keep(10**6), 0.0):
            return True
        return any(keepout_hit(q, self.keep(int(t0) + 2 * i), 0.0) for i, q in enumerate(p1.q[::2]))

    def _carry_routes(
        self,
        q0,
        R0,
        x_w: float,
        heavy: bool,
        limit: int,
        support: bool = False,
        keep: bool = True,
        t0: int | None = None,
    ) -> list:
        p_w = np.array([x_w, Y_CARRY, Z_CARRY])
        p_d0 = np.array([HOLD_TCP[0] - DROP_FROM_BAR, HOLD_TCP[1], Z_DROP])
        routes = []
        for k, Rc in enumerate(support_rotations(R0) if support else carry_rotations(R0)):
            if k >= 24 or len(routes) >= limit:
                break
            p1 = held_object_line(q0, _pose(Rc, p_w), heavy)
            if not p1.ok:
                continue
            p2 = held_object_line(p1.q[-1], _pose(Rc, p_d0), heavy)
            if p2.ok and (not (keep and self.keep is not None and self._route_hits(p1, t0))):
                routes.append((len(p1.q) + len(p2.q), k, Rc, p1, len(p2.q)))
        return routes

    def _plan_grasp(
        self,
        arm: Arm,
        tr: "Track",
        asset: str,
        t_now: int,
        obst: np.ndarray | None = None,
        exclude=frozenset(),
        t_lead: int = 30,
    ):
        if obst is None:
            obst = np.zeros((0, 3))
        sh = CS.shape(asset)
        f = tr.fit
        pts = CS.posed(sh, 0.0, 0.0, f.yaw)
        heavy = bool(sh.mass >= HEAVY_MASS)
        deep = bool(sh.mass >= DEEP_MASS)
        cands = grasp_candidates(pts, sh.extent[2], heavy=heavy, deep=deep)
        if not cands:
            cands = ring_candidates(pts)
            tr.ring = bool(cands)
        if not cands:
            cands = grasp_candidates(pts, sh.extent[2], lateral=CHORD_OFFSETS, heavy=heavy, deep=deep)
        if not cands:
            return None
        x_now = f.x + V_BELT * (t_now - tr.step)
        t0 = t_now + t_lead
        if x_now + V_BELT * (t0 - t_now) < X_GRASP[0]:
            t0 = t_now + int(math.ceil((X_GRASP[0] - x_now) / V_BELT))
        best = None
        x_mid = float(np.clip(x_now + V_BELT * (t0 + 40 - t_now), X_GRASP[0], X_GRASP[1]))
        tried, reach, blocked = set(), [], []
        for c in cands:
            key = (c["phi"], c["tilt"], c["roll"], c["side"]) + ((c["lateral"],) if c.get("ring") else ())
            if key in tried or key in exclude:
                continue
            if len(tried) >= (N_ROT_CHECK_RING if c.get("ring") else N_ROT_CHECK) or len(reach) >= N_GRASP_POOL:
                break
            tried.add(key)
            p_mid = np.array([x_mid + c["tcp"][0], f.y + c["tcp"][1], c["tcp"][2]])
            r1 = K.solve_near("left", _pose(c["R"], p_mid), arm.q, offset=K.TCP_OFFSET, n_random=2)
            if r1.ok:
                ok = (
                    self._approach_clear(
                        c["R"], p_mid, r1.q, grip_for_gap(c["gap"]), obst + np.array([x_mid, f.y, 0.0])
                    )
                    >= 0.0
                )
                (reach if ok else blocked).append(c)
        n_blocked = len(blocked)
        reach += blocked[: max(0, N_GRASP_POOL - len(reach))]
        if not reach:
            seen = set()
            for c in cands:
                key = (c["phi"], c["tilt"], c["roll"], c["side"]) + ((c["lateral"],) if c.get("ring") else ())
                if key not in seen and key not in exclude and (len(seen) < N_GRASP_IK):
                    seen.add(key)
                    reach.append(c)
        for c in reach:
            Rg, rel = c["R"], c["tcp"]
            for t_close in range(t0, t0 + 160, 10):
                x = x_now + V_BELT * (t_close + 1 + LEAD_OBJ - t_now)
                if x > X_GRASP[1]:
                    break
                p_g = np.array([x + rel[0], f.y + rel[1], rel[2]])
                ik_g = K.solve_near("left", _pose(Rg, p_g), arm.q, offset=K.TCP_OFFSET, n_random=3)
                if not ik_g.ok:
                    continue
                lift_ok = True
                p_e = p_g + np.array([V_BELT * (N_CLOSE_OBJ + N_LIFT_DECEL // 2), 0.0, 0.0])
                ik_e = K.solve("left", _pose(Rg, p_e), ik_g.q, offset=K.TCP_OFFSET)
                ik_l = K.solve(
                    "left",
                    _pose(Rg, p_e + np.array([0.0, 0.0, LIFT_OBJ])),
                    ik_e.q if ik_e.ok else ik_g.q,
                    offset=K.TCP_OFFSET,
                )
                lift_ok = bool(ik_e.ok and ik_l.ok)
                if self.keep is not None:
                    hit = None
                    p_e = p_g + np.array([V_BELT * (N_CLOSE_OBJ + 4), 0.0, 0.0])
                    for p_k, t_k in (
                        (p_e, t_close + N_CLOSE_OBJ + 4),
                        (p_e + np.array([0.0, 0.0, LIFT_OBJ]), t_close + N_CLOSE_OBJ + lift_steps(asset)),
                    ):
                        ik_k = K.solve("left", _pose(Rg, p_k), ik_g.q, offset=K.TCP_OFFSET)
                        q_k = ik_k.q if ik_k.ok else ik_g.q
                        hit = hit or keepout_hit(q_k, self.keep(t_k), grip_for_gap(c["gap"]))
                    if hit:
                        break
                p_pre = (
                    p_g
                    - np.array([V_BELT * N_ADV_OBJ, 0.0, 0.0])
                    - PRE_DIST_OBJ * Rg[:, 0]
                    + np.array([0.0, 0.0, PRE_UP_OBJ])
                )
                ik_p = K.solve_near("left", _pose(Rg, p_pre), arm.q, offset=K.TCP_OFFSET, n_random=3)
                if not ik_p.ok:
                    continue
                n_tr = len(motion.joint_move(arm.q, ik_p.q, joint_step=TRANSIT_STEP).q)
                if t_close - N_ADV_OBJ - n_tr < t_now:
                    continue
                clear = self._approach_clear(Rg, p_g, ik_g.q, grip_for_gap(c["gap"]), obst + np.array([x, f.y, 0.0]))
                late = late_close_cost(x + V_BELT * (N_CLOSE_OBJ - 1))
                score = (
                    t_close
                    + COST_STEPS * c["cost"]
                    + (NB_PENALTY if clear < 0.0 else 0.0)
                    + (0.0 if lift_ok else LIFT_PENALTY)
                    + late
                )
                if heavy or best is None or score < best[0]:
                    plan = {
                        "R": Rg,
                        "t_close": int(t_close),
                        "x": float(p_g[0]),
                        "y": float(p_g[1]),
                        "z": float(p_g[2]),
                        "phi": c["phi"],
                        "tilt": c["tilt"],
                        "roll": c["roll"],
                        "side": c["side"],
                        "depth": c["depth"],
                        "lateral": c.get("lateral", 0.0),
                        "ring": bool(c.get("ring")),
                        "wedge": round(float(c.get("wedge", 0.0)), 4),
                        "dwedge": round(float(c.get("dwedge", 0.0)), 4),
                        "engage": round(float(c.get("engage", 0.0)), 4),
                        "area": round(float(c.get("area", 0.0)) * 10000.0, 2),
                        "hang": round(float(c.get("hang", 0.0)) * 1000.0, 1),
                        "width": round(c["width"], 4),
                        "gap": round(c["gap"], 4),
                        "grip_open": grip_for_gap(c["gap"]),
                        "n_transit": n_tr,
                        "L": (pts - c["tcp"]) @ Rg,
                        "x0": float(x_now + rel[0]),
                        "step0": int(t_now),
                        "track": tr.id,
                        "n_cands": len(cands),
                        "clear": round(float(clear), 4),
                        "rel": np.asarray(rel, float),
                        "cy": float(f.y),
                        "fit": [
                            round(float(f.x), 4),
                            round(float(f.y), 4),
                            round(math.degrees(float(f.yaw)), 1),
                            int(tr.step),
                        ],
                        "n_blocked": n_blocked,
                        "n_obst": int(len(obst) // max(1, NB_POINTS)),
                        "lift_ok": lift_ok,
                        "late": round(float(late), 1),
                    }
                    if best is None or score < best[0]:
                        best = score, plan
                break
        if best is None:
            return None
        return best[1]


F_A0, F_A1 = -0.055, 0.013
F_B = 0.011
F_T = 0.01
PALM_A, PALM_B = -0.08, 0.025
CLEAR = 0.004
GAP_MAX = 2 * (0.0004 + 0.054 - 0.01)
GAP_SPARE = 0.012
TIP_CLEAR = 0.003
PAD_COVER = 0.008
ENGAGE_MIN = 0.006
ENGAGE_REF, W_ENG = 0.015, 0.5
PATCH_DEPTH = 0.003
A_REF = 0.012 * 0.018
TALL = 0.06
TOP_MARGIN = 0.03
W_RATIO, W_AREA, W_TOP = 1.0, 0.5, 0.6
GRASP_ROLLS = 0, 25, -25
ROLL_PENALTY = 2.0
SIDE_TILTS = 30, 45
SIDE_DEPTHS = 0.0, 0.02
N_GRASP_IK = 10
COST_STEPS = 40


def grip_for_gap(gap: float) -> float:
    return float(np.clip((gap / 2 - 0.0004 + 0.01) / 0.054, 0.0, 1.0))


def grasp_check(L: np.ndarray, d_rel: np.ndarray, gap: float, n_s: int = 6):
    for s in np.linspace(0.0, 1.0, n_s):
        P = L + (1.0 - s) * d_rel
        a, c, b = P[:, 0], P[:, 1], P[:, 2]
        ac = np.abs(c)
        fin = (
            (np.abs(b) < F_B + CLEAR)
            & (a > F_A0 - CLEAR)
            & (a < F_A1 + CLEAR)
            & (ac > gap / 2 - CLEAR)
            & (ac < gap / 2 + F_T + CLEAR)
        )
        palm = (np.abs(b) < PALM_B) & (a > PALM_A) & (a < F_A0 + CLEAR) & (ac < gap / 2 + F_T)
        if fin.any() or palm.any():
            return None
    a, c, b = L[:, 0], L[:, 1], L[:, 2]
    pad = (np.abs(b) < F_B) & (a > F_A0 + 0.01) & (a < F_A1) & (np.abs(c) < gap / 2)
    if np.count_nonzero(pad) < 6:
        return None
    lo, hi = float(c[pad].min()), float(c[pad].max())
    if lo > -0.004 or hi < 0.004:
        return None
    cover = float(np.ptp(b[pad]))
    if cover < PAD_COVER:
        return None
    area = 1.0
    for side in (c > 0.0, c < 0.0):
        sp = pad & side
        cs = np.abs(c[sp])
        if cs.size == 0:
            return None
        patch = cs > cs.max() - PATCH_DEPTH
        area = min(area, (float(np.ptp(a[sp][patch])) + 0.002) * (float(np.ptp(b[sp][patch])) + 0.002))
    mid = pad & (np.abs(b) < F_B / 2)
    engage = float(np.ptp(a[mid])) if np.count_nonzero(mid) >= 3 else 0.0
    if engage < ENGAGE_MIN:
        return None
    return hi - lo, 0.5 * (hi + lo), cover, area, engage


def tip_wedge(L: np.ndarray, gap: float, look: float = 0.01) -> float:
    a, c, b = L[:, 0], L[:, 1], L[:, 2]
    inb = np.abs(b) < F_B
    pad = inb & (a > F_A0 + 0.01) & (a <= F_A1) & (np.abs(c) < gap / 2)
    out = inb & (a > F_A1) & (a <= F_A1 + look)
    worst = -1.0
    for sgn in (1.0, -1.0):
        sp, so = pad & (sgn * c > 0), out & (sgn * c > 0)
        if not sp.any() or not so.any():
            continue
        worst = max(worst, float((sgn * c[so]).max() - (sgn * c[sp]).max()))
    return worst


def down_wedge(L: np.ndarray, R: np.ndarray, gap: float, look: float = 0.01) -> float:
    a, c, b = L[:, 0], L[:, 1], L[:, 2]
    bs = 1.0 if R[2, 2] < 0 else -1.0
    ina = (a > F_A0 + 0.01) & (a <= F_A1)
    pad = ina & (np.abs(b) < F_B) & (np.abs(c) < gap / 2)
    out = ina & (bs * b >= F_B) & (bs * b < F_B + look) & (np.abs(c) < gap / 2 + 0.03)
    worst = -1.0
    for sgn in (1.0, -1.0):
        sp, so = pad & (sgn * c > 0), out & (sgn * c > 0)
        if sp.any() and so.any():
            worst = max(worst, float((sgn * c[so]).max() - (sgn * c[sp]).max()))
    return worst


def contact_above_centroid(L: np.ndarray, R: np.ndarray, gap: float) -> float:
    a, c, b = L[:, 0], L[:, 1], L[:, 2]
    pad = (np.abs(b) < F_B) & (a > F_A0 + 0.01) & (a < F_A1) & (np.abs(c) < gap / 2)
    centres = []
    for side in (c > 0.0, c < 0.0):
        sp = pad & side
        if not sp.any():
            return 0.0
        cs = np.abs(c[sp])
        centres.append(L[sp][cs > cs.max() - PATCH_DEPTH].mean(axis=0))
    u = centres[0] - centres[1]
    u = u / max(float(np.linalg.norm(u)), 1e-09)
    r = L.mean(axis=0) - 0.5 * (centres[0] + centres[1])
    r = r - r @ u * u
    g = R.T @ np.array([0.0, 0.0, -1.0])
    g = g - g @ u * u
    n = float(np.linalg.norm(g))
    return float(r @ g) / n if n > 1e-06 else 0.0


def finger_min_z(R: np.ndarray, tcp: np.ndarray, gap: float) -> float:
    corners = np.array(
        [[a, k * (gap / 2 + t), b] for a in (F_A0, F_A1) for k in (-1, 1) for t in (0.0, F_T) for b in (-F_B, F_B)]
    )
    return float(tcp[2] + np.min(corners @ R[2, :]))


def slice_min_widths(pts: np.ndarray, z_belt: float = CS.BELT_Z) -> dict[int, float]:
    out = {}
    dirs = np.array([[math.cos(math.radians(d)), math.sin(math.radians(d))] for d in range(0, 180, 15)])
    for zc in range(1, 13):
        sl = pts[np.abs(pts[:, 2] - z_belt - zc / 100.0) < 0.01, :2]
        if len(sl) >= 5:
            out[zc] = float(np.min(np.ptp(sl @ dirs.T, axis=0)))
    return out


def lift_steps(asset: str) -> int:
    return N_LIFT_OBJ_HEAVY if CS.shape(asset).mass >= HEAVY_MASS else N_LIFT_OBJ


def held_object_line(q0, goal, heavy: bool):
    if heavy:
        return motion.line(
            "left",
            q0,
            goal,
            offset=K.TCP_OFFSET,
            lin_step=HEAVY_CARRY_LIN_STEP,
            ang_step=math.radians(HEAVY_ANG_STEP),
            joint_step=HEAVY_JOINT_STEP,
        )
    step = CARRY_LIN_STEP
    pl = motion.line("left", q0, goal, offset=K.TCP_OFFSET, lin_step=step, ang_step=math.radians(3.0))
    if not pl.ok and step > FEAS_LIN_STEP:
        pl = motion.line("left", q0, goal, offset=K.TCP_OFFSET, lin_step=FEAS_LIN_STEP, ang_step=math.radians(3.0))
    return pl


def drop_poses(R_c: np.ndarray, L: np.ndarray, hold) -> list:
    out = []
    hx, hy = float(hold[0]), float(hold[1])
    for turn in DROP_TURNS:
        R = K.rot_z(math.radians(turn)) @ R_c
        W = L @ R.T
        x0, x1 = float(W[:, 0].min()), float(W[:, 0].max())
        y0, y1 = float(W[:, 1].min()), float(W[:, 1].max())
        lo_x, hi_x = hx - FREE_FAR - x0, hx - FREE_NEAR - x1
        lo_y, hi_y = hy - FREE_HALF_W - y0, hy + FREE_HALF_W - y1
        want_x = hx - DROP_FROM_BAR - 0.5 * (x0 + x1)
        want_y = hy - 0.5 * (y0 + y1)
        over = max(0.0, lo_x - hi_x) + max(0.0, lo_y - hi_y)
        tx = float(np.clip(want_x, min(lo_x, hi_x), max(lo_x, hi_x))) if lo_x <= hi_x else 0.5 * (lo_x + hi_x)
        ty = float(np.clip(want_y, min(lo_y, hi_y), max(lo_y, hi_y))) if lo_y <= hi_y else 0.5 * (lo_y + hi_y)
        info = {
            "turn": turn,
            "overflow": round(over, 4),
            "len_x": round(x1 - x0, 3),
            "len_y": round(y1 - y0, 3),
            "x": round(tx, 3),
            "y": round(ty, 3),
        }
        out.append((over, abs(turn), R, np.array([tx, ty, Z_DROP]), info))
    out.sort(key=lambda t: (round(t[0], 3), t[1]))
    return [(R, p, info) for _, _, R, p, info in out]


def basket_keepout(t: int, bfit, sched: dict) -> list:
    boxes = []
    if t < sched["lift"] + KEEP_SLACK:
        boxes.append((bfit.x + V_BELT * (t + 1) - BASKET_HALF_X, -1.0))
    if t >= sched["lift"]:
        x_min = HOLD_X_MIN
        if "bar_x" in sched:
            x_min = max(HOLD_X_MIN, transit_x_min(t, sched["lift"], sched["bar_x"]))
        boxes.append((x_min, HOLD_Z_MIN))
    return boxes


def transit_x_min(t: int, t_lift: int, bar_x: float) -> float:
    t1 = t_lift + N_LIFT
    tau = min(1.0, max(0.0, (t + KEEP_T_MARGIN - t1) / N_TRANSIT_MIN))
    s_hi = 0.5 - 0.5 * math.cos(math.pi * tau)
    x_end = bar_x + V_BELT * N_LIFT_FOLLOW
    s = np.append(_S_GRID[_S_GRID < s_hi], s_hi)
    th = 0.5 * math.pi * s
    x = (
        x_end
        + s * (HOLD_TCP[0] - x_end)
        + 0.01 * s
        - BASKET_HALF_X * np.cos(th)
        - BASKET_HALF_Y * np.sin(th)
        - KEEP_SLIP
    )
    return float(np.min(x)) - 0.001


_S_GRID = np.linspace(0.0, 1.0, 201)


def keepout_hit(q, boxes, grip: float) -> bool:
    if not boxes:
        return False
    sp = CO.arm_spheres("left", q, grip)
    for x_min, z_min in boxes:
        if np.any((sp[:, 0] + sp[:, 3] + KEEP_MARGIN > x_min) & (sp[:, 2] + sp[:, 3] + KEEP_MARGIN > z_min)):
            return True
    return False


def approach_clearance(q, grip: float, obst_w: np.ndarray) -> float:
    s = CO.arm_spheres("left", q, grip)
    links, fingers = s[:-10], s[-10:]
    over = links[links[:, 1] > Y_OVER_BELT]
    out = float((over[:, 2] - over[:, 3]).min() - CS.BELT_Z - BELT_MARGIN) if len(over) else 9.0
    if len(obst_w):
        sp = np.vstack([over, fingers])
        d = np.linalg.norm(sp[:, None, :3] - obst_w[None, :, :], axis=2) - sp[:, 3:4]
        out = min(out, float(d.min()) - NB_MARGIN)
    return out


def release_height(R_c: np.ndarray, L: np.ndarray, z_hold: float, s_bar: float = 0.0) -> float:
    hang = float(np.min((L @ R_c.T)[:, 2]))
    z_b = z_hold - (BK.HANDLE_TOP - BK.HANDLE_THICK / 2)
    floor = z_b + BK.FLOOR_IN
    wall = z_b + BK.WALL_TOP_SIDE - max(0.0, s_bar) * math.sin(math.radians(HOLD_TILT))
    pitch = math.asin(float(np.clip(-R_c[2, 0], -1.0, 1.0)))
    z_body = wall + WALL_CLEAR - WALL_DIST * math.tan(max(pitch, 0.0))
    return float(min(Z_DROP, max(floor + RELEASE_ABOVE - hang, z_body)))


def grasp_candidates(
    pts: np.ndarray, h: float, z_belt: float = CS.BELT_Z, lateral=(0.0,), heavy: bool = False, deep: bool | None = None
) -> list[dict]:
    deep = heavy if deep is None else deep
    out = []
    z_pref = min(0.5 * h, 0.055)
    wmin = slice_min_widths(pts, z_belt)
    heights = np.arange(0.012, min(h, 0.095) + 0.0081, 0.01)
    rots = []
    for phi in range(0, 180, 15):
        for sgn in (1.0, -1.0):
            for tau in GRASP_TILTS:
                R0 = GR.grasp_rotation(math.radians(phi), sgn * math.radians(tau))
                for roll in GRASP_ROLLS:
                    R = K.axis_angle(R0[:, 0], math.radians(roll)) @ R0 if roll else R0
                    rots.append((R, phi, sgn * tau, roll, 0.0, (0.0,)))
            for eps in SIDE_TILTS:
                Rt = K.top_down(math.radians(phi))
                R = K.axis_angle(Rt[:, 2], sgn * math.radians(eps)) @ Rt
                rots.append((R, phi, 0.0, 0, sgn * eps, SIDE_DEPTHS))
    for R, phi, tilt, roll, side, depths in rots:
        ah = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-09)
        if np.linalg.norm(R[:2, 0]) > 0.05 and abs(math.degrees(math.atan2(ah[0], ah[1]))) > MAX_HEADING:
            continue
        tau = abs(tilt) if tilt else 45.0
        tau_app = abs(tilt) if tilt else abs(side)
        c_heavy = HEAVY_TILT_W * max(0.0, HEAVY_TILT_MIN - tau_app) / 10.0 if heavy else 0.0
        d = PRE_DIST_OBJ * R[:, 0] - np.array([0.0, 0.0, PRE_UP_OBJ])
        d_rel = R.T @ d
        n_lat = np.array([-R[1, 1], R[0, 1], 0.0])
        n_lat = n_lat / max(float(np.linalg.norm(n_lat)), 1e-09)
        for depth, s_lat in [(d_, l_) for d_ in depths for l_ in lateral]:
            z_low = TIP_CLEAR + 0.001 - finger_min_z(R, np.zeros(3), GAP_MAX)
            for zt in sorted(set([round(float(z_low), 4)] + [float(z) for z in heights if z > z_low])):
                tcp = np.array([s_lat * n_lat[0], s_lat * n_lat[1], z_belt + zt]) - depth * R[:, 0]
                r = grasp_check((pts - tcp) @ R, d_rel, GAP_MAX)
                if r is None:
                    continue
                tcp = tcp + r[1] * R[:, 1]
                L = (pts - tcp) @ R
                gap = min(GAP_MAX, r[0] + 2 * GAP_SPARE)
                r2 = grasp_check(L, d_rel, gap)
                if r2 is None:
                    gap = GAP_MAX
                    r2 = grasp_check(L, d_rel, gap)
                if r2 is None or r2[0] > gap - 2 * CLEAR:
                    continue
                if finger_min_z(R, tcp, gap) < z_belt + TIP_CLEAR:
                    continue
                w = r2[0]
                wedge = tip_wedge(L, gap)
                dwedge = down_wedge(L, R, gap)
                hang = contact_above_centroid(L, R, gap) if heavy and h < TALL else 0.0
                w_ref = wmin.get(int(round(zt * 100)), w)
                top = max(0.0, zt - (h - TOP_MARGIN)) / 0.01 if h >= TALL else 0.0
                cost = (
                    0.5 * max(0.0, GAP_SPARE - (gap - w) / 2) / (GAP_SPARE - CLEAR)
                    + W_RATIO * max(0.0, w / max(w_ref, 0.001) - 1.0)
                    + W_AREA * (1.0 - min(r2[3], A_REF) / A_REF)
                    + W_TOP * top
                    + W_ENG * max(0.0, ENGAGE_REF - r2[4]) / ENGAGE_REF
                    + (W_ENG_HEAVY * max(0.0, ENGAGE_REF_HEAVY - r2[4]) / ENGAGE_REF_HEAVY if deep else 0.0)
                    + 0.4 * (1.0 - min(r2[2], 2 * F_B) / (2 * F_B))
                    + 0.5 * abs(zt - z_pref) / max(h, 0.03)
                    + 0.3 * float(np.hypot(*tcp[:2])) / 0.03
                    + 0.02 * abs(tau - 45) / 5
                    + 0.2 * abs(roll) / 25
                    + ((ROLL_PENALTY_HEAVY if heavy else ROLL_PENALTY) if roll else 0.0)
                    + 0.4 * abs(side) / 45
                    + (W_CHORD * max(0.0, CHORD_W_REF - w) / 0.01 if s_lat else 0.0)
                    + (WEDGE_PENALTY + W_WEDGE * (wedge - WEDGE_MAX) / 0.001 if wedge > WEDGE_MAX else 0.0)
                    + (W_WEDGE_POS * max(0.0, wedge) / 0.001 if not heavy else 0.0)
                    + (
                        W_DOWN_WEDGE_HEAVY * max(0.0, dwedge - DOWN_WEDGE_OK_HEAVY)
                        if heavy
                        else W_DOWN_WEDGE * max(0.0, dwedge - DOWN_WEDGE_OK)
                    )
                    / 0.001
                    + c_heavy
                    + (
                        SHALLOW_ENG_PENALTY
                        if heavy and tau_app >= SHALLOW_ENG_TILT and (r2[4] < ENGAGE_REF_HEAVY)
                        else 0.0
                    )
                    + (W_AREA_HEAVY * (1.0 - min(r2[3], A_REF) / A_REF) if heavy else 0.0)
                    + W_HANG_HEAVY * max(0.0, hang - HANG_OK) / 0.001
                )
                out.append(
                    {
                        "cost": float(cost),
                        "R": R,
                        "tcp": tcp,
                        "gap": float(gap),
                        "width": float(r2[0]),
                        "phi": phi,
                        "tilt": float(tilt),
                        "roll": roll,
                        "side": float(side),
                        "zt": float(zt),
                        "depth": float(depth),
                        "lateral": float(s_lat),
                        "wedge": float(wedge),
                        "dwedge": float(dwedge),
                        "engage": float(r2[4]),
                        "area": float(r2[3]),
                        "hang": float(hang),
                    }
                )
    out.sort(key=lambda c: c["cost"])
    return out


RING_TILTS = 0, 10, 20
RING_OFFSETS = 0.028, -0.028, 0.032, -0.032
RING_HEIGHTS = 0.026, 0.028, 0.03, 0.032, 0.034, 0.036
RING_GAPS = 0.056, 0.062
RING_CLEAR = 0.001
RING_ANGLE_MAX = (
    12.0
)
RING_TIP_W, RING_TIP_DW = 0.0053, 0.27
RING_TIP_T, RING_TIP_DT = 0.002, 0.3
RING_RETRY_STEPS = 10


def ring_check(L: np.ndarray, d_rel: np.ndarray, gap: float, n_s: int = 6):
    for s in np.linspace(0.0, 1.0, n_s):
        P = L + (1.0 - s) * d_rel
        a, c, b = P[:, 0], P[:, 1], P[:, 2]
        ac = np.abs(c)
        back = np.clip(F_A1 - a, 0.0, None)
        fb = np.minimum(RING_TIP_W + RING_TIP_DW * back, F_B + 0.006)
        ft = np.minimum(RING_TIP_T + RING_TIP_DT * back, F_T)
        fin = (
            (np.abs(b) < fb + RING_CLEAR)
            & (a > F_A0 - RING_CLEAR)
            & (a < F_A1 + RING_CLEAR)
            & (ac > gap / 2 - RING_CLEAR)
            & (ac < gap / 2 + ft + RING_CLEAR)
        )
        palm = (np.abs(b) < PALM_B) & (a > PALM_A) & (a < F_A0 + RING_CLEAR) & (ac < gap / 2 + F_T)
        if fin.any() or palm.any():
            return None
    a, c, b = L[:, 0], L[:, 1], L[:, 2]
    fb = np.minimum(RING_TIP_W + RING_TIP_DW * np.clip(F_A1 - a, 0.0, None), F_B + 0.006)
    pad = (np.abs(b) < fb) & (a > F_A0 + 0.01) & (a < F_A1) & (np.abs(c) < gap / 2)
    if np.count_nonzero(pad) < 6:
        return None
    lo, hi = float(c[pad].min()), float(c[pad].max())
    if lo > -0.004 or hi < 0.004:
        return None
    return hi - lo, pad


def surface_normals(pts: np.ndarray, k: int = 14) -> np.ndarray:
    sq = (pts**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * pts @ pts.T
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    q = pts[nn] - pts[nn].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))[1][:, :, 0]


def ring_candidates(pts: np.ndarray, z_belt: float = CS.BELT_Z) -> list[dict]:
    nrm = surface_normals(pts)
    out = []
    for phi in range(0, 180, 15):
        Rt = K.top_down(math.radians(phi))
        for eps in RING_TILTS:
            for sgn in (1.0,) if eps == 0 else (1.0, -1.0):
                R = K.axis_angle(Rt[:, 2], sgn * math.radians(eps)) @ Rt if eps else Rt
                ah = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-09)
                if np.linalg.norm(R[:2, 0]) > 0.05 and abs(math.degrees(math.atan2(ah[0], ah[1]))) > MAX_HEADING:
                    continue
                d = PRE_DIST_OBJ * R[:, 0] - np.array([0.0, 0.0, PRE_UP_OBJ])
                d_rel = R.T @ d
                n_al = np.array([R[0, 1], R[1, 1], 0.0])
                n_al = n_al / max(float(np.linalg.norm(n_al)), 1e-09)
                for off in RING_OFFSETS:
                    for zt in RING_HEIGHTS:
                        tcp = np.array([off * n_al[0], off * n_al[1], z_belt + zt])
                        L = (pts - tcp) @ R
                        for gap in RING_GAPS:
                            r = ring_check(L, d_rel, gap)
                            if r is None or r[0] > gap - 2 * RING_CLEAR:
                                continue
                            if finger_min_z(R, tcp, gap) < z_belt + TIP_CLEAR:
                                continue
                            w, pad = r
                            c = L[:, 1]
                            angs = []
                            for s in (1.0, -1.0):
                                idx = np.where(pad & (s * c > 0.0))[0]
                                cs = s * c[idx]
                                patch = idx[cs > cs.max() - PATCH_DEPTH]
                                n = nrm[patch] @ R
                                n = (n * np.sign(s * n[:, 1] + 1e-12)[:, None]).mean(axis=0)
                                angs.append(math.degrees(math.atan2(abs(n[0]), abs(n[1]))))
                            ang = max(angs)
                            if ang > RING_ANGLE_MAX:
                                continue
                            cost = ang / 10.0 + abs(zt - float(np.mean(RING_HEIGHTS))) / 0.01 + 0.2 * eps / 10.0
                            out.append(
                                {
                                    "cost": float(cost),
                                    "R": R,
                                    "tcp": tcp,
                                    "gap": float(gap),
                                    "width": float(w),
                                    "phi": phi,
                                    "tilt": 0.0,
                                    "roll": 0,
                                    "side": float(sgn * eps),
                                    "zt": float(zt),
                                    "depth": 0.0,
                                    "lateral": float(off),
                                    "wedge": 0.0,
                                    "dwedge": 0.0,
                                    "engage": 0.0,
                                    "area": 0.0,
                                    "hang": 0.0,
                                    "ring": True,
                                    "angle": round(ang, 1),
                                }
                            )
                            break
    out.sort(key=lambda c: c["cost"])
    return out


STEP_LIMIT = 700
HEAVY_MASS = 0.3
HEAVY_TILT_MIN = 55.0
HEAVY_TILT_W = 0.0
ENGAGE_REF_HEAVY = 0.03
W_ENG_HEAVY = 1.5
SHALLOW_ENG_TILT = 45.0
SHALLOW_ENG_PENALTY = 2.0
W_AREA_HEAVY = 2.5
W_HANG_HEAVY = 1.2
HANG_OK = 0.0005
ROLL_PENALTY_HEAVY = 6.0
DEEP_MASS = 0.3
HEAVY_TURN_OK = 30.0
HEAVY_TURN_STEPS = 15.0
N_ROUTES_HEAVY = 6
WEDGE_MAX = 0.003
WEDGE_PENALTY = 3.0
W_WEDGE = 0.05
DOWN_WEDGE_OK, W_DOWN_WEDGE = 0.003, 0.6
DOWN_WEDGE_OK_HEAVY, W_DOWN_WEDGE_HEAVY = 0.001, 0.5
W_WEDGE_POS = 0.4
W_CHORD = 0.3
CHORD_W_REF = 0.08
CHORD_OFFSETS = (
    0.04,
    -0.04,
    0.045,
    -0.045,
)
N_ADV_OBJ = 12
N_CLOSE_OBJ = 13
N_LIFT_OBJ = 12
LIFT_OBJ = 0.06
N_LIFT_DECEL = 8
CARRY_LIN_STEP = 0.011
HEAVY_CARRY_LIN_STEP = 0.008
HEAVY_ANG_STEP = 3.0
HEAVY_JOINT_STEP = 0.05
N_LIFT_OBJ_HEAVY = 12
FEAS_LIN_STEP = 0.008
N_REGRASP = 1
G_EMPTY = 0.03
N_BACK, BACK_DIST, BACK_UP = 8, 0.05, 0.03
REGRASP_LEAD = 20
REGRASP_WAIT = 18
PRE_DIST_OBJ = 0.08
PRE_UP_OBJ = 0.02
GRASP_TILTS = (
    45,
    35,
    55,
    65,
    75,
    25,
    15,
)
MAX_HEADING = 95.0
X_GRASP = -0.55, -0.2
CARRY_PITCH = 20, 30, 10, 40
SUPPORT_ROLLS = 90.0, -90.0
DROP_GUARD = 0.01
BASKET_HALF_X = 0.12
KEEP_SLACK = 15
HOLD_X_MIN = HOLD_TCP[0] + 0.01 - 0.2
HOLD_Z_MIN = 0.9
KEEP_MARGIN = 0.02
BASKET_HALF_Y = 0.18
KEEP_SLIP = 0.02
N_TRANSIT_MIN = motion.n_steps_for(0.0, 0.5 * math.pi, 0.006, math.radians(2.0))
KEEP_T_MARGIN = 10
HOLD_TILT = 14.5
FREE_NEAR, FREE_FAR, FREE_HALF_W = 0.035, 0.145, 0.085
DROP_TURNS = 0, 30, -30, 60, -60, 90, -90
N_DROP_TRY = 6
DROP_MIN_CLEAR = 0.02
READY_TCP = -0.4, -0.25, 0.94
READY_PHI, READY_TILT = 165, -45
TRANSIT_STEP = 0.08
N_ROT_CHECK = 40
N_ROT_CHECK_RING = 150
N_GRASP_POOL = 16
MAX_JOINT_FALLBACK = 1.2
FALLBACK_WAIT_X = -0.25, -0.2, -0.35
MAX_JOINT_FALLBACK_LAST = 1.8
FALLBACK_ROTS = 30
FALLBACK_BUDGET_S = 12.0
LIFT_PENALTY = 300.0
LATE_CLOSE_X = -0.3
LATE_CLOSE_W = 0.4
NB_MARGIN = 0.008
NB_RANGE = 0.3
NB_POINTS = 250
BELT_MARGIN = 0.005
Y_OVER_BELT = -0.3
NB_PENALTY = 1000
NB_SELF = 0.04
NB_STATIC = 0.25
Y_CARRY = -0.05
Z_CARRY = 1.15
Z_DROP = 1.13
RELEASE_ABOVE = 0.03
WALL_CLEAR = 0.04
WALL_DIST = 0.12
DROP_FROM_BAR = 0.11
DET_START = 36
DET_EVERY = 6
X_SEE = -0.8, -0.16
ASSOC = 0.06, 0.05
ASSOC_CAT = 0.05
MAX_BLOB_AREA = 3500
MOVE_FRAC_LONG = 0.25
BASKET_MIN_PX = 1500
BASKET_DILATE = 10
ROBOT_DILATE = 4
N_CONF = 2
FIT_YAWS = np.radians(np.arange(-35.0, 35.1, 10.0))
T_ALL_SEEN = int((X_SEE[0] + 1.4) / V_BELT) + 3 * DET_EVERY + 12
MARGIN_MIN = 0.05
MARGIN_REL = -0.15
N_OBJECTS = 4
X_LEAVE = -0.52
GAP_REL = 0.25
W_COL = 0.5
TRACE_CHUNK = 120
SEG_SCORE = 0.5
SEG_OVERLAP = 0.3
SEG_VOTES = 2
SEG_MARGIN = -0.25
SEG_IGNORE = {"basket_grasp", "photo", "camera_stand"}
SIGS = ASSETS / "photo_signatures.json"


def _category(asset: str) -> str:
    return asset.split("/")[0]


def basket_region(img: np.ndarray) -> np.ndarray:
    import cv2

    m = BK.colour_mask(img)
    big = BK.largest_component(m)
    out = m.copy()
    if np.count_nonzero(big) >= BASKET_MIN_PX:
        pts = cv2.findNonZero(big.astype(np.uint8))
        hull = cv2.convexHull(pts)
        h = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(h, hull, 1)
        h = cv2.dilate(h, np.ones((2 * BASKET_DILATE + 1, 2 * BASKET_DILATE + 1), np.uint8))
        out |= h > 0
    return out


def support_rotations(R0: np.ndarray) -> list[np.ndarray]:
    out = []
    for R in carry_rotations(R0):
        for roll in SUPPORT_ROLLS:
            Rs = R @ K.rot_x(math.radians(roll))
            out.append((K.rot_angle(R0, Rs), Rs))
    out.sort(key=lambda t: t[0])
    return [R for _, R in out]


def carry_rotations(R0: np.ndarray) -> list[np.ndarray]:
    out = []
    c0 = R0[:, 1]
    for head in range(0, 181, 15):
        h = np.array([math.cos(math.radians(head)), math.sin(math.radians(head)), 0.0])
        c = np.array([-h[1], h[0], 0.0])
        if float(c @ c0) < 0:
            c = -c
        for pitch in CARRY_PITCH:
            th = math.radians(pitch)
            a = np.array([math.cos(th) * h[0], math.cos(th) * h[1], -math.sin(th)])
            R = np.stack([a, c, np.cross(a, c)], axis=1)
            out.append((K.rot_angle(R0, R) + 0.01 * CARRY_PITCH.index(pitch), R))
    out.sort(key=lambda t: t[0])
    return [R for _, R in out]


class Track:
    _next = 0

    def __init__(self, step, xy):
        Track._next += 1
        self.id = Track._next
        self.step, self.x, self.y = int(step), float(xy[0]), float(xy[1])
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.n = 0
        self.msum = 0.0
        self.x_first, self.step_first = self.x, self.step
        self.fit = None
        self.ofit = None
        self.rejected = False
        self.votes: dict[str, int] = {}
        self.agree: dict[str, int] = {}
        self.ssum = 0.0

    def predict(self, step) -> float:
        return self.x + V_BELT * (step - self.step)

    def mean(self, a) -> float:
        return self.sums.get(a, 0.0) / max(self.counts.get(a, 0), 1)


class Tracker:
    def __init__(self, cam, target: str):
        self.cam, self.target = cam, target
        self.bg = BO.BeltBackground()
        self.tracks: list[Track] = []
        self.all_tracks: list[Track] = []
        sig = json.loads(SIGS.read_text())["signatures"]
        self.sigs = {a: np.asarray(v, float) for a, v in sig.items() if a in CS.assets()}
        self.tcat = _category(target)
        self.cats = {_category(a) for a in CS.assets()}
        self.siblings = [a for a in CS.assets() if a != target and _category(a) == self.tcat]
        self.cat_assets: dict[str, list[str]] = {}
        for a in CS.assets():
            self.cat_assets.setdefault(_category(a), []).append(a)

    def _seg(self, img, bad) -> list:
        out = []
        for ins in seg.segment(img, camera="head"):
            if (
                ins.category not in SEG_IGNORE
                and ins.score >= SEG_SCORE
                and (int((ins.mask & ~bad).sum()) >= 0.5 * max(ins.area, 1))
            ):
                out.append(ins)
        return out

    def sib_margin(self, tr: Track) -> float:
        return tr.ssum / max(tr.n, 1) if self.siblings else 1.0

    def seg_kind(self, tr: Track) -> str | None:
        n_t = tr.votes.get(self.tcat, 0)
        n_o = max((v for c, v in tr.votes.items() if c != self.tcat and c in self.cats), default=0)
        n_ag = max((v for c, v in tr.agree.items()), default=0)
        if n_t >= SEG_VOTES and n_t > n_o and (self.margin(tr) > SEG_MARGIN):
            return "target"
        if n_ag >= SEG_VOTES and n_t <= 1:
            return "other"

    def seg_contra(self, tr: Track) -> bool:
        if tr.votes.get(self.tcat, 0) > 0:
            return False
        return max((v for c, v in tr.votes.items() if c != self.tcat and c in self.cats), default=0) >= SEG_VOTES

    def seg_open(self, tr: Track) -> bool:
        return self.seg_kind(tr) != "other" and sum(tr.votes.values()) >= 1

    def update(self, step, img, st, bfit, shared) -> bool:
        self.bg.add(step, img)
        if step < DET_START or step % DET_EVERY or (not self.bg.ready()):
            return False
        import cv2

        cam = self.cam
        bad = robot_mask.mask(
            cam,
            st["left_arm_joint_state"],
            st["right_arm_joint_state"],
            float(st["left_ee_joint_state"][0]),
            float(st["right_ee_joint_state"][0]),
            dilate=ROBOT_DILATE,
        )
        bad = bad | basket_region(img)
        strong, weak = BO.foreground(img, self.bg.image(step), bad=bad, cam=cam)
        insts = self._seg(img, bad)
        k9 = np.ones((9, 9), np.uint8)
        regions = []
        per_inst: dict[int, list] = {}
        dil = {}
        for b in BO.blobs(strong):
            near = cv2.dilate(b.mask.astype(np.uint8), k9) > 0
            mine = [
                k
                for k, ins in enumerate(insts)
                if int((ins.mask & b.mask).sum()) >= SEG_OVERLAP * min(b.area, ins.area)
            ]
            if mine:
                for k in mine:
                    if k not in dil:
                        dil[k] = cv2.dilate(insts[k].mask.astype(np.uint8), k9) > 0
                    part = near & dil[k]
                    r = [b.mask & part, strong & part, weak & part]
                    per_inst[k] = r if k not in per_inst else [a | c for a, c in zip(per_inst[k], r)]
            elif b.area <= MAX_BLOB_AREA:
                regions.append((b.mask, strong & near, weak & near, None))
        for k, r in per_inst.items():
            m = insts[k].mask & ~bad
            regions.append((m, r[1] | m, r[2] | m, insts[k]))
        for k, ins in enumerate(insts):
            if k not in per_inst and ins.category in self.cats:
                m = ins.mask & ~bad
                regions.append((m, m, m, ins))
        dets = []
        for pix, s_m, w_m, ins in regions:
            if pix.sum() < BO.MIN_AREA:
                continue
            xy = CS.blob_xy(cam, pix)
            if not X_SEE[0] <= xy[0] <= X_SEE[1]:
                continue
            if not shared.get("basket_lifted") and abs(xy[0] - (bfit.x + V_BELT * step)) < 0.2:
                continue
            sg = BO.signature(img[pix])
            col = {a: BO.similarity(sg, s) for a, s in self.sigs.items()}
            others = sorted((a for a in col if a != self.target and a not in self.siblings), key=lambda a: -col[a])[
                :N_CONF
            ]
            scores, fits = {}, {}
            vcat = None if ins is None else ins.category
            voted = [a for a in self.cat_assets.get(vcat, []) if vcat != self.tcat and a not in others]
            for a in [self.target] + self.siblings + others + voted:
                xy_a = CS.blob_xy(cam, pix, height=float(CS.shape(a).extent[2]))
                f = CS.fit(cam, s_m, w_m, a, xy_a, step=0.02, yaws=FIT_YAWS)
                fits[a] = f
                scores[a] = f.score + W_COL * col[a]
            vote = None if ins is None else (ins.category, round(float(ins.score), 2))
            if vote is not None and vcat in self.cats and (vcat != self.tcat):
                vote = vote + (bool(max(scores[a] for a in self.cat_assets[vcat]) > scores[self.target]),)
            sib = scores[self.target] - max(scores[a] for a in self.siblings) if self.siblings else 0.0
            dets.append((xy, scores, fits[self.target], fits[max(scores, key=scores.get)], vote, sib))
        pairs = []
        for i, d in enumerate(dets):
            for tr in self.tracks:
                dx, dy = d[0][0] - tr.predict(step), d[0][1] - tr.y
                if abs(dx) < ASSOC[0] and abs(dy) < ASSOC[1]:
                    major = max(tr.votes, key=tr.votes.get) if tr.votes else None
                    clash = d[4] is not None and major is not None and (d[4][0] != major)
                    pairs.append((abs(dx) + abs(dy) + (ASSOC_CAT if clash else 0.0), i, tr))
        match, took = {}, {}
        for _, i, tr in sorted(pairs, key=lambda p: p[0]):
            if i in match:
                continue
            vote = dets[i][4]
            if tr.id in took:
                prev = took[tr.id]
                if vote is not None and prev is not None and (vote[0] != prev):
                    continue
            match[i] = tr
            if tr.id not in took or took[tr.id] is None:
                took[tr.id] = None if vote is None else vote[0]
        for i, (xy, scores, ft, fo, vote, sib) in enumerate(dets):
            best = match.get(i)
            if best is None:
                best = Track(step, xy)
                self.tracks.append(best)
                self.all_tracks.append(best)
            best.step, best.x, best.y = step, xy[0], xy[1]
            best.n += 1
            best.msum += scores[self.target] - max(v for a, v in scores.items() if a != self.target)
            best.fit = ft
            best.ofit = fo
            for a, s in scores.items():
                best.sums[a] = best.sums.get(a, 0.0) + s
                best.counts[a] = best.counts.get(a, 0) + 1
            if self.siblings:
                best.ssum += sib
            if vote is not None:
                best.votes[vote[0]] = best.votes.get(vote[0], 0) + 1
                if len(vote) > 2 and vote[2]:
                    best.agree[vote[0]] = best.agree.get(vote[0], 0) + 1
        self.tracks = [tr for tr in self.tracks if step - tr.step <= 4 * DET_EVERY]
        return True

    def obstacles(self, tr: Track) -> np.ndarray:
        f = tr.fit
        out = []
        for o in self.tracks:
            if (
                o is tr
                or o.ofit is None
                or (o.n >= 2 and o.x - o.x_first <= NB_STATIC * V_BELT * (o.step - o.step_first))
            ):
                continue
            fo = o.ofit
            dx = fo.x - f.x + V_BELT * (tr.step - o.step)
            dy = fo.y - f.y
            if not NB_SELF <= math.hypot(dx, dy) <= NB_RANGE:
                continue
            pts = CS.posed(CS.shape(fo.asset), dx, dy, fo.yaw)
            out.append(pts[:: max(1, len(pts) // NB_POINTS)])
        return np.vstack(out) if out else np.zeros((0, 3))

    def margin(self, tr: Track) -> float:
        return tr.msum / max(tr.n, 1)

    @staticmethod
    def moving(tr: Track) -> bool:
        frac = 0.5 if tr.n < 3 else MOVE_FRAC_LONG
        return tr.n >= 2 and tr.x - tr.x_first > frac * V_BELT * (tr.step - tr.step_first)

    def choose(self, step) -> "Track | None":
        live = [
            tr
            for tr in self.tracks
            if self.moving(tr) and (not tr.rejected) and (tr.step == step) and (tr.predict(step) < X_GRASP[1] - 0.15)
        ]
        if not live:
            return None
        kind = {t.id: self.seg_kind(t) for t in live}
        cand = [t for t in live if kind[t.id] == "target" and self.sib_margin(t) > 0]
        if cand:
            tr = max(cand, key=lambda t: (t.votes.get(self.tcat, 0), self.margin(t)))
            return tr
        live = [t for t in live if kind[t.id] is None and self.seg_open(t)]
        if not live:
            return None
        tr = max(live, key=self.margin)
        m = self.margin(tr)
        early_ok = [t for t in live if not self.seg_contra(t)]
        tr_e = max(early_ok, key=self.margin) if early_ok else None
        m_e = self.margin(tr_e) if tr_e is not None else -1000000000.0
        seen = sorted((t for t in self.all_tracks if self.moving(t)), key=self.margin, reverse=True)
        best_seen = len(seen) >= N_OBJECTS - 1 and seen[0] is tr
        clear_best = len(seen) >= 2 and seen[0] is tr and (m - self.margin(seen[1]) >= GAP_REL)
        leaving = tr.predict(step) > X_LEAVE
        forced = (
            step >= T_ALL_SEEN
            and m > MARGIN_REL
            or ((best_seen or clear_best) and leaving and (m > MARGIN_REL) and (tr.n >= 5))
        )
        if not forced and tr_e is not None and (m_e > MARGIN_MIN and tr_e.n >= 2 or (m_e > 0 and tr_e.n >= 4)):
            tr, m = tr_e, m_e
        elif not forced:
            return None
        return tr


class HoldTrace:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows: list = []

    def step(self, arm: Arm, phase) -> None:
        if phase is None:
            self.flush()
            return
        st = self.ctx.obs.state
        q_obs = np.asarray(st["left_arm_joint_state"], float)
        tcp = K.point_of(K.pose_to_mat(np.asarray(st["left_ee_pose"], float)), K.TCP_OFFSET)
        self.rows.append(
            (
                int(self.ctx.obs.step),
                str(phase),
                round(float(st["left_ee_joint_state"][0]), 4),
                round(float(np.max(np.abs(np.asarray(arm.q, float) - q_obs))), 4),
                [round(float(v), 4) for v in q_obs],
                [round(float(v), 4) for v in tcp],
            )
        )
        if len(self.rows) >= TRACE_CHUNK:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.rows = []


def late_close_cost(x: float) -> float:
    return COST_STEPS * LATE_CLOSE_W * max(0.0, float(x) - LATE_CLOSE_X) / 0.01
