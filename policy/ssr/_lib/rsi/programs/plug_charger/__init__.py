from __future__ import annotations

import math

import numpy as np

from ...core import actions
from ...perception import cameras as C
from . import charger as CH
from ...perception import robot_mask as RM
from ...planning import kinematics as K
from ...planning import motion
from ...skills import dual as D
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

TCP = K.TCP_OFFSET
LOOK_STEP = 0.1
LOAD_STEP = 0.09
SETTLE = 4
CLEAR_DQ = {"left": np.array([0.45, 0.0, 0.4, 0.0, 0.0, 0.0]), "right": np.array([-0.45, 0.0, 0.4, 0.0, 0.0, 0.0])}
CHARGER_MAX_YAW = math.radians(45)
MIN_IOU = 0.85
LOOK_H = 0.13
MIN_SLOTS = 8
BETAS = tuple(math.radians(b) for b in (20, 15, 25, 30))
G_BODY = 0.024
GRASP_DZ = 0.0055
OPEN_G = 0.75
HOVER_Z = 0.09
STRIP_TOP = K.TABLE_Z + CH.SOCKET_Z - K.TABLE_Z + CH.STRIP_TOP_LOCAL
ABOVE_APPROACH = 0.045
ABOVE_SLOW = 0.021
FAST_DOWN = 0.0015
ABOVE_CONTACT = 0.0165
ABOVE_DEEP = 0.011
SLOW_STEP = 0.0008
SEAT_ABOVE = -0.008
BLOCK_LAG = 0.0012
INSERTED_ABOVE = 0.008
ENTRY_ABOVE = 0.012
ENTRY_DEFICIT = 0.0022
SEAT_TARGET = -0.0025
SEATED_MAX = 0.003
MAX_SLIDE = 0.018
SURFACE_MIN = 0.009
SEARCH = (0.0, 0.0), (0.35, 0.0), (-0.35, 0.0), (0.0, 0.7), (0.0, -0.7), (0.7, 0.0), (-0.7, 0.0)
HANDOFF_X = 0.1, 0.06, 0.14, 0.03
HANDOFF_Y = -0.23, -0.27, -0.19
Z = np.array([0.0, 0.0, 1.0])
HELD_BOX = np.array([[x, y, z] for x in (-0.0233, 0.0233) for y in (-0.0169, 0.0531) for z in (-0.015, 0.015)])
TRANSFER_UP = 0.03
STRIP_MARGIN = 0.02
MIN_CLEAR = 0.012
HOME_UP = 0.05


def match_instruction(text: str) -> bool:
    s = text.lower()
    return "charger" in s and ("plug" in s or "insert" in s)


def frame(x, z) -> np.ndarray:
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    return np.stack([x, np.cross(z, x), z], axis=1)


def grasp_rot(h, beta: float) -> np.ndarray:
    h = np.asarray(h, float)
    return frame(-(math.cos(beta) * Z + math.sin(beta) * h), -math.sin(beta) * Z + math.cos(beta) * h)


def plug_rot(f, beta: float) -> np.ndarray:
    f = np.asarray(f, float)
    return frame(math.cos(beta) * f - math.sin(beta) * Z, math.cos(beta) * Z + math.sin(beta) * f)


def plug_tcp(outlet_xy, f, above: float, g: float = G_BODY, dz: float = GRASP_DZ) -> np.ndarray:
    f = np.asarray(f, float)
    xy = np.asarray(outlet_xy, float)[:2] - dz * f[:2]
    return np.array([xy[0], xy[1], STRIP_TOP + above + g])


def ik(arm: str, R, p, q_cur, seeds=()):
    r = K.solve_near(arm, tcp_pose(R, p), q_cur, TCP, extra_seeds=seeds, n_random=3)
    return r.q if r.ok else None


def look_joints(arm: str, target, height: float, q_cur, axis_yaw: float | None = None):
    Rm, tm = C.WRIST_MOUNT[:3, :3], C.WRIST_MOUNT[:3, 3]
    a_loc = Rm @ np.array([0.0, 0.0, -1.0])
    tilt = math.atan2(-a_loc[2], a_loc[0])
    q_cur = np.asarray(q_cur, dtype=float)
    cam = np.asarray(target, float) + height * Z
    best = None
    for psi in np.radians(np.arange(-180.0, 180.0, 15.0)):
        R0 = K.top_down(float(psi))
        R = K.axis_angle(R0[:, 1], -tilt) @ R0
        if (R @ a_loc)[2] > -0.999:
            R = K.axis_angle(R0[:, 1], tilt) @ R0
        T = tcp_pose(R, cam - R @ tm)
        r = K.solve_near(arm, T, q_cur, 0.0, n_random=1)
        if not r.ok:
            continue
        cost = float(np.sum(K.JOINT_COST_W * np.abs(r.q - q_cur)))
        if axis_yaw is not None:
            cost += 2.0 * abs(math.sin(float(psi) - axis_yaw))
        if best is None or cost < best[0]:
            best = cost, r.q
    return None if best is None else best[1]


def observed_tcp(arm: Arm) -> np.ndarray:
    return K.point_of(K.pose_to_mat(arm.ctx.obs.state[f"{arm.name}_ee_pose"]), TCP)


def held(arm: Arm) -> float:
    return D.held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]), 0.0)


def strip_normal(yaw: float) -> np.ndarray:
    n = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return n if n[1] >= 0 else -n


def charger_h(yaw_x: float) -> np.ndarray:
    h = np.array([-math.sin(yaw_x), math.cos(yaw_x), 0.0])
    return h if h[1] >= 0 else -h


def look_strip(arm: Arm, ctx, sock):
    best = None
    for k, (dx, h) in enumerate(((0.0, LOOK_H), (0.0, LOOK_H + 0.04))):
        q = look_joints(arm.name, (sock.x, sock.y, STRIP_TOP), h, arm.q, sock.yaw)
        if q is None:
            continue
        yield from arm.joints_to(q, joint_step=LOOK_STEP)
        yield from arm.hold(SETTLE)
        img = ctx.obs.images.get(f"cam_{arm.name}_wrist")
        ee = np.asarray(ctx.obs.state[f"{arm.name}_ee_pose"])
        fit = CH.fit_strip_wrist(img, C.wrist(ee), sock.x, sock.y, sock.yaw) if img is not None else None
        if fit is not None and (best is None or fit.n_matched > best.n_matched):
            best = fit
        if best is not None and best.n_matched >= MIN_SLOTS:
            break
    return best


def grasp(arm: Arm, body_xy, h, beta: float):
    R = grasp_rot(h, beta)
    x, y = float(body_xy[0]), float(body_xy[1])
    q_low = ik(arm.name, R, (x, y, low_z()), arm.q)
    q, hz = None, None
    for hz in (HOVER_Z, 0.07, 0.055, 0.042):
        for seed in (arm.q,) if q_low is None else (q_low, arm.q):
            q = ik(arm.name, R, (x, y, K.TABLE_Z + hz), seed)
            if q is not None:
                break
        if q is not None:
            break
    if q is None or q_low is None:
        arm.status = "fail"
        return
    yield from arm.joints_to(q, grip=OPEN_G, joint_step=LOOK_STEP)
    yield from arm.line_to(tcp_pose(R, (x, y, low_z())), lin_step=0.01, fallback_joint=False)
    if arm.status != "ok":
        yield from arm.joints_to(q_low, joint_step=0.03)
    yield from arm.set_grip(0.0, n=8)
    yield from arm.hold(3)
    w = held(arm)
    yield from arm.line_to(tcp_pose(R, (x, y, K.TABLE_Z + hz)), lin_step=0.01, fallback_joint=False)
    if arm.status != "ok":
        yield from arm.joints_to(q, joint_step=0.04)
    arm.status = "ok" if 0.03 < w < 0.05 else "fail"


def choose_beta(arm: str, body_xy, h, strip: CH.StripFit | None, f, outlets, q_cur):
    for beta in BETAS:
        Rg = grasp_rot(h, beta)
        if ik(arm, Rg, (body_xy[0], body_xy[1], K.TABLE_Z + 0.0135 + GRASP_DZ), q_cur) is None:
            continue
        if strip is None:
            return beta, None
        k = pick_outlet(arm, strip, f, beta, outlets, q_cur)
        if k is not None:
            return beta, k
    return None, None


def pick_outlet(arm: str, strip: CH.StripFit, f, beta: float, outlets, q_cur):
    Rp = plug_rot(f, beta)
    for k in outlets:
        o = strip.outlet(k)
        qa = ik(arm, Rp, plug_tcp(o, f, ABOVE_APPROACH), q_cur)
        if qa is not None and ik(arm, Rp, plug_tcp(o, f, SEAT_ABOVE), qa) is not None:
            return k


def held_in_tcp(beta: float):
    h = np.array([0.0, 1.0, 0.0])
    Rg = grasp_rot(h, beta)
    Rc = np.stack([np.cross(h, Z), h, Z], axis=1)
    return Rg.T @ Rc, Rg.T @ (-GRASP_DZ * Z - G_BODY * h)


def held_frame_link6(beta: float):
    Rr, pr = held_in_tcp(beta)
    return Rr, np.array([TCP, 0.0, 0.0]) + pr


def held_corners(arm: str, q, beta: float) -> np.ndarray:
    T = K.fk(arm, q)
    Rr, pr = held_in_tcp(beta)
    return K.point_of(T, TCP) + (T[:3, :3] @ (pr[:, None] + Rr @ HELD_BOX.T)).T


def path_clear(arm: str, qs, beta: float, strip: CH.StripFit) -> float:
    ex = np.array([math.cos(strip.yaw), math.sin(strip.yaw)])
    ey = np.array([-ex[1], ex[0]])
    worst = 1.0
    for q in qs:
        c = held_corners(arm, q, beta)
        d = c[:, :2] - np.array([strip.x, strip.y])
        over = (np.abs(d @ ex) < 0.0786 + STRIP_MARGIN) & (np.abs(d @ ey) < 0.0182 + STRIP_MARGIN)
        worst = min(worst, float(c[:, 2].min()) - K.TABLE_Z)
        if over.any():
            worst = min(worst, float(c[over, 2].min()) - STRIP_TOP)
    return worst


def transfer(arm: Arm, Rp, p_pre, beta: float, strip: CH.StripFit):
    q0 = arm.q.copy()
    for up in (TRANSFER_UP, TRANSFER_UP + 0.04):
        T_hi = tcp_pose(Rp, np.asarray(p_pre) + up * Z)
        plan = motion.line(arm.name, q0, T_hi, TCP, lin_step=0.014, ang_step=math.radians(6), joint_step=LOAD_STEP)
        cands = []
        if plan.ok:
            cands.append(("line", plan))
        q_hi = ik(arm.name, Rp, T_hi[:3, 3], q0)
        if q_hi is not None:
            cands.append(("joint", motion.joint_move(q0, q_hi, joint_step=LOAD_STEP)))
        for kind, pl in cands:
            clr = path_clear(arm.name, pl.q, beta, strip)
            if clr >= MIN_CLEAR:
                ok = yield from arm.follow(pl)
                arm.status = "ok" if ok else "blocked"
                if arm.status == "ok":
                    yield from arm.line_to(tcp_pose(Rp, p_pre), lin_step=0.01, fallback_joint=False)
                    if arm.status != "ok":
                        q = ik(arm.name, Rp, p_pre, arm.q)
                        if q is not None:
                            yield from arm.joints_to(q, joint_step=0.03)
                return
    q = ik(arm.name, Rp, p_pre, q0)
    if q is None:
        arm.status = "fail"
        return
    yield from arm.joints_to(q, joint_step=LOAD_STEP)


def insert(arm: Arm, strip: CH.StripFit, k: int, f, beta: float):
    Rp = plug_rot(f, beta)
    o = strip.outlet(k)
    ex = np.array([math.cos(strip.yaw), math.sin(strip.yaw)])
    ey = np.array([-ex[1], ex[0]])
    yield from transfer(arm, Rp, plug_tcp(o, f, ABOVE_APPROACH), beta, strip)
    if arm.status == "fail":
        return
    for n_try, (sx, sy) in enumerate(SEARCH):
        oo = o + (sx * ex + sy * ey) / 1000.0
        yield from arm.hold(1)
        s = measure_slide(arm, beta)
        g = G_BODY - (0.0 if s is None else s)
        yield from arm.line_to(tcp_pose(Rp, plug_tcp(oo, f, ABOVE_SLOW, g=g)), lin_step=0.008)
        yield from arm.hold(1)
        above = ABOVE_SLOW
        deficit = None
        while above > SEAT_TARGET:
            step = SLOW_STEP if ABOVE_DEEP < above <= ABOVE_CONTACT + 1e-09 else FAST_DOWN
            nxt = max(SEAT_TARGET, above - step)
            check = s is not None and deficit is None and (nxt <= ENTRY_ABOVE)
            if check:
                nxt = ENTRY_ABOVE
            above = nxt
            qq = ik(arm.name, Rp, plug_tcp(oo, f, above, g=g), arm.q)
            if qq is None:
                break
            arm.q = qq
            yield
            if check:
                yield from arm.hold(2)
                s2 = measure_slide(arm, beta)
                lag = observed_tcp(arm)[2] - plug_tcp(oo, f, above, g=g)[2]
                deficit = lag + (0.0 if s2 is None else s2 - s)
                if deficit > ENTRY_DEFICIT:
                    break
                continue
            lag = observed_tcp(arm)[2] - plug_tcp(oo, f, above, g=g)[2]
            if lag > BLOCK_LAG:
                break
        yield from arm.hold(3)
        face, measured = seat_height(arm, beta, s)
        for push in range(2):
            on_surface = deficit is not None and deficit > ENTRY_DEFICIT
            if not measured or on_surface or (not SEATED_MAX <= face <= SURFACE_MIN):
                break
            s3 = measure_slide(arm, beta)
            s_now = s3 if s3 is not None else 0.0 if s is None else s
            g = G_BODY - s_now
            qq = ik(arm.name, Rp, plug_tcp(oo, f, SEAT_TARGET - 0.002, g=g), arm.q)
            if qq is None:
                break
            yield from arm.joints_to(qq, joint_step=0.01)
            yield from arm.hold(3)
            face, measured = seat_height(arm, beta, s)
        if face < (SEATED_MAX if measured else INSERTED_ABOVE):
            arm.status = "ok"
            return
        qq = ik(arm.name, Rp, plug_tcp(oo, f, ABOVE_SLOW, g=g), arm.q)
        if qq is not None:
            yield from arm.joints_to(qq, joint_step=0.03)
    arm.status = "fail"


def measure_slide(arm: Arm, beta: float):
    img = arm.ctx.obs.images.get(f"cam_{arm.name}_wrist")
    R_c, o_c = held_frame_link6(beta)
    fit = CH.held_slide(img, R_c, o_c) if img is not None else None
    if fit is None or abs(fit.slide) > MAX_SLIDE:
        return None
    return fit.slide


def seat_height(arm: Arm, beta: float, s_pre):
    s = measure_slide(arm, beta)
    z = observed_tcp(arm)[2] - G_BODY - STRIP_TOP
    if s is not None:
        return z + s, True
    return z + (0.0 if s_pre is None else s_pre), False


def release_and_home(arm: Arm, beta: float, f):
    yield from arm.set_grip(OPEN_G, n=10)
    yield from arm.hold(3)
    R = plug_rot(f, beta)
    p = K.point_of(arm.T, TCP)
    yield from rise(arm, R, p, (0.07 + HOME_UP, 0.07, 0.05, 0.035), lin_step=0.01)
    yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)


def rise(arm: Arm, R, p, dzs, lin_step: float = 0.012):
    for dz in dzs:
        yield from arm.line_to(tcp_pose(R, (p[0], p[1], p[2] + dz)), lin_step=lin_step, fallback_joint=False)
        if arm.status == "ok":
            return
    for dz in dzs:
        q = ik(arm.name, R, (p[0], p[1], p[2] + dz), arm.q)
        if q is not None and float(np.abs(q - arm.q).max()) < 0.6:
            yield from arm.joints_to(q, joint_step=0.03)
            return


def low_z() -> float:
    return K.TABLE_Z + 0.0135 + GRASP_DZ


def plan_handoff(carrier: Arm, inserter: Arm, chg, h, sock):
    sgn = -1.0 if inserter.name == "left" else 1.0
    ex = np.array([math.cos(sock.yaw), math.sin(sock.yaw)])
    ey = np.array([-ex[1], ex[0]])
    for y in HANDOFF_Y:
        for ax in HANDOFF_X:
            p = np.array([sgn * ax, y])
            d = p - np.array([sock.x, sock.y])
            du, dv = max(0.0, abs(float(d @ ex)) - 0.079), max(0.0, abs(float(d @ ey)) - 0.018)
            if math.hypot(du, dv) < 0.065:
                continue
            if not any(
                ik(inserter.name, grasp_rot(h, b), (p[0], p[1], low_z()), inserter.q) is not None
                and ik(inserter.name, grasp_rot(h, b), (p[0], p[1], K.TABLE_Z + 0.07), inserter.q) is not None
                for b in BETAS
            ):
                continue
            for b in BETAS:
                R = grasp_rot(h, b)
                if ik(carrier.name, R, (chg.x, chg.y, low_z()), carrier.q) is None:
                    continue
                q_hover = ik(carrier.name, R, (p[0], p[1], K.TABLE_Z + HOVER_Z), carrier.q)
                if q_hover is None:
                    continue
                q_low = ik(carrier.name, R, (p[0], p[1], low_z() + 0.002), q_hover)
                if q_low is None:
                    continue
                return p, (b, q_hover, q_low)
    return None, None


def regrasp(arm: Arm, beta: float, state: dict, tag: str):
    ctx = arm.ctx
    yield from arm.set_grip(OPEN_G, n=6)
    yield from arm.joints_to(arm.q_home + CLEAR_DQ[arm.name], joint_step=LOOK_STEP)
    yield from arm.hold(SETTLE)
    _, chg = head_detect(ctx, C.head())
    if chg is None:
        arm.status = "fail"
        return None
    h = charger_h(chg.yaw)
    for b in (beta,) + tuple(x for x in BETAS if x != beta):
        if ik(arm.name, grasp_rot(h, b), (chg.x, chg.y, low_z()), arm.q) is None:
            continue
        yield from grasp(arm, chg.xy, h, b)
        return (chg.xy, h, b) if arm.status == "ok" else None
    arm.status = "fail"


def carry_task(arm: Arm, chg, h, spot, plan, state: dict):
    beta, q_hover, q_low = plan
    yield from grasp(arm, chg.xy, h, beta)
    if arm.status != "ok":
        r = yield from regrasp(arm, beta, state, "carry")
        if r is None:
            state["handoff"] = "fail"
            yield from arm.set_grip(OPEN_G, n=6)
            yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)
            return
        _, h_new, beta = r
        R = grasp_rot(h_new, beta)
        q_hover = ik(arm.name, R, (spot[0], spot[1], K.TABLE_Z + HOVER_Z), arm.q)
        q_low = None if q_hover is None else ik(arm.name, R, (spot[0], spot[1], low_z() + 0.002), q_hover)
        if q_low is None:
            state["handoff"] = "fail"
            yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)
            return
        state["handoff_h"] = h_new
        h = h_new
    R = grasp_rot(h, beta)
    yield from arm.joints_to(q_hover, joint_step=LOAD_STEP)
    if arm.status != "ok":
        yield from arm.joints_to(q_hover, joint_step=LOAD_STEP, force=True)
    yield from arm.line_to(tcp_pose(R, (spot[0], spot[1], low_z() + 0.002)), lin_step=0.01, fallback_joint=False)
    if arm.status != "ok":
        yield from arm.joints_to(q_low, joint_step=0.03, force=True)
    yield from arm.set_grip(OPEN_G, n=8)
    yield from arm.hold(2)
    p = observed_tcp(arm)
    state["handoff_xy"] = p[:2].copy()
    yield from rise(arm, R, p, (0.08, 0.06, 0.045))
    state["handoff"] = "done"
    yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)


def insert_task(arm: Arm, ctx, sock, chg, h, state: dict, carried: bool):
    strip = yield from look_strip(arm, ctx, sock)
    if strip is None or strip.n_matched < 4:
        strip = CH.StripFit(sock.x, sock.y, sock.yaw, 99.0, 0, 0)
    state["strip"] = strip
    if carried:
        yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)
        yield from arm.wait_until(lambda: state.get("handoff") in ("done", "fail"), max_steps=250)
        if state.get("handoff") != "done":
            return
        body_xy = state["handoff_xy"]
        h = state.get("handoff_h", h)
    else:
        body_xy = chg.xy
    f = strip_normal(strip.yaw)
    outlets = (1, 0, 2) if arm.name == "right" else (1, 2, 0)
    beta, k = choose_beta(arm.name, body_xy, h, strip, f, outlets, arm.q)
    if beta is None or k is None:
        yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)
        return
    yield from grasp(arm, body_xy, h, beta)
    if arm.status != "ok":
        r = yield from regrasp(arm, beta, state, "insert")
        if r is None:
            yield from arm.set_grip(OPEN_G, n=6)
            yield from arm.joints_to(arm.q_home, joint_step=LOOK_STEP)
            return
        body_xy, h, beta = r
        k = pick_outlet(arm.name, strip, f, beta, (k,) + tuple(o for o in outlets if o != k), arm.q)
        if k is None:
            yield from release_and_home(arm, beta, f)
            return
    yield from insert(arm, strip, k, f, beta)
    yield from release_and_home(arm, beta, f)


def head_detect(ctx, cam):
    img, st = ctx.obs.images.get("cam_head"), ctx.obs.state
    robot = RM.mask(
        cam,
        st["left_arm_joint_state"],
        st["right_arm_joint_state"],
        float(st["left_ee_joint_state"][0]),
        float(st["right_ee_joint_state"][0]),
    )
    sock, chg = CH.detect_head(img, cam, robot, CHARGER_MAX_YAW) if img is not None else (None, None)
    return sock, chg


@register
class PlugCharger(Program):
    name = "plug_charger"

    def match(self, instruction: str, first_obs) -> float:
        return 0.9 if match_instruction(instruction) else 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        cam = C.head()
        sock, chg = head_detect(ctx, cam)
        link_arms(arms, None)
        if sock is None or chg is None or sock.iou < MIN_IOU or (chg.iou < MIN_IOU):
            yield from run_arms(
                arms, {a: arms[a].joints_to(arms[a].q_home + CLEAR_DQ[a], joint_step=LOOK_STEP) for a in K.ARMS}
            )
            for _ in range(SETTLE):
                yield [hold_action(arms)]
            s2, c2 = head_detect(ctx, cam)
            if s2 is not None:
                sock = s2
            if c2 is not None:
                chg = c2
        if sock is None or chg is None:
            yield from run_arms(arms, {a: arms[a].joints_to(arms[a].q_home, joint_step=LOOK_STEP) for a in K.ARMS})
            while True:
                yield [hold_action(arms)]
        ins = "left" if sock.x < 0 else "right"
        other = "right" if ins == "left" else "left"
        carried = (chg.x < 0) != (sock.x < 0)
        h = charger_h(chg.yaw)
        state: dict = {}
        tasks = {ins: insert_task(arms[ins], ctx, sock, chg, h, state, carried)}
        if carried:
            spot, plan_c = plan_handoff(arms[other], arms[ins], chg, h, sock)
            if spot is None:
                state["handoff"] = "fail"
                tasks[other] = arms[other].joints_to(arms[other].q_home, joint_step=LOOK_STEP)
            else:
                tasks[other] = carry_task(arms[other], chg, h, spot, plan_c, state)
        else:
            tasks[other] = arms[other].joints_to(arms[other].q_home, joint_step=LOOK_STEP)
        yield from run_arms(arms, tasks)
        yield from run_arms(arms, {a: arms[a].joints_to(arms[a].q_home, joint_step=LOOK_STEP) for a in K.ARMS})
        while True:
            yield [hold_action(arms)]


def hold_action(arms) -> dict:
    return actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)
