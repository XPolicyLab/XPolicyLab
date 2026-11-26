
from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.geometry.transforms import axis_angle, make_T, rot_angle, rot_err, wrap

STEP_M = 0.012
STEP_RAD = 0.06
SLOW_M = 0.005
MAX_LINE_JUMP = 0.25
TRANSIT_M = 0.10
TRANSIT_RAD = 0.6
SAFE_Z = 0.86

class PlanError(Exception):
    pass

def tcp(arm, q, offset=kin.TCP_X):
    return kin.tcp_from_link6(kin.fk(arm, q), offset)

def plan_line(arm, q_start, T_goal, step_m=STEP_M, step_rad=STEP_RAD, offset=kin.TCP_X):
    T0 = tcp(arm, q_start, offset)
    p0, p1 = T0[:3, 3], T_goal[:3, 3]
    rv = rot_err(T_goal[:3, :3], T0[:3, :3])
    n = int(max(1, np.ceil(np.linalg.norm(p1 - p0) / step_m), np.ceil(np.linalg.norm(rv) / step_rad)))
    q = np.array(q_start, float)
    out = []
    for i in range(1, n + 1):
        t = i / n
        R = axis_angle(rv, np.linalg.norm(rv) * t) @ T0[:3, :3] if np.linalg.norm(rv) > 1e-9 else T0[:3, :3]
        T6 = kin.link6_from_tcp(make_T(R, p0 + t * (p1 - p0)), offset)
        qn, ep, er = kin.ik(arm, T6, q)
        if ep > 2e-3 or er > 0.03:
            qn, ep, er = kin.ik_multi(arm, T6, q, extra_seeds=_SEEDS)
            if ep > 2e-3 or er > 0.03:
                raise PlanError(f"{arm} line IK failed at {i}/{n}: ep={ep:.4f} er={er:.3f}")
        qn = q + wrap(qn - q)
        jump = float(np.max(np.abs(qn - q)))
        if jump > MAX_LINE_JUMP:
            raise PlanError(f"{arm} line IK jumped {jump:.2f} rad at {i}/{n}")
        q = qn
        out.append(q.copy())
    return out

def plan_joint(q_start, q_goal, max_dq=0.06):
    q0, q1 = np.asarray(q_start, float), np.asarray(q_goal, float)
    n = int(max(1, np.ceil(np.max(np.abs(q1 - q0)) / max_dq)))
    return [q0 + (q1 - q0) * (i / n) for i in range(1, n + 1)]

_SEEDS = (
    np.array([0.0, 1.6, 1.0, -0.9, 0.0, 1.5]),
    np.array([0.0, 2.1, 1.6, -1.1, 0.0, 1.5]),
    np.array([0.0, 1.3, 1.3, -1.5, 0.0, 1.5]),
)

def follow(ctx, plans, settle=True, tol=0.012, max_settle=10):
    n = max((len(v) for v in plans.values()), default=0)
    for i in range(n):
        for arm, qs in plans.items():
            if i < len(qs):
                ctx.cmd_q[arm] = qs[i]
        yield
    if settle:
        yield from settle_arms(ctx, list(plans), tol, max_settle)

def settle_arms(ctx, arms, tol=0.012, max_steps=10):
    for _ in range(max_steps):
        if all(np.max(np.abs(wrap(ctx.q(a) - ctx.cmd_q[a]))) < tol for a in arms):
            return
        yield

def plan_move(arm, q_start, T, step_m=STEP_M, step_rad=STEP_RAD):
    try:
        return plan_line(arm, q_start, T, step_m, step_rad)
    except PlanError:
        T0 = tcp(arm, q_start)
        if np.linalg.norm(T[:3, 3] - T0[:3, 3]) < 0.06 and rot_angle(T[:3, :3], T0[:3, :3]) < 0.8:
            return plan_joint(q_start, solve(arm, T, q_start), 0.05)
        raise

def move(ctx, targets, step_m=STEP_M, step_rad=STEP_RAD, settle=True, max_settle=10):
    plans = {arm: plan_move(arm, ctx.cmd_q[arm], T, step_m, step_rad) for arm, T in targets.items()}
    yield from follow(ctx, plans, settle=settle, max_settle=max_settle)

def move_joints(ctx, targets, max_dq=0.06, settle=True):
    plans = {arm: plan_joint(ctx.cmd_q[arm], q, max_dq) for arm, q in targets.items()}
    yield from follow(ctx, plans, settle=settle)

def grip(ctx, arms, value, max_steps=10, min_steps=3):
    if isinstance(arms, str):
        arms = (arms,)
    for a in arms:
        ctx.cmd_g[a] = float(value)
    prev = None
    for i in range(max_steps):
        yield
        cur = np.array([ctx.g(a) for a in arms])
        if i + 1 >= min_steps and prev is not None and np.max(np.abs(cur - prev)) < 0.004:
            break
        prev = cur

def holding(ctx, arm, threshold=0.05):
    return ctx.cmd_g[arm] < 0.2 and ctx.g(arm) > threshold

def hold(ctx, n):
    for _ in range(n):
        yield

def home(ctx, arms=("left", "right"), max_dq=0.07, settle=True):
    yield from move_joints(ctx, {a: ctx.home_q[a] for a in arms}, max_dq=max_dq, settle=settle)

def down_pose(p, yaw):
    return make_T(kin.down_rotation(yaw), np.asarray(p, float))

def best_down_yaw(arm, yaw_obj, q_ref, period=np.pi):
    cands = [yaw_obj + k * period for k in range(-2, 3)] if period else [yaw_obj]
    T6 = kin.fk(arm, q_ref)
    cur_y = T6[:3, 1]
    cur_yaw = np.arctan2(cur_y[1], cur_y[0])
    return min(cands, key=lambda y: abs(wrap(y - cur_yaw)))

def parallel(*gens):
    # Lazy import avoids the runtime -> kinematics -> motion import cycle.
    from ...runtime.session import SwitchRequest

    gens = list(gens)
    while gens:
        alive = []
        for g in gens:
            try:
                # A paused terminal child must bubble before advancing another
                # arm. Pure-code comparison resumes it within the same code tick.
                while True:
                    yielded = next(g)
                    if not isinstance(yielded, SwitchRequest):
                        break
                    yield yielded
                alive.append(g)
            except StopIteration:
                pass
        gens = alive
        if gens:
            yield

def reachable(arm, poses, q_seed=None):
    q = np.zeros(6) if q_seed is None else np.asarray(q_seed, float)
    for T in poses:
        q, ep, er = kin.ik_multi(arm, kin.link6_from_tcp(T), q, extra_seeds=_SEEDS)
        if ep > 2e-3 or er > 0.03:
            return False
    return True

def approach_pose(p, azimuth, pitch):
    a = np.array([np.cos(pitch) * np.cos(azimuth), np.cos(pitch) * np.sin(azimuth), -np.sin(pitch)])
    j = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return make_T(np.stack([a, j, np.cross(a, j)], axis=1), np.asarray(p, float))

def solve(arm, T, q_ref):
    seeds = (np.asarray(q_ref, float),) + _SEEDS + (np.zeros(6),)
    best = None
    for s in seeds:
        q, ep, er = kin.ik(arm, kin.link6_from_tcp(T), s)
        if ep < 2e-3 and er < 0.02:
            d = np.max(np.abs(wrap(q - q_ref)))
            if best is None or d < best[0]:
                best = (d, q)
    if best is None:
        raise PlanError(f"{arm} pose unreachable at {np.round(T[:3, 3], 3)}")
    return q_ref + wrap(best[1] - q_ref)

READY_TCP = {"left": (-0.24, -0.20, 0.95), "right": (0.24, -0.20, 0.95)}
_READY_Q = {}
STAND = (-0.07, 0.07, -0.60, -0.36)

def ready_q(arm):
    if arm not in _READY_Q:
        _READY_Q[arm] = solve(arm, down_pose(READY_TCP[arm], np.pi / 2), np.zeros(6))
    return _READY_Q[arm]

def is_down(T, tol=0.6):
    return float(T[:3, 0] @ np.array([0.0, 0.0, -1.0])) > np.cos(tol)

def joint_path_ok(arm, qs, z_floor):
    for q in qs[::2] + qs[-1:]:
        frames = kin.fk_chain(arm, q)
        pts = [T[:3, 3] for T in frames[2:]] + [kin.tcp_from_link6(frames[-1], kin.FINGER_TIP_X)[:3, 3]]
        for p in pts:
            if p[2] < z_floor:
                return False
            if STAND[0] < p[0] < STAND[1] and STAND[2] < p[1] < STAND[3]:
                return False
    return True

def _lines(arm, q, poses, step_m, step_rad):
    out = []
    for T in poses:
        seg = plan_line(arm, q, T, step_m, step_rad)
        out += seg
        q = seg[-1] if seg else q
    return out

def plan_goto(arm, q_start, T, step_m=STEP_M, step_rad=STEP_RAD, max_dq=0.08):
    q0 = np.asarray(q_start, float)
    T0 = tcp(arm, q0)
    dist = np.linalg.norm(T[:3, 3] - T0[:3, 3])
    turn = rot_angle(T[:3, :3], T0[:3, :3])
    if dist <= TRANSIT_M and turn <= TRANSIT_RAD:
        try:
            return plan_line(arm, q0, T, step_m, step_rad)
        except PlanError:
            pass
    if is_down(T0) and is_down(T):
        z = max(SAFE_Z, T0[2, 3], T[2, 3])
        up, over = T0.copy(), T.copy()
        up[2, 3] = z
        over[2, 3] = z
        poses = [P for P, needed in ((up, T0[2, 3] < z - 1e-3), (over, True), (T, T[2, 3] < z - 1e-3)) if needed]
        try:
            return _lines(arm, q0, poses, step_m, step_rad)
        except PlanError:
            pass
    q1 = solve(arm, T, q0)
    floor = min(T0[2, 3], T[2, 3]) - 0.03
    direct = plan_joint(q0, q1, max_dq)
    if joint_path_ok(arm, direct, floor):
        return direct
    qr = ready_q(arm)
    via = plan_joint(q0, qr, max_dq)
    if joint_path_ok(arm, via, min(T0[2, 3], 0.9) - 0.03):
        rest = plan_goto(arm, qr, T, step_m, step_rad, max_dq) if is_down(T) else plan_joint(qr, solve(arm, T, qr), max_dq)
        return via + rest
    return direct

def ready(ctx, arms=("left", "right"), max_dq=0.11):
    yield from move_joints(ctx, {a: ready_q(a) for a in arms}, max_dq=max_dq, settle=False)

def goto(ctx, targets, step_m=STEP_M, step_rad=STEP_RAD, max_dq=0.08, settle=True, max_settle=10):
    plans = {arm: plan_goto(arm, ctx.cmd_q[arm], T, step_m, step_rad, max_dq) for arm, T in targets.items()}
    yield from follow(ctx, plans, settle=settle, max_settle=max_settle)

def look_pose(xy, height, yaw):
    from ...lib.geometry.transforms import TABLE_Z

    up = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    T_cam = make_T(np.stack([x, up, z], axis=1), np.array([xy[0], xy[1], TABLE_Z + height]))
    return kin.tcp_from_link6(kin.link6_for_camera(T_cam))

def best_look(arm, xy, height, q_ref, yaws=None):
    yaws = np.deg2rad(np.arange(0, 360, 30)) if yaws is None else yaws
    best = None
    for yaw in yaws:
        T = look_pose(xy, height, yaw)
        try:
            q = solve(arm, T, q_ref)
        except PlanError:
            continue
        d = np.max(np.abs(wrap(q - q_ref)))
        if best is None or d < best[0]:
            best = (d, T, q)
    return (None, None) if best is None else (best[1], best[2])


def settle(ctx, arm, tol=0.01, max_steps=8):
    for _ in range(max_steps):
        if np.max(np.abs(wrap(ctx.q(arm) - ctx.cmd_q[arm]))) < tol:
            return
        yield


def tcp_now(ctx, arm):
    return tcp(arm, ctx.cmd_q[arm])
