from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.motion import plan as m
from ...lib.geometry.transforms import axis_angle, make_T
from . import it_see as S
from . import it_task as G

def tcp_now(ctx, arm):
    return m.tcp(arm, ctx.q(arm))

def lag(ctx, arm):
    return float(np.max(np.abs(ctx.q(arm) - ctx.cmd_q[arm])))

def park(ctx, arms, max_dq=0.1):
    yield from m.move_joints(ctx, {a: G.SURVEY_Q for a in arms}, max_dq=max_dq, settle=False)

def _line(ctx, arm, T, step_m=m.STEP_M, step_rad=m.STEP_RAD, settle=True, q_fallback=None, max_dq=0.08):
    try:
        yield from m.move(ctx, {arm: T}, step_m=step_m, step_rad=step_rad, settle=settle, max_settle=6)
        return "line"
    except m.PlanError:
        if q_fallback is None:
            raise
    yield from m.move_joints(ctx, {arm: q_fallback}, max_dq=max_dq, settle=settle)
    return "joint"

def pick(ctx, arm, P, Q):
    ctx.cmd_g[arm] = G.OPEN_APPROACH
    yield from m.move_joints(ctx, {arm: Q["above"]}, max_dq=0.1, settle=True)
    ctx.snap("pre_pinch", cams=("cam_head", "cam_%s_wrist" % arm))
    yield from _line(ctx, arm, P["pinch"], step_m=0.008, q_fallback=Q["pinch"])
    yield from m.grip(ctx, arm, G.HOLD_CMD, max_steps=8)
    g0 = ctx.g(arm)
    ctx.log("pinch", arm=arm, g=round(g0, 3), tcp_err=round(float(np.linalg.norm(tcp_now(ctx, arm)[:3, 3] - P["pinch"][:3, 3])), 4),
            q=np.round(ctx.q(arm), 4), q_cmd=np.round(ctx.cmd_q[arm], 4))
    if g0 < G.HOLD_MIN:
        ctx.cmd_g[arm] = G.OPEN_APPROACH
        yield from m.hold(ctx, 3)
        yield from _line(ctx, arm, P["above"], step_m=0.015, settle=False, q_fallback=Q["above"])
        return False
    yield from _line(ctx, arm, P["lift1"], step_m=0.008, settle=False, q_fallback=Q["lift1"])
    yield from _line(ctx, arm, P["lift"], step_m=0.015, q_fallback=Q["lift"])
    g1 = ctx.g(arm)
    ctx.log("lifted", arm=arm, g=round(g1, 3))
    return g1 >= G.HOLD_MIN

def stand(ctx, arm, P, Q):
    how = yield from _line(ctx, arm, P["stand"], step_m=0.01, step_rad=0.08, q_fallback=Q["stand"], max_dq=0.07)
    g = ctx.g(arm)
    P["cap_ref"] = held_cap_now(ctx, arm, "stood")
    ctx.log("stood", arm=arm, how=how, g=round(g, 3), tip_z=round(G.tip_z_of(tcp_now(ctx, arm)), 4))
    return g >= G.HOLD_MIN

def descend(ctx, arm, T_from, tip_target, watch=None, deadline=None):
    T = T_from.copy()
    n_lag = 0
    n_slip = 0
    blocked = False
    slip = None
    peak, lost = 0.0, 0
    cam = "cam_%s_wrist" % arm
    for _ in range(90):
        tip_cmd = G.tip_z_of(T)
        if tip_cmd <= tip_target + 1e-6:
            break
        T[2, 3] -= min(G.DESCENT_STEP, tip_cmd - tip_target)
        ctx.cmd_q[arm] = m.solve(arm, T, ctx.cmd_q[arm])
        yield
        held_up = T[2, 3] - float(tcp_now(ctx, arm)[2, 3])
        if held_up > G.BLOCK_LAG:
            n_lag += 1
            if n_lag >= 2:
                blocked = True
                break
        else:
            n_lag = 0
        if watch is None:
            continue
        now = S.held_cap(ctx.rgb(cam))
        sl = None if now is None else S.cap_slide(watch, now)
        if now is None:
            lost += 1
        else:
            peak = max(peak, sl)
        bad = now is None or sl > G.SLIDE_MAX or abs(now["cx"] - watch["cx"]) > G.CAP_SHIFT_MAX_PX
        n_slip = n_slip + 1 if bad else 0
        if n_slip >= 2 and (deadline is None or ctx.step + G.REINSERT_STEPS <= deadline):
            slip = G.SLIDE_MAX if sl is None else sl
            break
    tip = G.tip_z_of(tcp_now(ctx, arm))
    if watch is not None:
        ctx.log("descent", arm=arm, tip_z=round(tip, 4), peak=round(peak, 4), lost=lost, slip=None if slip is None else round(slip, 4))
    return tip, blocked, slip

def held_cap_now(ctx, arm, tag):
    cam = "cam_%s_wrist" % arm
    ctx.snap(tag, cams=("cam_head", cam))
    return S.held_cap(ctx.rgb(cam))

def cap_state(ctx, arm, ref, tag):
    now = held_cap_now(ctx, arm, tag)
    if ref is None:
        return False, None, now
    if now is None:
        return True, None, None
    slide = S.cap_slide(ref, now)
    return abs(slide) > G.SLIDE_MAX or abs(now["cx"] - ref["cx"]) > G.CAP_SHIFT_MAX_PX, slide, now

def lean_fix(ctx, arm, tag, gate=None):
    cap = held_cap_now(ctx, arm, tag)
    q = ctx.q(arm)
    fit = S.held_lean(cap, kin.wrist_camera_T(arm, q), m.tcp(arm, q))
    if fit is None:
        ctx.log("lean_fit", arm=arm, at=tag, fit=None)
        return None
    dx, dy = fit["tx"] - G.LEAN_TX_REF, fit["ty"] - G.LEAN_TY_REF
    drop = float(np.clip(-(fit["slide"] - G.LEAN_SLIDE_REF), 0.0, G.CAP_DROP_MAX))
    dev = abs(dy) if gate is None else max(abs(dx), abs(dy))
    act = fit["res"] <= G.LEAN_RES_MAX and dev > (G.LEAN_GATE if gate is None else gate)
    ctx.log("lean_fit", arm=arm, at=tag, tx=round(np.degrees(fit["tx"]), 2), ty=round(np.degrees(fit["ty"]), 2),
            slide=round(fit["slide"], 4), res=round(fit["res"], 1), act=bool(act), drop=round(drop, 4) if act else 0.0)
    if not act:
        return None
    R = m.tcp(arm, q)[:3, :3]
    up = np.array([0.0, 0.0, 1.0])
    a = R @ (axis_angle([1, 0, 0], dx) @ axis_angle([0, 1, 0], dy) @ (R.T @ up))
    k = np.cross(a, up)
    ang = float(np.arcsin(min(1.0, np.linalg.norm(k))))
    R_align = np.eye(3) if ang < 1e-6 else axis_angle(k / np.linalg.norm(k), ang)
    return R_align, drop, fit

def turned(T, R_align):
    c = G.contact_at(T)
    return make_T(R_align @ T[:3, :3], c + R_align @ (T[:3, 3] - c))

def insert(ctx, arm, P, Q, bore, deadline=None):
    yield from _line(ctx, arm, P["carry"], step_m=0.018, step_rad=0.08, q_fallback=Q["carry"], max_dq=0.08)
    ref = P.get("cap_ref")
    E = P["entry"].copy()
    drop = 0.0


    fix = lean_fix(ctx, arm, "carry")
    if fix is not None:
        R_align, drop_fit, _ = fix
        Tc, Te = turned(P["carry"], R_align), turned(P["entry"], R_align)
        Ts = Te.copy()
        Ts[2, 3] -= G.ENTRY_TIP_Z - G.SEAT_TIP_Z
        deg = round(float(np.degrees(np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1, 1)))), 2)
        try:
            q1 = m.solve(arm, Tc, ctx.cmd_q[arm])
            q2 = m.solve(arm, Te, q1)
            m.solve(arm, Ts, q2)
        except m.PlanError as exc:
            ctx.log("lean_turn", arm=arm, deg=deg, skipped=str(exc))
            fix = None
        else:
            yield from _line(ctx, arm, Tc, step_m=0.006, step_rad=0.02, q_fallback=q1)
            E, drop, q_entry = Te, drop_fit, q2
            ctx.log("lean_turn", arm=arm, deg=deg, drop=round(drop, 4))
    if fix is None:
        q_entry = Q["entry"]
    yield from _line(ctx, arm, E, step_m=0.012, q_fallback=q_entry)
    at_entry = held_cap_now(ctx, arm, "entry")
    ctx.log("entry_cap", arm=arm, ref=ref, now=at_entry, drop=drop,
            slide=None if ref is None or at_entry is None else round(S.cap_slide(ref, at_entry), 4))
    tip, blocked, slip = yield from descend(ctx, arm, E, G.SEAT_TIP_Z + drop, watch=at_entry, deadline=deadline)
    if slip is None and blocked and tip - drop > G.DEEP_ENOUGH:

        ctx.log("blocked", arm=arm, tip_z=round(tip, 4), retry=0)
        yield from _line(ctx, arm, E, step_m=0.008, q_fallback=q_entry)
        yield from m.hold(ctx, 2)
        tip, blocked, _ = yield from descend(ctx, arm, E, G.SEAT_TIP_Z + drop)
    depth = seated_depth(ctx, arm, ref, at_entry, tip, drop, blocked, "bottom")
    tries = 0
    while depth > G.DEEP_ENOUGH and tries < G.REINSERT_TRIES and ctx.g(arm) >= G.HOLD_MIN:
        if deadline is not None and ctx.step + G.REINSERT_STEPS > deadline:
            ctx.log("reinsert_skip", arm=arm, step=ctx.step, deadline=deadline)
            break
        tries += 1
        again = yield from realign(ctx, arm, E, q_entry, drop, at_entry, "retry%d" % tries)
        if again is None:
            break
        E, q_entry, drop, at_entry = again
        tip, blocked, _ = yield from descend(ctx, arm, E, G.SEAT_TIP_Z + drop)
        depth = seated_depth(ctx, arm, ref, at_entry, tip, drop, blocked, "bottom%d" % tries)
    ok = depth <= G.DEEP_ENOUGH
    ctx.log("descended", arm=arm, tip_z=round(depth, 4), blocked=bool(blocked), into_bore=round(S.RACK_TOP - depth, 4),
            g=round(ctx.g(arm), 3), held=round(ctx.g(arm), 3) >= G.HOLD_MIN, ok=bool(ok), tries=tries)
    return ok, tip, blocked

def seated_depth(ctx, arm, ref, at_entry, tip, drop, blocked, tag):
    moved, push, now = cap_state(ctx, arm, at_entry, tag)
    slide = None if ref is None or now is None else S.cap_slide(ref, now)
    push = max(0.0, push) if moved and push is not None else 0.0
    ctx.log("bottom", arm=arm, at=tag, tip_z=round(tip, 4), drop=drop, slide=None if slide is None else round(slide, 4),
            push=round(push, 4), moved=bool(moved), blocked=bool(blocked), cap_seen=now is not None)
    return tip - drop + push

def realign(ctx, arm, E, q_entry, drop, at_entry, tag):
    yield from _line(ctx, arm, E, step_m=0.02, q_fallback=q_entry)
    now = held_cap_now(ctx, arm, tag)
    if at_entry is not None and now is None:
        ctx.log("realign", arm=arm, at=tag, cap_seen=False, step=ctx.step)
        return None
    push = 0.0 if at_entry is None else max(0.0, S.cap_slide(at_entry, now))
    fix = lean_fix(ctx, arm, tag, gate=G.REALIGN_GATE)
    if fix is not None:
        Te = turned(E, fix[0])
        Ts = Te.copy()
        Ts[2, 3] -= G.ENTRY_TIP_Z - G.SEAT_TIP_Z - drop + push
        try:
            q2 = m.solve(arm, Te, ctx.cmd_q[arm])
            m.solve(arm, Ts, q2)
        except m.PlanError as exc:
            ctx.log("realign", arm=arm, at=tag, skipped=str(exc))
            fix = None
        else:
            yield from _line(ctx, arm, Te, step_m=0.006, step_rad=0.02, q_fallback=q2)
            E, q_entry = Te, q2
    drop = float(np.clip(drop - push, -G.CAP_DROP_MAX, G.CAP_DROP_MAX))
    cap = held_cap_now(ctx, arm, tag + "_entry") if fix is not None else now
    ctx.log("realign", arm=arm, at=tag, push=round(push, 4), turned=fix is not None, drop=round(drop, 4), step=ctx.step)
    return E, q_entry, drop, cap

def release_and_clear(ctx, arm, P, Q):
    ctx.cmd_g[arm] = G.RELEASE
    for _ in range(8):
        yield
        if ctx.g(arm) >= G.RELEASE - 0.05:
            break
    ctx.snap("released", cams=("cam_head", "cam_%s_wrist" % arm))
    ctx.log("released", arm=arm, g=round(ctx.g(arm), 3))
    yield from _line(ctx, arm, P["rise"], step_m=0.01, settle=False, q_fallback=Q["rise"])
    yield from _line(ctx, arm, P["back"], step_m=0.015, settle=False, q_fallback=Q["back"])

def abandon(ctx, arm, P, Q):
    up = P["entry"].copy()
    up[2, 3] += 0.06
    yield from _line(ctx, arm, up, step_m=0.012, settle=False, q_fallback=Q["carry"])
    down = P["stand"].copy()
    yield from _line(ctx, arm, down, step_m=0.018, settle=True, q_fallback=Q["stand"])
    ctx.cmd_g[arm] = 1.0
    yield from m.hold(ctx, 4)
    up2 = down.copy()
    up2[2, 3] += 0.05
    yield from _line(ctx, arm, up2, step_m=0.015, settle=False, q_fallback=Q["stand"])
