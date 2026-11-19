from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...core import actions
from ...perception import cameras
from . import scene as SS
from ...planning import collision as CO
from ...planning import kinematics as K
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, Zone, link_arms, run_arms, tcp_pose
from .. import Program, register

STEP_LIMIT = 1000
HOME_RESERVE = 50
ITEM_STEPS = 78
LATE_SURVEY_MIN = 100
LATE_PICK_MIN = 90
IGNORE = "broom_shovel", "broom", "camera_stand"
OBJ_REGION = (-0.05, 0.45), (-0.24, 0.14)
HANDLE_GRASP_YS = 0.07, 0.1, 0.12, 0.055
HANDLE_W = 0.03
HANDLE_OPEN = grip_for_gap(HANDLE_W + 0.024)
WALL_X = 0.0905
WALL_POINTS = (-0.06, 0.0114), (-0.07, 0.0088), (-0.05, 0.0136)
WALL_W = 0.004
WALL_OPEN = grip_for_gap(0.036)
WALL_LIFT = 0.004
DUSTPAN_LIFT = 0.01
SC_TARGET = np.array(
    [-0.06, -0.25]
)
CLEAR = 0.03
DROP_Y_MIN = -0.08
DROP_R_SAFE = 0.056
WALL_IN = 0.088, 0.016
FINGER_OUT = 0.02
RELEASE_GAP = 0.02
DROP_CLEAR = 0.015
DROP_BACK_W = 6.0
DROP_BEHIND_PENALTY = 0.3
DROP_BEHIND_MARGIN = 0.02
DROP_BEHIND_DY = 0.005
DROP_R = 0.01, 0.035
_DROP_GRID = np.stack(np.meshgrid(np.arange(-0.08, 0.0801, 0.005), np.arange(-0.13, 0.0201, 0.005)), -1).reshape(-1, 2)
DROP_ABOVE = 0.015
PAN_FLOOR = K.TABLE_Z + 0.003
WALL_TOP = K.TABLE_Z + 0.05
CARRY_CLEAR = 0.075
HOVER_DZ = 0.06
FAST_STEP = 0.1
ITEM_JOINT_STEP = 0.13
APPROACH_STEP = 0.014
LIFT_STEP = 0.02
DROP_STEP = 0.018
BROOM_GRASP = 0.0, 0.0102, -0.03
BROOM_AXIS_Y = 0.0036, -0.2303
BROOM_CLOSE_DYAW = math.radians(12.9)


def broom_axis_y(lz: float) -> float:
    return BROOM_AXIS_Y[0] + BROOM_AXIS_Y[1] * lz


def aligned_broom(br: "SS.Fit", tcp_xy, lz: float) -> "SS.Fit":
    yaw = br.yaw - BROOM_CLOSE_DYAW
    off = SS.Fit(SS.BROOM, 0.0, 0.0, yaw, br.iou).to_world((0.0, broom_axis_y(lz), lz))[0][:2]
    x, y = np.asarray(tcp_xy[:2], dtype=float) - off
    return SS.Fit(SS.BROOM, float(x), float(y), yaw, br.iou, dict(br.extra))


BROOM_GRASP_ZS = -0.03, -0.06
BROOM_GRASP_ZS_LAST = (
    BROOM_GRASP_ZS + (-0.09,)
)
BROOM_W = 0.028
BROOM_OPEN = grip_for_gap(BROOM_W + 0.026)
PITCHES = math.radians(10.0), math.radians(20.0), math.radians(30.0)
BROOM_PITCHES = PITCHES[:2]
BROOM_PITCH_GAP = BROOM_W + 0.018
BROOM_PITCH_OPEN = grip_for_gap(BROOM_PITCH_GAP)
BROOM_CARRY = K.TABLE_Z + 0.09
BROOM_CARRY_HI = K.TABLE_Z + 0.15
BROOM_CARRY_ZS = BROOM_CARRY_HI, K.TABLE_Z + 0.12
STAND_BOX = (-0.17, 0.17), (-0.6, -0.37)
SURVEY_ROUNDS = 3
ZONE_R = 0.09
TABLE_REGION = (-0.55, 0.55), (-0.4, 0.2)
INSIDE_MARGIN = 0.005


@dataclass
class Item:
    category: str
    xy: np.ndarray
    half_h: float
    width: float
    yaw: float
    length: float = 0.0
    state: str = "todo"
    tries: int = 0


@dataclass
class Shared:
    dustpan: SS.Fit | None = None
    broom: SS.Fit | None = None
    dustpan_placed: bool = False
    drops_open: bool = False
    broom_done: bool = False
    left_homing: bool = False
    pick_open: bool = False
    broom_recovery: bool = False
    target: SS.Fit | None = None
    items: list = field(default_factory=list)
    in_pan: list = field(default_factory=list)
    rollouts: list = field(default_factory=list)
    rollout_looks: int = 0
    log: list = field(default_factory=list)


def now(arm: Arm) -> int:
    return int(arm.ctx.obs.step)


def g_obs(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


THIN_G = 0.005


def holds(arm: Arm) -> tuple[bool, float, float]:
    g = g_obs(arm)
    w = held_width(g)
    return w >= 0.004 or g > THIN_G, w, g


def broom_held(arm: Arm) -> bool:
    return holds(arm)[0]


def steps_left(arm: Arm) -> int:
    return STEP_LIMIT - now(arm)


def go(arm: Arm, make, tries: int = 3, retreat=None):
    for _ in range(tries):
        yield from make()
        if arm.status != "blocked":
            break
        if retreat is not None:
            yield from retreat()
        yield from arm.hold(6)
    return arm.status == "ok"


def think(gen, arms):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        yield [actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)]


def grasps_iter(arm_name: str, center, size: float, yaw: float, q_cur, *, tilts=G.TILTS, **kw):
    for i, t in enumerate(tilts):
        if i:
            yield
        gs = G.grasp_candidates(arm_name, center, size, yaw, q_cur, tilts=(t,), **kw)
        if gs:
            return gs
    return []


def footprint(cam, mask, z):
    import cv2

    vv, uu = np.nonzero(mask)
    xy = cam.on_plane_z(np.c_[uu, vv].astype(float), z)[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    (cx, cy), (w, h), ang = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    a = math.radians(ang)
    if w <= h:
        return np.array([cx, cy]) / 1000.0, w / 1000.0, a, h / 1000.0
    return np.array([cx, cy]) / 1000.0, h / 1000.0, a + math.pi / 2, w / 1000.0


THICK_END = ("earbuds",)
THICK_LEVEL = 0.85


def thickest_point(cam, mask, z: float):
    import cv2

    d = cv2.distanceTransform(np.ascontiguousarray(mask, dtype=np.uint8), cv2.DIST_L2, 5)
    dm = float(d.max())
    if dm <= 0.0:
        return None
    vv, uu = np.nonzero(d >= THICK_LEVEL * dm)
    p = cam.on_plane_z(np.array([[float(uu.mean()), float(vv.mean())]]), z)[0, :2]
    return p if np.isfinite(p).all() else None


def detect_items(cam, insts, dustpan: SS.Fit | None = None):
    items = []
    for inst in insts:
        if inst.category in IGNORE:
            continue
        hh = SS.OBJECT_HALF_H.get(inst.category, SS.DEFAULT_HALF_H)
        c, w, yaw, ln = footprint(cam, inst.mask, K.TABLE_Z + hh)
        (x0, x1), (y0, y1) = OBJ_REGION if dustpan is None else TABLE_REGION
        if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
            continue
        if dustpan is not None and in_dustpan(dustpan, c, INSIDE_MARGIN):
            continue
        g = c
        if inst.category in THICK_END:
            p = thickest_point(cam, inst.mask, K.TABLE_Z + hh)
            if p is not None and float(np.linalg.norm(p - c)) <= 0.5 * ln:
                g = p
        items.append(Item(inst.category, g, hh, float(np.clip(w, 0.012, 0.07)), yaw, float(ln)))
    return items


def pan_objects(cam, insts, dustpan: SS.Fit):
    out = []
    for inst in insts:
        if inst.category in IGNORE:
            continue
        hh = SS.OBJECT_HALF_H.get(inst.category, SS.DEFAULT_HALF_H)
        c, w, _, ln = footprint(cam, inst.mask, K.TABLE_Z + hh)
        if in_dustpan(dustpan, c, INSIDE_MARGIN):
            out.append((to_local(dustpan, c), float(np.clip(0.25 * (max(ln, w) + w), *DROP_R))))
    return out


def in_dustpan(dp: SS.Fit, xy, margin: float = 0.0) -> bool:
    return float(np.linalg.norm(np.asarray(xy)[:2] - SS.support_center(dp))) < SS.SUPPORT_R - margin


def perceive(ctx):
    from ...perception import seg

    cam = cameras.head()
    img = ctx.obs.images["cam_head"]
    insts = seg.segment(img, camera="head")
    yield
    dp = seg.of_category(insts, SS.DUSTPAN)
    br = seg.of_category(insts, SS.BROOM)
    dustpan = (yield from SS.fit_iter(cam, dp[0].mask, SS.DUSTPAN)) if dp else None
    yield
    broom = (yield from SS.fit_iter(cam, br[0].mask, SS.BROOM)) if br else None
    items = detect_items(cam, insts)
    return dustpan, broom, items


def footprint_dist(pts_xy: np.ndarray, items, broom_xy=None) -> float:
    d = [float(np.min(np.linalg.norm(pts_xy - it.xy, axis=1))) - 0.5 * max(it.width, it.length) for it in items]
    if broom_xy is not None:
        d.append(float(np.min(np.linalg.norm(pts_xy[:, None, :] - broom_xy[None, :, :], axis=2))))
    return min(d + [1.0])


def dustpan_target(dp: SS.Fit, items, broom: SS.Fit | None) -> tuple[float, float, float]:
    sc_local = np.array([SS.SUPPORT_C[0], SS.SUPPORT_C[1]])
    broom_xy = None if broom is None else SS.footprint_points(SS.BROOM, broom.x, broom.y, broom.yaw)
    best = None
    for dx in (0.0, -0.03, 0.03, -0.06):
        for dy in (0.0, -0.03, 0.03, -0.06):
            sc = SC_TARGET + [dx, dy]
            c, s = math.cos(dp.yaw), math.sin(dp.yaw)
            org = sc - np.array([c * sc_local[0] - s * sc_local[1], s * sc_local[0] + c * sc_local[1]])
            pts = SS.footprint_points(SS.DUSTPAN, org[0], org[1], dp.yaw)
            if pts[:, 1].min() < STAND_BOX[1][1] + 0.01 and abs(org[0]) < 0.25:
                continue
            dmin = footprint_dist(pts, items, broom_xy)
            if dmin >= CLEAR:
                return float(org[0]), float(org[1]), dp.yaw
            if best is None or dmin > best[0]:
                best = dmin, float(org[0]), float(org[1])
    if best is None:
        best = _dustpan_target_fallback(dp, items, broom_xy, sc_local)
    return best[1], best[2], dp.yaw


STAND_MARGIN = 0.01


def stand_clearance(pts: np.ndarray) -> float:
    (x0, x1), (y0, y1) = STAND_BOX
    dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - y1), 0.0)
    out = np.hypot(dx, dy)
    inside = out == 0.0
    if inside.any():
        p = pts[inside]
        depth = np.minimum(np.minimum(p[:, 0] - x0, x1 - p[:, 0]), np.minimum(p[:, 1] - y0, y1 - p[:, 1]))
        return -float(depth.max())
    return float(out.min())


def _dustpan_target_fallback(dp: SS.Fit, items, broom_xy, sc_local):
    c, s = math.cos(dp.yaw), math.sin(dp.yaw)
    clear_stand = far_stand = None
    for dx in (0.0, 0.03, -0.03, -0.06):
        for dy in (0.0, 0.03, 0.06, 0.09, -0.03):
            sc = SC_TARGET + [dx, dy]
            org = sc - np.array([c * sc_local[0] - s * sc_local[1], s * sc_local[0] + c * sc_local[1]])
            pts = SS.footprint_points(SS.DUSTPAN, org[0], org[1], dp.yaw)
            ds = stand_clearance(pts)
            if far_stand is None or ds > far_stand[0]:
                far_stand = ds, float(org[0]), float(org[1])
            if ds < STAND_MARGIN:
                continue
            dmin = footprint_dist(pts, items, broom_xy)
            if dmin >= CLEAR:
                return dmin, float(org[0]), float(org[1])
            if clear_stand is None or dmin > clear_stand[0]:
                clear_stand = dmin, float(org[0]), float(org[1])
    return clear_stand if clear_stand is not None else far_stand


def handle_grasps(arm: Arm, dp_pose: tuple[float, float, float], q_cur):
    x, y, yaw = dp_pose
    R = K.rot_z(yaw)
    o = np.array([x, y, K.TABLE_Z + SS.ORIGIN_DZ[SS.DUSTPAN]])
    for ly in HANDLE_GRASP_YS:
        p = o + R @ np.array([0.0, ly, 0.5 * (SS.HANDLE_Z[0] + SS.HANDLE_Z[1])])
        gs = yield from grasps_iter(arm.name, p, HANDLE_W, yaw, q_cur, blocked_yaws=(yaw + math.pi / 2,), hover_dz=0.06)
        if gs:
            yield (gs[0], "handle")
        yield
    walls = sorted((-WALL_X, WALL_X), key=lambda lx: float((R @ np.array([lx, 0.0, 0.0]))[1]))
    for lx in walls:
        for ly, ztop in WALL_POINTS:
            p = o + R @ np.array([lx, ly, ztop - 0.005])
            gs = yield from grasps_iter(
                arm.name, p, WALL_W, yaw, q_cur, blocked_yaws=(yaw + math.pi / 2,), hover_dz=0.06
            )
            if gs:
                yield (gs[0], "wall")
            yield
    for ly in HANDLE_GRASP_YS:
        p = o + R @ np.array([0.0, ly, 0.5 * (SS.HANDLE_Z[0] + SS.HANDLE_Z[1])])
        gs = yield from pitched_grasps(arm.name, p, (yaw,), q_cur, HANDLE_W + 0.024, 0.06)
        if gs:
            yield (gs[0], "handle")
        yield


def move_dustpan(arm: Arm, sh: Shared, refit: bool = True):
    dp = sh.dustpan
    yield
    grasps = handle_grasps(arm, (dp.x, dp.y, dp.yaw), arm.q.copy())
    tx, ty, tyaw = sh.target.x, sh.target.y, sh.target.yaw
    if abs(math.remainder(dp.yaw - tyaw, 2 * math.pi)) > 1e-06:
        sc = np.asarray(SS.support_center(sh.target), dtype=float)[:2]
        c, s = math.cos(dp.yaw), math.sin(dp.yaw)
        tx = float(sc[0] - (c * SS.SUPPORT_C[0] - s * SS.SUPPORT_C[1]))
        ty = float(sc[1] - (s * SS.SUPPORT_C[0] + c * SS.SUPPORT_C[1]))
        tyaw = dp.yaw
        sh.target = SS.Fit(SS.DUSTPAN, tx, ty, tyaw, 1.0)
    tried = 0
    for item in grasps:
        if item is None:
            yield
            continue
        gr, kind = item
        if tried >= 2:
            break
        tried += 1
        g_open, min_w, lift = (HANDLE_OPEN, 0.01, DUSTPAN_LIFT) if kind == "handle" else (WALL_OPEN, 0.0015, WALL_LIFT)
        yield
        hover = tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], gr.hover_z])
        ok = yield from go(
            arm,
            lambda: arm.transit(
                hover, min(K.TABLE_Z + 0.1, gr.hover_z), grip=g_open, joint_step=FAST_STEP, seeds=(gr.q_hover,)
            ),
        )
        ok = ok and (yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, gr.tcp), lin_step=0.008)))
        if not ok:
            yield from go(arm, lambda: arm.line_to(hover, lin_step=0.012))
            continue
        yield from arm.set_grip(0.0, 10)
        ok_h, w, _ = holds(arm)
        if kind == "handle" and (not ok_h or w < min_w):
            yield from arm.set_grip(g_open, 6)
            yield from go(arm, lambda: arm.line_to(hover, lin_step=0.012))
            if refit:
                refit = False
                yield from go(arm, lambda: arm.joints_to(arm.q_home, joint_step=FAST_STEP), tries=3)
                yield from refit_dustpan(arm, sh)
                if sh.dustpan is not dp:
                    return (yield from move_dustpan(arm, sh, refit=False))
            continue
        d = np.array([tx - dp.x, ty - dp.y, 0.0])
        up = gr.tcp + [0, 0, lift]
        yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, up), lin_step=0.004))
        yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, up + d), lin_step=0.008), tries=6)
        yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, gr.tcp + d), lin_step=0.004))
        yield from arm.set_grip(g_open, 8)
        yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, gr.tcp + d + [0, 0, 0.07]), lin_step=0.01))
        sh.dustpan = SS.Fit(SS.DUSTPAN, tx, ty, tyaw, dp.iou, {"moved": True})
        sh.dustpan_placed = True
        return True
    return False


BROOM_DP_CLEAR = 0.03
BROOM_TARGET_CLEAR = 0.05
BROOM_TARGET_PENALTY = 0.05
PARK_CLEAR = 0.05
MAX_IK_CANDS = 30
MAX_REACH_XY = 0.55


def ik_first(arm: str, T: np.ndarray, seeds, iters: int = 60):
    for sd in seeds:
        r = K.solve(arm, T, sd, K.TCP_OFFSET, iters=iters)
        if r.ok:
            return r


def dustpan_sweep(sh: Shared, swept: bool = True) -> np.ndarray:
    dp, d0 = sh.target, sh.dustpan
    sweep = [SS.footprint_points(SS.DUSTPAN, dp.x, dp.y, dp.yaw)]
    if swept and d0 is not None and (not sh.dustpan_placed):
        for t in (0.0, 0.5):
            sweep.append(
                SS.footprint_points(
                    SS.DUSTPAN, d0.x + t * (dp.x - d0.x), d0.y + t * (dp.y - d0.y), d0.yaw + t * (dp.yaw - d0.yaw)
                )
            )
    return np.concatenate(sweep)


def broom_target(sh: Shared, arm: Arm, gr, swept: bool = True, br: "SS.Fit | None" = None):
    from scipy.spatial import cKDTree

    br, dp = sh.broom if br is None else br, sh.target
    dp_tree = cKDTree(dustpan_sweep(sh, swept))
    tgt_tree = cKDTree(SS.footprint_points(SS.DUSTPAN, dp.x, dp.y, dp.yaw))
    q_other = arm.other.q_home if arm.other is not None and (not sh.drops_open) else None
    todo = [it for it in sh.items if it.state != "dropped"]
    rel = gr.tcp[:2] - np.array([br.x, br.y])
    cands = []
    for k, dyaw in enumerate(np.radians((0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0))):
        if k == 4:
            yield
        yaw = br.yaw + dyaw
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        for x in np.arange(dp.x + 0.14, dp.x + 0.3, 0.03):
            for y in (-0.31, -0.28, -0.34, -0.25, -0.22, -0.37):
                p_tcp = np.array([x, y]) + [c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]]
                pts = SS.footprint_points(SS.BROOM, x, y, yaw)
                (sx0, sx1), (sy0, sy1) = STAND_BOX
                if np.any((pts[:, 0] > sx0) & (pts[:, 0] < sx1) & (pts[:, 1] > sy0) & (pts[:, 1] < sy1)):
                    continue
                if footprint_dist(pts, todo) < 0.02 or float(dp_tree.query(pts)[0].min()) < BROOM_DP_CLEAR:
                    continue
                cost = abs(dyaw) * 0.05 + abs(x - (dp.x + 0.15)) + 0.2 * abs(y + 0.31)
                if float(tgt_tree.query(pts)[0].min()) < BROOM_TARGET_CLEAR:
                    cost += BROOM_TARGET_PENALTY
                cands.append((cost, float(x), float(y), dyaw, p_tcp))
    yield
    cands.sort(key=lambda t: t[0])
    bx, by = K.BASE_POS[arm.name][:2]
    n_ik = 0
    for cost, x, y, dyaw, p_tcp in cands:
        if math.hypot(p_tcp[0] - bx, p_tcp[1] - by) > MAX_REACH_XY:
            continue
        if n_ik >= MAX_IK_CANDS:
            break
        n_ik += 1
        if n_ik % 2 == 0:
            yield
        rot_new = K.rot_z(dyaw) @ gr.rot
        seeds = gr.q, gr.q + np.array([0, 0, 0, 0, 0, dyaw]), G.base_seed(arm.name, p_tcp)
        r = ik_first(arm.name, tcp_pose(rot_new, [p_tcp[0], p_tcp[1], BROOM_CARRY]), seeds)
        if r is None:
            continue
        r2 = ik_first(arm.name, tcp_pose(rot_new, [p_tcp[0], p_tcp[1], gr.tcp[2] + 0.008]), (r.q,))
        if r2 is None:
            continue
        if q_other is not None and min(CO.arm_distance(r.q, q_other), CO.arm_distance(r2.q, q_other)) < PARK_CLEAR:
            continue
        return x, y, dyaw, p_tcp, rot_new, br.yaw + dyaw


def pitched_grasps(arm_name: str, p, yaws, q_cur, gap: float, hover_dz: float = 0.06, pitches=PITCHES):
    bx, by = K.BASE_POS[arm_name][:2]
    rad = np.array([p[0] - bx, p[1] - by])
    rad /= max(1e-09, float(np.linalg.norm(rad)))
    seed0 = G.base_seed(arm_name, p[:2])
    out = []
    for i, th in enumerate(pitches):
        if i:
            yield
        z = max(float(p[2]), K.TABLE_Z + G.TIP_CLEARANCE + G.FINGER_DEPTH * math.cos(th) + 0.5 * gap * math.sin(th))
        tcp = np.array([p[0], p[1], z])
        for c0 in yaws:
            for c in (c0, c0 + math.pi):
                r0 = K.top_down(c)
                for sgn in (1.0, -1.0):
                    rot = K.axis_angle(r0[:, 2], sgn * th) @ r0
                    if float(np.dot(rot[:2, 0], rad)) < 0.5 * math.sin(th):
                        continue
                    seeds = (q_cur, seed0) + tuple(
                        seed0 + np.array([0, 0, 0, 0, 0, a]) for a in (c, c + math.pi, c - math.pi)
                    )
                    r = ik_first(arm_name, tcp_pose(rot, tcp), seeds)
                    if r is None:
                        continue
                    rh = ik_first(arm_name, tcp_pose(rot, tcp + [0, 0, hover_dz]), (r.q,))
                    if rh is None:
                        continue
                    g = G.Grasp(
                        arm_name,
                        rot,
                        c,
                        sgn * th,
                        tcp,
                        r.q,
                        z + hover_dz,
                        rh.q,
                        float(np.sum(K.JOINT_COST_W * np.abs(rh.q - np.asarray(q_cur, dtype=float)))),
                    )
                    g.pitch = sgn * th
                    out.append(g)
        if out:
            break
    out.sort(key=lambda g: g.cost)
    return out


def broom_pitched_grasps(arm_name: str, p, yaw: float, q_cur, hover_dz: float = 0.06):
    return (yield from pitched_grasps(arm_name, p, (yaw,), q_cur, BROOM_PITCH_GAP, hover_dz, BROOM_PITCHES))


def plan_broom(arm: Arm, sh: Shared, swept: bool = True, zs=BROOM_GRASP_ZS, skip=()):
    br = sh.broom
    any_grasp = False
    for pitched in (False, True):
        if pitched and any_grasp:
            break
        for lz in zs:
            if (lz, pitched) in skip:
                continue
            p = br.to_world((BROOM_GRASP[0], BROOM_GRASP[1], lz))[0]
            p[2] = K.TABLE_Z + 0.012
            if pitched:
                gs = yield from broom_pitched_grasps(arm.name, p, br.yaw, arm.q)
            else:
                gs = yield from grasps_iter(
                    arm.name, p, BROOM_W, br.yaw, arm.q, blocked_yaws=(br.yaw + math.pi / 2,), hover_dz=0.06
                )
            if not gs:
                continue
            any_grasp = True
            yield
            tgt = yield from broom_target(sh, arm, gs[0], swept, br=aligned_broom(br, gs[0].tcp, lz))
            if tgt is None:
                continue
            return gs[0], tgt, (lz, pitched)


def exec_broom(arm: Arm, sh: Shared, gr, tgt):
    br = sh.broom
    tx, ty, dyaw, p_new, rot_new = tgt[:5]
    yaw_placed = tgt[5] if len(tgt) > 5 else br.yaw + dyaw
    yield
    home = lambda: arm.joints_to(arm.q_home)
    hover = tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], gr.hover_z])
    open_g = BROOM_PITCH_OPEN if getattr(gr, "pitch", 0.0) else BROOM_OPEN
    ok = yield from go(
        arm,
        lambda: arm.transit(
            hover, min(K.TABLE_Z + 0.1, gr.hover_z), grip=open_g, joint_step=FAST_STEP, seeds=(gr.q_hover,)
        ),
        tries=6,
        retreat=home,
    )
    ok = ok and (yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, gr.tcp), lin_step=0.008), tries=6))
    if not ok:
        yield from go(arm, lambda: arm.line_to(hover, lin_step=0.012))
        return False
    yield from arm.set_grip(0.0, 10)
    w = held_width(g_obs(arm))
    if w < 0.01:
        yield from arm.set_grip(open_g, 6)
        yield from go(arm, lambda: arm.line_to(hover, lin_step=0.012))
        return False
    z_lift = next(
        (
            z
            for z in BROOM_CARRY_ZS
            if ik_first(arm.name, tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], z]), (arm.q,)) is not None
        ),
        BROOM_CARRY,
    )
    yield
    seeds = arm.q, arm.q + np.array([0, 0, 0, 0, 0, dyaw]), G.base_seed(arm.name, p_new)
    z_place, r_place = BROOM_CARRY, None
    for z in BROOM_CARRY_ZS:
        r_place = ik_first(arm.name, tcp_pose(rot_new, [p_new[0], p_new[1], z]), seeds)
        if r_place is not None:
            z_place = z
            break
    yield
    for i, z in enumerate([z_lift] + [z for z in (BROOM_CARRY, gr.hover_z) if z < z_lift - 0.005]):
        if i:
            yield
        yield from go(arm, lambda z=z: arm.line_to(tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], z]), lin_step=0.01))
        if arm.status == "ok":
            z_lift = z
            break
    q_seeds = (r_place.q,) if r_place is not None else ()
    via = broom_via(arm, sh, gr, rot_new, p_new, max(z_lift, z_place), q_seeds)
    yield
    if via is not None:
        T_via, r_via, _ = via
        yield from arm.move_to(T_via, min_z=min(z_lift, z_place) - 0.01, seeds=(r_via.q,))
    ok = yield from go(
        arm,
        lambda: arm.move_to(
            tcp_pose(rot_new, [p_new[0], p_new[1], z_place]), min_z=min(z_lift, z_place) - 0.01, seeds=q_seeds
        ),
        tries=6,
        retreat=home,
    )
    if not ok and arm.status == "fail" and (r_place is not None):
        ok = yield from go(arm, lambda: arm.joints_to(r_place.q), tries=6, retreat=home)
    if not broom_held(arm):
        yield from arm.set_grip(open_g, 6)
        return "lost"
    yield from go(arm, lambda: arm.line_to(tcp_pose(rot_new, [p_new[0], p_new[1], gr.tcp[2] + 0.008]), lin_step=0.008))
    yield from arm.set_grip(open_g, 8)
    yield from go(arm, lambda: arm.line_to(tcp_pose(rot_new, [p_new[0], p_new[1], gr.tcp[2] + 0.07]), lin_step=0.01))
    sh.broom = SS.Fit(SS.BROOM, float(tx), float(ty), float(yaw_placed), br.iou, {"moved": True})
    return True


BROOM_VIA_ZS = K.TABLE_Z + 0.21, K.TABLE_Z + 0.18
BROOM_VIA_GAIN = 0.025
BROOM_VIA_MARGIN = 0.02


def broom_via(arm: Arm, sh: Shared, gr, rot_new, p_new, z_carry: float, seeds=()):
    br, dp = sh.broom, sh.target if sh.dustpan_placed else sh.dustpan
    if br is None or dp is None:
        return None
    p0 = np.asarray(gr.tcp[:2], dtype=float)
    reach = float(np.max(np.linalg.norm(SS.footprint_points(SS.BROOM, br.x, br.y, br.yaw) - p0, axis=1)))
    fp = SS.footprint_points(SS.DUSTPAN, dp.x, dp.y, dp.yaw)
    ts = np.linspace(0.0, 1.0, 21)
    path = p0 + ts[:, None] * (np.asarray(p_new[:2], dtype=float) - p0)
    d = np.min(np.linalg.norm(path[:, None, :] - fp[None, :, :], axis=2), axis=1)
    if float(d.min()) > reach + BROOM_VIA_MARGIN:
        return None
    close = np.flatnonzero(d <= float(d.min()) + 0.02)
    k = int(close[len(close) // 2])
    for z in BROOM_VIA_ZS:
        if z < z_carry + BROOM_VIA_GAIN:
            break
        for rot in (rot_new, gr.rot):
            T = tcp_pose(rot, [path[k][0], path[k][1], z])
            r = ik_first(arm.name, T, (arm.q,) + tuple(seeds))
            if r is not None:
                return T, r, float(d[k])


def drop_radius(it: Item) -> float:
    return float(np.clip(0.25 * (max(it.length, it.width) + it.width), *DROP_R))


def release_gap(width: float) -> float:
    return min(0.085, width + RELEASE_GAP)


def drop_spot(in_pan, r: float, width: float, exclude=()) -> np.ndarray:
    c = _DROP_GRID
    m = max(r + 0.01, release_gap(width) / 2 + FINGER_OUT)
    ok = (
        (np.abs(c[:, 0]) <= WALL_IN[0] - m)
        & (c[:, 1] <= WALL_IN[1] - m)
        & (c[:, 1] >= DROP_Y_MIN)
        & (np.linalg.norm(c - SS.SUPPORT_C, axis=1) <= DROP_R_SAFE)
    )
    for e in exclude:
        ok &= np.linalg.norm(c - np.asarray(e, dtype=float)[:2], axis=1) > DROP_ALT_EXCLUDE
    if not ok.any():
        return SS.SUPPORT_C.copy()
    c = c[ok]
    clear = np.full(len(c), DROP_CLEAR)
    behind = np.zeros(len(c))
    for p, rj in in_pan:
        p = np.asarray(p, dtype=float)[:2]
        d = np.linalg.norm(c - p, axis=1)
        clear = np.minimum(clear, d - r - rj)
        behind += (p[1] < c[:, 1] - DROP_BEHIND_DY) & (d < r + rj + DROP_BEHIND_MARGIN)
    score = (
        20.0 * np.minimum(clear, 0.0)
        + 6.0 * clear
        + DROP_BACK_W * c[:, 1]
        + 0.1 * np.abs(c[:, 0])
        - DROP_BEHIND_PENALTY * behind
    )
    return c[int(np.argmax(score))].copy()


def to_local(fit: SS.Fit, xy) -> np.ndarray:
    o = fit.origin
    return ((np.array([xy[0], xy[1], o[2]]) - o) @ fit.R)[:2]


DROP_ALT_SPOTS = 3
DROP_ALT_EXCLUDE = 0.03


def find_drop(arm: Arm, sh: Shared, it: Item, w_held: float, rot_grasp, p, carry_z: float):
    o = arm.other
    q_o, g_o = (None, 1.0) if o is None else (o.q, o.g)
    level = level_rotation(rot_grasp)
    rots = (rot_grasp,) if level is None else (rot_grasp, level)
    first = p, sh.in_pan[-1]
    cands = [first]
    passes = (True, False) if q_o is not None else (False,)
    for need_clear in passes:
        for k in range(1 + DROP_ALT_SPOTS):
            if k == len(cands):
                sh.in_pan.pop()
                pk = drop_point(sh, it, w_held, exclude=[c[1][0] for c in cands])
                cands.append((pk, sh.in_pan.pop()))
                sh.in_pan.append(first[1])
            pk, booked = cands[k]
            for r0 in rots:
                if k or r0 is not rot_grasp or need_clear != passes[0]:
                    yield
                rot, qs = drop_rotation(arm, r0, pk, carry_z, q_o, g_o, need_clear=need_clear)
                if rot is not None:
                    sh.in_pan.pop()
                    sh.in_pan.append(booked)
                    return rot, qs, pk, booked[0]
    sh.in_pan.pop()
    sh.in_pan.append(first[1])
    return None, None, first[0], first[1][0]


def level_rotation(rot: np.ndarray):
    y = np.asarray(rot, dtype=float)[:2, 1]
    if float(np.linalg.norm(y)) < 1e-06:
        return None
    lv = G.grasp_rotation(math.atan2(float(y[1]), float(y[0])), 0.0)
    return None if np.allclose(lv, rot, atol=1e-06) else lv


def drop_point(sh: Shared, it: Item, width: float, exclude=()) -> np.ndarray:
    dp = sh.dustpan
    r = drop_radius(it)
    spot = drop_spot(sh.in_pan, r, width, exclude)
    sh.in_pan.append((spot, r))
    p = dp.to_world([spot[0], spot[1], 0.0])[0]
    return np.array([p[0], p[1], PAN_FLOOR + DROP_ABOVE + it.half_h])


def lift_item(arm: Arm, gr, carry_z: float):
    hz = float(gr.hover_z)
    zs = [carry_z] + ([0.5 * (carry_z + hz), hz] if hz < carry_z - 0.005 else [])
    for i, z in enumerate(zs):
        if i:
            yield
        yield from go(
            arm,
            lambda z=z: arm.line_to(
                tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], z]), lin_step=LIFT_STEP, fallback_joint=False
            ),
        )
        if arm.status == "ok":
            return z
    return float(K.point_of(arm.T, K.TCP_OFFSET)[2])


FINGER_REACH = 0.02
FINGER_HALF_W = 0.015


def finger_blocked_yaws(it: Item, sh: Shared, gap: float) -> tuple:
    br = sh.broom
    if br is None:
        return ()
    d = SS.footprint_points(SS.BROOM, br.x, br.y, br.yaw) - np.asarray(it.xy, dtype=float)[:2]
    reach = gap / 2 + FINGER_REACH
    d = d[np.linalg.norm(d, axis=1) < reach + FINGER_HALF_W]
    if not len(d):
        return ()
    out = []
    for k in range(12):
        th = k * math.pi / 12
        u = np.array([math.cos(th), math.sin(th)])
        if np.any((np.abs(d @ u) < reach) & (np.abs(d @ np.array([-u[1], u[0]])) < FINGER_HALF_W)):
            out.append(th)
    return tuple(out)


DROP_ARM_CLEAR = 0.05


def arm_dist(arm: Arm, q, q_other, g_other: float = 1.0) -> float:
    if arm.name == "left":
        return CO.arm_distance(q, q_other, arm.g, g_other)
    return CO.arm_distance(q_other, q, g_other, arm.g)


def drop_rotation(
    arm: Arm,
    rot_grasp,
    p,
    carry_z: float,
    q_other=None,
    g_other: float = 1.0,
    yaw_steps: int = 12,
    need_clear: bool = False,
):
    q_cur = np.asarray(arm.q, dtype=float)
    best = clear = None
    for i in range(yaw_steps):
        d = (i + 1) // 2 * (2 * math.pi / yaw_steps) * (1 if i % 2 else -1) if i else 0.0
        rot, qs = G.place_rotation(
            arm.name, K.rot_z(d) @ rot_grasp, p, q_cur, extra_z=(0.0, carry_z - p[2]), yaw_steps=1
        )
        if rot is None:
            continue
        cost = float(np.sum(K.JOINT_COST_W * np.abs(qs[0] - q_cur))) + 0.3 * abs(d)
        if best is None or cost < best[0]:
            best = cost, rot, qs
        if q_other is None or (clear is not None and cost >= clear[0]):
            continue
        path = [q_cur + (qs[1] - q_cur) * t for t in np.linspace(0.0, 1.0, 9)] + [qs[0]]
        if min(arm_dist(arm, q, q_other, g_other) for q in path) >= DROP_ARM_CLEAR:
            clear = cost, rot, qs
    if best is None or (need_clear and q_other is not None and (clear is None)):
        return None, None
    out = clear if clear is not None else best
    return out[1], out[2]


def pick_item(arm: Arm, sh: Shared, it: Item):
    it.tries += 1
    z = K.TABLE_Z + it.half_h
    blocked = (it.yaw + math.pi / 2,) if it.length > 1.3 * it.width else ()
    open_gap = min(0.085, it.width + 0.024)
    fingers = finger_blocked_yaws(it, sh, open_gap)
    gs = yield from grasps_iter(
        arm.name, (it.xy[0], it.xy[1], z), it.width, it.yaw, arm.q, hover_dz=HOVER_DZ, blocked_yaws=blocked + fingers
    )
    if fingers:
        if not gs:
            gs = yield from grasps_iter(
                arm.name, (it.xy[0], it.xy[1], z), it.width, it.yaw, arm.q, hover_dz=HOVER_DZ, blocked_yaws=blocked
            )
    if not gs:
        yaws = (it.yaw,) if blocked else (it.yaw, it.yaw + math.pi / 2)
        gs = yield from pitched_grasps(arm.name, (it.xy[0], it.xy[1], z), yaws, arm.q, open_gap, HOVER_DZ)
    if not gs:
        it.state = "unreachable"
        return False
    gr = gs[0]
    yield
    carry_z = K.TABLE_Z + CARRY_CLEAR + 2 * it.half_h
    open_g = grip_for_gap(open_gap)
    hover = tcp_pose(gr.rot, [gr.tcp[0], gr.tcp[1], gr.hover_z])
    ok = yield from go(
        arm,
        lambda: arm.transit(
            hover, min(carry_z, gr.hover_z), grip=open_g, seeds=(gr.q_hover,), joint_step=ITEM_JOINT_STEP
        ),
    )
    if ok and len(sh.in_pan) >= ROLL_AFTER and sh.drops_open and (sh.rollout_looks < ROLL_LOOKS):
        sh.rollout_looks += 1
        sh.rollouts.extend((yield from look_rollouts(arm, sh, it)))
    ok = ok and (yield from go(arm, lambda: arm.line_to(tcp_pose(gr.rot, gr.tcp), lin_step=APPROACH_STEP)))
    if not ok:
        yield from go(arm, lambda: arm.line_to(hover, lin_step=0.012))
        it.state = "missed"
        return False
    yield from arm.set_grip(0.0, 8)
    h1, _, _ = holds(arm)
    lift_z = yield from lift_item(arm, gr, carry_z)
    h2, w2, _ = holds(arm)
    if not (h1 and h2):
        yield from arm.set_grip(1.0, 4)
        it.state = "missed"
        return False
    w_held = max(w2, it.width)
    yield from wait_dustpan(arm, sh, carry_z, gr.rot, it.half_h, lift_z=lift_z, item=it, width=w_held)
    yield from take_zone(arm)
    p = drop_point(sh, it, w_held)
    yield
    rot, qs, p, _ = yield from find_drop(arm, sh, it, w_held, gr.rot, p, carry_z)
    if rot is None:
        rot = gr.rot
    yield
    q_seed = () if qs is None else (qs[1],)
    yield from go(
        arm,
        lambda: arm.move_to(
            tcp_pose(rot, [p[0], p[1], carry_z]), min_z=WALL_TOP + it.half_h, joint_step=ITEM_JOINT_STEP, seeds=q_seed
        ),
        tries=6,
    )
    yield from go(arm, lambda: arm.line_to(tcp_pose(rot, p), lin_step=DROP_STEP))
    yield from arm.set_grip(grip_for_gap(release_gap(w_held)), 6)
    yield from arm.hold(3)
    yield from go(arm, lambda: arm.line_to(tcp_pose(rot, [p[0], p[1], carry_z]), lin_step=LIFT_STEP))
    free_zone(arm)
    it.state = "dropped"
    return True


PARK_TRIES = 3


def left_away(sh: Shared) -> bool:
    return sh.dustpan_placed or sh.left_homing


STAGE_OFF = np.array([0.14, -0.03])


def wait_dustpan(
    arm: Arm,
    sh: Shared,
    carry_z: float | None = None,
    rot_grasp=None,
    half_h: float = 0.02,
    max_steps: int = 500,
    lift_z: float | None = None,
    item: Item | None = None,
    width: float = 0.04,
):
    o = arm.other
    n = parks = 0
    at_home = sh.drops_open
    staged = tried_stage = False

    def stage():
        sxy = np.asarray(SS.support_center(sh.target), dtype=float)[:2] + STAGE_OFF
        T = arm.T.copy()
        if rot_grasp is not None:
            r = drop_radius(item) if item is not None else DROP_R[1]
            spot = drop_spot(sh.in_pan, r, width)
            pd = sh.target.to_world([spot[0], spot[1], 0.0])[0]
            pd = np.array([pd[0], pd[1], PAN_FLOOR + DROP_ABOVE + half_h])
            rot, _ = drop_rotation(arm, rot_grasp, pd, carry_z, None if o is None else o.q_home)
            yield
            if rot is not None:
                T = tcp_pose(rot, [0.0, 0.0, 0.0])
        T[:3, 3] = [sxy[0], sxy[1], carry_z]
        z_min = carry_z if lift_z is None else min(carry_z, lift_z)
        yield from arm.move_to(T, min_z=z_min - 0.02, joint_step=ITEM_JOINT_STEP)
        return arm.status == "ok"

    can_stage = sh.target is not None and carry_z is not None
    deferred = False
    if not at_home and can_stage and sh.broom_done:
        if left_away(sh):
            tried_stage = True
            staged = at_home = yield from stage()
        else:
            deferred = True
    while not sh.drops_open and n < max_steps:
        if not tried_stage and can_stage and at_home and (sh.left_homing or (deferred and left_away(sh))):
            tried_stage = True
            staged = yield from stage()
            at_home = True
            continue
        if staged and o is not None and o.waiting and (not o.waiting_zone):
            staged = at_home = False
        if (
            not at_home
            and parks < PARK_TRIES
            and (parks == 0 or (o is not None and o.waiting and (not o.waiting_zone)))
        ):
            parks += 1
            yield from arm.joints_to(arm.q_home, joint_step=FAST_STEP)
            at_home = arm.status == "ok"
            continue
        n += 1
        arm.waiting = True
        yield
    arm.waiting = False


def take_zone(arm: Arm, max_steps: int = 600):
    z = arm.zone
    if z is None:
        return True
    n = 0
    while not z.take(arm.name) and n < max_steps:
        n += 1
        arm.waiting = arm.waiting_zone = True
        yield
    arm.waiting = arm.waiting_zone = False
    return z.owner == arm.name


def free_zone(arm: Arm):
    if arm.zone is not None:
        arm.zone.free(arm.name)


BROOM_RIGHT = 0.15


def broom_path_clear(sh: Shared) -> bool:
    from scipy.spatial import cKDTree

    pts = SS.footprint_points(SS.BROOM, sh.broom.x, sh.broom.y, sh.broom.yaw)
    return float(cKDTree(dustpan_sweep(sh)).query(pts)[0].min()) >= 0.015


REFIT_IOU = 0.7
REFIT_SHIFT = 0.015, math.radians(5.0)
PLACED_REFIT_SPAN = math.radians(
    45.0
)
REFIT_IOU_LOW = 0.45
REFIT_GAIN = 0.25


def think_iou(cam, mask, f: SS.Fit):
    yield
    return SS.pose_iou(cam, mask, f.name, f.x, f.y, f.yaw)


def refit_dustpan(arm: Arm, sh: Shared, placed: bool = False, yaw_span: float = math.radians(30)):
    from ...perception import seg

    img = arm.ctx.obs.images.get("cam_head")
    if img is None or sh.dustpan is None:
        return
    yield
    insts = seg.segment(img, camera="head")
    yield
    dp = seg.of_category(insts, SS.DUSTPAN)
    if not dp:
        return
    old = sh.dustpan
    fit = yield from SS.fit_iter(cameras.head(), dp[0].mask, SS.DUSTPAN, yaw0=old.yaw, yaw_span=yaw_span)
    if fit is None or fit.iou <= REFIT_IOU:
        full = yield from SS.fit_iter(cameras.head(), dp[0].mask, SS.DUSTPAN)
        if full is not None and (fit is None or full.iou > fit.iou):
            fit = full
    if fit is None:
        return
    shift = math.hypot(fit.x - old.x, fit.y - old.y)
    turn = abs(math.remainder(fit.yaw - old.yaw, 2 * math.pi))
    old_iou = yield from think_iou(cameras.head(), dp[0].mask, old)
    moved = shift > REFIT_SHIFT[0] or turn > REFIT_SHIFT[1]
    use = moved and (fit.iou > REFIT_IOU or (fit.iou > REFIT_IOU_LOW and fit.iou - old_iou > REFIT_GAIN))
    if not use:
        return
    sh.dustpan = SS.Fit(SS.DUSTPAN, fit.x, fit.y, fit.yaw, fit.iou, dict(old.extra))
    if placed and arm.zone is not None:
        arm.zone.xy = np.asarray(SS.support_center(sh.dustpan), dtype=float)[:2]


BROOM_TRIES = 2


def try_broom(arm: Arm, sh: Shared, plan, tried: set, swept: bool, zs):
    n = lost = 0
    while plan is not None and n < BROOM_TRIES:
        n += 1
        gr, tgt, key = plan
        tried.add(key)
        res = yield from exec_broom(arm, sh, gr, tgt)
        if res is True:
            return True
        if res == "lost":
            sh.broom_recovery = True
            seen = yield from find_lost_broom(arm, sh)
            tried.clear()
            if seen and sh.target is not None and (sh.broom.x - sh.target.x > BROOM_RIGHT):
                return True
            if not seen or lost >= BROOM_LOST_TRIES:
                return False
            lost += 1
            n -= 1
            plan = yield from plan_broom(arm, sh, swept=swept, zs=zs)
            continue
        plan = yield from plan_broom(arm, sh, swept=swept, zs=zs, skip=tried)
    return False


BROOM_LOST_TRIES = 1
BROOM_FIX_DX = 0.12
BROOM_FIX_IOU = 0.5
BROOM_FIX_STEPS = 180


def broom_fix_needed(bfit, dp, steps_left: int) -> bool:
    return (
        bfit is not None
        and dp is not None
        and (bfit.iou > BROOM_FIX_IOU)
        and (bfit.x - dp.x < BROOM_FIX_DX)
        and (steps_left >= BROOM_FIX_STEPS + HOME_RESERVE)
    )


def fix_broom(arm: Arm, sh: Shared):
    old_target, old_drops = sh.target, sh.drops_open
    sh.target = SS.Fit(SS.DUSTPAN, sh.dustpan.x, sh.dustpan.y, sh.dustpan.yaw, 1.0)
    sh.drops_open = False
    ok, plan = False, None
    try:
        plan = yield from plan_broom(arm, sh, swept=False, zs=BROOM_GRASP_ZS_LAST)
        ok = yield from try_broom(arm, sh, plan, set(), swept=False, zs=BROOM_GRASP_ZS_LAST)
    finally:
        sh.target, sh.drops_open = old_target, old_drops
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=FAST_STEP)
    return ok


def find_lost_broom(arm: Arm, sh: Shared):
    from ...perception import seg

    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=FAST_STEP)
    yield from refit_dustpan(arm, sh, placed=sh.dustpan_placed)
    img = arm.ctx.obs.images.get("cam_head")
    if img is None:
        return False
    yield
    insts = seg.segment(img, camera="head")
    yield
    br = seg.of_category(insts, SS.BROOM)
    fit = (yield from SS.fit_iter(cameras.head(), br[0].mask, SS.BROOM)) if br else None
    if fit is None:
        return False
    sh.broom = SS.Fit(SS.BROOM, fit.x, fit.y, fit.yaw, fit.iou, {"lost": True})
    return True


def left_task(arm: Arm, sh: Shared):
    yield from take_zone(arm)
    need_broom = sh.broom is not None and sh.target is not None
    if need_broom and sh.broom.x - sh.target.x > BROOM_RIGHT:
        need_broom = False
    tried = set()
    b_swept, b_zs = True, BROOM_GRASP_ZS
    if need_broom:
        plan = yield from plan_broom(arm, sh)
        if plan is None and sh.dustpan is not None:
            tx, ty, tyaw = dustpan_target(sh.dustpan, sh.items, sh.broom)
            old = sh.target
            sh.target = SS.Fit(SS.DUSTPAN, tx, ty, tyaw, 1.0)
            if broom_path_clear(sh):
                sh.pick_open = True
                if arm.zone is not None:
                    arm.zone.xy = np.asarray(SS.support_center(sh.target), dtype=float)[:2]
            else:
                sh.target = old
                b_swept, b_zs = False, BROOM_GRASP_ZS_LAST
                plan = yield from plan_broom(arm, sh, swept=b_swept, zs=b_zs)
        need_broom = not (yield from try_broom(arm, sh, plan, tried, swept=b_swept, zs=b_zs))
    sh.broom_done = not need_broom
    sh.pick_open = True
    if sh.dustpan is not None:
        yield from move_dustpan(arm, sh)
    sh.dustpan_placed = True
    if need_broom:
        plan = yield from plan_broom(arm, sh, zs=BROOM_GRASP_ZS_LAST, skip=tried)
        yield from try_broom(arm, sh, plan, tried, swept=True, zs=BROOM_GRASP_ZS_LAST)
    sh.pick_open = sh.broom_done = sh.left_homing = True
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=ITEM_JOINT_STEP)
    yield from refit_dustpan(arm, sh, placed=True, yaw_span=PLACED_REFIT_SPAN)
    sh.drops_open = True
    free_zone(arm)


RELOOKS = 2
RELOOK_MATCH = 0.08
RELOOK_PAN_MARGIN = 0.04


def match_items(seen, todo, max_d: float = RELOOK_MATCH):
    pairs = sorted(
        (float(np.linalg.norm(s.xy - t.xy)), i, j)
        for i, s in enumerate(seen)
        for j, t in enumerate(todo)
        if s.category == t.category
    )
    used_s, used_t = set(), set()
    for d, i, j in pairs:
        if d > max_d or i in used_s or j in used_t:
            continue
        used_s.add(i)
        used_t.add(j)
    return [s for i, s in enumerate(seen) if i not in used_s]


def relook_items(arm: Arm, sh: Shared, todo: list):
    from ...perception import seg

    img = arm.ctx.obs.images.get("cam_head")
    if img is None or sh.dustpan is None:
        return []
    yield
    insts = seg.segment(img, camera="head")
    yield
    seen = detect_items(cameras.head(), insts, dustpan=sh.dustpan)
    sc = np.asarray(SS.support_center(sh.dustpan), dtype=float)[:2]
    seen = [s for s in seen if float(np.linalg.norm(s.xy - sc)) > SS.SUPPORT_R + RELOOK_PAN_MARGIN]
    new = match_items(seen, todo)
    return new


ROLL_LOOKS = 8
ROLL_AFTER = 3
ROLL_LOOK_R = 0.2
ROLL_OUT_MARGIN = 0.01
ROLL_ARM_CLEAR = 0.1


def look_rollouts(arm: Arm, sh: Shared, it: Item):
    from ...perception import seg

    img = arm.ctx.obs.images.get("cam_head")
    if img is None or sh.dustpan is None:
        return []
    yield
    insts = seg.segment(img, camera="head")
    yield
    seen = detect_items(cameras.head(), insts, dustpan=sh.dustpan)
    sc = np.asarray(SS.support_center(sh.dustpan), dtype=float)[:2]
    tcp = np.asarray(K.point_of(K.fk(arm.name, arm.q), K.TCP_OFFSET), dtype=float)[:2]
    known = [i for i in sh.items if i is it or i.state in ("todo", "unreachable")]
    n, new = rollout_candidates(seen, sc, tcp, known)
    if not n:
        return []
    for s in new:
        s.state = "todo"
    sh.items.extend(new)
    return new


def rollout_candidates(seen, sc, tcp, known):
    near = [
        s
        for s in seen
        if SS.SUPPORT_R + ROLL_OUT_MARGIN < float(np.linalg.norm(s.xy - sc)) < ROLL_LOOK_R
        and float(np.linalg.norm(s.xy - tcp)) > ROLL_ARM_CLEAR
    ]
    return len(near), match_items(near, known) if near else []


def right_task(arm: Arm, sh: Shared, items, first_far: bool = False, reserve: int = HOME_RESERVE + ITEM_STEPS):
    order = sorted(items, key=lambda i: float(np.linalg.norm(i.xy - SC_TARGET)))
    if first_far and order:
        far = max(order, key=lambda i: float(i.xy[0] - SC_TARGET[0]) + float(i.xy[1] - SC_TARGET[1]))
        order = [far] + [i for i in order if i is not far]
    if first_far:
        n_wait = [0]

        def picks_open():
            n_wait[0] += 1
            return sh.pick_open or (n_wait[0] > 400 and (not sh.broom_recovery))

        yield from arm.wait_until(picks_open, max_steps=600)
    queue, k, missed, looks = list(order), 0, False, 0
    while k < len(queue):
        it = queue[k]
        k += 1
        if steps_left(arm) < reserve:
            break
        ok = yield from pick_item(arm, sh, it)
        if sh.rollouts:
            queue[k:k] = sh.rollouts
            sh.rollouts.clear()
        if not ok:
            missed = missed or it.state == "missed"
        elif missed and looks < RELOOKS:
            looks += 1
            missed = False
            new = yield from relook_items(arm, sh, queue[k:])
            sh.items.extend(new)
            queue.extend(new)
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=ITEM_JOINT_STEP)


@register
class SweepIntoDustpan(Program):
    name = "sweep_into_dustpan"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if ("dustpan" in s or "dust pan" in s) and ("sweep" in s or "broom" in s):
            return 0.95
        return 0.0

    def run(self, ctx):
        from ...perception import seg

        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        dustpan, broom, items = yield from think(perceive(ctx), arms)
        zone, target = None, None
        if dustpan is not None:
            tx, ty, tyaw = dustpan_target(dustpan, items, None)
            target = SS.Fit(SS.DUSTPAN, tx, ty, tyaw, 1.0)
            zone = Zone(SS.support_center(target), ZONE_R)
        link_arms(arms, zone)
        sh = Shared(dustpan=dustpan, broom=broom, items=items, target=target)
        yield from run_arms(
            arms,
            {"left": left_task(arms["left"], sh), "right": right_task(arms["right"], sh, sh.items, first_far=True)},
        )
        for rnd in range(SURVEY_ROUNDS):
            if sh.dustpan is None or STEP_LIMIT - ctx.obs.step < LATE_SURVEY_MIN:
                break
            yield from run_arms(arms, {a: arms[a].hold(4) for a in K.ARMS})
            cam = cameras.head()
            img = ctx.obs.images["cam_head"]
            insts = seg.segment(img, camera="head")
            yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            dp = seg.of_category(insts, SS.DUSTPAN)
            fit = (
                (
                    yield from think(
                        SS.fit_iter(cam, dp[0].mask, SS.DUSTPAN, yaw0=sh.dustpan.yaw, yaw_span=math.radians(30)), arms
                    )
                )
                if dp
                else None
            )
            if fit is not None and fit.iou > 0.6:
                sh.dustpan = fit
            left_out = detect_items(cam, insts, dustpan=sh.dustpan)
            sh.in_pan = pan_objects(cam, insts, sh.dustpan)
            sh.items.extend(left_out)
            if rnd == 0:
                br = seg.of_category(insts, SS.BROOM)
                bfit = (yield from think(SS.fit_iter(cam, br[0].mask, SS.BROOM), arms)) if br else None
                fix = broom_fix_needed(bfit, sh.dustpan, STEP_LIMIT - ctx.obs.step)
                if fix:
                    sh.broom = SS.Fit(SS.BROOM, bfit.x, bfit.y, bfit.yaw, bfit.iou, {"survey": True})
                    yield from run_arms(arms, {"left": fix_broom(arms["left"], sh)})
                    if left_out and STEP_LIMIT - ctx.obs.step < HOME_RESERVE + ITEM_STEPS:
                        break
                elif bfit is not None and bfit.iou > BROOM_FIX_IOU:
                    sh.broom = SS.Fit(SS.BROOM, bfit.x, bfit.y, bfit.yaw, bfit.iou, {"survey": True})
            if not left_out:
                break
            yield from run_arms(arms, {"right": right_task(arms["right"], sh, left_out, reserve=LATE_PICK_MIN)})
        yield from run_arms(arms, {a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True) for a in K.ARMS})
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
