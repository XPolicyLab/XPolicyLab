from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from . import demo_watch as DW
from ...perception import robot_mask as RM
from . import tabletop as TT
from ...planning import collision as CO
from ...planning import kinematics as K
from ...skills import grasp as G
from ...skills.dual import held_width
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

V0, V1 = 30, 270
FAR_AREA = (-0.62, 0.62), (-0.06, 0.47)
NEAR_AREA = (-0.5, 0.52), (-0.4, 0.05)
WATCH_MIN_STEPS = 80
WATCH_MAX_STEPS = 1000
BACK_STILL = 8
WATCH_MIN_STEPS_STILL = 300
LONG_STILL = 30
N_OBJECTS = 5
GRASP_Z_MIN = 0.016
GRASP_Z_MAX = 0.03
HOVER_DZ = 0.06
LIFT_Z = TT.TABLE_Z + 0.12
DROP_Z = TT.TABLE_Z + 0.16
RELEASE_Z = TT.TABLE_Z + 0.09
LONG_DROP = 0.08
DROP_OFFSETS = (0.0, 0.0), (-0.035, 0.075), (0.035, -0.075), (-0.035, -0.075), (0.035, 0.075)
OPEN_MARGIN = 0.03
MIN_HELD = 0.008
STAGE_SPOTS = (
    (-0.08, -0.31),
    (0.08, -0.31),
    (-0.1, -0.35),
    (0.1, -0.35),
    (0.0, -0.36),
    (0.0, -0.24),
    (-0.07, -0.24),
    (0.07, -0.24),
)
STAGE_CLEAR = 0.07
STAGE_SEP = 0.14
N_STAGE_SPOTS = 2
LEFT_MAX_X = 0.05
FIRST_LEFT_MAX_X = 0.12
FIRST_MAX_TILT = math.radians(15)
MAX_ATTEMPTS = 3
LOOK_Q = {
    "left": np.array([0.08, 0.15, 0.56, 0.32, -0.61, 0.0]),
    "right": np.array([-0.08, 0.15, 0.56, 0.32, 0.61, 0.0]),
}


@dataclass
class Scene:
    img0: np.ndarray
    boxes: list
    watch: DW.DemoWatch
    exclude: np.ndarray

    @staticmethod
    def first(img0: np.ndarray, cam: C.Camera, robot: np.ndarray, step0: int = 0) -> "Scene":
        import cv2

        lab = TT.lab_image(img0)
        boxes = TT.white_boxes(img0, cam, lab)[:2]
        exclude = robot.copy()
        for b in boxes:
            exclude |= cv2.dilate(b.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        region = TT.region_with_height(cam, img0.shape, FAR_AREA, 0.1) & TT.region_with_height(
            cam, img0.shape, ((-0.7, 0.7), (-0.1, 0.47)), 0.0
        )
        edge_v = int(cam.project([0.0, FAR_AREA[1][1], TT.TABLE_Z])[0][0][1])
        band = np.zeros(img0.shape[:2], bool)
        band[: max(1, edge_v - 3), 120:520] = True
        watch = DW.DemoWatch(img0, step0, region, exclude, band, V0, V1)
        return Scene(img0, boxes, watch, exclude)


@dataclass
class Analysis:
    holes: list
    order: list
    near: list
    assign: list
    margin: float
    boxes: list
    our_box: object = None
    costs: np.ndarray | None = None


def analyse(
    scene: Scene,
    img: np.ndarray,
    cam: C.Camera,
    robot: np.ndarray,
    img_near: np.ndarray | None = None,
    robot_near: np.ndarray | None = None,
) -> Analysis:
    import cv2

    lab = TT.lab_image(img)
    table_area = TT.plane_mask(cam, img.shape)
    nt = TT.not_table(img, cam, table_area & ~scene.exclude, lab)
    holes = scene.watch.holes(img)
    t_lab = TT.table_colour(lab, table_area & ~scene.exclude)
    t_chroma = float(np.hypot(t_lab[1], t_lab[2]))

    def bare_table(h):
        px = lab[h.mask]
        chroma = np.hypot(px[:, 1], px[:, 2])
        return float(np.median(chroma)) >= 0.5 * t_chroma

    holes = [h for h in holes if bare_table(h) and h.first_gone >= 0]
    if len(holes) > N_OBJECTS:
        holes = sorted(holes, key=lambda h: -h.area)[:N_OBJECTS]
    retime(scene.watch, holes, img)
    order = DW.order(holes)
    if img_near is None:
        img_near, robot_near, lab_n, nt_n = img, robot, lab, nt
        excl_n, boxes_n = scene.exclude, scene.boxes
    else:
        lab_n = TT.lab_image(img_near)
        boxes_n = TT.white_boxes(img_near, cam, lab_n)[:2] or scene.boxes
        excl_n = robot_near.copy()
        for b in boxes_n:
            excl_n |= cv2.dilate(b.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        nt_n = TT.not_table(img_near, cam, table_area & ~excl_n, lab_n)
    regions = near_objects(
        img_near, cam, robot_near, excl_n, nt_n, lab_n, max(len(holes), 1), boxes_n, split_short=False
    )
    lab0 = TT.lab_image(scene.img0)
    bg = TT.table_background(lab_n, table_area & ~excl_n)
    da = [TT.appearance(h.mask, h.colours, lab[h.mask], cam) for h in holes]
    dn = [near_appearance(r, nt_n & ~excl_n, lab_n, bg, cam) for r in regions]
    assign, margin, costs = [], 0.0, None
    if holes and regions:
        costs = np.array([[TT.appearance_cost(a, b) for b in dn] for a in da])
        drop = same_pick(holes, order, costs)
        if drop:
            keep = [i for i in range(len(holes)) if i not in drop]
            holes = [holes[i] for i in keep]
            da = [da[i] for i in keep]
            costs = costs[keep]
            order = DW.order(holes)
        for _ in range(2):
            if len(regions) >= len(holes):
                break
            split = best_split(regions, dn, da, costs, cam, lab_n, nt_n & ~excl_n, bg)
            if split is None:
                break
            regions, dn, costs = split
        cnn = cnn_distances(scene.img0, holes, img_near, regions)
        assign, _, margin = TT.match(costs + objectness_penalty(regions)[None, :] + CNN_W * cnn)
    else:
        cnn = None
    our = None
    if scene.boxes:
        lab_0 = lab0
        changes = []
        for b in scene.boxes:
            d = DW.delta_e(lab[b.mask], lab_0[b.mask])
            changes.append(float(np.mean(d > DW.HOLE_DE)))
        our = scene.boxes[int(np.argmin(changes))]
    an = Analysis(holes, order, regions, assign, margin, scene.boxes, our, costs)
    an.far_desc = da
    an.near_desc = dn
    an.boxes_near = boxes_n
    an.exclude_near = excl_n & ~robot_near
    return an


MIN_PICK_DT = 40
STILL_THERE_FRAC = 0.25
RETIME_DT = 20
RETIME_GAP = 80


def retime(watch, holes, img) -> None:
    holes = [h for h in holes if h.first_gone - h.last_present > RETIME_GAP]
    if not holes or not watch.frames:
        return
    watch.timeline(holes, img)
    for h in holes:
        still = [s for s, p in h.present if p >= STILL_THERE_FRAC]
        if not still or still[-1] <= h.first_gone + RETIME_DT:
            continue
        last = still[-1]
        gone = [s for s, g in h.gone if s > last and g >= DW.GONE_FRAC]
        h.last_present, h.first_gone = last, gone[0] if gone else last + watch.every


def same_pick(holes, order, costs) -> set:
    drop = set()
    for a, b in zip(order[:-1], order[1:]):
        if a in drop or b in drop:
            continue
        if abs(holes[b].first_gone - holes[a].first_gone) < MIN_PICK_DT:
            drop.add(a if float(np.min(costs[a])) > float(np.min(costs[b])) else b)
    return drop


OBJ_IOU = 0.8
OBJ_W = 0.5


def objectness_penalty(regions) -> np.ndarray:
    iou = np.array([r.fit.iou if getattr(r, "fit", None) is not None else 0.0 for r in regions], float)
    return OBJ_W * np.maximum(0.0, OBJ_IOU - iou)


CNN_W = 2.0
CNN_LAYER = "layer3"


def cnn_distances(img_far: np.ndarray, holes, img_near: np.ndarray, regions) -> np.ndarray:
    from ...perception import detector as D

    fh = D.embed(img_far, [h.mask for h in holes], layer=CNN_LAYER)
    fn = D.embed(img_near, [r.mask for r in regions], layer=CNN_LAYER)
    return 1.0 - fh @ fn.T


MISSING_COST = 0.9
SPLIT_GAIN = 0.1


def match_total(costs: np.ndarray) -> float:
    _, total, _ = TT.match(costs)
    return float(total) + MISSING_COST * max(0, costs.shape[0] - costs.shape[1])


def best_split(regions, dn, da, costs, cam, lab, fg, bg):
    base = match_total(costs)
    best = None
    for k, r in enumerate(regions):
        if r.area < 2 * MIN_OBJ_PX:
            continue
        parts = [p for p in TT.split_region(r, cam, lab, accept=lambda p: object_sized(p, cam)) if object_sized(p, cam)]
        if len(parts) != 2:
            continue
        dp = [near_appearance(p, fg, lab, bg, cam) for p in parts]
        cp = np.array([[TT.appearance_cost(a, b) for b in dp] for a in da])
        c2 = np.concatenate([np.delete(costs, k, axis=1), cp], axis=1)
        t = match_total(c2)
        if t < base - SPLIT_GAIN and (best is None or t < best[0]):
            regs = [x for j, x in enumerate(regions) if j != k] + parts
            best = t, regs, [x for j, x in enumerate(dn) if j != k] + dp, c2
    return None if best is None else best[1:]


def near_appearance(r, fg, lab, bg, cam):
    m = r.mask & fg
    if m.sum() < 0.5 * r.mask.sum():
        m = r.mask
    return TT.appearance(m, lab[m], bg[m], cam)


MIN_OBJ_PX = 120
MIN_OBJ_WIDTH = 0.018
MAX_OBJ_ASPECT = 4.5
MAX_OBJ_LENGTH = 0.15
PRIOR_SLACK = 1.0
PRIOR_MIN_IOU = 0.6
BOX_MARGIN_PX = 5


def object_sized(r, cam) -> bool:
    if r.fit is None:
        r.fit = TT.fit_footprint(r.mask, cam, prior_slack=PRIOR_SLACK, prior_min_iou=PRIOR_MIN_IOU)
    return r.fit is not None and r.fit.width >= MIN_OBJ_WIDTH and (r.fit.length <= MAX_OBJ_ASPECT * r.fit.width)


def near_objects(
    img,
    cam,
    robot,
    exclude,
    nt,
    lab,
    n_expected: int,
    boxes=None,
    area=NEAR_AREA,
    keep_all: bool = False,
    split_short: bool = True,
) -> list:
    import cv2

    table_area = TT.plane_mask(cam, img.shape)
    near_area = TT.region_with_height(cam, img.shape, area, 0.08) & table_area
    excl = exclude | (cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    fg = nt & near_area & ~excl
    bg = TT.table_background(lab, table_area & ~excl)
    vs, us = np.nonzero(fg)
    ok = np.isfinite(bg[vs, us, 0])
    sh = np.zeros(len(vs), bool)
    sh[ok] = DW.shadow_like(lab[vs[ok], us[ok]], bg[vs[ok], us[ok]])
    fg[vs[sh], us[sh]] = False
    near_box = None
    if boxes:
        bm = np.zeros(img.shape[:2], np.uint8)
        for b in boxes:
            bm |= b.mask.astype(np.uint8)
        near_box = cv2.dilate(bm, np.ones((2 * BOX_MARGIN_PX + 1, 2 * BOX_MARGIN_PX + 1), np.uint8)) > 0
    regions = [r for r in TT.segment(fg, cam, min_px=MIN_OBJ_PX, join_px=1, fill_holes=True) if r.front_xy[1] < 0.06]
    if near_box is not None:
        regions = [r for r in regions if float(np.mean(near_box[r.mask])) < 0.7]

    def good(r):
        return object_sized(r, cam)

    regions = [r for r in regions if good(r)]
    for _ in range(4):
        if not regions:
            break
        big = max(regions, key=lambda r: r.fit.length)
        if big.fit.length <= MAX_OBJ_LENGTH:
            if len(regions) >= n_expected or not split_short:
                break
            big = max(regions, key=lambda r: r.area)
            others = [r.area for r in regions if r is not big]
            if others and big.area < 1.6 * float(np.median(others)):
                break
        parts = [p for p in TT.split_region(big, cam, lab, accept=good) if good(p)]
        if len(parts) != 2:
            break
        regions = [r for r in regions if r is not big] + parts
    regions.sort(key=lambda r: -r.area)
    return regions if keep_all else regions[: n_expected + 3]


@dataclass
class Item:
    rank: int
    region: TT.Region
    xy: np.ndarray
    yaw_long: float
    width: float
    height: float
    arm: str = "left"
    stage: tuple | None = None
    state: str = "todo"
    close_yaw: float | None = None
    length: float = 0.0

    def __post_init__(self):
        if self.region is not None and getattr(self.region, "fit", None) is not None:
            self.length = float(self.region.fit.length)


def grasp_z(h: float) -> float:
    return TT.TABLE_Z + float(np.clip(0.5 * h - 0.004, GRASP_Z_MIN, GRASP_Z_MAX))


def solve_down(arm: str, xy, z: float, yaw: float, q_seed, tilt: float = 0.0) -> np.ndarray | None:
    rot = grasp_rot(arm, xy, yaw, tilt)
    if rot is None:
        return None
    seeds = [np.asarray(q_seed, float), G.base_seed(arm, xy), G.base_seed(arm, xy) + np.array([0, 0, 0, 0, 0, yaw])]
    for s in seeds:
        r = K.solve(arm, tcp_pose(rot, [xy[0], xy[1], z]), s, K.TCP_OFFSET, iters=80)
        if r.ok:
            return r.q


TILTS = 0.0, math.radians(15), math.radians(30)


def grasp_rot(arm: str, xy, yaw: float, tilt: float):
    if not tilt:
        return K.top_down(yaw)
    bx, by = K.BASE_POS[arm][:2]
    to_base = np.array([bx - xy[0], by - xy[1]])
    to_base /= max(1e-09, float(np.linalg.norm(to_base)))
    for sgn in (1.0, -1.0):
        rot = G.grasp_rotation(yaw, sgn * tilt)
        if float(np.dot(-rot[:2, 0], to_base)) > 0.2:
            return rot


def grasp_plan(arm: str, xy, yaws, zs, q_seed):
    bx, by = K.BASE_POS[arm][:2]
    if math.hypot(xy[0] - bx, xy[1] - by) > G.MAX_REACH_XY + 0.05:
        return None
    for tilt in TILTS:
        for k in range(0, len(yaws), 2):
            best = None
            for yaw in yaws[k : k + 2]:
                qs = []
                seed = q_seed
                for z in zs:
                    q = solve_down(
                        arm, xy, z + (G.FINGER_HALF_WIDTH * math.sin(tilt) if tilt else 0.0), yaw, seed, tilt
                    )
                    if q is None:
                        break
                    qs.append(q)
                    seed = q
                if len(qs) != len(zs):
                    continue
                cost = float(np.sum(K.JOINT_COST_W * np.abs(qs[0] - np.asarray(q_seed))))
                if best is None or cost < best[0]:
                    best = cost, yaw, qs
            if best is not None:
                return best[1], tilt, best[2]


YAW_DEVS = tuple(math.radians(d) for d in (0, 15, -15, 30, -30))


def close_yaws(yaw_long: float) -> list[float]:
    out = []
    for d in YAW_DEVS:
        y = yaw_long + math.pi / 2 + d
        out += [math.remainder(y, 2 * math.pi), math.remainder(y + math.pi, 2 * math.pi)]
    return out


def g_obs(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


REACH_TOL = 0.008
MAX_HELD = 0.084


def reached(arm: Arm, p) -> float:
    q_obs = np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], dtype=float)
    return float(np.linalg.norm(K.point_of(K.fk(arm.name, q_obs), K.TCP_OFFSET) - np.asarray(p)))


def pick(arm: Arm, xy, yaw_long: float, width: float, height: float, yaws=None, lift_z: float = LIFT_Z, check=None):
    arm.held_up_h = None
    zg = grasp_z(height)
    zh = TT.TABLE_Z + max(height, 0.06) + HOVER_DZ
    yaws = close_yaws(yaw_long) if yaws is None else yaws
    plan = grasp_plan(arm.name, xy, yaws, [zh, zg], arm.q)
    if plan is None:
        return None
    yaw, tilt, qs = plan
    arm.last_close_yaw = yaw
    rot = grasp_rot(arm.name, xy, yaw, tilt)
    dz = G.FINGER_HALF_WIDTH * math.sin(tilt) if tilt else 0.0
    yield from move_retry(
        arm,
        lambda: arm.transit(
            tcp_pose(rot, [xy[0], xy[1], zh + dz]),
            z_safe=max(zh, TT.TABLE_Z + 0.1),
            grip=1.0,
            seeds=(qs[0],),
            joint_step=FREE_JOINT_STEP,
            up_step=UP_LIN,
        ),
    )
    if arm.status != "ok":
        return None
    if check is not None and (not check()):
        return "check"
    p = np.array([xy[0], xy[1], zg + dz])
    yield from move_retry(arm, lambda: arm.line_to(tcp_pose(rot, p), lin_step=DESCENT_LIN))
    if arm.status != "ok":
        yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], zh + dz]), lin_step=UP_LIN)
        return None
    yield from arm.hold(3)
    err = reached(arm, p)
    if err > REACH_TOL:
        q_obs = np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], dtype=float)
        z_tcp = K.point_of(K.fk(arm.name, q_obs), K.TCP_OFFSET)[2]
        arm.held_up_h = float(np.clip(z_tcp - G.FINGER_DEPTH - TT.TABLE_Z, 0.01, 0.09))
        yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], zh + dz]), lin_step=UP_LIN)
        return None
    yield from arm.set_grip(0.0, 8)
    yield from arm.hold(3)
    w = held_width(g_obs(arm))
    if w < MIN_HELD or w > MAX_HELD:
        yield from arm.set_grip(1.0, 5)
        yield from arm.line_to(tcp_pose(rot, [xy[0], xy[1], zh + dz]), lin_step=UP_LIN)
        return None
    ok = yield from move_retry(arm, lambda: arm.line_to(tcp_pose(rot, [xy[0], xy[1], lift_z + dz]), lin_step=LIFT_LIN))
    if not ok and lift_z > LIFT_Z:
        yield from move_retry(arm, lambda: arm.line_to(tcp_pose(rot, [xy[0], xy[1], LIFT_Z + dz]), lin_step=LIFT_LIN))
    w2 = held_width(g_obs(arm))
    if w2 < MIN_HELD:
        yield from arm.set_grip(1.0, 5)
        return None
    return yaw


CARRY_JOINT_STEP = 0.1
FREE_JOINT_STEP = 0.11
DESCENT_LIN = 0.02
LIFT_LIN = 0.024
UP_LIN = 0.02
MOVE_RETRIES = 8
GIVE_WAY_AFTER = 2
GIVE_WAY_WAIT = 60
MOVED = 0.02


def _moved(arm: Arm, q) -> float:
    return float(np.max(np.abs(np.asarray(arm.q, float) - np.asarray(q, float))))


def yields_way(arm: Arm, o) -> bool:
    free, o_free = arm.g >= 0.5, o.g >= 0.5
    return free and (not o_free) or (free == o_free and arm.name == "right")


def move_retry(arm: Arm, gen_fn):
    q_start = np.asarray(arm.q, float).copy()
    cleared = False
    for k in range(MOVE_RETRIES):
        yield from gen_fn()
        if arm.status == "ok":
            return True
        if arm.status == "fail":
            return False
        o = arm.other
        if (
            not cleared
            and k + 1 >= GIVE_WAY_AFTER
            and (o is not None)
            and o.active
            and (o.status == "blocked")
            and yields_way(arm, o)
        ):
            cleared = True
            back = yield from clear_way(arm, o, q_start)
            if not back:
                return False
            continue
        yield from arm.hold(4 + 2 * k)
    return False


def clear_way(arm: Arm, o, q_start):
    q0 = np.asarray(arm.q, float).copy()
    p = arm.tcp()
    if p[2] < LIFT_Z - 0.003:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], LIFT_Z]), lin_step=UP_LIN)
    yield from arm.joints_to(LOOK_Q[arm.name], grip=arm.g, joint_step=FREE_JOINT_STEP)
    if _moved(arm, q0) > MOVED:
        yield from arm.wait_until(lambda: o.status != "blocked" or not o.active, max_steps=GIVE_WAY_WAIT)
    if _moved(arm, q_start) <= MOVED:
        return True
    L6 = K.fk(arm.name, q_start)
    T0 = tcp_pose(L6[:3, :3], K.point_of(L6, K.TCP_OFFSET))
    z0 = float(T0[2, 3])
    for _ in range(3):
        yield from arm.transit(T0, z_safe=max(z0, LIFT_Z), grip=arm.g, seeds=(q_start,), joint_step=FREE_JOINT_STEP)
        if arm.status == "ok":
            return True
        yield from arm.hold(8)
    return False


def least_turn(arm: Arm, qs):
    qs = [q for q in qs if q is not None]
    if not qs:
        return None
    return min(qs, key=lambda q: float(np.max(np.abs(np.asarray(q) - arm.q))))


def drop_along(sh, length: float) -> float | None:
    rect = getattr(sh, "basket_rect", None)
    if rect is None or length < LONG_DROP:
        return None
    return math.pi / 2 if rect[3] - rect[1] >= rect[2] - rect[0] else 0.0


LONG_DROP_OFF = 0.05


def drop_point(sh, length: float, width: float = 0.05, img=None, robot=None) -> np.ndarray:
    off = np.asarray(DROP_OFFSETS[sh.n_drops % len(DROP_OFFSETS)], float)
    along = drop_along(sh, length)
    k = None
    if along is not None:
        k = 1 if abs(math.sin(along)) > 0.5 else 0
        off[k] = float(np.clip(off[k], -LONG_DROP_OFF, LONG_DROP_OFF))
    centre = np.asarray(sh.basket_xy, float)
    ref = getattr(sh, "basket_ref", None)
    if img is None or ref is None:
        return centre + off
    best = free_drop_offset(img, robot, ref, centre, off, k, length, width)
    if best is None:
        return centre + off
    if not np.allclose(best[0], off) and (not drop_reachable(centre + best[0], along)):
        return centre + off
    return centre + best[0]


def drop_reachable(xy, along) -> bool:
    yaws = (
        (along + math.pi / 2, along - math.pi / 2) if along is not None else (0.0, math.pi / 2, math.pi, -math.pi / 2)
    )
    for y in yaws:
        q = solve_down("left", xy, DROP_Z, math.remainder(y, 2 * math.pi), LOOK_Q["left"])
        if q is not None and solve_down("left", xy, RELEASE_Z, math.remainder(y, 2 * math.pi), q) is not None:
            return True
    return False


def head_view(arm: Arm):
    img = arm.ctx.obs.images.get("cam_head")
    if img is None:
        return None, None
    q = {arm.name: arm.q, arm.other.name: arm.other.q}
    g = {arm.name: arm.g, arm.other.name: arm.other.g}
    return img, RM.mask(C.head(), q["left"], q["right"], g["left"], g["right"], dilate=6)


FREE_DE = 15.0
FREE_MARGIN = 0.015
FREE_KNOWN = 0.6
FREE_GAIN = 0.05
FLOOR_Z = TT.TABLE_Z + 0.01


def free_drop_offset(img, robot, ref, centre, off_default, k_long, length, width):
    import cv2

    lab_ref, robot_ref = ref
    lab = TT.lab_image(img)
    cam = C.head()
    unknown = robot_ref.copy()
    if robot is not None:
        unknown |= robot
    changed = DW.delta_e(lab, lab_ref) > FREE_DE
    xs = sorted({round(o[0], 4) for o in DROP_OFFSETS})
    ys = sorted({round(o[1], 4) for o in DROP_OFFSETS})
    gx = np.linspace(min(xs), max(xs), 3)
    gy = np.linspace(min(ys), max(ys), 5)
    if k_long == 1:
        gy = np.linspace(-LONG_DROP_OFF, LONG_DROP_OFF, 5)
    elif k_long == 0:
        gx = np.linspace(-LONG_DROP_OFF, LONG_DROP_OFF, 5)
    big = max(length, width)
    if k_long is None:
        hx = hy = 0.5 * big + FREE_MARGIN
    elif k_long == 1:
        hx, hy = 0.5 * width + FREE_MARGIN, 0.5 * length + FREE_MARGIN
    else:
        hx, hy = 0.5 * length + FREE_MARGIN, 0.5 * width + FREE_MARGIN
    res = []
    for dx in gx:
        for dy in gy:
            o = np.array([dx, dy])
            c = centre + o
            pts = np.array(
                [
                    [c[0] - hx, c[1] - hy, FLOOR_Z],
                    [c[0] + hx, c[1] - hy, FLOOR_Z],
                    [c[0] + hx, c[1] + hy, FLOOR_Z],
                    [c[0] - hx, c[1] + hy, FLOOR_Z],
                ]
            )
            uv, _ = cam.project(pts)
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
            m = m > 0
            n = int(m.sum())
            if n == 0:
                continue
            known = m & ~unknown
            nk = int(known.sum())
            if nk < FREE_KNOWN * n:
                continue
            res.append((o, float(changed[known].sum()) / nk))
    if not res:
        return None
    d = [r for r in res if np.allclose(r[0], off_default, atol=0.004)]
    occ_def = d[0][1] if d else None
    o, occ = min(res, key=lambda r: (r[1], float(np.linalg.norm(r[0] - off_default))))
    if occ_def is not None and occ > occ_def - FREE_GAIN:
        return off_default, occ_def, occ_def
    return o, occ, occ_def


def drop_into(arm: Arm, xy, along: float | None = None):
    r0 = arm.T[:3, :3]
    yaw0 = math.atan2(r0[1, 1], r0[0, 1])
    p = arm.tcp()
    if p[2] < DROP_Z - 0.003:
        up = tcp_pose(r0, [p[0], p[1], DROP_Z])
        yield from move_retry(arm, lambda: arm.line_to(up, lin_step=LIFT_LIN))
    q_drop = None
    if along is not None:
        c = along + math.pi / 2
        q_drop = least_turn(
            arm, [solve_down(arm.name, xy, DROP_Z, math.remainder(c + d, 2 * math.pi), arm.q) for d in (0.0, math.pi)]
        )
    if q_drop is None:
        q_drop = least_turn(
            arm,
            [
                solve_down(arm.name, xy, DROP_Z, math.remainder(yaw0 + d, 2 * math.pi), arm.q)
                for d in (0.0, math.pi / 2, -math.pi / 2, math.pi)
            ],
        )
    if q_drop is None:
        return False
    ok = yield from move_retry(arm, lambda: arm.joints_to(q_drop, grip=0.0, joint_step=CARRY_JOINT_STEP))
    if not ok:
        return False
    if held_width(g_obs(arm)) < MIN_HELD:
        yield from arm.set_grip(1.0, 4)
        return "lost"
    T = arm.T.copy()
    p = arm.tcp()
    yield from arm.line_to(tcp_pose(T[:3, :3], [p[0], p[1], RELEASE_Z]), lin_step=LIFT_LIN)
    yield from arm.set_grip(1.0, 4)
    yield from arm.hold(2)
    yield from move_retry(arm, lambda: arm.line_to(tcp_pose(T[:3, :3], [p[0], p[1], DROP_Z]), lin_step=0.03))
    return True


CARRY_MIN_Z = TT.TABLE_Z + 0.1


def joint_path_min_z(arm: Arm, q_goal, n: int = 12) -> float:
    q0 = np.asarray(arm.q, float)
    q1 = np.asarray(q_goal, float)
    return min(float(K.point_of(K.fk(arm.name, q0 + (q1 - q0) * t), K.TCP_OFFSET)[2]) for t in np.linspace(0.0, 1.0, n))


def place_at(arm: Arm, xy, height: float):
    r0 = arm.T[:3, :3]
    yaw0 = math.atan2(r0[1, 1], r0[0, 1])
    zg = grasp_z(height) + 0.004
    p = arm.tcp()
    if p[2] < DROP_Z - 0.003:
        yield from move_retry(arm, lambda: arm.line_to(tcp_pose(r0, [p[0], p[1], DROP_Z]), lin_step=LIFT_LIN))
    yaw, q_high = yaw0, None
    for z_carry in (DROP_Z, LIFT_Z):
        cands = []
        for d in (
            0.0,
            math.pi,
            math.pi / 2,
            -math.pi / 2,
            math.pi / 4,
            -math.pi / 4,
            3 * math.pi / 4,
            -3 * math.pi / 4,
        ):
            y = math.remainder(yaw0 + d, 2 * math.pi)
            qh = solve_down(arm.name, xy, z_carry, y, arm.q)
            if qh is not None and solve_down(arm.name, xy, zg, y, qh) is not None:
                cands.append((float(np.max(np.abs(qh - arm.q))), y, qh))
        if cands:
            _, yaw, q_high = min(cands, key=lambda c: c[0])
            break
    rot = K.top_down(yaw)
    if q_high is not None and joint_path_min_z(arm, q_high) >= CARRY_MIN_Z:
        ok = yield from move_retry(arm, lambda: arm.joints_to(q_high, grip=0.0, joint_step=CARRY_JOINT_STEP))
    else:
        seeds = () if q_high is None else (q_high,)
        ok = yield from move_retry(
            arm,
            lambda: arm.transit(
                tcp_pose(rot, [xy[0], xy[1], DROP_Z]), z_safe=LIFT_Z, seeds=seeds, joint_step=CARRY_JOINT_STEP
            ),
        )
    if not ok:
        return None
    if held_width(g_obs(arm)) < MIN_HELD:
        yield from arm.set_grip(1.0, 4)
        return "lost"
    yield from move_retry(arm, lambda: arm.line_to(tcp_pose(rot, [xy[0], xy[1], zg]), lin_step=DESCENT_LIN))
    if arm.status != "ok":
        return None
    yield from arm.set_grip(1.0, 6)
    yield from arm.hold(2)
    yield from move_retry(arm, lambda: arm.line_to(tcp_pose(rot, [xy[0], xy[1], LIFT_Z]), lin_step=UP_LIN))
    return yaw


def go_home(arm: Arm):
    p = arm.tcp()
    if p[2] < TT.TABLE_Z + 0.1:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], TT.TABLE_Z + 0.12]), lin_step=UP_LIN)
    yield from arm.joints_to(arm.q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP)


H_UNKNOWN = 0.035


def taken_xy(sh, it) -> list:
    taken = [
        np.asarray(i.xy, float)
        for i in sh.items
        if i is not it and i.xy is not None and (i.state in ("todo", "staging", "staged"))
    ]
    taken += [
        np.asarray(i.stage, float) for i in sh.items if i is not it and i.state == "staged" and (i.stage is not None)
    ]
    return taken


TAKEN_R = 0.05
LOST_MAX_COST = 1.0


def search(arm: Arm, sh: Shared, it: Item):
    p = arm.tcp()
    if p[2] < LIFT_Z - 0.003:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], LIFT_Z]), lin_step=UP_LIN)
    yield from arm.joints_to(LOOK_Q[arm.name], grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
    yield from arm.hold(2)
    img = arm.ctx.obs.images.get("cam_head")
    if img is None:
        return None
    cam = C.head()
    q = {arm.name: arm.q, arm.other.name: arm.other.q}
    g = {arm.name: arm.g, arm.other.name: arm.other.g}
    robot = RM.mask(cam, q["left"], q["right"], g["left"], g["right"], dilate=4)
    excl = robot if sh.exclude is None else robot | sh.exclude
    table_area = TT.plane_mask(cam, img.shape)
    lab = TT.lab_image(img)
    nt = TT.not_table(img, cam, table_area & ~excl, lab)
    regions = near_objects(img, cam, robot, excl, nt, lab, N_OBJECTS, getattr(sh, "boxes", None))
    taken = taken_xy(sh, it)
    bg = TT.table_background(lab, table_area & ~excl)
    best, bc = None, math.inf
    for r in regions:
        if any(np.linalg.norm(r.fit.xy - np.asarray(t)) < TAKEN_R for t in taken):
            continue
        c = TT.appearance_cost(it.desc, near_appearance(r, nt & ~excl, lab, bg, cam))
        if c < bc:
            best, bc = r, c
    it.search_cost = bc
    return None if best is None else best.fit


MID_X = 0.17
MID_OUT = 0.04
MID_WAIT = 700
MID_SEP = 0.18
MID_CROSS = 0.04
MID_CLEAR = 0.02
_CONFS: dict = {}
_CLEAR: dict = {}


def target_confs(arm: str, xy) -> list:
    key = arm, round(float(xy[0]), 2), round(float(xy[1]), 2)
    c = _CONFS.get(key)
    if c is None:
        c = []
        p = np.array(key[1:])
        for z in (TT.TABLE_Z + 0.12, grasp_z(H_UNKNOWN)):
            for yaw in (0.0, math.pi / 2):
                pl = grasp_plan(arm, p, [yaw, yaw + math.pi], [z], LOOK_Q[arm])
                if pl is not None:
                    c.append(pl[2][0])
        _CONFS[key] = c
    return c


def target_clearance(xy_left, xy_right) -> float:
    key = (
        round(float(xy_left[0]), 2),
        round(float(xy_left[1]), 2),
        round(float(xy_right[0]), 2),
        round(float(xy_right[1]), 2),
    )
    d = _CLEAR.get(key)
    if d is None:
        a, b = target_confs("left", key[:2]), target_confs("right", key[2:])
        d = min((CO.arm_distance(qa, qb, 1.0, 1.0) for qa in a for qb in b), default=math.inf)
        _CLEAR[key] = d
    return d


class Mid:
    def __init__(self):
        self.claims: dict[str, np.ndarray] = {}
        self.intent: tuple | None = None

    def blocked_by_intent(self, name: str, xy, rank) -> bool:
        if name != "right" or self.intent is None or rank is None or (self.intent[0] >= rank):
            return False
        return any(not self.compatible(ixy, xy) for ixy in self.intent[1])

    @staticmethod
    def compatible(xy_left, xy_right) -> bool:
        return (
            float(xy_right[0]) - float(xy_left[0]) >= MID_CROSS
            and float(np.linalg.norm(np.asarray(xy_right[:2], float) - np.asarray(xy_left[:2], float))) >= MID_SEP
            and (target_clearance(xy_left, xy_right) >= MID_CLEAR)
        )

    def take(self, name: str, xy, rank=None) -> bool:
        xy = np.asarray(xy, float)[:2]
        if self.blocked_by_intent(name, xy, rank):
            return False
        other = "right" if name == "left" else "left"
        o = self.claims.get(other)
        if o is not None and (not (self.compatible(xy, o) if name == "left" else self.compatible(o, xy))):
            return False
        self.claims[name] = xy
        return True

    def free(self, name: str) -> None:
        self.claims.pop(name, None)

    def holds(self, name: str) -> bool:
        return name in self.claims


def out_of_mid(arm: Arm) -> bool:
    side = -1.0 if arm.name == "left" else 1.0
    return side * float(arm.tcp()[0]) > MID_X + MID_OUT


WAIT_BEFORE_RETREAT = 30
RETREAT_XY = {"left": (-0.33, -0.25), "right": (0.33, -0.25)}


def retreat(arm: Arm):
    p = arm.tcp()
    side = -1.0 if arm.name == "left" else 1.0
    if side * float(p[0]) >= abs(RETREAT_XY[arm.name][0]) - 0.02:
        return
    r0 = arm.T[:3, :3]
    yaw0 = math.atan2(r0[1, 1], r0[0, 1])
    if p[2] < DROP_Z - 0.003:
        yield from move_retry(arm, lambda: arm.line_to(tcp_pose(r0, [p[0], p[1], DROP_Z]), lin_step=LIFT_LIN))
    q = least_turn(
        arm,
        [
            solve_down(arm.name, RETREAT_XY[arm.name], DROP_Z, math.remainder(yaw0 + d, 2 * math.pi), arm.q)
            for d in (0.0, math.pi / 2, -math.pi / 2, math.pi)
        ],
    )
    if q is not None:
        yield from move_retry(arm, lambda: arm.joints_to(q, grip=arm.g, joint_step=CARRY_JOINT_STEP))


def wait_retreating(arm: Arm, sh: "Shared", cond, max_steps: int):
    yield from arm.wait_until(cond, max_steps=WAIT_BEFORE_RETREAT)
    if not cond():
        free_mid(arm, sh)
        yield from retreat(arm)
        yield from arm.wait_until(cond, max_steps=max_steps)


def refresh_intent(sh: "Shared") -> None:
    if sh.mid is None or sh.mid.intent is not None:
        return
    nxt = next((i for i in sh.items if i.state not in ("done", "failed")), None)
    if nxt is not None and nxt.state == "staged" and (nxt.stage is not None):
        sh.mid.intent = nxt.rank, [np.asarray(nxt.stage, float)[:2]]


def take_mid(arm: Arm, sh: "Shared", xy, rank=None):
    if arm.name == "right":
        refresh_intent(sh)
    if sh.mid is None or sh.mid.take(arm.name, xy, rank):
        return
    if sh.mid.blocked_by_intent(arm.name, xy, rank):
        free_mid(arm, sh)
        yield from retreat(arm)
    yield from wait_retreating(arm, sh, lambda: sh.mid.take(arm.name, xy, rank), MID_WAIT)


def freeing_mid(arm: Arm, sh: "Shared", gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            if sh.mid is not None and out_of_mid(arm):
                sh.mid.free(arm.name)
            return e.value
        if sh.mid is not None and sh.mid.holds(arm.name) and out_of_mid(arm):
            sh.mid.free(arm.name)
        yield


@dataclass
class Shared:
    items: list
    basket_xy: np.ndarray
    exclude: np.ndarray | None = None
    spot_busy: dict = field(default_factory=dict)
    mid: Mid | None = None
    basket_rect: tuple | None = None
    stray_fixes: int = 0
    n_drops: int = -1


RELOCATE_RADIUS = 0.08


def relocate(arm: Arm, sh: Shared, xy, h_prior: float | None = None, it: Item | None = None):
    p = arm.tcp()
    if p[2] < LIFT_Z - 0.003:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], LIFT_Z]), lin_step=UP_LIN)
    yield from arm.joints_to(LOOK_Q[arm.name], grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
    yield from arm.hold(2)
    avoid = None if it is None else taken_xy(sh, it)
    best, _ = find_near(arm, sh, xy, RELOCATE_RADIUS, h_prior, dilate=4, avoid=avoid)
    return best


def find_near(
    arm: Arm,
    sh: Shared,
    xy,
    radius: float,
    h_prior: float | None = None,
    dilate: int = 6,
    avoid=None,
    info: dict | None = None,
):
    img = arm.ctx.obs.images.get("cam_head")
    if img is None:
        return None, radius
    cam = C.head()
    q = {arm.name: arm.q, arm.other.name: arm.other.q}
    g = {arm.name: arm.g, arm.other.name: arm.other.g}
    robot = RM.mask(cam, q["left"], q["right"], g["left"], g["right"], dilate=dilate)
    excl = robot if sh.exclude is None else robot | sh.exclude
    table_area = TT.plane_mask(cam, img.shape)
    lab = TT.lab_image(img)
    nt = TT.not_table(img, cam, table_area & ~excl, lab)
    regions = near_objects(img, cam, robot, excl, nt, lab, 1, getattr(sh, "boxes", None), keep_all=True)
    best, bd, best_r = None, radius, None
    for r in regions:
        f = r.fit
        if h_prior is not None:
            f2 = TT.fit_footprint(r.mask, cam, h_prior=h_prior, prior_slack=PRIOR_SLACK, prior_min_iou=PRIOR_MIN_IOU)
            if f2 is not None:
                f = f2
        d = float(np.linalg.norm(f.xy - np.asarray(xy)[:2]))
        if avoid and any(float(np.linalg.norm(f.xy - np.asarray(t)[:2])) < min(TAKEN_R, d) for t in avoid):
            continue
        if d < bd:
            best, bd, best_r = f, d, r
    if info is not None:
        info["robot_frac"] = 0.0 if best_r is None else robot_border(best_r.mask, robot)
    return best, bd


def robot_border(mask: np.ndarray, robot: np.ndarray, ring: int = 3) -> float:
    import cv2

    k = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    out = (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask
    n = int(out.sum())
    return 0.0 if n == 0 else float((out & robot).sum()) / n


STAGED_RADIUS = 0.09
STAGED_SHIFT = 0.01
STAGED_CUT = 0.02
STAGED_LOOKS = 6
STAGED_LOOK_DT = 3
STAGED_MIN_IOU = 0.75
STRAY_AREA = (-0.66, 0.0), (-0.38, 0.08)
STRAY_DIST = 0.1
STRAY_SPOT_CLEAR = 0.1
STRAY_KNOWN = 0.05
MAX_STRAY_FIXES = 1
STRAY_ROBOT_PX = 12
STRAY_MAX_COST = 0.75
FINAL_CHECK_BEFORE = 1450


def basket_rect(box, cam) -> tuple | None:
    vs, us = np.nonzero(box.mask)
    if len(vs) == 0:
        return None
    p = cam.on_plane_z(np.stack([us, vs], axis=1).astype(float) + 0.5, TT.TABLE_Z + 0.04)
    p = p[np.isfinite(p[:, 0])]
    if len(p) == 0:
        return None
    return (
        float(np.percentile(p[:, 0], 2)),
        float(np.percentile(p[:, 0], 98)),
        float(np.percentile(p[:, 1], 2)),
        float(np.percentile(p[:, 1], 98)),
    )


def rect_dist(xy, rect) -> float:
    dx = max(rect[0] - xy[0], 0.0, xy[0] - rect[1])
    dy = max(rect[2] - xy[1], 0.0, xy[1] - rect[3])
    return math.hypot(dx, dy)


def find_stray(arm: Arm, sh: Shared):
    img = arm.ctx.obs.images.get("cam_head")
    if img is None or getattr(sh, "basket_rect", None) is None:
        return None
    done = [i for i in sh.items if i.state == "done" and getattr(i, "desc", None) is not None]
    if not done:
        return None
    rest = [i for i in sh.items if i.state in ("todo", "staging", "staged") and getattr(i, "desc", None) is not None]
    spots = [tuple(s) for s in sh.spot_busy] + [
        tuple(i.stage) for i in sh.items if i.state in ("staging", "staged") and i.stage is not None
    ]
    cam = C.head()
    q = {arm.name: arm.q, arm.other.name: arm.other.q}
    g = {arm.name: arm.g, arm.other.name: arm.other.g}
    robot = RM.mask(cam, q["left"], q["right"], g["left"], g["right"], dilate=6)
    excl = robot if sh.exclude is None else robot | sh.exclude
    table_area = TT.plane_mask(cam, img.shape)
    lab = TT.lab_image(img)
    nt = TT.not_table(img, cam, table_area & ~excl, lab)
    regions = near_objects(img, cam, robot, excl, nt, lab, N_OBJECTS, getattr(sh, "boxes", None), area=STRAY_AREA)
    bg = TT.table_background(lab, table_area & ~excl)
    import cv2

    near_robot = cv2.dilate(robot.astype(np.uint8), np.ones((2 * STRAY_ROBOT_PX + 1,) * 2, np.uint8)) > 0
    for r in regions:
        f = r.fit
        if float(np.mean(near_robot[r.mask])) > 0.05:
            continue
        dist = rect_dist(f.xy, sh.basket_rect)
        if dist < 0.01 or dist > STRAY_DIST:
            continue
        if any(i.xy is not None and float(np.linalg.norm(np.asarray(f.xy) - i.xy)) < STRAY_KNOWN for i in rest):
            continue
        if any(float(np.linalg.norm(np.asarray(f.xy) - np.asarray(s))) < STRAY_SPOT_CLEAR for s in spots):
            continue
        d = near_appearance(r, nt & ~excl, lab, bg, cam)
        c_done = min(TT.appearance_cost(i.desc, d) for i in done)
        c_rest = min((TT.appearance_cost(i.desc, d) for i in rest), default=math.inf)
        if c_done < min(c_rest, STRAY_MAX_COST):
            return f


def put_back(arm: Arm, sh: Shared, f):
    free_mid(arm, sh)
    sh.stray_fixes += 1
    xy = np.asarray(f.xy, float)
    yield from take_mid(arm, sh, xy)
    yaw = yield from pick(arm, xy, f.yaw, f.width, f.height, lift_z=DROP_Z)
    if yaw is None:
        free_mid(arm, sh)
        return False
    sh.n_drops += 1
    drop_xy = drop_point(sh, f.length, f.width, *head_view(arm))
    ok = yield from freeing_mid(arm, sh, drop_into(arm, drop_xy, drop_along(sh, f.length)))
    free_mid(arm, sh)
    return ok is True


def left_task(arm: Arm, sh: Shared):
    for it in sh.items:
        if it.state == "failed":
            break
        if it.xy is None:
            f = yield from search(arm, sh, it)
            if f is None or grasp_plan(arm.name, f.xy, close_yaws(f.yaw), [LIFT_Z, grasp_z(f.height)], arm.q) is None:
                it.state = "failed"
                break
            it.xy, it.yaw_long, it.width, it.height = np.asarray(f.xy, float), f.yaw, f.width, f.height
            it.length = float(f.length)
        if it.stage is not None:
            yield from arm.wait_until(lambda it=it: it.state in ("staged", "failed") or it.stage is None, max_steps=600)
            if it.stage is not None and it.state != "staged":
                it.state = "failed"
                break
        if it.stage is not None:
            xy, yaws = np.asarray(it.stage), close_yaws(it.close_yaw - math.pi / 2)
        else:
            xy, yaws = it.xy, None
        set_intent(sh, it, xy)
        yield from take_mid(arm, sh, xy)
        yl, w, h = it.yaw_long, it.width, it.height
        if it.stage is not None:
            f, d, cut = None, STAGED_RADIUS, 0.0
            for look in range(STAGED_LOOKS):
                info = {}
                f, d = find_near(arm, sh, xy, STAGED_RADIUS, avoid=taken_xy(sh, it), info=info)
                cut = info.get("robot_frac", 0.0)
                if f is not None and cut <= STAGED_CUT and (f.iou >= STAGED_MIN_IOU):
                    break
                if look + 1 < STAGED_LOOKS:
                    yield from arm.hold(STAGED_LOOK_DT)
            if f is not None and cut > STAGED_CUT:
                f = None
            if f is not None and f.iou < STAGED_MIN_IOU:
                f = None
            if f is not None:
                it.length = max(it.length, float(f.length))
            if f is not None and d > STAGED_SHIFT:
                xy = np.asarray(f.xy, float)
                if f.length > 1.25 * f.width:
                    yaws, yl, w = None, f.yaw, f.width
                set_intent(sh, it, xy)
                yield from take_mid(arm, sh, xy)
        yaw = None
        stray = {}
        view = {}

        def check():
            view["v"] = head_view(arm)
            if sh.stray_fixes >= MAX_STRAY_FIXES:
                return True
            f = find_stray(arm, sh)
            if f is None:
                return True
            stray["f"] = f
            return False

        attempt = 0
        while attempt < MAX_ATTEMPTS:
            yaw = yield from pick(arm, xy, yl, w, h, yaws, lift_z=DROP_Z, check=check)
            if yaw == "check":
                yield from put_back(arm, sh, stray.pop("f"))
                yield from take_mid(arm, sh, xy)
                yaw = None
                continue
            attempt += 1
            if yaw is not None or attempt == MAX_ATTEMPTS:
                break
            f = yield from relocate(arm, sh, xy, getattr(arm, "held_up_h", None), it)
            if f is None and getattr(it, "desc", None) is not None:
                f = yield from search(arm, sh, it)
                if (
                    f is not None
                    and grasp_plan(arm.name, f.xy, close_yaws(f.yaw), [LIFT_Z, grasp_z(f.height)], arm.q) is None
                ):
                    f = None
                if f is not None:
                    yield from take_mid(arm, sh, f.xy)
            if f is None:
                break
            xy, yl, w, yaws = np.asarray(f.xy, float), f.yaw, f.width, None
            it.length = max(it.length, float(f.length))
            set_intent(sh, it, xy)
            yield from take_mid(arm, sh, xy)
            if getattr(arm, "held_up_h", None):
                h = max(h, arm.held_up_h)
                if getattr(arm, "last_close_yaw", None) is not None:
                    yaws = close_yaws(arm.last_close_yaw)
        if sh.mid is not None:
            sh.mid.intent = None
        if yaw is None:
            it.state = "failed"
            free_mid(arm, sh)
            break
        it.state = "carried"
        if it.stage is not None:
            sh.spot_busy.pop(tuple(it.stage), None)
        sh.n_drops += 1
        drop_xy = drop_point(sh, it.length, it.width, *(view.get("v") or head_view(arm)))
        ok = yield from freeing_mid(arm, sh, drop_into(arm, drop_xy, drop_along(sh, it.length)))
        for _ in range(2):
            if ok != "lost" or getattr(it, "desc", None) is None:
                break
            f = yield from search(arm, sh, it)
            if f is None or getattr(it, "search_cost", 0.0) > LOST_MAX_COST:
                ok = True
                break
            yield from take_mid(arm, sh, f.xy)
            it.length = max(it.length, float(f.length))
            yaw = yield from pick(arm, np.asarray(f.xy, float), f.yaw, f.width, max(f.height, it.height), lift_z=DROP_Z)
            if yaw is None:
                ok = True
                break
            ok = yield from freeing_mid(arm, sh, drop_into(arm, drop_xy, drop_along(sh, it.length)))
        free_mid(arm, sh)
        if ok is not True:
            it.state = "failed"
            break
        it.state = "done"
    free_mid(arm, sh)
    if (
        all(i.state == "done" for i in sh.items)
        and sh.stray_fixes < MAX_STRAY_FIXES
        and (arm.ctx.obs.step < FINAL_CHECK_BEFORE)
    ):
        yield from arm.joints_to(LOOK_Q[arm.name], grip=1.0, force=True, joint_step=FREE_JOINT_STEP)
        yield from arm.hold(1)
        f = find_stray(arm, sh)
        if f is not None:
            yield from put_back(arm, sh, f)
    yield from go_home(arm)


def free_mid(arm: Arm, sh: Shared) -> None:
    if sh.mid is not None:
        sh.mid.free(arm.name)


def set_intent(sh: Shared, it: Item, xy) -> None:
    if sh.mid is not None:
        sh.mid.intent = it.rank, [np.asarray(xy, float)[:2]]


def right_task(arm: Arm, sh: Shared):
    def failed():
        return any(i.state == "failed" for i in sh.items)

    for it in sh.items:
        if it.stage is None:
            continue
        if failed():
            break
        spots = [tuple(s) for s in getattr(it, "spots", [it.stage])]

        def free_spots():
            return [s for s in spots if s not in sh.spot_busy]

        if not free_spots():
            yield from freeing_mid(arm, sh, park(arm))
            free_mid(arm, sh)
            yield from arm.wait_until(lambda: bool(free_spots()) or failed(), max_steps=800)
        if failed():
            break
        it.state = "staging"
        res = yield from stage_one(arm, sh, it, free_spots, failed)
        if res == "failed":
            it.state = "failed"
            free_mid(arm, sh)
            break
    yield from freeing_mid(arm, sh, park(arm))
    free_mid(arm, sh)
    yield from arm.wait_until(
        lambda: not arm.other.active or all(i.state in ("done", "failed") for i in sh.items), max_steps=1600
    )
    yield from go_home(arm)


MAX_CARRY_LOSSES = 2


def stage_one(arm: Arm, sh: Shared, it: Item, free_spots, failed):
    xy, yl, w = it.xy, it.yaw_long, it.width
    for loss in range(MAX_CARRY_LOSSES + 1):
        yield from take_mid(arm, sh, xy, it.rank)
        yaw = None
        yaws, h = None, it.height
        for attempt in range(MAX_ATTEMPTS):
            yaw = yield from pick(arm, xy, yl, w, h, yaws, lift_z=DROP_Z)
            if yaw is not None or attempt + 1 == MAX_ATTEMPTS:
                break
            held_up = getattr(arm, "held_up_h", None)
            f = yield from relocate(arm, sh, xy, held_up, it)
            if f is None and getattr(it, "desc", None) is not None:
                f = yield from search(arm, sh, it)
                if f is not None:
                    fxy = np.asarray(f.xy, float)
                    zs = [TT.TABLE_Z + max(h, 0.06) + HOVER_DZ, grasp_z(h)]
                    if (
                        fxy[0] <= LEFT_MAX_X
                        and grasp_plan("left", fxy, close_yaws(f.yaw), zs, LOOK_Q["left"]) is not None
                    ):
                        free_mid(arm, sh)
                        it.xy, it.yaw_long, it.width = fxy, f.yaw, f.width
                        it.stage = None
                        it.state = "todo"
                        return "left"
                    if grasp_plan(arm.name, fxy, close_yaws(f.yaw), zs, arm.q) is None:
                        f = None
            if f is None:
                break
            xy, yl, w, yaws = np.asarray(f.xy, float), f.yaw, f.width, None
            it.length = max(it.length, float(f.length))
            if held_up:
                h = max(h, held_up)
                if getattr(arm, "last_close_yaw", None) is not None:
                    yaws = close_yaws(arm.last_close_yaw)
            yield from take_mid(arm, sh, xy, it.rank)
        if yaw is None:
            return "failed"
        chosen = []

        def grant():
            refresh_intent(sh)
            for s in free_spots():
                if sh.mid is None or sh.mid.take(arm.name, s, it.rank):
                    chosen.append(s)
                    return True
            return False

        if not grant():
            yield from wait_retreating(arm, sh, lambda: grant() or failed(), MID_WAIT)
            if not chosen:
                chosen = free_spots()[:1] or [tuple(it.stage)]
        spot = chosen[0]
        sh.spot_busy[spot] = it.rank
        res = yield from place_at(arm, spot, it.height)
        if res is None:
            sh.spot_busy.pop(spot, None)
            return "failed"
        if res != "lost":
            it.stage, it.close_yaw = spot, res
            it.state = "staged"
            return "staged"
        sh.spot_busy.pop(spot, None)
        free_mid(arm, sh)
        f = yield from search(arm, sh, it)
        if f is None:
            return "failed"
        fxy = np.asarray(f.xy, float)
        zs = [TT.TABLE_Z + max(it.height, 0.06) + HOVER_DZ, grasp_z(it.height)]
        if fxy[0] <= LEFT_MAX_X and grasp_plan("left", fxy, close_yaws(f.yaw), zs, LOOK_Q["left"]) is not None:
            it.xy, it.yaw_long, it.width = fxy, f.yaw, f.width
            it.stage = None
            it.state = "todo"
            return "left"
        xy, yl, w = fxy, f.yaw, f.width
    return "failed"


def park(arm: Arm):
    p = arm.tcp()
    if p[2] < LIFT_Z - 0.003:
        yield from move_retry(arm, lambda: arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], LIFT_Z]), lin_step=UP_LIN))
    yield from arm.joints_to(LOOK_Q[arm.name], grip=1.0, force=True, joint_step=FREE_JOINT_STEP)


def footprint_dist(p, f) -> float:
    d = np.asarray(p, float)[:2] - np.asarray(f.xy, float)[:2]
    u = np.array([math.cos(f.yaw), math.sin(f.yaw)])
    a = max(0.0, abs(float(d @ u)) - 0.5 * f.length)
    b = max(0.0, abs(float(d[0] * -u[1] + d[1] * u[0])) - 0.5 * f.width)
    return math.hypot(a, b)


GAP_FACTOR = 1.6


def missing_pick(times) -> int | None:
    if len(times) < 3:
        return None
    gaps = np.diff(np.asarray(times, float))
    k = int(np.argmax(gaps))
    med = float(np.median(np.delete(gaps, k)))
    if med < 40 or gaps[k] < GAP_FACTOR * med:
        return None
    return k + 1


def plan_items(an: Analysis) -> list[Item] | None:
    items = []
    assigned = {j for j in an.assign if j is not None}
    spare = [
        j
        for j, r in enumerate(an.near)
        if j not in assigned
        and r.fit is not None
        and (r.fit.length <= MAX_OBJ_LENGTH)
        and (r.fit.width >= MIN_OBJ_WIDTH)
    ]
    insert_at = None
    if len(an.holes) < N_OBJECTS and spare:
        insert_at = missing_pick([an.holes[hi].first_gone for hi in an.order])
    for rank, hi in enumerate(an.order):
        j = an.assign[hi]
        if j is None:
            it = Item(rank, None, None, 0.0, 0.05, H_UNKNOWN)
            it.desc = an.far_desc[hi]
            it.arm = "search"
            items.append(it)
            continue
        r = an.near[j]
        f = r.fit
        it = Item(rank, r, np.asarray(f.xy, float), f.yaw, f.width, f.height)
        it.desc = an.far_desc[hi]
        items.append(it)
    if insert_at is not None:
        r = an.near[spare[0]]
        f = r.fit
        it = Item(insert_at, r, np.asarray(f.xy, float), f.yaw, f.width, f.height)
        it.desc = an.near_desc[spare[0]] if getattr(an, "near_desc", None) else None
        items.insert(insert_at, it)
        for k, i in enumerate(items):
            i.rank = k
    others = [r.fit for r in an.near]
    spot_ok = {}
    spots = [s for s in STAGE_SPOTS if all(footprint_dist(s, f) > STAGE_CLEAR for f in others)]
    used: list = []
    n_staged = 0
    for it in items:
        if it.xy is None:
            continue
        zs = [TT.TABLE_Z + max(it.height, 0.06) + HOVER_DZ, grasp_z(it.height)]
        right_ok = grasp_plan("right", it.xy, close_yaws(it.yaw_long), zs, LOOK_Q["right"]) is not None
        left_plan = grasp_plan("left", it.xy, close_yaws(it.yaw_long), zs, LOOK_Q["left"])
        first = (
            it.rank == 0
            and it.xy[0] <= FIRST_LEFT_MAX_X
            and (left_plan is not None)
            and (left_plan[1] <= FIRST_MAX_TILT + 1e-06)
        )
        if left_plan is not None and (it.xy[0] <= LEFT_MAX_X or not right_ok or first):
            it.arm = "left"
            continue
        if not right_ok:
            return None
        it.arm = "right"
        zz = LIFT_Z, grasp_z(it.height)
        if len(used) < N_STAGE_SPOTS:
            for s in spots:
                if s in used or any(np.linalg.norm(np.asarray(s) - np.asarray(u)) < STAGE_SEP for u in used):
                    continue
                key = s, zz
                if key not in spot_ok:
                    spot_ok[key] = all(
                        grasp_plan(a, s, close_yaws(0.0) + close_yaws(math.pi / 2), list(zz), LOOK_Q[a]) is not None
                        for a in ("left", "right")
                    )
                if spot_ok[key]:
                    used.append(s)
                    break
        if not used:
            return None
        it.stage = used[n_staged % len(used)]
        n_staged += 1
    for it in items:
        if it.stage is not None:
            it.spots = list(used)
    return items


@register
class ImitateSequence(Program):
    name = "imitate_sequence"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if (
            "order" in s
            and "basket" in s
            and ("observe" in s or "watch" in s)
            and ("same order" in s or "remember" in s)
        ):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        cam = C.head()
        robot = RM.mask(cam, arms["left"].q, arms["right"].q, 1.0, 1.0, dilate=3)
        img0 = o.images.get("cam_head")
        for _ in range(5):
            if img0 is not None and float(np.std(img0)) > 5.0:
                break
            yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            o = ctx.obs
            img0 = o.images.get("cam_head")
        scene = Scene.first(img0, cam, robot, o.step)
        while True:
            yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            o = ctx.obs
            img = o.images.get("cam_head")
            if img is None:
                continue
            scene.watch.update(img, o.step)
            steps = o.step - scene.watch.step0
            if steps >= WATCH_MIN_STEPS and scene.watch.back_for(o.step) >= BACK_STILL:
                break
            if steps >= WATCH_MIN_STEPS_STILL and scene.watch.still_for(o.step) >= LONG_STILL:
                break
            if steps >= WATCH_MAX_STEPS:
                break
        img = ctx.obs.images.get("cam_head")
        yield from run_arms(
            arms, {a: arms[a].joints_to(LOOK_Q[a], grip=1.0, force=True, joint_step=FREE_JOINT_STEP) for a in K.ARMS}
        )
        yield from run_arms(arms, {a: arms[a].hold(2) for a in K.ARMS})
        img_look = ctx.obs.images.get("cam_head")
        robot_look = RM.mask(cam, arms["left"].q, arms["right"].q, 1.0, 1.0, dilate=3)
        an = analyse(scene, img, cam, robot, img_look, robot_look)
        if not (not an.holes or not an.assign or an.our_box is None):
            items = plan_items(an)
            if items is not None:
                sh = Shared(items, np.asarray(an.our_box.xy, float), exclude=an.exclude_near, mid=Mid())
                sh.boxes = an.boxes_near
                sh.basket_ref = (
                    TT.lab_image(img_look),
                    RM.mask(cam, arms["left"].q, arms["right"].q, 1.0, 1.0, dilate=6),
                )
                if an.boxes_near:
                    b = min(an.boxes_near, key=lambda b: float(np.linalg.norm(np.asarray(b.xy) - sh.basket_xy)))
                    if float(np.linalg.norm(np.asarray(b.xy) - sh.basket_xy)) < 0.08:
                        sh.basket_rect = basket_rect(b, cam)
                link_arms(arms)
                yield from run_arms(arms, {"left": left_task(arms["left"], sh), "right": right_task(arms["right"], sh)})
        yield from run_arms(
            arms,
            {a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True, joint_step=FREE_JOINT_STEP) for a in K.ARMS},
        )
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
