from __future__ import annotations

import math
import re
import time
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ...core import actions
from ...perception import cameras as C
from . import number_tiles as N
from ...perception import robot_mask as RM
from ...planning import kinematics as K
from ...planning import motion
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

STEP_LIMIT = 1050
LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
FREE_JOINT_STEP = 0.11
LOOK_SETTLE = 4
PER_PIECE = 110
HOME_RESERVE = 60
Z_CARRY = N.TABLE_Z + 0.065
TIP_CLEAR = 0.003
RELEASE_GAP = 0.008
CARRY_MIN_Z = 0.05
TILTS = 0.0, math.radians(15), math.radians(30)
WEAK_CONTACT = 0.002
TILT_WEAK = math.radians(15)
HANDOFF_SPOTS = [(x, y) for y in (-0.17, -0.12, -0.22, -0.08) for x in (0.0, -0.04, 0.04, -0.08, 0.08)]
HANDOFF_SPOTS_WIDE = HANDOFF_SPOTS + [(x, y) for y in (-0.17, -0.12, -0.22) for x in (-0.12, 0.12, -0.16, 0.16)]
HANDOFF_SEARCH_S = 8.0
FREE_R = 0.065
READY_DX, READY_DZ = 0.2, 0.1
MOVE_STEP = 0.09
FREE_V, FREE_A = 0.13, 0.03


def parse_instruction(text: str):
    s = text.lower()
    if "number" not in s and "digit" not in s:
        return None
    m = re.search(r"\b(largest|biggest|greatest|smallest|least)\b(?:\s+possible)?\s+(?:number|integer|value)", s)
    if not m:
        return None
    return "largest" if m.group(1) in ("largest", "biggest", "greatest") else "smallest"


def target_order(digits, mode: str = "largest") -> list[int]:
    ds = sorted(int(d) for d in digits)
    if mode == "largest":
        return ds[::-1]
    if ds and ds[0] == 0 and (len(ds) > 1):
        k = next((i for i, d in enumerate(ds) if d != 0), 0)
        ds = [ds[k]] + ds[:k] + ds[k + 1 :]
    return ds


JOINT_TOP_K = 12
CLASH_FRAC = 0.2


def footprints_clash(a, b) -> bool:
    if getattr(a, "xy", None) is None or getattr(b, "xy", None) is None:
        return False
    if float(np.linalg.norm(np.asarray(a.xy) - np.asarray(b.xy))) > 0.06:
        return False
    if getattr(a, "digit", None) is None or getattr(b, "digit", None) is None:
        return False
    res, n = 0.001, 140
    c = (np.asarray(a.xy) + np.asarray(b.xy)) / 2

    def raster(t):
        m = np.zeros((n, n), np.uint8)
        pts = N.boundary(int(t.digit)) @ _rot2(float(t.yaw)).T + np.asarray(t.xy) - c
        px = np.round(pts / res + n / 2).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [px], 1)
        return m.astype(bool)

    ma, mb = raster(a), raster(b)
    return np.count_nonzero(ma & mb) > CLASH_FRAC * min(np.count_nonzero(ma), np.count_nonzero(mb))


COLOUR_SPREAD = 12.0
COLOUR_ERODE = 3


def top_colour(tv, t):
    if getattr(t, "digit", None) is None or getattr(t, "xy", None) is None:
        return None
    top = N.Tile(
        digit=int(t.digit),
        xy=np.asarray(t.xy, dtype=float) + N._sweep(tv, t.xy),
        yaw=float(t.yaw),
        score=0.0,
        margin=0.0,
        scores={},
        z_support=tv.z_support,
    )
    m = N.tile_pixels(tv, top, erode_px=COLOUR_ERODE)
    if m.sum() < 30:
        return None
    return [round(float(v), 1) for v in np.median(tv.lab[m], axis=0)]


def colour_spread(cols) -> float:
    cs = np.array([c for c in cols if c is not None], dtype=float)
    if len(cs) < 2:
        return 0.0
    d = (cs - np.median(cs, axis=0)) * np.array([0.5, 1.0, 1.0])
    return float(np.max(np.linalg.norm(d, axis=1)))


TILE_SPAWN = (-0.4, 0.4), (-0.25, -0.05)
SPAWN_TOL = 0.02


def in_spawn(t) -> bool:
    xy = getattr(t, "xy", None)
    if xy is None:
        return True
    (x0, x1), (y0, y1) = TILE_SPAWN
    return x0 - SPAWN_TOL <= float(xy[0]) <= x1 + SPAWN_TOL and y0 - SPAWN_TOL <= float(xy[1]) <= y1 + SPAWN_TOL


def read_digits(tiles, n: int, use_colour: bool = True):
    pool = [t for t in tiles if getattr(t, "scores", None) and in_spawn(t)]
    whole = [t for t in pool if not getattr(t, "clipped", False)]
    if len(whole) >= n:
        pool = whole
    cands = sorted(pool, key=lambda t: -t.score)[:JOINT_TOP_K]
    if n <= 0 or len(cands) < n:
        return None
    sc = np.array([[float(t.scores.get(d, 0.0)) for d in range(10)] for t in cands])
    col = [t.extra.get("colour") if use_colour and hasattr(t, "extra") else None for t in cands]
    clash = [[i != j and footprints_clash(cands[i], cands[j]) for j in range(len(cands))] for i in range(len(cands))]
    best = None
    for sub in combinations(range(len(cands)), n):
        if any(clash[i][j] for i in sub for j in sub):
            continue
        if use_colour and (0 not in sub or colour_spread([col[i] for i in sub]) > COLOUR_SPREAD):
            continue
        m = sc[list(sub)]
        r, c = linear_sum_assignment(-m)
        tot = float(m[r, c].sum())
        if best is None or tot > best[0]:
            best = tot, [(cands[sub[i]], int(c[k])) for k, i in enumerate(r)]
    if best is None:
        return read_digits(tiles, n, use_colour=False) if use_colour else None
    return best[1]


def assign(tiles, pads, mode: str = "largest"):
    if tiles and all(getattr(t, "scores", None) for t in tiles):
        sel = read_digits(tiles, len(pads))
        if sel is None:
            return None
        for t, d in sel:
            t.digit = d
        tiles = [t for t, _ in sel]
    best = {}
    for t in tiles:
        if t.digit is not None and (t.digit not in best or t.score > best[t.digit].score):
            best[t.digit] = t
    if len(best) < len(pads):
        return None
    chosen = sorted(best.values(), key=lambda t: -t.score)[: len(pads)]
    order = target_order([t.digit for t in chosen], mode)
    by_digit = {t.digit: t for t in chosen}
    pad_idx = sorted(range(len(pads)), key=lambda i: float(pads[i].xy[0]))
    return [(by_digit[d], pad_idx[k]) for k, d in enumerate(order)]


def robot_mask(o, cam):
    st = o.state
    return RM.mask(
        cam,
        st["left_arm_joint_state"],
        st["right_arm_joint_state"],
        float(st["left_ee_joint_state"][0]),
        float(st["right_ee_joint_state"][0]),
    )


PAD_SMOOTH, PAD_SMOOTH_MIN = 0.6, 0.4
TILE_CUES = "multi"
PAD_SOFT, PAD_SOFT_MIN = 30.0, 0.25


class Scene:
    def __init__(self, tiles, pads):
        self.tiles, self.pads, self.discs = list(tiles), list(pads), []
        self.head_tv = None


PAD_CUES = (
    (dict(min_score=0.4), 0.4),
    (dict(min_score=PAD_SOFT_MIN, soft=PAD_SOFT), PAD_SOFT_MIN),
    (dict(min_score=PAD_SMOOTH_MIN, smooth=PAD_SMOOTH), PAD_SMOOTH_MIN),
)
EDGE_ROW_MIN = 1.2
EDGE_N_RATIO = 2.5


def find_pads(img, cam, robot):
    pads, by_n = N.pad_row_edges(img, cam, robot, radial=True)
    if pads:
        v = sorted(by_n.values(), reverse=True)
        if v[0] >= EDGE_ROW_MIN and (len(v) < 2 or v[0] >= EDGE_N_RATIO * max(v[1], 0.1)):
            return pads, [], {"cue": "edges", "by_n": by_n}
    fits, cands_all = [by_n], []
    for kw, row_min in PAD_CUES:
        info = {}
        cands = N.detect_pads(img, cam, robot, info=info, max_n=12, **kw)
        cands_all.append(cands)
        pads, by_n = N.pad_row_model(info["score_map"])
        fits.append(by_n)
        if pads and max(by_n.values()) >= row_min:
            return pads, cands, {"cue": len(fits) - 2, "by_n": fits}
    for cands in cands_all:
        pads = N.pad_row(cands)
        if pads:
            return pads, cands, {"cue": None, "by_n": fits}
    return [], cands_all[0], {"cue": None, "by_n": fits}


def perceive_head(ctx) -> Scene:
    o = ctx.obs
    cam = C.head()
    img = o.images["cam_head"]
    robot = robot_mask(o, cam)
    tiles = N.detect_tiles(img, cam, robot, cues=TILE_CUES)
    pads, _, _ = find_pads(img, cam, robot)
    sc = Scene(tiles, pads)
    sc.head_tv = N.top_view(img, cam, robot)
    return sc


PAD_EXCLUDE_R = N.PAD_R + 0.01
WRIST_LOOK_H = 0.3
WRIST_LOOK_XY = {"left": (-0.22, -0.15), "right": (0.22, -0.15)}
FUSE_R = 0.012


def look_target(arm: str, q_cur, xy, height: float, other_q=None):
    x, y = xy
    for tilt in (0.0, math.radians(10), math.radians(20)):
        best = None
        for yaw in (0.0, math.radians(20), -math.radians(20), math.radians(45), -math.radians(45)):
            for dz in (0.0, -0.03):
                t_cam = np.eye(4)
                t_cam[:3, :3] = K.rot_z(yaw) @ K.rot_x(tilt)
                t_cam[:3, 3] = [x, y, K.TABLE_Z + height + dz]
                r = K.solve_near(arm, t_cam @ np.linalg.inv(C.WRIST_MOUNT), q_cur, 0.0)
                if r.ok and (other_q is None or clear_of(arm, r.q, other_q)):
                    cost = (
                        float(np.sum(K.JOINT_COST_W * np.abs(r.q - np.asarray(q_cur))))
                        + 3.0 * abs(dz) / 0.03
                        + 20.0 * abs(yaw)
                    )
                    if best is None or cost < best[0]:
                        best = cost, r.q
        if best is not None:
            return best[1]


def fuse_views(head_tiles, wrist_dets):
    dets = sorted(list(head_tiles) + list(wrist_dets), key=lambda d: -d.score)
    clusters = []
    for d in dets:
        c = next((c for c in clusters if np.linalg.norm(np.asarray(d.xy) - np.asarray(c[0].xy)) < FUSE_R), None)
        if c is None:
            clusters.append([d])
        else:
            c.append(d)
    out = []
    for c in clusters:
        whole = [d for d in c if not d.clipped] or c
        per_view = {}
        for d in whole:
            if d.src not in per_view or d.score > per_view[d.src].score:
                per_view[d.src] = d
        views = list(per_view.values())
        fused = {k: round(float(np.mean([float(v.scores.get(k, 0.0)) for v in views])), 3) for k in range(10)}
        best = max(fused, key=fused.get)
        wr = [d for d in whole if d.src != "head"]
        ref = max(wr, key=lambda d: d.score) if wr else whole[0]
        t = c[0]
        t.extra["views"] = [(v.src, int(v.digit), round(float(v.score), 3), bool(v.clipped)) for v in views]
        t.scores, t.digit, t.score = fused, int(best), float(fused[best])
        t.margin = float(fused[best] - max(v for k, v in fused.items() if k != best))
        t.xy, t.yaw, t.clipped = np.asarray(ref.xy, dtype=float), float(ref.yaw), all(d.clipped for d in c)
        t.src = "+".join(sorted(per_view))
        out.append(t)
    return out


def read_wrists(ctx, sc) -> None:
    o = ctx.obs
    views = []
    for arm in K.ARMS:
        cam = C.wrist(o.state[f"{arm}_ee_pose"])
        img = o.images[f"cam_{arm}_wrist"]
        dets = N.detect_tiles(img, cam, robot_mask(o, cam), cues=TILE_CUES)
        for d in dets:
            d.src = f"wrist_{arm}"
        views += dets
    for t in sc.tiles:
        t.src = "head"
    tiles = fuse_views(sc.tiles, views)
    sc.tiles = [t for t in tiles if all(np.linalg.norm(np.asarray(t.xy) - p.xy) >= PAD_EXCLUDE_R for p in sc.pads)]
    if sc.head_tv is not None:
        for t in sc.tiles:
            t.extra["colour"] = top_colour(sc.head_tv, t)


class Piece:
    kind = "tile"

    def __init__(self, xy, yaw, digit, z: float = N.TABLE_Z):
        self.xy, self.yaw = np.asarray(xy, dtype=float).copy(), float(yaw)
        self.digit = int(digit)
        self.z = float(z)

    def grasps(self):
        return [
            (
                phi,
                pt,
                ch,
                gap,
                N.lever_arm(self.digit, phi, pt),
                N.contacts_inside(self.digit, phi, pt),
                TILT_WEAK if N.contact_offset(self.digit, phi, pt) > WEAK_CONTACT else TILTS[-1],
            )
            for phi, pt, ch, _, gap in N.tile_grasps(self.digit)
        ]

    @property
    def height(self):
        return N.TILE_T

    def label(self):
        return str(self.digit)


class Obstacles:
    X0, Y0, W, H, RES = -0.52, -0.36, 1040, 440, 0.001

    def __init__(self, sc, exclude_xy=None, exclude_r: float = 0.02):
        m = np.zeros((self.H, self.W), np.uint8)
        for t in sc.tiles:
            if t.digit is None:
                continue
            if exclude_xy is not None and float(np.linalg.norm(np.asarray(t.xy) - np.asarray(exclude_xy))) < exclude_r:
                continue
            cv2.fillPoly(m, [self._px(N.boundary(int(t.digit)) @ _rot2(t.yaw).T + np.asarray(t.xy))], 1)
        self.m = m

    def _px(self, pts):
        pts = np.asarray(pts, dtype=float)
        return (
            np.stack([(pts[:, 0] - self.X0) / self.RES, (pts[:, 1] - self.Y0) / self.RES], axis=1)
            .round()
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )

    def finger_overlap(self, g) -> float:
        u = np.array([math.cos(g["heading"]), math.sin(g["heading"])])
        v = np.array([-u[1], u[0]])
        c = np.asarray(g["tcp"][:2], dtype=float)
        a0, a1 = g["gap"] / 2, g["gap"] / 2 + FINGER_T
        tot = 0
        for sg in (-1.0, 1.0):
            corners = np.array(
                [
                    c + sg * a0 * u - FINGER_W2 * v,
                    c + sg * a1 * u - FINGER_W2 * v,
                    c + sg * a1 * u + FINGER_W2 * v,
                    c + sg * a0 * u + FINGER_W2 * v,
                ]
            )
            f = np.zeros_like(self.m)
            cv2.fillPoly(f, [self._px(corners)], 1)
            tot += int(np.count_nonzero(f & self.m))
        return tot * self.RES * self.RES


MEASURE_HALF = 0.08


def measure_tile(arm: Arm, piece: Piece, z_support: float):
    o = arm.ctx.obs
    cam = C.wrist(o.state[f"{arm.name}_ee_pose"])
    x, y = float(piece.xy[0]), float(piece.xy[1])
    region = (x - MEASURE_HALF, x + MEASURE_HALF), (y - MEASURE_HALF, y + MEASURE_HALF)
    img = o.images[f"cam_{arm.name}_wrist"]
    ts = N.detect_tiles(img, cam, robot_mask(o, cam), region=region, z_support=z_support, cues=TILE_CUES)
    best = None
    for t in ts:
        d = float(np.linalg.norm(np.asarray(t.xy) - piece.xy))
        if d > REFINE_MAX_SHIFT:
            continue
        key = float(t.scores.get(piece.digit, 0.0)) - 2.0 * d
        if best is None or key > best[0]:
            best = key, t
    if best is None:
        return None
    t = best[1]
    same = int(t.digit) == int(piece.digit)
    return {
        "score": float(t.score) if same else 0.0,
        "xy": np.asarray(t.xy, dtype=float),
        "yaw": float(t.yaw),
        "clipped": bool(t.clipped),
        "scores": dict(t.scores),
        "digit": int(t.digit),
    }


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _ik(arm, T, seeds):
    for s in seeds:
        r = K.solve(arm, T, s, K.TCP_OFFSET, iters=80)
        if r.ok:
            return r.q


def plan_pick(arm: str, piece: Piece, q_cur, z_support=N.TABLE_Z, all_tilts: bool = False, any_tilt: bool = False):
    out = []
    seed0 = G.base_seed(arm, piece.xy)
    for phi, pt, chord, gap, lever, tilt_ok, max_tilt in piece.grasps():
        p_xy = piece.xy + _rot2(piece.yaw) @ pt
        if math.hypot(p_xy[0] - K.BASE_POS[arm][0], p_xy[1] - K.BASE_POS[arm][1]) > G.MAX_REACH_XY:
            continue
        head0 = piece.yaw + phi
        for tilt in TILTS:
            if tilt and (not (tilt_ok or any_tilt)):
                break
            if tilt > max_tilt + 1e-09 and (not any_tilt):
                break
            found = False
            for k in (0, 1):
                heading = head0 + k * math.pi
                for sgn in (1.0,) if tilt == 0 else (1.0, -1.0):
                    rot = G.grasp_rotation(heading, sgn * tilt)
                    to_base = np.asarray(K.BASE_POS[arm][:2]) - p_xy
                    if tilt and float(np.dot(-rot[:2, 0], to_base / np.linalg.norm(to_base))) < 0.2:
                        continue
                    z = z_support + G.FINGER_DEPTH * math.cos(tilt) + G.FINGER_HALF_WIDTH * math.sin(tilt) + TIP_CLEAR
                    T = tcp_pose(rot, [p_xy[0], p_xy[1], z])
                    seeds = [q_cur, seed0, seed0 + np.array([0, 0, 0, 0, 0, heading])]
                    q = _ik(arm, T, seeds)
                    if q is None:
                        continue
                    qh = _ik(arm, tcp_pose(rot, [p_xy[0], p_xy[1], Z_CARRY]), [q])
                    if qh is None:
                        continue
                    cost = float(np.sum(K.JOINT_COST_W * np.abs(qh - np.asarray(q_cur)))) + 3.0 * tilt
                    out.append(
                        {
                            "arm": arm,
                            "phi": phi,
                            "pt": pt,
                            "chord": chord,
                            "gap": gap,
                            "lever": lever,
                            "weak": max_tilt < TILTS[-1] - 1e-09,
                            "heading": heading,
                            "tilt": sgn * tilt,
                            "rot": rot,
                            "tcp": np.array([p_xy[0], p_xy[1], z]),
                            "q": q,
                            "q_hover": qh,
                            "cost": cost,
                            "dz": z - z_support,
                        }
                    )
                    found = True
            if found and (not all_tilts):
                break
    out.sort(key=lambda g: g["cost"])
    return out


def plan_place(arm: str, grasp: dict, piece_yaw: float, goal_xy, z_support, q_cur, final_yaw: float = 0.0):
    turn = final_yaw - piece_yaw
    heading = grasp["heading"] + turn
    p_xy = np.asarray(goal_xy, dtype=float) + _rot2(final_yaw) @ grasp["pt"]
    if math.hypot(p_xy[0] - K.BASE_POS[arm][0], p_xy[1] - K.BASE_POS[arm][1]) > G.MAX_REACH_XY:
        return None
    seed0 = G.base_seed(arm, p_xy)
    z = z_support + RELEASE_GAP + grasp["dz"]
    rot = G.grasp_rotation(heading, grasp["tilt"])
    seeds = [q_cur, seed0, seed0 + np.array([0, 0, 0, 0, 0, heading])]
    q = _ik(arm, tcp_pose(rot, [p_xy[0], p_xy[1], z]), seeds)
    if q is None:
        return None
    qa = _ik(arm, tcp_pose(rot, [p_xy[0], p_xy[1], Z_CARRY]), [q])
    if qa is None:
        return None
    cost = float(np.sum(K.JOINT_COST_W * np.abs(qa - np.asarray(q_cur))))
    return {"rot": rot, "tcp": np.array([p_xy[0], p_xy[1], z]), "q": q, "q_above": qa, "cost": cost}


FINAL_YAWS = 0.0, math.radians(15), -math.radians(15), math.radians(30), -math.radians(30)
REFINE_ITERS = 2
REFINE_SETTLE = 2
REFINE_MAX_SHIFT = 0.04
REFINE_MIN_SCORE = 0.8
LIFT_SETTLE = 2
LIFT_TILT_STEP = 0.006
GRASP_TOL = 0.005
EMPTY_W = 0.003
LOST_FRAC = 0.5
CARRY_TILT_STEP = 0.05
CARRY_LIN_STEP, CARRY_LIN_STEP_TILT = 0.02, 0.012


def move_steps(q0, q1) -> int:
    return math.ceil(float(np.max(np.abs(np.asarray(q1) - np.asarray(q0)))) * math.pi / 2 / MOVE_STEP)


TILT_COST = 20.0
LEVER_COST = 500.0
FINGER_T, FINGER_W2 = 0.012, 0.014
OVERLAP_COST = 200000.0
FINAL_YAW_COST = 30.0


def choose(
    arm_name: str,
    piece: Piece,
    q_cur,
    z_from,
    goal_xy,
    z_to,
    other_q=None,
    n: int = 24,
    any_tilt: bool = False,
    obstacles: Obstacles | None = None,
    final_yaws=None,
):
    best = None
    for g in plan_pick(arm_name, piece, q_cur, z_from, all_tilts=True, any_tilt=any_tilt)[:n]:
        if other_q is not None and (
            not (clear_of(arm_name, g["q"], other_q) and clear_of(arm_name, g["q_hover"], other_q))
        ):
            continue
        g["overlap"] = obstacles.finger_overlap(g) if obstacles is not None else 0.0
        t_pick = (
            len(free_plan(q_cur, g["q_hover"]))
            + TILT_COST * abs(g["tilt"])
            + LEVER_COST * g["lever"]
            + OVERLAP_COST * g["overlap"]
        )
        if best is not None and t_pick >= best[0]:
            continue
        for fy in FINAL_YAWS if final_yaws is None else final_yaws:
            pl = plan_place(arm_name, g, piece.yaw, goal_xy, z_to, g["q"], final_yaw=fy)
            if pl is None:
                continue
            if other_q is not None and (
                not (clear_of(arm_name, pl["q"], other_q) and clear_of(arm_name, pl["q_above"], other_q))
            ):
                continue
            t = t_pick + move_steps(g["q_hover"], pl["q_above"]) + FINAL_YAW_COST * abs(fy)
            if best is None or t < best[0]:
                pl["final_yaw"], pl["t"] = fy, t
                best = t, g, pl
    return (best[1], best[2]) if best is not None else (None, None)


REREAD_MIN, REREAD_MARGIN = 0.9, 0.05


def pick(
    arm: Arm,
    piece: Piece,
    z_from: float,
    goal_xy,
    z_to: float,
    info: dict,
    other_q=None,
    retry_before: int = 0,
    any_tilt: bool = False,
    obstacles: Obstacles | None = None,
    final_yaws=None,
):
    def replan():
        g, pl = choose(
            arm.name,
            piece,
            arm.q,
            z_from,
            goal_xy,
            z_to,
            other_q,
            any_tilt=any_tilt,
            obstacles=obstacles,
            final_yaws=final_yaws,
        )
        if g is None and final_yaws is not None:
            g, pl = choose(
                arm.name, piece, arm.q, z_from, goal_xy, z_to, other_q, any_tilt=any_tilt, obstacles=obstacles
            )
        return g, pl

    g, pl = replan()
    if g is None:
        info["fail"] = "no_grasp"
        return None
    for attempt in range(2):
        open_g = grip_for_gap(g["gap"])
        T_hover = tcp_pose(g["rot"], [g["tcp"][0], g["tcp"][1], Z_CARRY])
        yield from free_move_to(arm, T_hover, min_z=N.TABLE_Z + 0.04, grip=open_g, seeds=[g["q_hover"]])
        if arm.status != "ok":
            info["fail"] = f"hover_{arm.status}"
            return None
        if piece.kind == "tile":
            for it in range(REFINE_ITERS):
                yield from arm.hold(REFINE_SETTLE)
                r = measure_tile(arm, piece, z_from)
                shift = None if r is None else float(np.linalg.norm(r["xy"] - piece.xy))
                if (
                    r is not None
                    and r["score"] == 0.0
                    and (piece.kind == "tile")
                    and (not r["clipped"])
                    and (shift <= REFINE_MAX_SHIFT)
                ):
                    sc_ = r["scores"] or {}
                    alt = int(r["digit"])
                    if (
                        sc_.get(alt, 0.0) >= REREAD_MIN
                        and sc_.get(alt, 0.0) >= sc_.get(piece.digit, 0.0) + REREAD_MARGIN
                    ):
                        piece.digit = alt
                        r["score"] = float(sc_[alt])
                if r is None or r["score"] < REFINE_MIN_SCORE or shift > REFINE_MAX_SHIFT:
                    break
                dyaw = abs(math.remainder(r["yaw"] - piece.yaw, 2 * math.pi))
                piece.xy, piece.yaw = np.asarray(r["xy"], dtype=float), float(r["yaw"])
                if shift < 0.002 and dyaw < math.radians(3):
                    break
                g2, pl2 = replan()
                if g2 is None:
                    break
                g, pl = g2, pl2
                open_g = grip_for_gap(g["gap"])
                T_hover = tcp_pose(g["rot"], [g["tcp"][0], g["tcp"][1], Z_CARRY])
                yield from free_move_to(arm, T_hover, min_z=N.TABLE_Z + 0.04, grip=open_g, seeds=[g["q_hover"]])
                if arm.status != "ok":
                    info["fail"] = f"rehover_{arm.status}"
                    return None
                if not r["clipped"] and shift < 0.01 and (dyaw < math.radians(10)):
                    break
        yield from arm.set_grip(open_g, 2)
        yield from arm.line_to(tcp_pose(g["rot"], g["tcp"]), lin_step=0.009)
        if arm.status != "ok":
            info["fail"] = f"descend_{arm.status}"
            return None
        yield from arm.hold(1)
        yield from arm.set_grip(0.0, 8)
        yield from arm.hold(2)
        g_obs = float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])
        w = held_width(g_obs)
        ok = abs(w - g["chord"]) <= GRASP_TOL
        info["width"] = w
        if ok or attempt == 1 or arm.ctx.obs.step > retry_before:
            tilted = piece.kind == "tile" and abs(g["tilt"]) > 1e-06
            yield from arm.line_to(T_hover, lin_step=LIFT_TILT_STEP if tilted else 0.012)
            if w < EMPTY_W:
                info["fail"] = "empty"
                return None
            yield from arm.hold(LIFT_SETTLE)
            w_up = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]))
            if w_up >= EMPTY_W:
                return g, pl
            if attempt == 1 or arm.ctx.obs.step > retry_before:
                info["fail"] = "dropped"
                return None
            yield from arm.set_grip(open_g, 4)
        else:
            yield from arm.set_grip(open_g, 4)
            yield from arm.line_to(T_hover, lin_step=0.012)
        g2, pl2 = replan()
        if g2 is not None:
            g, pl = g2, pl2


def place(arm: Arm, pl: dict, g: dict, info: dict, kind: str | None = None):
    open_g = grip_for_gap(g["gap"])
    T_above = tcp_pose(pl["rot"], [pl["tcp"][0], pl["tcp"][1], Z_CARRY])
    step = CARRY_TILT_STEP if kind == "tile" and abs(g["tilt"]) > 1e-06 and g.get("weak") else MOVE_STEP
    w_lost = max(EMPTY_W, LOST_FRAC * held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])))

    def carry():
        lin = CARRY_LIN_STEP_TILT if step == CARRY_TILT_STEP else CARRY_LIN_STEP
        plan = motion.line(arm.name, arm.q, T_above, K.TCP_OFFSET, lin_step=lin, joint_step=step)
        if plan.ok:
            ok = yield from arm.follow(plan)
            arm.status = "ok" if ok else "blocked"
        else:
            yield from arm.move_to(T_above, min_z=N.TABLE_Z + CARRY_MIN_Z, seeds=[pl["q_above"]], joint_step=step)

    mv = carry()
    for _ in mv:
        yield
        w = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]))
        if w < w_lost:
            mv.close()
            info["lost_xy"] = np.round(arm.tcp()[:2], 4).tolist()
            info["fail"] = "lost"
            return False
    if arm.status != "ok":
        info["fail"] = f"carry_{arm.status}"
        return False
    yield from arm.line_to(tcp_pose(pl["rot"], pl["tcp"]), lin_step=0.007)
    yield from arm.hold(1)
    w = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]))
    if w > 0.0:
        yield from arm.set_grip(grip_for_gap(w), 2)
    yield from arm.set_grip(open_g, 4)
    yield from arm.line_to(T_above, lin_step=0.012)
    return True


PLACE_FIX_DIST = 0.012
PLACE_FIX_YAW = math.radians(36)
PLACE_FIX_YAW_SCORE = 0.88


def check_place(arm: Arm, piece: Piece, pad_xy, other_q=None, obstacles: Obstacles | None = None, fix_before: int = 0):
    if piece.kind != "tile":
        return
    yield from arm.hold(REFINE_SETTLE)
    r = measure_tile(arm, Piece(pad_xy, 0.0, piece.digit), N.PAD_TOP)
    if r is None or r["score"] < REFINE_MIN_SCORE:
        return
    err = float(np.linalg.norm(np.asarray(r["xy"], dtype=float) - np.asarray(pad_xy, dtype=float)))
    yaw = abs(math.remainder(float(r["yaw"]), 2 * math.pi))
    sc = r["scores"] or {}
    top = int(max(sc, key=sc.get)) if sc else int(piece.digit)
    bad = err > PLACE_FIX_DIST or (yaw > PLACE_FIX_YAW and r["score"] >= PLACE_FIX_YAW_SCORE)
    if not bad:
        return
    if r["clipped"] or top != int(piece.digit) or err > N.PAD_R or (arm.ctx.obs.step > fix_before):
        return
    fix = {}
    pc = Piece(r["xy"], float(r["yaw"]), piece.digit)
    gp = yield from pick(
        arm, pc, N.PAD_TOP, pad_xy, N.PAD_TOP, fix, other_q=other_q, obstacles=obstacles, retry_before=fix_before
    )
    if gp is not None:
        yield from place(arm, gp[1], gp[0], fix, kind="tile")


PARK_J1 = {"left": 0.6, "right": -0.6}
MIN_CLEAR = 0.035


def free_plan(q0, q1, v: float = FREE_V, a: float = FREE_A) -> motion.Plan:
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    d = float(np.max(np.abs(q1 - q0)))
    if d < 1e-09:
        return motion.Plan([q1.copy()])
    if d <= v * v / a:
        t_acc = math.sqrt(d / a)
        t_tot, v_pk = 2 * t_acc, a * t_acc
    else:
        t_acc = v / a
        t_tot, v_pk = d / v + t_acc, v
    n = max(1, math.ceil(t_tot))

    def dist(t):
        if t <= t_acc:
            return 0.5 * a * t * t
        if t >= t_tot - t_acc:
            r = max(0.0, t_tot - t)
            return d - 0.5 * a * r * r
        return 0.5 * a * t_acc * t_acc + v_pk * (t - t_acc)

    out = [q0 + (q1 - q0) * min(1.0, dist(t_tot * k / n) / d) for k in range(1, n + 1)]
    out[-1] = q1.copy()
    return motion.Plan(out)


def free_joints_to(arm: Arm, q_goal, grip=None):
    ok = yield from arm.follow(free_plan(arm.q, q_goal), grip)
    arm.status = "ok" if ok else "blocked"


def free_move_to(arm: Arm, T_goal, *, min_z: float, grip=None, seeds=()):
    r = K.solve_near(arm.name, T_goal, arm.q, K.TCP_OFFSET, extra_seeds=seeds, n_random=3)
    if r.ok:
        plan = free_plan(arm.q, r.q)
        zs = [K.point_of(K.fk(arm.name, q), K.TCP_OFFSET)[2] for q in plan.q[:: max(1, len(plan.q) // 8)]]
        if min(zs + [10.0]) >= min_z:
            ok = yield from arm.follow(plan, grip)
            arm.status = "ok" if ok else "blocked"
            return
    yield from arm.move_to(T_goal, min_z=min_z, grip=grip, seeds=seeds, joint_step=MOVE_STEP)


def park_q(arm: str) -> np.ndarray:
    return np.array([PARK_J1[arm], 0.0, 0.0, 0.0, 0.0, 0.0])


def clear_of(arm: str, q, other_q) -> bool:
    from ...planning import collision as CO

    d = CO.arm_distance(q, other_q, 1.0, 1.0) if arm == "left" else CO.arm_distance(other_q, q, 1.0, 1.0)
    return d >= MIN_CLEAR


def go_park(arm: Arm):
    yield from free_joints_to(arm, park_q(arm.name), grip=1.0)


def wait_for_release(arm: Arm, first: Arm, info: dict, max_steps: int = 400):
    n, moved = 0, False
    while not (info.get("released") or info.get("aborted")) and n < max_steps:
        if first.waiting and (not moved):
            moved = True
            arm.waiting = False
            yield from go_park(arm)
            continue
        n += 1
        arm.waiting = True
        yield
    arm.waiting = False


def ready_move(arm: Arm, q_goal, first: Arm, max_wait: int = 24):
    for q in free_plan(arm.q, q_goal).q:
        q = np.asarray(q, dtype=float)
        w = 0
        while not arm._may_step(q):
            if first.waiting or w >= max_wait:
                break
            w += 1
            yield
        if first.waiting or w >= max_wait:
            return False
        arm.q = q
        yield
    return True


def go_home(arm: Arm):
    p = arm.tcp()
    if p[2] < Z_CARRY - 0.01:
        yield from arm.line_to(tcp_pose(arm.T[:3, :3], [p[0], p[1], Z_CARRY]), lin_step=0.012)
    yield from free_joints_to(arm, arm.q_home, grip=1.0)


def free_spots(sc, avoid=(), spots=HANDOFF_SPOTS):
    pts = [t.xy for t in sc.tiles] + [d.xy for d in sc.discs] + [p.xy for p in sc.pads] + list(avoid)
    return [
        np.asarray(s, dtype=float)
        for s in spots
        if all(np.linalg.norm(np.asarray(s, dtype=float) - p) >= FREE_R for p in pts)
    ]


SPOT_TRUST = 0.8


class _Blockers:
    def __init__(self, sc, keep):
        self.tiles = [t for t in sc.tiles if keep(t)]
        self.discs, self.pads = sc.discs, sc.pads


def handoff_spot_levels(sc):
    planned = lambda t: bool(getattr(t, "extra", {}).get("planned"))
    out, seen = [], []
    for level, blk in (
        (0, sc),
        (1, _Blockers(sc, lambda t: planned(t) or float(t.score) >= SPOT_TRUST)),
        (2, _Blockers(sc, planned)),
    ):
        new = [x for x in free_spots(blk, spots=HANDOFF_SPOTS_WIDE) if all(np.linalg.norm(x - y) > 1e-06 for y in seen)]
        seen += new
        out.append((level, new))
    return out


MAX_RETRIES = 2
RETRY_MIN_STEPS = 300
RETRY_MIN_SCORE = 0.8


def relocate(ctx, sc, failed):
    o = ctx.obs
    cam = C.head()
    img = o.images["cam_head"]
    dets = N.detect_tiles(img, cam, robot_mask(o, cam), cues=TILE_CUES)
    held = [p for p in sc.pads if any(np.linalg.norm(np.asarray(t.xy) - p.xy) < 0.005 for t in sc.tiles)]
    dets = [d for d in dets if all(np.linalg.norm(np.asarray(d.xy) - p.xy) >= PAD_EXCLUDE_R for p in held)]
    best = None
    for it in failed:
        _, _, t = it
        for d in dets:
            v = float(d.scores.get(int(t.digit), 0.0))
            if v < RETRY_MIN_SCORE:
                continue
            key = v - 0.5 * float(np.linalg.norm(np.asarray(d.xy) - np.asarray(t.xy)))
            if best is None or key > best[0]:
                best = key, d, it
    if best is None:
        return None
    _, d, it = best
    t = it[2]
    t.xy = np.asarray(d.xy, dtype=float).copy()
    t.yaw = float(d.yaw)
    on_pad = any(np.linalg.norm(t.xy - p.xy) < N.PAD_R for p in sc.pads)
    return Piece(t.xy, t.yaw, t.digit, z=N.PAD_TOP if on_pad else N.TABLE_Z), it[1], t, it


def retry_budget(n_after: int) -> int:
    return STEP_LIMIT - HOME_RESERVE - PER_PIECE * (n_after + 1)


def execute(arms: dict, sc, best, pad, n_after: int):
    kind, pc, g = best[1], best[2], best[3]
    info, info2 = {}, {}
    res = {"ok": False, "piece": pc}
    a1 = arms[g["arm"]]
    other = arms["right" if g["arm"] == "left" else "left"]
    rb = retry_budget(n_after)

    def direct(arm: Arm):
        gp = yield from pick(
            arm,
            pc,
            pc.z,
            pad.xy,
            N.PAD_TOP,
            info,
            other_q=park_q(other.name),
            retry_before=rb,
            obstacles=Obstacles(sc, pc.xy),
        )
        ok = False
        if gp is not None:
            ok = yield from place(arm, gp[1], gp[0], info, kind=pc.kind)
            if ok:
                res["final_yaw"] = gp[1]["final_yaw"]
                yield from check_place(
                    arm, pc, pad.xy, other_q=park_q(other.name), obstacles=Obstacles(sc, pc.xy), fix_before=rb
                )
        res["ok"] = ok
        yield from go_home(arm)

    if kind == "direct":
        yield from run_arms(arms, {a1.name: direct(a1), other.name: go_park(other)})
        return res
    mid, g2 = best[5], best[6]

    def first(arm: Arm):
        gp = yield from pick(
            arm,
            pc,
            pc.z,
            mid.xy,
            N.TABLE_Z,
            info,
            retry_before=rb - 60,
            obstacles=Obstacles(sc, pc.xy),
            final_yaws=(best[4]["final_yaw"],),
        )
        ok = False
        if gp is not None:
            ok = yield from place(arm, gp[1], gp[0], info, kind=pc.kind)
            info["mid_yaw"] = gp[1]["final_yaw"]
        info["released" if ok else "aborted"] = True
        yield from go_home(arm)

    def second(arm: Arm):
        side = -1.0 if arm.name == "left" else 1.0
        T_ready = tcp_pose(g2["rot"], [g2["tcp"][0] + side * READY_DX, g2["tcp"][1], Z_CARRY + READY_DZ])
        r = K.solve_near(arm.name, T_ready, g2["q_hover"], K.TCP_OFFSET)
        if r.ok:
            yield from ready_move(arm, r.q, a1)
        yield from wait_for_release(arm, a1, info)
        if not info.get("released"):
            yield from go_home(arm)
            return
        piece2 = Piece(mid.xy, info.get("mid_yaw", 0.0), pc.digit)
        res["piece"] = piece2
        gp = yield from pick(arm, piece2, N.TABLE_Z, pad.xy, N.PAD_TOP, info2, retry_before=rb, obstacles=Obstacles(sc))
        ok = False
        if gp is not None:
            ok = yield from place(arm, gp[1], gp[0], info2, kind=pc.kind)
            if ok:
                res["final_yaw"] = gp[1]["final_yaw"]
                yield from check_place(arm, piece2, pad.xy, obstacles=Obstacles(sc, pc.xy), fix_before=rb)
        res["ok"] = ok
        yield from go_home(arm)

    yield from run_arms(arms, {a1.name: first(a1), other.name: second(other)})
    return res


def direct_plans(sc, arms: dict, todo):
    out = []
    for item in todo:
        pc, pad = item[0], item[1]
        obst = Obstacles(sc, pc.xy)
        best = None
        for a in K.ARMS:
            b = "right" if a == "left" else "left"
            g, pl = choose(a, pc, arms[a].q, pc.z, pad.xy, N.PAD_TOP, other_q=park_q(b), obstacles=obst)
            if g is not None and (best is None or pl["t"] < best[0]):
                best = pl["t"], "direct", pc, g, pl
        if best is not None:
            out.append((best, item))
    out.sort(key=lambda p: p[0][0])
    return out


def handoff_plan(sc, arms: dict, pc, pad):
    t0 = time.monotonic()
    levels = handoff_spot_levels(sc)
    tries = [(lv, sp) for lv, spots in levels for sp in spots[:8]]
    for level, spot in tries:
        for a in K.ARMS:
            b = "right" if a == "left" else "left"
            for fy in FINAL_YAWS:
                if time.monotonic() - t0 > HANDOFF_SEARCH_S:
                    return None
                g, pl1 = choose(
                    a, pc, arms[a].q, N.TABLE_Z, spot, N.TABLE_Z, obstacles=Obstacles(sc, pc.xy), final_yaws=(fy,)
                )
                if g is None:
                    continue
                mid = Piece(spot, fy, pc.digit)
                g2, pl2 = choose(b, mid, arms[b].q, N.TABLE_Z, pad.xy, N.PAD_TOP, obstacles=Obstacles(sc))
                if g2 is not None:
                    pl1["spot_level"] = level
                    return pl1["t"] + pl2["t"], "handoff", pc, g, pl1, mid, g2, pl2


@register
class ArrangeDigits(Program):
    name = "arrange_digits"

    def match(self, instruction: str, first_obs) -> float:
        return 0.95 if parse_instruction(instruction) else 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        for chunk in self._run(ctx, arms):
            o = ctx.obs
            yield chunk

    def _run(self, ctx, arms):
        mode = parse_instruction(ctx.instruction) or "largest"

        def hold_forever():
            while True:
                yield [actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)]

        def home_all():
            yield from run_arms(arms, {a: go_home(arms[a]) for a in K.ARMS})

        yield from run_arms(
            arms, {a: arms[a].joints_to(LOOK_Q, force=True, joint_step=FREE_JOINT_STEP) for a in K.ARMS}
        )
        yield from run_arms(arms, {a: arms[a].hold(LOOK_SETTLE) for a in K.ARMS})
        sc = perceive_head(ctx)
        q_look = {}
        for a in K.ARMS:
            b = "right" if a == "left" else "left"
            q_look[a] = look_target(a, arms[a].q, WRIST_LOOK_XY[a], WRIST_LOOK_H, other_q=q_look.get(b))

        def look_task(arm: Arm):
            if q_look[arm.name] is not None:
                yield from free_joints_to(arm, q_look[arm.name], grip=1.0)
            yield from arm.hold(LOOK_SETTLE)

        yield from run_arms(arms, {a: look_task(arms[a]) for a in K.ARMS})
        read_wrists(ctx, sc)
        plan = assign(sc.tiles, sc.pads, mode) if len(sc.pads) >= 2 else None
        if plan is None:
            yield from home_all()
            yield from hold_forever()
            return
        for t, _ in plan:
            if hasattr(t, "extra"):
                t.extra["planned"] = True
        todo = [(Piece(t.xy, t.yaw, t.digit), sc.pads[i], t) for t, i in plan]
        failed = []
        while todo:
            plans = direct_plans(sc, arms, todo)
            if plans:
                best, item = plans[0]
            else:
                best = item = None
                for it in todo:
                    best = handoff_plan(sc, arms, it[0], it[1])
                    if best is not None:
                        item = it
                        break
            if best is None:
                break
            todo.remove(item)
            res = yield from execute(arms, sc, best, item[1], n_after=len(todo))
            t = item[2]
            if res["ok"]:
                t.xy = np.asarray(item[1].xy, dtype=float).copy()
                t.yaw = float(res.get("final_yaw", 0.0))
            else:
                t.xy = np.asarray(res["piece"].xy, dtype=float).copy()
            if not res["ok"]:
                failed.append(item)
            if ctx.obs.step > STEP_LIMIT - HOME_RESERVE:
                break
        tries = 0
        while failed and tries < MAX_RETRIES and (STEP_LIMIT - ctx.obs.step > RETRY_MIN_STEPS):
            tries += 1
            yield from run_arms(
                arms, {a: arms[a].joints_to(LOOK_Q, force=True, joint_step=FREE_JOINT_STEP) for a in K.ARMS}
            )
            yield from run_arms(arms, {a: arms[a].hold(LOOK_SETTLE) for a in K.ARMS})
            item = relocate(ctx, sc, failed)
            if item is None:
                break
            failed.remove(item[3])
            todo = [item[:3]]
            plans = direct_plans(sc, arms, todo)
            best = plans[0][0] if plans else handoff_plan(sc, arms, item[0], item[1])
            if best is None:
                continue
            res = yield from execute(arms, sc, best, item[1], n_after=0)
            t = item[2]
            if res["ok"]:
                t.xy = np.asarray(item[1].xy, dtype=float).copy()
                t.yaw = float(res.get("final_yaw", 0.0))
            else:
                t.xy = np.asarray(res["piece"].xy, dtype=float).copy()
                failed.append(item[:3])
        yield from home_all()
        yield from hold_forever()
