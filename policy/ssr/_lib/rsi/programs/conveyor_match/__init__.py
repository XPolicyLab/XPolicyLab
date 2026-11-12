from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ...core import actions
from ...perception import cameras as C
from . import conveyor as CV
from . import objects as O
from ...perception import robot_mask as RM
from . import wrist_fit as WF
from ...planning import kinematics as K
from ...planning import motion
from ...skills import grasp as G
from . import shape_grasp as SG
from .. import Program, register

PARK_Q = {"left": np.array([1.2, 0.5, 0.4, 0.0, 0.0, 0.0]), "right": np.array([-1.2, 0.5, 0.4, 0.0, 0.0, 0.0])}
ROI_ROWS = 55, 330
BG_STEP = 100
BG_RIGHT_STEP = 140
PERCEIVE_EVERY = 2
FIT_ALL_EVERY = 10
MAX_FIT_ALL = 4
MAX_FIT_ALL_SIMILAR = 7
EXTRA_FIT_SIM = 0.9
REFIT_EVERY = 6
GOOD_FIT = 0.8
FIT0_MARGIN = 0.12
TIE_MARGIN = 0.05
TWIN_CAT_MIN_FITS = 3
TWIN_CAT_FRAC = 0.75
TWIN_CAT_MARGIN = 0.07
ALT_MIN_SIM = 0.75
DARK_PRIOR_MIN = 0.5
DARK_CATEGORIES = "camera", "garage"
BRIGHT_CATEGORIES = "bottle", "plate", "ice_cream", "puppet", "game_machine", "phone"
START_POSE_TOL = 0.01


def dark_allowed(categories, dark: float | None):
    cats = list(categories)
    if dark is None:
        return cats
    drop = BRIGHT_CATEGORIES if dark >= DARK_PRIOR_MIN else DARK_CATEGORIES
    kept = [c for c in cats if c not in drop]
    return kept or cats


def with_low_hover(combos, low_ok: bool):
    out = []
    i = 0
    while i < len(combos):
        j = i
        while j < len(combos) and combos[j][0] == combos[i][0]:
            j += 1
        out += [(ci, xg, False) for ci, xg in combos[i:j]]
        if low_ok:
            out += [(ci, xg, True) for ci, xg in combos[i:j]]
        i = j
    return out


def axis_offset(heading: float, yaw: float) -> float:
    d = (heading - yaw) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


CLEAR_WIN = 0.05
CLEAR_MIN = 0.55
CLEAR_S0 = 0.9
PLAUSIBLE = 0.06
PART_SIM = 0.85
REACQUIRE_STEPS = 8
REACQUIRE_PX = 40.0
SPLIT_RATIO = 0.6
MIN_MATCH_SIM = 0.65
PART_TRIGGER_SIM = 0.55
WIN_MARGIN = 0.005
SWITCH_WEAK = 0.9
SWITCH_MARGIN = 0.1
GRASP_X = 0.25, 0.3, 0.2, 0.35, 0.15, 0.4, 0.45, 0.5, 0.55
TILTS = tuple(math.radians(a) for a in (0.0, 20.0, 35.0, 50.0))
LIFT_BACK = 0.08
HOVER_DZ = 0.07
HOVER_TOP_CLEAR = 0.05
DESCEND_STEPS = 16
CLOSE_STEPS = 6
LIFT_STEPS = 47
LIFT_DZ = 0.26
LIFT_NEED = 0.12
LIFT_TILT_STEP = math.radians(5.0)
LIFT_TILT_MAX = math.radians(45.0)
LIFT_ROLL_MAX = math.radians(45.0)
HOVER_DZ_LOW = 0.03
HOVER_TOP_CLEARS_LOW = 0.03, 0.02
REPLAN_EVERY = 6
COMMIT_LEAD = 2
FOLLOW_LIFT = 8
V_PRIOR = 0.004
SPEED_AGREE = 0.03
SPEED_OFF = 0.25
POSE_MAD = 0.005
YAW_MAD = math.radians(6.0)
POSE_CHECK_DH = 0.03
POSE_CHECK_IOU = 0.75
POSE_CHECK_GAIN = 0.08
POSE_CHECK_START_R = 0.06
POSE_FIT_SCORE = 0.95


def pose_hover_top(shape: O.Shape) -> float:
    top = float(shape.extent[2])
    for s_ in O.shapes(shape.category):
        if s_.key != shape.key and abs(float(s_.extent[2]) - float(shape.extent[2])) > POSE_CHECK_DH:
            top = max(top, float(s_.extent[2]))
    return top


NUDGE_DZ = 0.008
NUDGE_DZ_MAX = 0.035
NUDGE_TOL = 0.005
NUDGE_SPAN = 0.03
NUDGE_RES = 0.002
NUDGE_MIN = 0.006
NUDGE_LIFT = 0.01
NUDGE_STEPS = 8
NUDGE_MAX = 2
VIEW_LOCK_RANGE = math.radians(10.0)
VIEW_LOCK_RATIO = 0.5
VIEW_LOCK_MIN_YAW = math.radians(2.0)
VIEW_LOCK_MIN_FITS = 4
MAX_ATTEMPTS = 3
COLOUR_ONLY_SIM = 0.75
COLOUR_ONLY_MARGIN = 0.05
COLOUR_ONLY_X = 0.0
COLOUR_ONLY_X_MAX = 0.15
GONE_X = 0.47
MAX_ATTEMPTS_TOTAL = 5
MISS_CHECK = 12
FINGERS_MET_G = 0.003
BACK_OFF_DZ = 0.07
BLOCK_CLOSE_DZ = 0.025
BLOCKED_HEADING = math.radians(25.0)
BLOCKED_TILT = math.radians(20.0)
BLOCKED_TCP = 0.02
BLOCKED_MEMORY_DZ = 0.01
BLOCKED_AWAY_TCP = 0.035
BLOCKED_AWAY_HEADING = math.radians(60.0)
BLOCKED_PER_TILT = 6
WRIST_MIN_IOU = 0.7
WRIST_MAX_SHIFT = 0.04
WRIST_BLEND = 4
WRIST_FADE_LEAD = 4
WRIST_GATE = 0.003
DARK_GATE = 0.6
DARK_RATIO = 0.5
SEARCH_Y_TOL = 0.006
SEARCH_Y_TOL_FALLBACK = 0.002
SYM_HEADING_OFFSETS = 2.0, -2.0, 4.0


def turned(v, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * float(v[0]) - s * float(v[1]), s * float(v[0]) + c * float(v[1])])


def default_search(pts_obj, to_base, sym: bool, height: float):
    found = SG.search(pts_obj, to_base, symmetric=sym)
    if found or not (sym and height > PINCH_MAX_H):
        return found, None
    for d in SYM_HEADING_OFFSETS:
        found = SG.search(pts_obj, turned(to_base, d), symmetric=sym)
        if found:
            return found, d
    return found, None


HANG_MOMENT = 0.0125
HANG_SWING_W = 0.4
HANG_Z_BAND = 0.007
HANG_PER_TILT = 6
HANG_ALIGN_TOL = math.radians(8.0)
RETRY_TURNS = 30.0, -30.0, 60.0, -60.0
DISH_DOWNSTREAM_MIN = 0.02
PINCH_MAX_H = 0.06
NEIGHBOUR_MARGIN = 0.012
NEIGHBOUR_PREFER_SCORE = 0.9
NEIGHBOUR_PREFER_ROBUST = 0.25


def neighbour_points(mem, v: float, t: int, x_target: float, min_score: float = 0.75) -> np.ndarray | None:
    pts = []
    for tr in mem.tracks:
        if tr is mem.target:
            continue
        good = [(s, f) for s, f in tr.fits if f.score > min_score]
        if not good:
            continue
        s0, f = good[-1]
        xn = f.x + v * (t - s0)
        if abs(xn - x_target) < 0.06:
            continue
        pts.append(f.shape.posed(xn, f.y, f.yaw, CV.BELT_Z, n=600))
    return np.concatenate(pts) if pts else None


def neighbour_hit(mem, v: float, t_close: int, rot, tcp_w, approach, grip: float, min_score: float = 0.75) -> bool:
    pts = neighbour_points(mem, v, t_close, float(tcp_w[0]), min_score)
    if pts is None:
        return False
    near = np.linalg.norm(pts[:, :2] - tcp_w[:2], axis=1) < 0.25
    if not near.any():
        return False
    pts = pts[near]
    boxes = SG.gripper_boxes(grip)
    for s in (0.0, HOVER_DZ * 0.5, HOVER_DZ):
        T = SG_link6(rot, tcp_w + np.asarray(approach) * s)
        p_l6 = (pts - T[:3, 3]) @ T[:3, :3]
        if SG.n_inside(p_l6, boxes, margin=NEIGHBOUR_MARGIN):
            return True
    return False


HOVER_NB_RADIUS = 0.25
HOVER_NB_BEHIND = 0.05
HOVER_NB_MAX = 0.16
TRANSIT_NB_SCORE = 0.75
TRANSIT_MARGIN = 0.03
TRANSIT_EVERY = 2
TRANSIT_MAX_TRY = 10
SIDE_DY = 0.13, 0.17
SIDE_MIN_Y = -0.25
SIDE_STEPS = 14
WRIST_BOX = -0.12, 0.0, -0.045, 0.045, -0.045, 0.045
MIRROR_DZ = 0.003
TRANSIT_BEST_GAIN = 0.003
NB_FIT_DOWN = 200.0
MAP_CLUSTER_R = 0.045
MAP_TARGET_R = 0.06
NB_FIT_UP = 45.0


def is_mirror(a, b) -> bool:
    return (
        abs(math.remainder(a.heading - b.heading - math.pi, 2 * math.pi)) <= math.radians(10.0)
        and abs(a.tilt + b.tilt) <= math.radians(5.0)
        and (a.approach == b.approach)
        and (float(np.linalg.norm(np.asarray(a.tcp, dtype=float) - np.asarray(b.tcp, dtype=float))) <= 0.01)
        and (abs(float(a.tcp[2]) - float(b.tcp[2])) <= MIRROR_DZ)
    )


def transit_neighbours(mem, v: float, x0_t: float, y_t: float, min_score: float = TRANSIT_NB_SCORE) -> list:
    return map_neighbours(getattr(mem, "belt_fits", []), v, x0_t, y_t, min_score)


def map_neighbours(belt_fits: list, v: float, x0_t: float, y_t: float, min_score: float = TRANSIT_NB_SCORE) -> list:
    good = sorted(((s_, f, tid) for s_, f, tid in belt_fits if f.score > min_score), key=lambda e: -e[1].score)
    clusters = []
    for s_, f, tid in good:
        p = f.x - v * s_, f.y
        for c in clusters:
            if math.hypot(p[0] - c[0][0], p[1] - c[0][1]) < MAP_CLUSTER_R:
                c[1].append((s_, f))
                break
        else:
            clusters.append([p, [(s_, f)], tid])
    out = []
    for _, fits, tid in clusters:
        nb = _transit_neighbour(tid, fits, v, x0_t, y_t, min_score, same_r=MAP_TARGET_R)
        if nb is not None:
            out.append(nb)
    return out


def _transit_neighbour(tid, fits, v, x0_t, y_t, min_score, same_r: float | None = None):
    good = [(s_, f) for s_, f in fits if f.score > min_score]
    if not good:
        return None
    votes: dict = {}
    for _, f in good:
        votes[f.shape.key] = votes.get(f.shape.key, 0) + 1
    key = max(votes, key=lambda k: (votes[k], max(f.score for _, f in good if f.shape.key == k)))
    kf = [(s_, f) for s_, f in good if f.shape.key == key]
    x0 = float(np.median([f.x - v * s_ for s_, f in kf]))
    y = float(np.median([f.y for _, f in kf]))
    if math.hypot(x0 - x0_t, y - y_t) < (SAME_PLACE_R if same_r is None else same_r):
        return None
    best = max((f for _, f in kf), key=lambda f: f.score)
    return tid, x0, y, float(best.yaw), best.shape


def transit_clearance(
    arm: str, steps: list, step0: int, t_until: int, nbs: list, v: float, g_open: float, cap: float = 0.05
):
    best = cap, None, None
    if not nbs:
        return best
    boxes = np.vstack([SG.gripper_boxes(g_open), np.asarray([WRIST_BOX], dtype=float)])
    lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]
    shapes = [(tid, sh.posed(0.0, y, yaw, CV.BELT_Z, n=300), x0) for tid, x0, y, yaw, sh in nbs]
    for i in range(0, len(steps), TRANSIT_EVERY):
        qq = steps[i][0]
        t = step0 + i + 1
        if t > t_until:
            break
        if qq is None:
            continue
        T = K.fk(arm, qq)
        for tid, pts, x0 in shapes:
            pw = pts + np.array([x0 + v * t, 0.0, 0.0])
            if float(np.min(np.linalg.norm(pw[:, :2] - T[:2, 3], axis=1))) > 0.35:
                continue
            p = (pw - T[:3, 3]) @ T[:3, :3]
            d = np.maximum(lo[None] - p[:, None], p[:, None] - hi[None]).max(axis=2)
            dmin = float(d.min())
            if dmin < best[0]:
                best = dmin, t, tid
    return best


def transit_hit(
    arm: str, steps: list, step0: int, t_until: int, nbs: list, v: float, g_open: float, margin: float = TRANSIT_MARGIN
):
    if not nbs:
        return None
    boxes = np.vstack([SG.gripper_boxes(g_open), np.asarray([WRIST_BOX], dtype=float)])
    shapes = [(tid, sh.posed(0.0, y, yaw, CV.BELT_Z, n=300), x0) for tid, x0, y, yaw, sh in nbs]
    for i in range(0, len(steps), TRANSIT_EVERY):
        qq = steps[i][0]
        t = step0 + i + 1
        if t > t_until:
            break
        if qq is None:
            continue
        T = K.fk(arm, qq)
        for tid, pts, x0 in shapes:
            pw = pts + np.array([x0 + v * t, 0.0, 0.0])
            if float(np.min(np.linalg.norm(pw[:, :2] - T[:2, 3], axis=1))) > 0.35:
                continue
            n = SG.n_inside((pw - T[:3, 3]) @ T[:3, :3], boxes, margin=margin)
            if n:
                return t, tid, int(n)


def neighbour_top(mem, v: float, t: int, xy, min_score: float = NEIGHBOUR_PREFER_SCORE) -> float:
    top = 0.0
    for tr in mem.tracks:
        if tr is mem.target:
            continue
        fits = tr.fits or getattr(tr, "split_fits", [])
        good = [(s, f) for s, f in fits if f.score > min_score]
        if not good:
            continue
        s0, f = good[-1]
        xn = f.x + v * (t - s0)
        if abs(xn - xy[0]) < 0.06 or xn < xy[0] - HOVER_NB_BEHIND:
            continue
        if math.hypot(xn - xy[0], f.y - xy[1]) > HOVER_NB_RADIUS:
            continue
        top = max(top, float(f.shape.extent[2]))
    return min(top, HOVER_NB_MAX)


def SG_link6(rot, tcp) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(tcp, dtype=float) - rot[:, 0] * K.TCP_OFFSET
    return t


def belt_speed(mem) -> float | None:
    slopes = []
    for t in mem.tracks + ([mem.target] if mem.target is not None and mem.target not in mem.tracks else []):
        pts = [(s, f.x) for s, f in t.fits if f.score > 0.85]
        if len(pts) >= 3 and pts[-1][0] - pts[0][0] >= 30:
            st = np.array([a for a, _ in pts], dtype=float)
            xs = np.array([b for _, b in pts])
            slopes.append(float(np.polyfit(st, xs, 1)[0]))
    return float(np.median(slopes)) if slopes else None


SAME_PLACE_R = 0.045


def last_place(t, min_score: float = 0.7):
    good = [(s, f) for s, f in t.fits if f.score > min_score]
    if not good:
        return None
    s_last = max(s for s, _ in good)
    f = max((f for s, f in good if s == s_last), key=lambda f: f.score)
    return s_last, float(f.x), float(f.y)


def twin_category(mem) -> str | None:
    t = mem.target
    wins = [max(cs, key=cs.get) for cs in t.cat_scores if cs]
    n = len(wins)
    if n < TWIN_CAT_MIN_FITS or not mem.scores0:
        return None
    cat = max(set(wins), key=wins.count)
    if (
        cat == mem.category
        or wins.count(cat) < TWIN_CAT_FRAC * n
        or wins.count(mem.category) > (1.0 - TWIN_CAT_FRAC) * n + 1e-09
    ):
        return None
    s0 = mem.scores0.get(cat)
    if s0 is None or max(mem.scores0.values()) - s0 > TWIN_CAT_MARGIN:
        return None
    return cat


def colour_only_candidate(mem, step: int, v: float = V_PRIOR):
    best = None
    for t in mem.tracks:
        if t.n_fit_all < MAX_FIT_ALL or t.last_step != step or rejected(mem, t, step):
            continue
        lp = last_place(t)
        if lp is None:
            continue
        x_now = lp[1] + v * (step - lp[0])
        if not COLOUR_ONLY_X <= x_now <= COLOUR_ONLY_X_MAX or t.sim() < COLOUR_ONLY_SIM:
            continue
        others = [u.sim() for u in mem.seen.values() if u is not t and u.sims and (not same_place(t, u, step, v))]
        margin = t.sim() - max(others, default=0.0)
        if margin >= COLOUR_ONLY_MARGIN and (best is None or t.sim() > best[0].sim()):
            best = t, margin
    return best


def same_place(a, b, step: int, v: float = V_PRIOR, r: float = SAME_PLACE_R) -> bool:
    pa, pb = last_place(a), last_place(b)
    if pa is None or pb is None:
        return False
    return math.hypot(pa[1] + v * (step - pa[0]) - pb[1] - v * (step - pb[0]), pa[2] - pb[2]) < r


def rejected(mem, t, step: int) -> bool:
    return any(t is r or same_place(r, t, step) for r in mem.rejected)


def grip_lost(closed_for: int, g_obs: float) -> str | None:
    met = g_obs <= FINGERS_MET_G
    if closed_for < MISS_CHECK or not met:
        return None
    if closed_for == MISS_CHECK:
        return "missed"
    return "slipped"


def like_blocked(
    g, blocked, tcp_r: float = BLOCKED_TCP, hd_tol: float = BLOCKED_HEADING, any_tilt: bool = False
) -> bool:
    h, t, tcp = blocked[:3]
    if float(np.linalg.norm(np.asarray(g.tcp, dtype=float) - tcp)) > tcp_r:
        return False
    tilt_tol = math.inf if any_tilt else BLOCKED_TILT
    same = abs(math.remainder(g.heading - h, 2 * math.pi)) <= hd_tol and abs(g.tilt - t) <= tilt_tol
    mirror = abs(math.remainder(g.heading - h - math.pi, 2 * math.pi)) <= hd_tol and abs(g.tilt + t) <= tilt_tol
    return same or mirror


def away_from_blocked(g, blocked) -> bool:
    if like_blocked(g, blocked):
        return False
    if len(blocked) > 3 and (not blocked[3]):
        return True
    return not like_blocked(g, blocked, BLOCKED_AWAY_TCP, BLOCKED_AWAY_HEADING, True)


def dish_downstream_first(grasps: list) -> list:
    return sorted(grasps, key=lambda g: float(g.tcp[0]) < DISH_DOWNSTREAM_MIN)


def steady_pose(fits, votes: dict, key: str, v: float) -> str:
    steady = {}
    for k in votes:
        kf = [(s, f) for s, f in fits if f.shape.key == k]
        if len(kf) < 3:
            continue
        x0s = np.array([f.x - v * s for s, f in kf])
        ys = np.array([f.y for _, f in kf])
        mad = max(float(np.median(np.abs(x0s - np.median(x0s)))), float(np.median(np.abs(ys - np.median(ys)))))
        ok = mad <= POSE_MAD
        if ok and k not in CV.SYMMETRIC:
            a = np.array([f.yaw for _, f in kf])
            ref = float(np.median(a))
            dev = np.abs(np.array([math.remainder(float(x) - ref, 2 * math.pi) for x in a]))
            ok = float(np.median(dev)) <= YAW_MAD
        steady[k] = ok
    if steady.get(key, True):
        return key
    alt = [k for k in votes if steady.get(k)]
    return max(alt, key=votes.get) if alt else key


def _wrap_half(a):
    return (np.asarray(a, dtype=float) + math.pi / 2) % math.pi - math.pi / 2


def _axial_mad(a) -> float:
    a = np.asarray(a, dtype=float)
    m = 0.5 * math.atan2(float(np.sin(2 * a).mean()), float(np.cos(2 * a).mean()))
    return float(np.median(np.abs(_wrap_half(a - m))))


def view_locked(fits, key: str) -> bool:
    if key in CV.SYMMETRIC:
        return False
    kf = [f for _, f in fits if f.shape.key == key]
    if len(kf) < VIEW_LOCK_MIN_FITS:
        return False
    cx, cy = float(C.head().pos[0]), float(C.head().pos[1])
    az = np.array([math.atan2(f.y - cy, f.x - cx) for f in kf])
    if float(az.max() - az.min()) < VIEW_LOCK_RANGE:
        return False
    yaw = np.array([f.yaw for f in kf])
    s_yaw = _axial_mad(yaw)
    return s_yaw >= VIEW_LOCK_MIN_YAW and _axial_mad(yaw - az) <= VIEW_LOCK_RATIO * s_yaw


def belt_speed_estimate(v_pool, v_own) -> float:
    if (
        v_pool is not None
        and v_own is not None
        and (abs(v_pool - v_own) <= SPEED_AGREE * V_PRIOR)
        and (abs(0.5 * (v_pool + v_own) - V_PRIOR) > SPEED_OFF * V_PRIOR)
    ):
        return 0.5 * (v_pool + v_own)
    return V_PRIOR


FIRST_VIEWS = 30, 42


def q_obs_both(obs):
    return np.asarray(obs.state["left_arm_joint_state"], dtype=float), np.asarray(
        obs.state["right_arm_joint_state"], dtype=float
    )


def best_by_category(fits) -> dict:
    out: dict = {}
    for f in fits:
        out[f.category] = max(out.get(f.category, 0.0), f.score)
    return out


def roi_mask() -> np.ndarray:
    m = np.zeros((C.H, C.W), bool)
    m[ROI_ROWS[0] : ROI_ROWS[1], :] = True
    return m


@dataclass
class Track:
    tid: int
    first_step: int
    last_step: int
    centroid: tuple
    du: float = 1.8
    fits: list = field(default_factory=list)
    sims: list = field(default_factory=list)
    last_fit_all: int = -1000
    n_fit_all: int = 0
    last_refit: int = -1000
    cat_scores: list = field(default_factory=list)
    part_fits: list = field(default_factory=list)
    part_sims: list = field(default_factory=list)
    part: bool = False
    dark: list = field(default_factory=list)
    fit_areas: list = field(default_factory=list)
    split_fits: list = field(default_factory=list)
    fit_keys: list = field(default_factory=list)

    def reset_identity(self):
        self.split_fits = self.split_fits + self.fits
        self.fits, self.sims, self.cat_scores = [], [], []
        self.part_fits, self.part_sims, self.fit_areas = [], [], []
        self.n_fit_all = 0
        self.last_fit_all = -1000

    def plausible(self, category: str, good_fit: float = GOOD_FIT, clear: bool = True) -> float:
        def informative(cs):
            if max(cs.values()) >= good_fit:
                return True
            if not clear:
                return False
            ranked = sorted(cs.values(), reverse=True)
            return (
                cs.get(category, 0.0) >= ranked[0]
                and ranked[0] >= CLEAR_MIN
                and (len(ranked) < 2 or ranked[0] - ranked[1] >= CLEAR_WIN)
            )

        good = [cs for cs in self.cat_scores if informative(cs)]
        if not good:
            return 0.0
        ok = [cs.get(category, 0.0) >= max(cs.values()) - PLAUSIBLE for cs in good]
        return float(np.mean(ok)) * len(good) / len(self.cat_scores)

    def votes(self) -> dict:
        v: dict = {}
        for _, f in self.fits:
            v[f.category] = v.get(f.category, 0.0) + f.score
        return v

    def category(self):
        v = self.votes()
        return max(v, key=v.get) if v else None

    def sim(self) -> float:
        return float(np.mean(self.sims)) if self.sims else 0.0


@dataclass
class Memory:
    bg: np.ndarray | None = None
    category: str | None = None
    score0: float = 1.0
    alt: dict = field(default_factory=dict)
    cat0: str | None = None
    score00: float = 1.0
    fit0: object = None
    hist0: np.ndarray | None = None
    tracks: list = field(default_factory=list)
    next_tid: int = 0
    target: Track | None = None
    last_combo: tuple | None = None
    search: list = field(default_factory=list)
    search_key: tuple | None = None
    search_y: float = 0.0
    failed: set = field(default_factory=set)
    blocked: list = field(default_factory=list)
    no_stable: bool = False
    target_sim: float = 0.0
    gen: int = 0
    gone: bool = False
    n_att: int = 0
    n_hover: int = 0
    first_shapes: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    seen: dict = field(default_factory=dict)
    scores0: dict = field(default_factory=dict)
    twin_cat_done: bool = False
    colour_only: bool = False
    retry: bool = False
    prefer: list = field(default_factory=list)
    rotated: bool = False
    belt_fits: list = field(default_factory=list)
    view_locked: bool = False
    pose_switch: tuple | None = None


@register
class ConveyorMatch(Program):
    name = "conveyor_match"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "conveyor" in s and ("first" in s or "remember" in s) and ("match" in s or "same" in s):
            return 0.95
        return 0.0

    def run(self, ctx):
        o = ctx.obs
        q = {a: np.asarray(o.state[f"{a}_arm_joint_state"], dtype=float).copy() for a in K.ARMS}
        g = {a: 1.0 for a in K.ARMS}
        q_start = {a: q[a].copy() for a in K.ARMS}
        frame0 = o.images["cam_head"].copy()
        wrist0 = None
        if float(np.max(np.abs(q["right"]))) < START_POSE_TOL:
            wrist0 = o.images["cam_right_wrist"].copy(), q["right"].copy(), float(o.state["right_ee_joint_state"][0])
        cam = C.head()
        roi = roi_mask()
        mem = Memory()

        def act():
            return [actions.joint_action(q["left"], q["right"], g["left"], g["right"])]

        plans = {a: motion.joint_move(q[a], PARK_Q[a], joint_step=0.06) for a in K.ARMS}
        n = max(len(p) for p in plans.values())
        early = []
        for i in range(n):
            for a in K.ARMS:
                if i < len(plans[a]):
                    q[a] = plans[a].q[i]
            if ctx.obs.step in FIRST_VIEWS:
                early.append((ctx.obs.images["cam_head"].copy(), q_obs_both(ctx.obs)))
            yield act()
        while ctx.obs.step < BG_STEP:
            if ctx.obs.step in FIRST_VIEWS:
                early.append((ctx.obs.images["cam_head"].copy(), q_obs_both(ctx.obs)))
            yield act()
        img = ctx.obs.images["cam_head"].copy()
        while ctx.obs.step < BG_RIGHT_STEP:
            yield act()
        img[:, C.W // 2 :] = ctx.obs.images["cam_head"][:, C.W // 2 :]
        rob_now = RM.mask(cam, q["left"], q["right"])
        rob0 = RM.mask(cam, q_start["left"], q_start["right"])
        diff = np.abs(img.astype(np.int16) - frame0.astype(np.int16)).max(axis=2) > CV.DIFF_WEAK
        bg = img.copy()
        left_half = np.zeros_like(diff)
        left_half[:, : C.W // 2] = True
        use0 = diff & left_half & ~rob0 | rob_now
        bg[use0] = frame0[use0]
        mem.bg = bg
        fg0 = CV.foreground(frame0, bg, valid=~rob0)
        right = roi & ~left_half
        cands = CV.blobs(fg0, roi=right)
        if cands:
            cands = object_pieces(cands)
            b0 = CV.merge_blobs(cands) if len(cands) > 1 else cands[0]
            fits = CV.fit_all(cam, fg0, b0)
            mem.fit0 = fits[0]
            hists = [CV.colour_hist(frame0, b0.mask)]
            views = [best_by_category(fits)]
            for img_k, (ql, qr) in early:
                rob_k = RM.mask(cam, ql, qr)
                fg_k = CV.foreground(img_k, bg, valid=~rob_k)
                ck = object_pieces(CV.blobs(fg_k, roi=right, robot=rob_k))
                if not ck:
                    continue
                bk = CV.merge_blobs(ck) if len(ck) > 1 else ck[0]
                if bk.touches_robot or bk.touches_border:
                    continue
                best_k = best_by_category(CV.fit_all(cam, fg_k, bk))
                if max(best_k.values()) < GOOD_FIT:
                    continue
                views.append(best_k)
                hists.append(CV.colour_hist(img_k, bk.mask))
            votes = {c: float(np.mean([v.get(c, 0.0) for v in views])) for c in views[0]}
            if wrist0 is not None:
                dark = None
                try:
                    dark = WF.object_darkness(wrist0[0], K.fk("right", wrist0[1]), wrist0[2])
                except Exception:
                    pass
                allowed = dark_allowed(votes, dark[0] if dark is not None else None)
                if len(allowed) < len(votes):
                    views = [{c: s for c, s in v.items() if c in allowed} for v in views]
                    votes = {c: s for c, s in votes.items() if c in allowed}
            mem.category = max(votes, key=votes.get)
            mem.score0 = float(views[0].get(mem.category, 0.0))
            mem.alt = {
                c: float(views[0].get(c, 0.0))
                for c, v in votes.items()
                if c != mem.category and v >= votes[mem.category] - TIE_MARGIN
            }
            best0 = max(views[0], key=views[0].get)
            if best0 != mem.category:
                mem.alt[best0] = float(views[0][best0])
            mem.cat0, mem.score00 = mem.category, mem.score0
            top3 = sorted(views[0], key=views[0].get, reverse=True)[:3] + sorted(votes, key=votes.get, reverse=True)[:3]
            mem.scores0 = {c: float(views[0].get(c, 0.0)) for c in top3}
            mem.first_shapes = [f.shape.key for f in fits[:3]]
            mem.hist0 = np.mean(hists, axis=0)
        for attempt in range(MAX_ATTEMPTS_TOTAL):
            plan = None
            last_try = -1000
            gen = mem.gen
            while True:
                step = ctx.obs.step
                if step % PERCEIVE_EVERY == 0:
                    self._perceive(ctx, cam, roi, mem, q)
                if mem.gen != gen:
                    gen = mem.gen
                    plan = None
                    last_try = -1000
                if mem.target is not None and mem.n_att >= MAX_ATTEMPTS and (not mem.gone):
                    mem.gone = True
                if mem.target is not None and step - last_try >= REPLAN_EVERY and (not mem.gone):
                    last_try = step
                    p = self._plan(ctx, mem, q)
                    if p is not None:
                        plan = p
                if plan is not None and plan["t_start"] - step <= COMMIT_LEAD:
                    break
                if step > 650:
                    break
                yield act()
            if plan is None:
                break
            arm = plan["arm"]
            arm2, steps2 = plan.get("arm2"), plan.get("steps2")
            mem.n_att += 1
            k0 = ctx.obs.step - plan["step0"]
            closed_for = 0
            missed = False
            pose_abort = False
            id_reject = False
            nudges = []
            q_cmd_prev = q[arm].copy()
            steps_list = plan["steps"]
            for i in range(len(steps_list) - k0):
                if arm2 is None and "wrist" in plan and (ctx.obs.step == plan["wrist"]["t_arr"]):
                    r_w = self._wrist_refine(ctx, mem, plan, arm, k0 + i, q[arm])
                    if r_w == "pose":
                        missed = pose_abort = True
                        break
                    if r_w == "identity":
                        missed = pose_abort = id_reject = True
                        break
                qq, gg = steps_list[k0 + i]
                q_cmd_prev = q[arm].copy()
                if steps2 is not None and k0 + i < len(steps2):
                    q2, g2 = steps2[k0 + i]
                    if q2 is not None:
                        q[arm2] = q2
                    if g2 is not None:
                        g[arm2] = g2
                if qq is not None:
                    q[arm] = qq
                if gg is not None:
                    if gg < 0.5 <= g[arm]:
                        q_obs = np.asarray(ctx.obs.state[f"{arm}_arm_joint_state"], dtype=float)
                        tcp_cmd = K.point_of(K.fk(arm, q[arm]), K.TCP_OFFSET)
                        tcp_obs = K.point_of(K.fk(arm, q_obs), K.TCP_OFFSET)
                        dz = float(tcp_obs[2] - tcp_cmd[2])
                        new = None
                        if arm2 is None and NUDGE_DZ < dz <= NUDGE_DZ_MAX and (len(nudges) < NUDGE_MAX):
                            try:
                                new = self._nudge(ctx, mem, plan, arm, k0 + i, q_cmd_prev, dz, nudges)
                            except Exception:
                                new = None
                        if new is not None:
                            steps_list[k0 + i :] = new
                            qq, gg = steps_list[k0 + i]
                            if qq is not None:
                                q[arm] = qq
                        elif (
                            arm2 is None
                            and dz > BLOCKED_MEMORY_DZ
                            and (mem.last_combo is not None)
                            and (mem.last_combo[0] < len(mem.search))
                        ):
                            gb = mem.search[mem.last_combo[0]]
                            mem.blocked.append(
                                (gb.heading, gb.tilt, np.asarray(gb.tcp, dtype=float).copy(), dz > BLOCK_CLOSE_DZ)
                            )
                        if new is None and arm2 is None and (dz > BLOCK_CLOSE_DZ):
                            missed = True
                            break
                    g[arm] = gg
                closed_for = closed_for + 1 if g[arm] < 0.5 else 0
                if arm2 is None and closed_for >= MISS_CHECK:
                    g_now = float(ctx.obs.state[f"{arm}_ee_joint_state"][0])
                    lost = grip_lost(closed_for, g_now)
                    if lost is not None:
                        missed = True
                        break
                yield act()
            if missed:
                g[arm] = 1.0
                mem.retry = True
                if mem.last_combo is not None and (not pose_abort):
                    mem.failed.add(mem.last_combo[0])
                for qq in self._back_off(arm, q[arm]):
                    q[arm] = qq
                    yield act()
                mem.last_combo = None
                if id_reject and mem.target is not None:
                    mem.target = None
                    mem.gen += 1
                    mem.search_key = None
                    mem.failed = set()
                    mem.blocked = []
                continue
            used = [arm] + ([arm2] if arm2 else [])
            for a in used:
                g[a] = 1.0
            for _ in range(4):
                yield act()
            moves = {a: motion.joint_move(q[a], PARK_Q[a], joint_step=0.08).q for a in used}
            for i in range(max(len(m) for m in moves.values())):
                for a, m in moves.items():
                    if i < len(m):
                        q[a] = m[i]
                yield act()
            mem.target = None
            mem.tracks = []
            mem.last_combo = None
            mem.retry = True
        while True:
            yield act()

    def _perceive(self, ctx, cam, roi, mem: Memory, q):
        o = ctx.obs
        step = o.step
        img = o.images["cam_head"]
        rob = RM.mask(cam, q["left"], q["right"])
        fg = CV.foreground(img, mem.bg, valid=~rob)
        bl = merge_columns(CV.blobs(fg, roi=roi, robot=rob))
        used = set()
        for b in sorted(bl, key=lambda b: -b.centroid[0]):
            best, best_d = None, 1000000000.0
            for t in mem.tracks:
                if t.tid in used:
                    continue
                du = t.du * (step - t.last_step)
                pu = t.centroid[0] + du
                d = math.hypot(b.centroid[0] - pu, b.centroid[1] - t.centroid[1])
                if d < best_d:
                    best, best_d = t, d
            if best is not None and best_d < 30.0 + 2.0 * (step - best.last_step):
                t = best
                dt = step - t.last_step
                if dt > 0 and (not b.touches_border):
                    t.du = 0.7 * t.du + 0.3 * max(0.5, (b.centroid[0] - t.centroid[0]) / dt)
            else:
                t = Track(mem.next_tid, step, step, b.centroid)
                mem.next_tid += 1
                mem.tracks.append(t)
            used.add(t.tid)
            t.centroid = b.centroid
            t.last_step = step
            clean = not b.touches_border and (not b.touches_robot)
            if not clean:
                continue
            if (
                t is not mem.target
                and t.fit_areas
                and (mem.category is not None)
                and (b.area < SPLIT_RATIO * float(np.median(t.fit_areas)))
                and (not self._twin_like(mem, t))
            ):
                t.reset_identity()
            n_max = MAX_FIT_ALL_SIMILAR if t.sims and t.sim() >= EXTRA_FIT_SIM else MAX_FIT_ALL
            nb_fit = mem.target is not None and t is not mem.target and self._near_target(mem, t, step)
            if (
                (mem.target is None or (t is not mem.target and self._switchable(mem)) or nb_fit)
                and t.n_fit_all < n_max
                and (step - t.last_fit_all >= FIT_ALL_EVERY)
            ):
                fits = CV.fit_all(cam, fg, b)
                t.fits.append((step, fits[0]))
                mem.belt_fits.append((step, fits[0], t.tid))
                t.fit_areas.append(b.area)
                cat_best: dict = {}
                for f in fits:
                    cat_best[f.category] = max(cat_best.get(f.category, 0.0), f.score)
                t.cat_scores.append(cat_best)
                t.fit_keys.append([f.shape.key for f in fits[:3]])
                t.n_fit_all += 1
                t.last_fit_all = step
                if mem.hist0 is not None:
                    t.sims.append(CV.hist_similarity(mem.hist0, CV.colour_hist(img, b.mask)))
                mem.seen[t.tid] = t
                top = max(cat_best.values())
                if (
                    mem.category is not None
                    and t.sims
                    and (t.sims[-1] >= PART_TRIGGER_SIM)
                    and (top < GOOD_FIT or cat_best.get(mem.category, 0.0) < top - PLAUSIBLE)
                ):
                    self._part_fit(ctx, cam, fg, img, b, t, mem, step)
            elif t is mem.target and step - t.last_refit >= REFIT_EVERY:
                if t.part:
                    b = CV.colour_part(img, b, mem.hist0)
                    if b is None:
                        continue
                    t.centroid = b.centroid
                shapes = O.shapes(mem.category)
                fits = sorted(
                    (CV.fit_shape(cam, fg, b, s, symmetric=s.key in CV.SYMMETRIC) for s in shapes),
                    key=lambda f: -f.score,
                )
                t.fits.extend((step, f) for f in fits)
                t.last_refit = step
                ratio = (img[b.mask].astype(np.float32) + 4.0) / (mem.bg[b.mask].astype(np.float32) + 4.0)
                t.dark.append(float((ratio.mean(axis=1) < DARK_RATIO).mean()) if len(ratio) else 0.0)
        mem.tracks = [t for t in mem.tracks if step - t.last_step < 40]
        if mem.target is not None:
            self._reacquire(ctx, mem, step)
        if mem.target is not None and mem.category is not None and (not mem.twin_cat_done):
            self._twin_category(ctx, mem, step)
        if mem.category is not None and (mem.target is None or self._switchable(mem)):
            self._select_twin(ctx, mem, step)

    @staticmethod
    def _twin_category(ctx, mem: Memory, step: int) -> None:
        cat = twin_category(mem)
        if cat is None:
            return
        mem.twin_cat_done = True
        mem.category = cat
        mem.score0 = mem.scores0[cat]

    @staticmethod
    def _near_target(mem: Memory, t: Track, step: int) -> bool:
        tgt = mem.target
        du = t.centroid[0] - (tgt.centroid[0] + tgt.du * (step - tgt.last_step))
        return -NB_FIT_UP <= du <= NB_FIT_DOWN

    @staticmethod
    def _reacquire(ctx, mem: Memory, step: int) -> None:
        tgt = mem.target
        if step - tgt.last_step < REACQUIRE_STEPS:
            return
        pu = tgt.centroid[0] + tgt.du * (step - tgt.last_step)
        pv = tgt.centroid[1]
        cands = [t for t in mem.tracks if t is not tgt and t.last_step == step and (t.first_step > tgt.first_step)]
        if not cands:
            return
        best = min(cands, key=lambda t: math.hypot(t.centroid[0] - pu, t.centroid[1] - pv))
        d = math.hypot(best.centroid[0] - pu, best.centroid[1] - pv)
        if d > REACQUIRE_PX:
            return
        best.fits = tgt.fits + best.fits
        best.sims = tgt.sims + best.sims
        best.cat_scores = tgt.cat_scores + best.cat_scores
        best.part, best.part_fits, best.part_sims = (
            tgt.part,
            tgt.part_fits + best.part_fits,
            tgt.part_sims + best.part_sims,
        )
        best.dark = tgt.dark + best.dark
        best.n_fit_all = max(best.n_fit_all, tgt.n_fit_all)
        mem.target = best

    @staticmethod
    def _twin_like(mem: Memory, t: Track) -> bool:
        good_fit = min(GOOD_FIT, mem.score0 - FIT0_MARGIN)
        return (
            len(t.cat_scores) >= 2
            and t.plausible(mem.category, good_fit, clear=mem.score0 < CLEAR_S0) >= 0.5
            or len(t.part_fits) >= 2
        )

    @staticmethod
    def _switchable(mem: Memory) -> bool:
        return mem.target is not None and (mem.target_sim < SWITCH_WEAK or mem.gone or mem.colour_only)

    def _select_twin(self, ctx, mem: Memory, step: int) -> None:
        cur = mem.target

        def options_for(cat, sc0, done_only=False, min_sim=MIN_MATCH_SIM):
            good_fit = min(GOOD_FIT, sc0 - FIT0_MARGIN)
            out = []
            for t in mem.tracks:
                if t is cur or rejected(mem, t, step):
                    continue
                wins = sum(
                    1
                    for cs in t.cat_scores
                    if cs.get(cat, 0.0) >= max([v for c, v in cs.items() if c != cat] + [0.0]) + WIN_MARGIN
                )
                need = 2 if wins else 3
                if len(t.cat_scores) >= need and t.sim() >= min_sim and (not done_only or t.n_fit_all >= MAX_FIT_ALL):
                    pl = t.plausible(cat, good_fit, clear=sc0 < CLEAR_S0)
                    if pl >= 0.5:
                        out.append((t.sim() + 0.3 * pl, t, cat))
            return out

        def take(t, sim, part, switch=False, cat=None):
            if cat is not None and cat != mem.category:
                mem.category = cat
                mem.score0 = mem.score00 if cat == mem.cat0 else mem.alt.get(cat, mem.score0)
            if part:
                t.part = True
                t.fits = list(t.part_fits)
            mem.target = t
            mem.target_sim = sim
            mem.colour_only = False
            mem.gen += 1
            mem.gone = False
            mem.n_att = 0
            mem.view_locked = False
            mem.pose_switch = None
            if switch:
                mem.search_key = None
                mem.failed = set()
                mem.blocked = []
                mem.last_combo = None

        options = options_for(mem.category, mem.score0)
        parts = [
            t
            for t in mem.tracks
            if t is not cur
            and t.n_fit_all >= MAX_FIT_ALL
            and (len(t.part_fits) >= 2)
            and (np.mean(t.part_sims) >= PART_SIM)
            and (not rejected(mem, t, step))
        ]
        if cur is not None and mem.colour_only and options:
            _, t, cat = max(options, key=lambda o: o[0])
            if not same_place(cur, t, step):
                take(t, t.sim(), False, switch=True, cat=cat)
                return
        if cur is not None:
            cands = [(t.sim(), t, cat, False) for _, t, cat in options] + [
                (float(np.mean(t.part_sims)), t, mem.category, True) for t in parts
            ]
            if mem.cat0 is not None and mem.category != mem.cat0:
                cands += [(t.sim(), t, mem.cat0, False) for _, t, _ in options_for(mem.cat0, mem.score00)]
            cands = [c for c in cands if not same_place(cur, c[1], step)]
            if cands:
                sim, t, cat, part = max(cands, key=lambda c: c[0])
                if sim >= mem.target_sim + SWITCH_MARGIN or mem.gone:
                    take(t, sim, part, switch=True, cat=cat)
            return
        if not options:
            for cat, sc0 in mem.alt.items():
                options += options_for(cat, sc0, done_only=True, min_sim=ALT_MIN_SIM)
        if options:
            _, t, cat = max(options, key=lambda o: o[0])
            take(t, t.sim(), False, cat=cat)
        elif parts:
            t = max(parts, key=lambda t: float(np.mean(t.part_sims)))
            take(t, float(np.mean(t.part_sims)), True)
        else:
            c = colour_only_candidate(mem, step)
            if c is not None:
                t, _ = c
                take(t, t.sim(), False)
                mem.colour_only = True

    def _back_off(self, arm, q_cur, n: int = 8) -> list:
        t6 = K.fk(arm, q_cur)
        rot, tcp = t6[:3, :3], np.asarray(K.point_of(t6, K.TCP_OFFSET), dtype=float)
        out, qp = [], q_cur
        for k in range(1, n + 1):
            r = K.solve(arm, pose(rot, tcp + np.array([0.0, 0.0, BACK_OFF_DZ * k / n])), qp, K.TCP_OFFSET, iters=40)
            if not r.ok:
                break
            out.append(r.q)
            qp = r.q
        return out

    def _part_fit(self, ctx, cam, fg, img, b, t, mem: Memory, step: int):
        sub = CV.colour_part(img, b, mem.hist0)
        if sub is None or sub.area > 0.9 * b.area:
            return
        pf = max(
            (CV.fit_shape(cam, fg, sub, s, symmetric=s.key in CV.SYMMETRIC) for s in O.shapes(mem.category)),
            key=lambda f: f.score,
        )
        ps = CV.hist_similarity(mem.hist0, CV.colour_hist(img, sub.mask))
        if pf.score >= GOOD_FIT and ps >= PART_SIM:
            t.part_fits.append((step, pf))
            t.part_sims.append(ps)

    def _motion(self, mem: Memory):
        fits = [(s, f) for s, f in mem.target.fits if f.category == mem.category and f.score > 0.7]
        if not fits:
            return None
        all_fits = fits
        top = max(f.score for _, f in fits)
        fits = [(s, f) for s, f in fits if f.score >= top - 0.08]
        votes: dict = {}
        for _, f in fits:
            votes[f.shape.key] = votes.get(f.shape.key, 0.0) + f.score
        key = max(votes, key=votes.get)
        key = steady_pose(fits, votes, key, V_PRIOR)
        if len(votes) > 1 and view_locked(all_fits, key):
            alt = [k for k in votes if k != key and (not view_locked(all_fits, k))]
            if alt:
                key = max(alt, key=votes.get)
                mem.view_locked = True
        sw = getattr(mem, "pose_switch", None)
        if sw is not None:
            sw_fits = [(s, f) for s, f in all_fits if f.shape.key == sw[0] and s >= sw[1]]
            if sw_fits:
                key, fits = sw[0], sw_fits
        best = max((f for _, f in fits if f.shape.key == key), key=lambda f: f.score)
        fits = [(s, f) for s, f in fits if f.shape.key == key]
        steps = np.array([s for s, _ in fits], dtype=float)
        xs = np.array([f.x for _, f in fits])
        span = float(steps.max() - steps.min()) if len(fits) else 0.0
        v_own = None
        keep = np.ones(len(xs), bool)
        if len(fits) >= 3 and span >= 40:
            for _ in range(2):
                v_fit, x0_fit = np.polyfit(steps[keep], xs[keep], 1)
                res = np.abs(xs - (x0_fit + v_fit * steps))
                keep = res <= max(0.006, 2.5 * float(np.median(res[keep])))
                if keep.sum() < 3:
                    keep[:] = True
                    break
            v_own = float(np.polyfit(steps[keep], xs[keep], 1)[0])
        v = belt_speed_estimate(belt_speed(mem), v_own)
        x0 = float(np.median(xs[keep] - v * steps[keep]))
        y = float(np.median([f.y for _, f in fits]))
        ys = np.array([f.y for _, f in fits])
        res = xs[keep] - v * steps[keep]
        self._spread = float(np.median(np.abs(res - np.median(res)))), float(np.median(np.abs(ys - np.median(ys))))
        return float(x0), float(v), y, best.yaw, best.shape, len(fits), float(steps.max() - steps.min())

    def _plan(self, ctx, mem: Memory, q):
        m = self._motion(mem)
        if m is None:
            return None
        x0, v, y, yaw, shape, nfit, span = m
        step = ctx.obs.step
        x_now = x0 + v * step
        if x_now > GONE_X and (not mem.gone):
            mem.gone = True
        if (nfit < 2 or span < 10) and x_now < -0.15:
            return None
        return self._plan_search(ctx, mem, q, x0, v, y, yaw, shape, nfit, step)

    def _plan_search(self, ctx, mem: Memory, q, x0, v, y, yaw, shape, nfit, step):
        bx, by = K.BASE_POS["right"][:2]
        sym = shape.key in CV.SYMMETRIC
        key = shape.key, 0 if sym else round(math.degrees(yaw) / 10.0), len(mem.blocked)
        y_tol = SEARCH_Y_TOL_FALLBACK if mem.no_stable else SEARCH_Y_TOL
        if mem.search_key != key or (y_tol and abs(y - mem.search_y) > y_tol):
            mem.search_y = y
            to_base = np.array([bx - 0.25, by - y])
            pts_obj = shape.posed(0.0, 0.0, 0.0 if sym else yaw, 0.0)
            mem.search, rot_off = default_search(pts_obj, to_base, sym, float(shape.extent[2]))
            mem.no_stable = not mem.search
            kind = "default" if rot_off is None else f"rotated{rot_off:+.0f}"
            if mem.blocked and mem.search:
                away = SG.search(
                    pts_obj,
                    to_base,
                    symmetric=sym,
                    per_tilt=BLOCKED_PER_TILT,
                    max_keep=2 * BLOCKED_PER_TILT,
                    exclude=lambda g: not all(away_from_blocked(g, b) for b in mem.blocked),
                )
                if away:
                    mem.search = away + mem.search
                    kind = "away"
            if not mem.search:
                extra = {"antipodal_check": False, "partial": True}
                mem.search = SG.search(pts_obj, to_base, symmetric=sym, normal_check=False, **extra)
                kind = "fallback"
                if sym and float(shape.extent[2]) <= PINCH_MAX_H:
                    mem.search = dish_downstream_first(mem.search)
            else:
                com = pts_obj.mean(axis=0)
                d0 = SG.hang_drop(mem.search[0].rot, mem.search[0].tcp, com)
                if O.mass(shape.category) * d0 > HANG_MOMENT:
                    n_tilt = 2 * HANG_PER_TILT
                    excl = None
                    if mem.blocked:
                        excl = lambda g: not all(away_from_blocked(g, b) for b in mem.blocked)
                    align = not sym
                    alt = SG.search(
                        pts_obj,
                        to_base,
                        symmetric=sym,
                        antipodal_check=False,
                        swing_w=HANG_SWING_W,
                        per_tilt=n_tilt,
                        max_keep=2 * n_tilt,
                        diverse_dz=HANG_Z_BAND - 0.0001,
                        exclude=excl,
                        heading_offset=yaw if align else 0.0,
                    )
                    alt = [g for g in alt if SG.hang_drop(g.rot, g.tcp, com) < 0.5 * d0]
                    alt.sort(
                        key=lambda g: (
                            SG.swing_angle(g.rot, g.tcp, com) > SG.SWING_FREE,
                            abs(g.tilt) > 0.001,
                            round(float(g.tcp[2]) / HANG_Z_BAND),
                            align and axis_offset(g.heading, yaw) > HANG_ALIGN_TOL,
                            -g.score,
                        )
                    )
                    if alt:
                        mem.search = alt + mem.search
                        kind = "hang"
            mem.search_key = key
            mem.search_kind = kind
            mem.failed = set()
            mem.prefer = []
            mem.rotated = False
        if not mem.search:
            return None
        if sym and mem.failed and (not mem.rotated):
            mem.rotated = True
            first = len(mem.search)
            for ci in sorted(mem.failed):
                g = mem.search[ci]
                if abs(g.tilt) > 0.001:
                    continue
                for d in RETRY_TURNS:
                    rz = K.axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(d))
                    mem.search.append(replace(g, rot=rz @ g.rot, tcp=rz @ g.tcp, heading=g.heading + math.radians(d)))
            mem.prefer = list(range(first, len(mem.search)))
        if mem.blocked:
            for ci, g in enumerate(mem.search):
                if any(like_blocked(g, b) for b in mem.blocked) or not all(
                    away_from_blocked(g, b) for b in mem.blocked
                ):
                    mem.failed.add(ci)
        combos = (
            [(ci, xg) for ci in mem.prefer for xg in GRASP_X if ci not in mem.failed]
            + [
                (ci, xg)
                for ci in range(len(mem.search))
                for xg in GRASP_X
                if ci not in mem.failed and ci not in mem.prefer
            ]
            + [(ci, xg) for ci in range(len(mem.search)) for xg in GRASP_X if ci in mem.failed]
        )
        hits = set()
        for ci in {c for c, _ in combos}:
            g = mem.search[ci]
            xg = GRASP_X[0]
            t_c = int(math.ceil((xg - g.tcp[0] - x0) / v))
            appr = np.array([0.0, 0.0, 1.0]) if g.approach == "up" else -g.rot[:, 0]
            tcp_w = np.array([x0 + v * t_c + g.tcp[0], y + g.tcp[1], CV.BELT_Z + g.tcp[2]])
            if neighbour_hit(mem, v, t_c, g.rot, tcp_w, appr, SG.grip_for_gap(g.gap), min_score=NEIGHBOUR_PREFER_SCORE):
                hits.add(ci)
        if mem.last_combo in combos and mem.last_combo[0] not in hits:
            combos.remove(mem.last_combo)
            combos.insert(0, mem.last_combo)
        elif hits and combos and (combos[0][0] in hits):
            r0 = mem.search[combos[0][0]].robust
            alt = next(
                (
                    c
                    for c in combos
                    if c[0] not in hits
                    and c[0] not in mem.failed
                    and (mem.search[c[0]].robust >= r0 - NEIGHBOUR_PREFER_ROBUST)
                ),
                None,
            )
            if alt is not None:
                combos = [c for c in combos if c[0] == alt[0]] + [c for c in combos if c[0] != alt[0]]
        roll = getattr(mem, "search_kind", None) == "hang"
        parked = float(np.max(np.abs(np.asarray(q["right"], dtype=float) - PARK_Q["right"]))) < 0.05
        nbs = transit_neighbours(mem, v, x0, y) if parked else []
        first = None
        best_fb = None
        tried = 0
        low_ok = roll
        for ci, xg, low in with_low_hover(combos, low_ok):
            if first is not None and tried >= TRANSIT_MAX_TRY:
                break
            g = mem.search[ci]
            t_close = int(math.ceil((xg - g.tcp[0] - x0) / v))
            appr = np.array([0.0, 0.0, 1.0]) if g.approach == "up" else -g.rot[:, 0]
            to_base = np.array([bx - xg, by - y])
            to_base /= max(1e-09, float(np.linalg.norm(to_base)))
            tcp_w = np.array([x0 + v * t_close + g.tcp[0], y + g.tcp[1], CV.BELT_Z + g.tcp[2]])
            top0 = float(shape.extent[2])
            top = top0
            if getattr(mem, "pose_switch", None) is None:
                top = pose_hover_top(shape)
            top_nb = neighbour_top(mem, v, t_close, tcp_w[:2])
            pl = None
            if top_nb > top:
                pl = self._intercept(
                    "right",
                    q["right"],
                    g.rot,
                    x0,
                    v,
                    y + g.tcp[1],
                    g.tcp[0],
                    CV.BELT_Z + g.tcp[2],
                    t_close,
                    step,
                    back=to_base,
                    approach=appr,
                    g_open=SG.grip_for_gap(g.gap),
                    top=top_nb,
                    roll=roll,
                )
            if pl is None:
                top_nb = 0.0
                pl = self._intercept(
                    "right",
                    q["right"],
                    g.rot,
                    x0,
                    v,
                    y + g.tcp[1],
                    g.tcp[0],
                    CV.BELT_Z + g.tcp[2],
                    t_close,
                    step,
                    back=to_base,
                    approach=appr,
                    g_open=SG.grip_for_gap(g.gap),
                    top=top,
                    roll=roll,
                    low_hover=low,
                )
            if pl is None and top > top0:
                top = top0
                pl = self._intercept(
                    "right",
                    q["right"],
                    g.rot,
                    x0,
                    v,
                    y + g.tcp[1],
                    g.tcp[0],
                    CV.BELT_Z + g.tcp[2],
                    t_close,
                    step,
                    back=to_base,
                    approach=appr,
                    g_open=SG.grip_for_gap(g.gap),
                    top=top,
                    roll=roll,
                    low_hover=low,
                )
            hit, side_dy, fold = None, 0.0, False
            if pl is not None and nbs:
                t_until = pl["wrist"]["t_arr"]
                hit = transit_hit("right", pl["steps"], step, t_until, nbs, v, SG.grip_for_gap(g.gap))
                if hit is not None:
                    tried_here = []
                    c0 = transit_clearance("right", pl["steps"], step, t_until, nbs, v, SG.grip_for_gap(g.gap))[0]
                    tried_here.append((c0, (pl, ci, g, t_close, 0.0, False)))
                    mirrors = [ci] + [
                        k for k, gk in enumerate(mem.search) if k != ci and k not in mem.failed and is_mirror(gk, g)
                    ]
                    variants = [(cj, False, dy) for cj in mirrors for dy in SIDE_DY]
                    variants += [(cj, True, dy) for cj in mirrors for dy in (0.0,) + tuple(SIDE_DY)]
                    found = None
                    for cj, fv, dy in variants:
                        gc = mem.search[cj]
                        tc = int(math.ceil((xg - gc.tcp[0] - x0) / v))
                        if dy and y + gc.tcp[1] - dy < SIDE_MIN_Y:
                            continue
                        ps = self._intercept(
                            "right",
                            q["right"],
                            gc.rot,
                            x0,
                            v,
                            y + gc.tcp[1],
                            gc.tcp[0],
                            CV.BELT_Z + gc.tcp[2],
                            tc,
                            step,
                            back=to_base,
                            approach=appr,
                            g_open=SG.grip_for_gap(gc.gap),
                            top=top_nb if top_nb else top,
                            side=np.array([0.0, -dy]) if dy else None,
                            fold=fv,
                            roll=roll,
                            low_hover=low,
                        )
                        if ps is None:
                            continue
                        if (
                            not fv
                            and transit_hit(
                                "right", ps["steps"], step, ps["wrist"]["t_arr"], nbs, v, SG.grip_for_gap(gc.gap)
                            )
                            is None
                        ):
                            found = ps, cj, gc, tc, dy, fv
                            break
                        c = transit_clearance(
                            "right", ps["steps"], step, ps["wrist"]["t_arr"], nbs, v, SG.grip_for_gap(gc.gap)
                        )[0]
                        if fv and c > TRANSIT_MARGIN:
                            found = ps, cj, gc, tc, dy, fv
                            break
                        tried_here.append((c, (ps, cj, gc, tc, dy, fv)))
                    if found is not None:
                        pl, ci, g, t_close, side_dy, fold = found
                    else:
                        tried += 1
                        for c, fields in tried_here:
                            if best_fb is None or c > best_fb[0] + TRANSIT_BEST_GAIN:
                                best_fb = c, fields + (xg, top_nb, hit)
                        if first is None:
                            first = pl, ci, xg, g, t_close, top_nb, hit
                        continue
            if pl is not None:
                clr = None
                if nbs and hit is None and (not side_dy) and (not fold):
                    clr = transit_clearance(
                        "right", pl["steps"], step, pl["wrist"]["t_arr"], nbs, v, SG.grip_for_gap(g.gap)
                    )[0]
                return self._commit_plan(
                    ctx,
                    mem,
                    pl,
                    "right",
                    ci,
                    xg,
                    g,
                    t_close,
                    x0,
                    v,
                    y,
                    yaw,
                    shape,
                    nfit,
                    step,
                    sym,
                    hits,
                    top_nb,
                    hit,
                    side_dy,
                    fold=fold,
                    clearance=clr,
                )
        if best_fb is not None:
            c, (pl, ci, g, t_close, side_dy, fold, xg, top_nb, hit) = best_fb
            return self._commit_plan(
                ctx,
                mem,
                pl,
                "right",
                ci,
                xg,
                g,
                t_close,
                x0,
                v,
                y,
                yaw,
                shape,
                nfit,
                step,
                sym,
                hits,
                top_nb,
                hit,
                side_dy,
                fold=fold,
                clearance=c,
            )
        if first is not None:
            pl, ci, xg, g, t_close, top_nb, hit = first
            return self._commit_plan(
                ctx, mem, pl, "right", ci, xg, g, t_close, x0, v, y, yaw, shape, nfit, step, sym, hits, top_nb, hit, 0.0
            )

    def _commit_plan(
        self,
        ctx,
        mem: Memory,
        pl,
        arm,
        ci,
        xg,
        g,
        t_close,
        x0,
        v,
        y,
        yaw,
        shape,
        nfit,
        step,
        sym,
        hits,
        top_nb,
        hit,
        side_dy,
        fold: bool = False,
        clearance=None,
    ):
        mem.last_combo = ci, xg
        spread = getattr(self, "_spread", (0.0, 0.0))
        dark = float(np.median(mem.target.dark)) if mem.target is not None and mem.target.dark else 0.0
        need = max(spread) > WRIST_GATE or dark >= DARK_GATE or mem.retry or mem.view_locked
        alts = []
        if getattr(mem, "pose_switch", None) is None:
            alts = [
                s_
                for s_ in O.shapes(shape.category)
                if s_.key != shape.key and abs(float(s_.extent[2]) - float(shape.extent[2])) > POSE_CHECK_DH
            ]
        pl["wrist"].update(
            shape=shape,
            x0=x0,
            v=v,
            y=y,
            yaw=yaw if not sym else 0.0,
            sym=sym,
            need=need,
            spread=spread,
            dark=dark,
            alt_shapes=alts,
        )
        pl["arm"] = arm
        return pl

    def _intercept(
        self,
        arm,
        q_cur,
        rot,
        x0,
        v,
        y,
        dx,
        z_g,
        t_close,
        step,
        back=None,
        approach=(0.0, 0.0, 1.0),
        g_open: float = 1.0,
        top: float | None = None,
        side=None,
        fold: bool = False,
        roll: bool = False,
        low_hover: bool = False,
    ):
        t_desc = t_close - DESCEND_STEPS
        t_end = t_close + CLOSE_STEPS + LIFT_STEPS
        t_need = t_close + CLOSE_STEPS + int(math.ceil(LIFT_STEPS * LIFT_NEED / LIFT_DZ))
        back = np.zeros(2) if back is None else np.asarray(back, dtype=float)
        appr = np.asarray(approach, dtype=float)
        hover_len = HOVER_DZ
        if top is not None and appr[2] > 0.9:
            hover_len = max(HOVER_DZ, CV.BELT_Z + top + HOVER_TOP_CLEAR - z_g)
        side_v = None if side is None else np.asarray(side, dtype=float)[:2]
        t_arr0 = t_desc - 2

        def target(t):
            xo = x0 + v * min(t, t_close + CLOSE_STEPS + FOLLOW_LIFT) + dx
            yo = y
            if t <= t_close:
                s = 0.0 if t <= t_desc else (t - t_desc) / DESCEND_STEPS
                a = hover_len * (1 - s) ** 1.5
                p = np.array([xo + appr[0] * a, yo + appr[1] * a, z_g + appr[2] * a])
                if side_v is not None and t < t_arr0:
                    u = min(1.0, (t_arr0 - t) / SIDE_STEPS)
                    p[:2] += side_v * (u * u * (3.0 - 2.0 * u))
                return p
            elif t <= t_close + CLOSE_STEPS:
                z = z_g
            else:
                s = min(1.0, (t - t_close - CLOSE_STEPS) / LIFT_STEPS)
                z = z_g + LIFT_DZ * s
                xo += LIFT_BACK * s * back[0]
                yo += LIFT_BACK * s * back[1]
            return np.array([xo, yo, z])

        def grip(t):
            return g_open if t <= t_close else 0.0

        t_arr = t_desc - 2
        seeds = [G.base_seed(arm, target(t_arr)[:2]) for a in (0.0, math.pi / 2, -math.pi / 2)]
        seeds += [sd + np.array([0, 0, 0, 0, 0, d]) for sd in seeds[:1] for d in (math.pi, -math.pi)]
        rg = None
        for s in seeds:
            rg = K.solve(arm, pose(rot, target(t_close)), s, K.TCP_OFFSET, iters=60)
            if rg.ok:
                break
        if rg is None or not rg.ok:
            return None
        r = K.solve(arm, pose(rot, target(t_arr)), rg.q, K.TCP_OFFSET, iters=60)
        if not r.ok and low_hover and (top is not None) and (appr[2] > 0.9):
            for clear in HOVER_TOP_CLEARS_LOW:
                hover_len = max(HOVER_DZ_LOW, CV.BELT_Z + top + clear - z_g)
                r = K.solve(arm, pose(rot, target(t_arr)), rg.q, K.TCP_OFFSET, iters=60)
                if r.ok:
                    break
        if not r.ok:
            return None
        t_j = t_arr
        if side_v is not None:
            t_j = t_arr - SIDE_STEPS
            r = K.solve(arm, pose(rot, target(t_j)), r.q, K.TCP_OFFSET, iters=60)
            if not r.ok:
                return None
        if fold:
            qv = np.array(q_cur, dtype=float).copy()
            qv[0], qv[5] = r.q[0], r.q[5]
            path = list(motion.joint_move(q_cur, qv, joint_step=0.07).q) + list(
                motion.joint_move(qv, r.q, joint_step=0.07).q
            )
        else:
            path = list(motion.joint_move(q_cur, r.q, joint_step=0.07).q)
        t_start = t_j - len(path)
        if t_start < step + 1:
            return None
        out = [(None, None)] * (t_start - step)
        out += [(qq, g_open) for qq in path]
        t_tilt = t_close + CLOSE_STEPS
        track = track_steps(arm, rot, target, grip, r.q, t_j, t_end, t_need, back, t_tilt=t_tilt, roll=roll)
        if track is None:
            return None
        out += track
        return {
            "steps": out,
            "t_start": t_start,
            "step0": step,
            "t_close": t_close,
            "wrist": {
                "t_arr": t_arr,
                "target": target,
                "grip": grip,
                "t_need": t_need,
                "rot": rot,
                "t_end": t_end,
                "back": back,
                "t_tilt": t_tilt,
                "roll": roll,
                "hover_len": hover_len,
            },
        }

    def _wrist_refine(self, ctx, mem: Memory, plan: dict, arm: str, idx: int, q_cmd) -> None:
        w = plan["wrist"]
        if "shape" not in w:
            return
        step = ctx.obs.step
        alts = w.get("alt_shapes") or []
        q_obs = np.asarray(ctx.obs.state[f"{arm}_arm_joint_state"], dtype=float)
        mem.n_hover += 1
        if not w.get("need", True) and (not alts):
            return
        T6 = K.fk(arm, q_obs)
        grip_now = float(ctx.obs.state[f"{arm}_ee_joint_state"][0])
        x_p = w["x0"] + w["v"] * step
        nb = neighbour_points(mem, w["v"], step, x_p)
        try:
            r = WF.refine(
                ctx.obs.images[f"cam_{arm}_wrist"],
                T6,
                grip_now,
                w["shape"],
                x_p,
                w["y"],
                w["yaw"],
                CV.BELT_Z,
                symmetric=w["sym"],
                others=[nb] if nb is not None else [],
            )
        except Exception:
            return
        if r is None:
            if alts and self._pose_check(ctx, mem, w, arm, T6, grip_now, x_p, nb, 0.0) == "pose":
                return "pose"
            return
        score, xr, yr, _ = r
        dx, dy = float(xr - x_p), float(yr - w["y"])
        ok = score >= WRIST_MIN_IOU and math.hypot(dx, dy) <= WRIST_MAX_SHIFT and w.get("need", True)
        if alts and self._pose_check(ctx, mem, w, arm, T6, grip_now, x_p, nb, score) == "pose":
            return "pose"
        if not ok or math.hypot(dx, dy) < 0.002:
            return
        steps = plan["steps"]
        s0 = plan["step0"]
        k_from = s0 + idx

        def shifted(t, base=w["target"]):
            b = min(1.0, (t - k_from) / WRIST_BLEND)
            return base(t) + np.array([b * dx, b * dy, 0.0])

        new = track_steps(
            arm,
            w["rot"],
            shifted,
            w["grip"],
            np.asarray(q_cmd, dtype=float),
            k_from,
            s0 + len(steps),
            w["t_need"],
            w["back"],
            t_tilt=w["t_tilt"],
            roll=w.get("roll", False),
        )
        if (new is None or len(new) != len(steps) - idx) and "t_close" in plan:
            faded = fade_shift(
                w["target"], dx, dy, k_from, plan["t_close"] + CLOSE_STEPS, w["t_need"] - WRIST_FADE_LEAD
            )
            new = track_steps(
                arm,
                w["rot"],
                faded,
                w["grip"],
                np.asarray(q_cmd, dtype=float),
                k_from,
                s0 + len(steps),
                w["t_need"],
                w["back"],
                t_tilt=w["t_tilt"],
                roll=w.get("roll", False),
            )
            if new is not None and len(new) == len(steps) - idx:
                shifted = faded
        if new is None or len(new) != len(steps) - idx:
            return
        steps[idx:] = new
        w["target_now"] = shifted

    def _nudge(self, ctx, mem: Memory, plan: dict, arm: str, idx: int, q_cmd, dz: float, done: list):
        w = plan.get("wrist")
        if w is None or mem.last_combo is None or mem.last_combo[0] >= len(mem.search) or ("shape" not in w):
            return None
        g = mem.search[mem.last_combo[0]]
        rot = np.asarray(w["rot"], dtype=float)
        if g.approach != "up" or float(rot[2, 0]) > -0.97:
            return None
        step = ctx.obs.step
        base = w.get("target_now", w["target"])
        grip = w.get("grip_now", w["grip"])
        g_open = float(grip(step))
        total = float(w.get("nudge_total", 0.0))
        pts = w["shape"].posed(0.0, 0.0, w["yaw"], 0.0, n=600)
        obs = list(done) + [(total, dz)]
        move = nudge_next(pts, rot, g.tcp, g_open, obs)
        if move is None:
            return None
        mv = (move - total) * closing_axis_xy(rot)
        tn, gn = nudge_target(base, grip, step, w["v"], dz + NUDGE_LIFT, mv)
        d = NUDGE_STEPS * (len(done) + 1)
        steps = plan["steps"]
        s0 = plan["step0"]
        new = track_steps(
            arm,
            rot,
            tn,
            gn,
            np.asarray(q_cmd, dtype=float),
            s0 + idx,
            s0 + len(steps),
            w["t_need"] + d,
            w["back"],
            t_tilt=w["t_tilt"] + d,
            roll=w.get("roll", False),
        )
        if new is None or len(new) != len(steps) - idx:
            return None
        w["target_now"], w["grip_now"], w["nudge_total"] = tn, gn, move
        done.append((total, dz))
        return new

    def _pose_check(self, ctx, mem: Memory, w: dict, arm: str, T6, grip: float, x_p: float, nb, s_plan: float):
        step = ctx.obs.step
        img = ctx.obs.images[f"cam_{arm}_wrist"]
        best = None
        for sh in w["alt_shapes"]:
            sx, sy, syaw = x_p, w["y"], w["yaw"]
            prev = [
                (s_, f)
                for s_, f in (mem.target.fits if mem.target is not None else [])
                if f.shape.key == sh.key and f.score > 0.7
            ]
            if prev:
                s_, f = prev[-1]
                fx = f.x + w["v"] * (step - s_)
                if math.hypot(fx - x_p, f.y - w["y"]) < POSE_CHECK_START_R:
                    sx, sy, syaw = fx, f.y, f.yaw
            try:
                ra = WF.refine(
                    img,
                    T6,
                    grip,
                    sh,
                    sx,
                    sy,
                    syaw,
                    CV.BELT_Z,
                    symmetric=sh.key in CV.SYMMETRIC,
                    others=[nb] if nb is not None else [],
                )
            except Exception:
                continue
            if ra is not None and (best is None or ra[0] > best[1][0]):
                best = sh, ra
        if best is None:
            return None
        sh, (sa, xa, ya, aa) = best
        switch = sa >= POSE_CHECK_IOU and sa >= s_plan + POSE_CHECK_GAIN and (mem.target is not None)
        if not switch:
            return None
        mem.target.fits.append((step, CV.Fit(sh, float(xa), float(ya), float(aa), POSE_FIT_SCORE)))
        mem.pose_switch = sh.key, step
        return "pose"


PIECE_PAD = 40


def object_pieces(bl: list) -> list:
    if len(bl) < 2:
        return list(bl)
    main = max(bl, key=lambda b: b.area)
    u0, v0, u1, v1 = main.bbox
    return [
        b
        for b in bl
        if b is main
        or not (
            b.bbox[0] > u1 + PIECE_PAD
            or b.bbox[2] < u0 - PIECE_PAD
            or b.bbox[1] > v1 + PIECE_PAD
            or (b.bbox[3] < v0 - PIECE_PAD)
        )
    ]


def fade_shift(target, dx: float, dy: float, k_from: int, t_fade0: int, t_fade1: int):
    t_fade1 = max(t_fade0 + 1, t_fade1)

    def shifted(t):
        b = min(1.0, (t - k_from) / WRIST_BLEND) * min(1.0, max(0.0, (t_fade1 - t) / (t_fade1 - t_fade0)))
        return target(t) + np.array([b * dx, b * dy, 0.0])

    return shifted


def track_steps(
    arm,
    rot,
    target,
    grip,
    q_start,
    k_from: int,
    k_to: int,
    t_need: int,
    back,
    t_tilt: int | None = None,
    roll: bool = False,
) -> list | None:
    t_tilt = t_need if t_tilt is None else t_tilt
    out = []
    qprev = np.asarray(q_start, dtype=float)
    rot_k, extra, sgn = rot, 0.0, None
    extra_r, sgn_r = 0.0, None
    back = np.zeros(2) if back is None else np.asarray(back, dtype=float)[:2]
    for k in range(k_from, k_to):
        rr = K.solve(arm, pose(rot_k, target(k + 1)), qprev, K.TCP_OFFSET, iters=40)
        ok = rr.ok and float(np.max(np.abs(rr.q - qprev))) <= 0.25
        if not ok and extra_r == 0.0 and (k + 1 > t_tilt) and (extra + LIFT_TILT_STEP <= LIFT_TILT_MAX + 1e-09):
            for s_ in (sgn,) if sgn is not None else (1.0, -1.0):
                cand = K.axis_angle(rot[:, 1], s_ * (extra + LIFT_TILT_STEP)) @ rot
                if float(np.dot(-cand[:2, 0], back)) <= float(np.dot(-rot_k[:2, 0], back)):
                    continue
                r2 = K.solve(arm, pose(cand, target(k + 1)), qprev, K.TCP_OFFSET, iters=40)
                if r2.ok and float(np.max(np.abs(r2.q - qprev))) <= 0.25:
                    rr, ok, rot_k, sgn = r2, True, cand, s_
                    extra += LIFT_TILT_STEP
                    break
        if not ok and roll and (k + 1 > t_tilt) and (extra_r + LIFT_TILT_STEP <= LIFT_ROLL_MAX + 1e-09):
            for s_ in (sgn_r,) if sgn_r is not None else (1.0, -1.0):
                cand = K.axis_angle(rot_k[:, 2], s_ * LIFT_TILT_STEP) @ rot_k
                if float(np.dot(-cand[:2, 0], back)) <= float(np.dot(-rot_k[:2, 0], back)):
                    continue
                r2 = K.solve(arm, pose(cand, target(k + 1)), qprev, K.TCP_OFFSET, iters=40)
                if r2.ok and float(np.max(np.abs(r2.q - qprev))) <= 0.25:
                    rr, ok, rot_k, sgn_r = r2, True, cand, s_
                    extra_r += LIFT_TILT_STEP
                    break
        if not ok:
            if k + 1 > t_need:
                out.append((qprev, grip(k + 1)))
                continue
            return None
        out.append((rr.q, grip(k + 1)))
        qprev = rr.q
    return out


def rest_height(pts_obj: np.ndarray, rot: np.ndarray, tcp, grip: float) -> float:
    p = (np.asarray(pts_obj, dtype=float) - np.asarray(tcp, dtype=float)) @ np.asarray(rot, dtype=float)
    p[:, 0] += K.TCP_OFFSET
    m = SG.COLLIDE_MARGIN
    h = 0.0
    for x0, x1, y0, y1, z0, z1 in SG.gripper_boxes(grip):
        sel = (p[:, 1] >= y0 - m) & (p[:, 1] <= y1 + m) & (p[:, 2] >= z0 - m) & (p[:, 2] <= z1 + m)
        if sel.any():
            h = max(h, float((x1 + m - p[sel, 0]).max()))
    return h


def nudge_next(pts_obj: np.ndarray, rot: np.ndarray, tcp, grip: float, obs: list) -> float | None:
    u = np.array([rot[0, 1], rot[1, 1], 0.0])
    n = float(np.linalg.norm(u))
    if n < 0.5:
        return None
    u /= n
    tcp = np.asarray(tcp, dtype=float)
    es = np.arange(-NUDGE_SPAN, NUDGE_SPAN + 1e-09, NUDGE_RES)
    rest = {}

    def rest_at(d):
        k = int(round(d / (0.5 * NUDGE_RES)))
        if k not in rest:
            rest[k] = rest_height(pts_obj, rot, tcp + k * 0.5 * NUDGE_RES * u, grip)
        return rest[k]

    tried = [m_k for m_k, _ in obs]

    def best(sub):
        ok = []
        for e in es:
            good = True
            for m_k, dz_k in sub:
                r = [rest_at(e + m_k + d) for d in (-0.5 * NUDGE_RES, 0.0, 0.5 * NUDGE_RES)]
                if not min(r) - NUDGE_TOL <= dz_k <= max(r) + NUDGE_TOL:
                    good = False
                    break
            ok.append(good)
        runs, cur = [], []
        for j in range(len(es)):
            if ok[j]:
                cur.append(float(es[j]))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        cands = []
        for run in runs:
            move = -0.5 * (run[0] + run[-1])
            if all(abs(move - t) >= NUDGE_MIN for t in tried):
                cands.append((min(abs(run[0]), abs(run[-1])), move))
        cands.sort(key=lambda c: c[0])
        return cands[0][1] if cands else None

    m = best(obs)
    if m is None and len(obs) > 1:
        m = best(obs[:1])
    return m


def closing_axis_xy(rot: np.ndarray) -> np.ndarray:
    u = np.array([rot[0, 1], rot[1, 1]], dtype=float)
    return u / max(1e-09, float(np.linalg.norm(u)))


def nudge_target(base, grip, t0: int, v: float, lift: float, move, steps: int = NUDGE_STEPS):
    p0 = np.asarray(base(t0), dtype=float).copy()
    mv = np.array([float(move[0]), float(move[1]), 0.0])
    up = max(1, steps // 4)
    across = max(1, steps // 4)
    down = max(1, steps - up - across)

    def target(t):
        k = t - t0
        if k <= 0:
            return base(t)
        if k <= steps:
            if k <= up:
                z, s = lift * k / up, 0.0
            elif k <= up + across:
                z, s = lift, (k - up) / across
            else:
                z, s = lift * (1.0 - (k - up - across) / down), 1.0
            return p0 + np.array([v * k, 0.0, z]) + s * mv
        return np.asarray(base(t - steps), dtype=float) + np.array([v * steps, 0.0, 0.0]) + mv

    def grip_n(t):
        return grip(t) if t <= t0 else grip(max(t0, t - steps))

    return target, grip_n


def merge_columns(bl: list, min_overlap: float = 0.6) -> list:
    out = list(bl)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                ov = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
                narrow = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
                if narrow > 0 and ov >= min_overlap * narrow:
                    out[i] = CV.merge_blobs([a, b])
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def pose(rot, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = p
    return t
