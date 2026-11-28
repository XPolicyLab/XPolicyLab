from __future__ import annotations

from ... import ASSETS

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from . import dolls as D
from ...perception import seg
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, Zone, link_arms, run_arms, tcp_pose
from .. import Program, register

LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
STEP_LIMIT = 1050
HOME_STEPS = 45
JOB_STEPS = 95
GAP_MAX = 0.0888
PAD_DOWN, PAD_UP = 0.013, 0.022
PALM_CLEAR = 0.054
SIDE_MARGIN = 0.011
WIDE_MARGIN = 0.007
GRIP_TOL = 0.0015
GRIP_SPAN_MIN = 0.008
GRIP_SPAN = 0.012
OPEN_EXTRA = 0.03
FINGER_T = 0.01
FINGER_W = 0.015
ROW_GAP = 0.025
ROW_YS = tuple(np.round(np.arange(-0.17, -0.0799, 0.01), 3))
ROW_Y = -0.13
ROW_REACH_MARGIN = 0.02
CLUTTER_H = 0.06
LIFT_CLEAR = 0.02
TOP_CLEAR = 0.028
PLACE_DROP = 0.003
DONE_TOL = 0.012
VERIFY_PASSES = 1
VERIFY_TOL = 0.015
VERIFY_WAIT = 10
VERIFY_MIN_STEPS = HOME_STEPS + JOB_STEPS + 60 + VERIFY_WAIT
RELAY_Y = -0.29
ZONE_R = 0.1
SIDE_XY = {"left": (-0.27, -0.24), "right": (0.27, -0.24)}
SIDE_Z = 0.19
SIDE_X = 0.07
SIDE_X_DIRECT = 0.16
DEG = math.pi / 180
PICK_PSIS = tuple(a * DEG for a in (0, 180, 30, -30, 150, -150, 60, -60, 120, -120, 90, -90))
DUCK_PSIS = tuple(a * DEG for a in (0, 180, 20, -20, 160, -160))
PLACE_PSIS = tuple(a * DEG for a in (90, -90, 70, -70, 110, -110))
FREE_STEP = 0.1
CARRY_LIN = 0.015
DUCK_GROUP = 2
GROUP_DEV_BAD = 0.0045
GROUP_DEV_OK = 0.0025
GROUP_VOTES = 2
HOUSING_DZ = 0.062
HOUSING_R = 0.09
HOUSING_NARROW = 0.045
SPOT_HOUSING_R = 0.07
ARM_CLEAR = 0.008
NEAR_CLEAR = 0.002
MAX_RELAYS = 2
SPOT_X = 0.06
ASIDE_STEP = 0.03
MAX_ASIDE = 2
SELF_RELAY_AFTER = (
    2
)
CROSS_MARGIN = 0.015
CROSS_WAIT = 40
RECHECK_MARGIN = 0.004


def carried_hits(board, d, a, b, zb: float) -> bool:
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    for e in board.dolls.values():
        if e is d or e.h + LIFT_CLEAR <= zb + 1e-06:
            continue
        need = d.R + e.R + RECHECK_MARGIN
        for c in [e.xy]:
            c = np.asarray(c, float)[:2]
            if np.linalg.norm(c - a) < need or np.linalg.norm(c - b) < need:
                continue
            if seg_dist(c, a, b) < need:
                return True
    return False


SPOT_X_STUCK = 0.12
ACCESS_BEHIND = 0.02
ACCESS_FAILS = 2
TILT_PICK = 10 * DEG, 15 * DEG
SLOT_SPOT_DZ = 0.02
SPOT_YS_FRONT = tuple(np.arange(-0.32, -0.179, 0.015))
SPOT_YS_ALL = tuple(np.arange(-0.32, -0.039, 0.015))


def radius_for_rank(rank: int) -> float:
    return max(D.model(g * D.N_RANKS + rank).max_halfwidth() for g in D.groups())


def grasp_width(m: D.DollModel, h: float, psi_rel: float = 0.0) -> float:
    zs = np.arange(max(0.0, h - PAD_DOWN), min(m.height, h + PAD_UP) + 1e-06, 0.003)
    return max(m.width_at(float(z), psi_rel) for z in zs)


def contact_span(m: D.DollModel, h: float, psi_rel: float = 0.0) -> tuple[float, float]:
    zs = np.arange(max(0.0, h - PAD_DOWN), min(m.height, h + PAD_UP) + 1e-06, 0.002)
    ws = np.array([m.width_at(float(z), psi_rel) for z in zs])
    c = zs[ws >= ws.max() - GRIP_TOL]
    return float(ws.max()), float(c.max() - c.min())


_GRASP_H: dict = {}


def grasp_heights(m: D.DollModel) -> tuple[float, float | None]:
    key = m.model_id
    if key not in _GRASP_H:
        lo = max(0.02, 0.45 * m.height, m.height - PALM_CLEAR)
        hs = [float(hg) for hg in np.arange(lo, 0.9 * m.height, 0.002)]
        base = next((hg for hg in hs if grasp_width(m, hg) <= GAP_MAX - 2 * SIDE_MARGIN), None)
        old = base if base is not None else 0.85 * m.height
        best = None
        if base is not None and (base < 0.5 * m.height or contact_span(m, base)[1] >= GRIP_SPAN_MIN):
            best = base
        for margin in (SIDE_MARGIN, WIDE_MARGIN):
            for hg in hs:
                if best is not None:
                    break
                w, span = contact_span(m, hg)
                if (base is None or hg >= base) and w <= GAP_MAX - 2 * margin and (span >= GRIP_SPAN):
                    best = hg
        _GRASP_H[key] = (old, None) if best is None or best == old else (best, old)
    return _GRASP_H[key]


@dataclass
class Doll:
    rank: int
    model: D.DollModel
    xy: np.ndarray
    done: bool = False
    busy: str | None = None
    dest: np.ndarray | None = None
    fails: int = 0
    yaw: float = 0.0
    turn: float = 0.0
    gave_up: bool = False
    plan_fails: int = 0
    arm_fails: dict = field(default_factory=dict)
    blocks: int = 0
    relays: int = 0
    asides: int = 0
    alt_grip: bool = False
    lost: bool = False
    model_next: object = None
    tilt_spot: bool = False

    @property
    def R(self) -> float:
        return self.model.max_halfwidth() + 0.003

    @property
    def h(self) -> float:
        return self.model.height

    @property
    def h_g(self) -> float:
        best, old = grasp_heights(self.model)
        return old if self.alt_grip and old is not None else best

    @property
    def w(self) -> float:
        return grasp_width(self.model, self.h_g)


CLUTTER_H_UNKNOWN = 0.15
CLUTTER_LEVEL_STEP = 0.02
_CAT_H: dict = {}
CLUTTER_LOG: list = []


def category_prior(cat: str):
    if not _CAT_H:
        import json

        f = ASSETS / "category_heights.json"
        _CAT_H.update(json.loads(f.read_text())["classes"])
    c = _CAT_H.get(cat)
    return None if c is None else (float(c["h"]), float(c["diag"]))


def clutter_height(p: np.ndarray, cat: str | None, cam) -> float:
    prior = category_prior(cat) if cat is not None else None
    h_cat, diag = prior if prior is not None else (CLUTTER_H_UNKNOWN, 0.0)
    hc = float(cam.pos[2]) - K.TABLE_Z
    r = np.linalg.norm(p - np.asarray(cam.pos[:2], float), axis=1)
    if len(r) == 0:
        return h_cat
    r_f, L = float(r.min()), float(r.max() - r.min())
    h_geo = hc * max(0.0, L - diag) / max(1e-06, L + r_f)
    return float(max(h_cat, min(h_geo, hc)))


def obstacle_cells(scene, cam, cell: float = 0.01):
    pts, hs = [], []
    CLUTTER_LOG.clear()
    cats = list(getattr(scene, "others_cat", []) or [])
    for k, mk in enumerate(scene.others):
        vs, us = np.nonzero(mk[::2, ::2])
        if len(us) == 0:
            continue
        p = cam.on_plane_z(np.c_[us * 2.0, vs * 2.0], K.TABLE_Z)
        p = p[np.isfinite(p[:, 0]), :2]
        if not len(p):
            continue
        cat = cats[k] if k < len(cats) else None
        h = clutter_height(p, cat, cam)
        CLUTTER_LOG.append((cat, round(float(h), 3), int(len(p))))
        pts.append(p)
        hs.append(np.full(len(p), h))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    q = np.round(np.concatenate(pts) / cell) * cell
    h = np.concatenate(hs)
    cells, inv = np.unique(q, axis=0, return_inverse=True)
    out = np.zeros(len(cells))
    np.maximum.at(out, inv.reshape(-1), h)
    return cells, out


def min_dist(points: np.ndarray, xy) -> float:
    if len(points) == 0:
        return 1000000000.0
    return float(np.min(np.linalg.norm(points - np.asarray(xy)[:2], axis=1)))


def seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab)), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def arms_for(x: float, lim: float = SIDE_X) -> tuple[str, ...]:
    out = []
    if x <= lim:
        out.append("left")
    if x >= -lim:
        out.append("right")
    return tuple(out)


def row_offsets(ranks=(4, 3, 2, 1, 0), radius=radius_for_rank) -> np.ndarray:
    offs = [0.0]
    for a, b in zip(ranks, ranks[1:]):
        offs.append(offs[-1] + radius(a) + radius(b) + ROW_GAP)
    return np.array(offs) - 0.5 * offs[-1]


def plan_row(dolls: dict, obstacles: np.ndarray):
    ranks = [4, 3, 2, 1, 0]
    offs = row_offsets(ranks, lambda r: dolls[r].R if r in dolls else radius_for_rank(r))
    cands = []
    for y in ROW_YS:
        for xc in np.arange(-0.08, 0.081, 0.02):
            slots = {r: np.array([xc + o, y]) for r, o in zip(ranks, offs)}
            if any(abs(s[0]) > 0.36 for s in slots.values()):
                continue
            cost = 2.0 * abs(y - ROW_Y)
            for r, s in slots.items():
                R = dolls[r].R if r in dolls else radius_for_rank(r)
                if min_dist(obstacles, s) < R + 0.01:
                    cost += 100.0
                for d in dolls.values():
                    if d.rank != r and np.linalg.norm(d.xy - s) < R + d.R + 0.01:
                        cost += 0.6
                if r in dolls:
                    d = dolls[r]
                    dist = float(np.linalg.norm(d.xy - s))
                    if dist > DONE_TOL:
                        cost += 0.5 + dist
                        if not set(arms_for(d.xy[0])) & set(arms_for(s[0])):
                            cost += 1.2
            cands.append((cost, len(cands), slots))
    cands.sort(key=lambda c: c[:2])

    def served(slots):
        return all(
            any(reach(a, s, dolls[r].h + TOP_CLEAR + ROW_REACH_MARGIN) for a in arms_for(s[0]))
            for r, s in slots.items()
            if r in dolls
        )

    for cost, _, slots in cands:
        if served(slots):
            return slots, cost
    return cands[0][2], cands[0][0]


_REACH: dict = {}


def reach(arm: str, xy, z: float) -> bool:
    key = arm, round(float(xy[0]), 2), round(float(xy[1]), 2), round(z, 2)
    if key not in _REACH:
        ok = False
        for psi in (90 * DEG, -90 * DEG, 0.0, math.pi):
            T = tcp_pose(G.grasp_rotation(psi, 0.0), (key[1], key[2], K.TABLE_Z + key[3]))
            seed = G.base_seed(arm, key[1:3])
            if K.solve(arm, T, seed, K.TCP_OFFSET).ok:
                ok = True
                break
        _REACH[key] = ok
    return _REACH[key]


_REACH_T: dict = {}


def reach_tilted(arm: str, xy, z: float) -> float | None:
    key = arm, round(float(xy[0]), 2), round(float(xy[1]), 2), round(z, 2)
    if key not in _REACH_T:
        bx, by = K.BASE_POS[arm][:2]
        psi_r = math.atan2(key[2] - by, key[1] - bx) + 0.5 * math.pi
        out = None
        for tilt in TILT_PICK:
            for psi in (psi_r, psi_r + math.pi, 90 * DEG, -90 * DEG, 0.0, math.pi):
                for sgn in (1.0, -1.0):
                    T = tcp_pose(G.grasp_rotation(psi, sgn * tilt), (key[1], key[2], K.TABLE_Z + key[3]))
                    if K.solve(arm, T, G.base_seed(arm, key[1:3]), K.TCP_OFFSET).ok:
                        out = float(tilt)
                        break
                if out is not None:
                    break
            if out is not None:
                break
        _REACH_T[key] = out
    return _REACH_T[key]


def ik(arm: str, psi: float, p, seed, tilt: float = 0.0):
    T = tcp_pose(G.grasp_rotation(psi, tilt), p)
    r = K.solve_near(arm, T, seed, K.TCP_OFFSET, extra_seeds=[G.base_seed(arm, p[:2])], n_random=0)
    return r.q if r.ok else None


@dataclass
class Job:
    doll: Doll
    dest: np.ndarray
    final: bool
    kind: str
    zone: bool = False
    psi_p: float = 0.0
    put: np.ndarray | None = None
    turn: float = 0.0


class PathZone(Zone):
    def __init__(self, xy, radius: float):
        self.paths: dict = {}
        super().__init__(xy, radius)

    @property
    def owner(self):
        return next(iter(self.paths)) if len(self.paths) == 1 else None

    @owner.setter
    def owner(self, name) -> None:
        if name is None:
            self.paths.clear()
        else:
            self.paths.setdefault(name, np.zeros((0, 2)))

    def holds(self, name: str) -> bool:
        return name in self.paths

    def fits(self, name: str, pts) -> bool:
        others = [(n, p) for n, p in self.paths.items() if n != name]
        if not others:
            return True
        if ZONE_SHARE_X is None or pts is None:
            return False
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        for n, p in others:
            left, right = (pts, p) if name == "left" else (p, pts)
            if not len(left) or not len(right) or float(right[:, 0].min() - left[:, 0].max()) < ZONE_SHARE_X:
                return False
        return True

    def take(self, name: str, pts=None) -> bool:
        pts = np.zeros((0, 2)) if pts is None else np.asarray(pts, dtype=float).reshape(-1, 2)
        if name in self.paths or self.fits(name, pts):
            self.paths[name] = pts
            return True
        return False

    def update(self, name: str, pts) -> None:
        if name in self.paths:
            self.paths[name] = np.asarray(pts, dtype=float).reshape(-1, 2)

    def free(self, name: str) -> None:
        self.paths.pop(name, None)


class Board:
    def __init__(self, ctx, dolls: dict, slots: dict, obstacles: np.ndarray, obstacle_h: np.ndarray | None = None):
        self.ctx = ctx
        self.dolls = dolls
        self.slots = slots
        self.obstacles = obstacles
        self.obstacle_h = np.full(len(obstacles), CLUTTER_H) if obstacle_h is None else np.asarray(obstacle_h, float)
        self.version = 0
        self.relax = 0
        self.changed = 0
        self.no_job_v: dict = {}
        self.grasp_devs: dict = {}
        self.regrouped = False
        self.locked: dict = {}
        self.idle: set = set()
        self.bad: dict = {}
        self.moves = 0
        self.held_off: dict = {}
        self.logged: set = set()
        self.zone_want: dict = {}
        self.shares: dict = {}
        self.side_q: dict = {}
        self.corridors: dict = {}
        self.cross_waiting: set = set()
        row_y = float(next(iter(slots.values()))[1]) if slots else RELAY_Y
        self.zone = PathZone((0.0, 0.5 * (RELAY_Y + row_y)), max(ZONE_R, 0.5 * abs(row_y - RELAY_Y) + 0.05))

    def steps_left(self) -> int:
        return STEP_LIMIT - HOME_STEPS - self.ctx.obs.step

    def pending(self):
        return [d for d in self.dolls.values() if not d.done]

    def blocks_slot(self, e, d, slot, points=None) -> bool:
        r_up = radius_above(e.model, d.h_g + PLACE_DROP + HOUSING_DZ - 0.01)
        if points is None:
            points = [e.xy] if e.dest is None else [e.xy, e.dest]
        for c in points:
            off = np.asarray(c, float) - np.asarray(slot, float)
            dist = float(np.linalg.norm(off))
            if dist < d.R + e.R + 0.008:
                return True
            if not e.done and r_up > 0 and (dist < HOUSING_R + r_up) and (abs(off[1]) >= 0.5 * abs(off[0])):
                return True
        return False

    def slot_blockers(self, d) -> list:
        return [e for e in self.dolls.values() if e is not d and self.blocks_slot(e, d, self.slots[d.rank])]

    def blockers(self, xy, R, exclude) -> list:
        out = []
        for e in self.dolls.values():
            if e is exclude:
                continue
            for p in [e.xy] if e.dest is None else [e.xy, e.dest]:
                if np.linalg.norm(p - xy) < R + e.R + 0.008:
                    out.append(e)
                    break
        return out

    def spot_free(self, xy, d, margin: float = 0.012, housing: int = 0, pending_slots: bool = False) -> bool:
        if min_dist(self.obstacles, xy) < d.R + margin:
            return False
        if self.blockers(xy, d.R + margin - 0.008, d):
            return False
        if housing:
            for e in self.dolls.values():
                r_up = radius_above(e.model, d.h_g + HOUSING_DZ - 0.01) if e is not d else 0.0
                if r_up > 0:
                    cs = [e.xy] if e.dest is None else [e.xy, e.dest]
                    if housing >= 2 and (not e.done):
                        cs.append(self.slots[e.rank])
                    hr = HOUSING_R if housing >= 3 else SPOT_HOUSING_R
                    for c in cs:
                        if np.linalg.norm(c - xy) < r_up + hr:
                            return False
            if housing >= 3:
                for e in self.dolls.values():
                    if e is d or e.done:
                        continue
                    if e.dest is None:
                        c = e.xy
                    elif np.linalg.norm(np.asarray(e.dest) - self.slots[e.rank]) > 1e-06:
                        c = e.dest
                    else:
                        continue
                    r_up = radius_above(d.model, e.h_g + HOUSING_DZ - 0.01)
                    if r_up > 0 and np.linalg.norm(np.asarray(c, float) - xy) < r_up + HOUSING_R:
                        return False
        for r, s in self.slots.items():
            if pending_slots and (not self.dolls[r].done):
                continue
            if r != d.rank and np.linalg.norm(s - xy) < d.R + self.dolls[r].R + margin:
                return False
        return True

    def crosses(self, arm, d, xy, now: dict | None = None, skip=(), live_only: bool = False) -> bool:
        xy = np.asarray(xy, float)
        for a, (poly, r_c, zb, live) in self.corridors.items():
            if a == arm or a in skip or zb >= d.h + LIFT_CLEAR - 1e-06 or (live_only and (not live)):
                continue
            if live and now is not None and (a in now) and (len(poly) >= 2):
                poly = np.vstack([np.asarray(now[a], float)[:2], poly[1:]])
            need = d.R + r_c + CROSS_MARGIN
            if len(poly) == 1:
                if float(np.linalg.norm(poly[0] - xy)) < need:
                    return True
            elif any(seg_dist(xy, p, q) < need for p, q in zip(poly[:-1], poly[1:])):
                return True
        return False

    def find_spot(
        self, d, arms_needed, arm=None, takers=None, strict: bool = True, wide: bool = False, slots_ok: bool = False
    ) -> np.ndarray | None:
        levels = 3, 2, 1, 0
        if strict and (not self.relax):
            levels = (3,)
        for tilt in (False, True) if TILT_PICK else (False,):
            for ys in (SPOT_YS_FRONT,) if tilt else (SPOT_YS_FRONT, SPOT_YS_ALL):
                for housing in levels:
                    for x_lim in (SPOT_X,):
                        p = self._find_spot(d, arms_needed, arm, housing, ys, takers, x_lim, tilt)
                        if p is not None:
                            return p
            if wide and self.relax:
                for ys in (SPOT_YS_FRONT,) if tilt else (SPOT_YS_FRONT, SPOT_YS_ALL):
                    for housing in levels:
                        p = self._find_spot(d, arms_needed, arm, housing, ys, takers, SPOT_X_STUCK, tilt)
                        if p is not None:
                            return p
        if slots_ok and self.clutter_locked(d):
            row_y = float(np.mean([s[1] for s in self.slots.values()]))
            ys = tuple(y for y in SPOT_YS_ALL if y <= row_y + 1e-06)
            for x_lim in (SPOT_X, SPOT_X_STUCK):
                for housing in levels:
                    p = self._find_spot(d, arms_needed, arm, housing, ys, takers, x_lim, bool(TILT_PICK), in_slots=True)
                    if p is not None:
                        return p

    def clutter_locked(self, d) -> bool:
        key = d.rank, d.model.model_id
        if key not in self.locked:
            x_lim = max(SPOT_X, SPOT_X_STUCK or 0.0)
            z = d.h + TOP_CLEAR
            slot = self.slots[d.rank]
            takers = [a for a in K.ARMS if a in arms_for(slot[0]) and reach(a, slot, z)]
            front = set(np.round(SPOT_YS_FRONT, 4))
            free = False
            for x in np.arange(-x_lim, x_lim + 0.001, 0.02):
                if not any(a in arms_for(x) for a in takers):
                    continue
                for y in SPOT_YS_ALL:
                    q = np.array([x, y])
                    if min_dist(self.obstacles, q) < d.R + 0.012:
                        continue
                    if any(
                        r != d.rank and np.linalg.norm(s - q) < d.R + self.dolls[r].R + 0.012
                        for r, s in self.slots.items()
                    ):
                        continue
                    tilt_ok = TILT_PICK and round(float(y), 4) in front
                    if all(reach(a, q, z) or (tilt_ok and reach_tilted(a, q, z) is not None) for a in K.ARMS):
                        free = [round(float(x), 3), round(float(y), 3)]
                        break
                if free:
                    break
            self.locked[key] = not free
        return self.locked[key]

    def tilt_only(self, d, p) -> bool:
        return bool(TILT_PICK) and (not all(reach(a, p, d.h + TOP_CLEAR) for a in K.ARMS))

    def _find_spot(
        self,
        d,
        arms_needed,
        arm,
        housing: int,
        ys,
        only=None,
        x_lim: float | None = None,
        tilt: bool = False,
        in_slots: bool = False,
    ) -> np.ndarray | None:
        best = None
        slot = self.slots[d.rank]
        takers = [
            a
            for a in arms_needed
            if a in arms_for(slot[0]) and reach(a, slot, d.h + TOP_CLEAR) and (only is None or a in only)
        ]
        x_lim = SPOT_X if x_lim is None else x_lim
        for x in np.arange(-x_lim, x_lim + 0.001, 0.02):
            for y in ys:
                p = np.array([x, y])
                if np.linalg.norm(p - d.xy) < 0.03:
                    continue
                if not any(a in arms_for(x) for a in takers):
                    continue
                if (
                    not self.spot_free(p, d, housing=housing, pending_slots=in_slots)
                    or self.bad.get((arm, d.rank, tuple(np.round(p, 2))), -1) >= self.version
                ):
                    continue
                if in_slots and (not any(x >= -1e-06 if a == "right" else x <= 1e-06 for a in takers)):
                    continue
                z_r = d.h + TOP_CLEAR + (SLOT_SPOT_DZ if in_slots else 0.0)
                if not all(reach(a, p, z_r) or (tilt and reach_tilted(a, p, z_r) is not None) for a in arms_needed):
                    continue
                c = abs(x) + abs(y - RELAY_Y)
                if best is None or c < best[0]:
                    best = c, p
        return None if best is None else best[1]

    def would_block(self, d, p) -> bool:
        xy, dest = d.xy, d.dest
        d.xy, d.dest = np.asarray(p, float), None
        try:
            return any(self.blocks_slot(d, e, self.slots[e.rank]) for e in self.pending() if e is not d)
        finally:
            d.xy, d.dest = xy, dest

    def find_aside(self, d, arm) -> np.ndarray | None:
        for housing in (3, 2, 1):
            p = self._find_aside(d, arm, housing)
            if p is not None:
                return p

    def _find_aside(self, d, arm, housing: int) -> np.ndarray | None:
        best = None
        xs = (
            np.arange(-0.44, SIDE_X + 0.001, ASIDE_STEP)
            if arm == "left"
            else np.arange(-SIDE_X, 0.44 + 0.001, ASIDE_STEP)
        )
        for x in xs:
            for y in np.arange(-0.32, -0.039, ASIDE_STEP):
                p = np.array([x, y])
                dist = float(np.linalg.norm(p - d.xy))
                if dist < 0.03 or (best is not None and dist >= best[0]):
                    continue
                if self.zone.inside(p) or float(np.linalg.norm(p - self.zone.xy)) < self.zone.radius + d.R:
                    continue
                if not self.spot_free(p, d, housing=housing) or self.would_block(d, p):
                    continue
                if self.bad.get((arm, d.rank, tuple(np.round(p, 2))), -1) >= self.version:
                    continue
                if not reach(arm, p, d.h + TOP_CLEAR):
                    continue
                best = dist, p
        return None if best is None else best[1]

    def loosen(self) -> bool:
        if self.relax >= 1:
            return False
        self.relax = 1
        self.bump()
        return True

    def aside_job(self, arm, d, blocking: int):
        if (
            self.relax < 1
            or not blocking
            or d.asides >= MAX_ASIDE
            or (arm not in arms_for(d.xy[0]))
            or (not self.pick_reach(arm, d))
        ):
            return None
        spot = self.find_aside(d, arm)
        return None if spot is None else Job(d, spot, False, "aside")

    def any_direct(self, exclude) -> bool:
        for e in self.pending():
            if e is exclude:
                continue
            if e.busy is not None:
                return True
            s = self.slots[e.rank]
            if not self.slot_blockers(e) and any(self.can_do(a, e, s) for a in K.ARMS):
                return True
        return False

    def needs_tilt(self, d) -> bool:
        if not TILT_PICK:
            return False
        lim = SIDE_X if SIDE_X_DIRECT is None else max(SIDE_X, SIDE_X_DIRECT)
        return not any(reach(a, d.xy, d.h + TOP_CLEAR) for a in arms_for(d.xy[0], lim))

    def pick_reach(self, arm, d) -> bool:
        if reach(arm, d.xy, d.h + TOP_CLEAR):
            return True
        return (self.needs_tilt(d) or d.tilt_spot) and reach_tilted(arm, d.xy, d.h + TOP_CLEAR) is not None

    def can_do(self, arm, d, dest, ignore_recent: bool = False) -> bool:
        bad = self.bad.get((arm, d.rank, tuple(np.round(dest, 2))), -1)
        if bad >= (10**9 if ignore_recent else self.version):
            return False
        lim = SIDE_X if SIDE_X_DIRECT is None else max(SIDE_X, SIDE_X_DIRECT)
        if arm not in arms_for(d.xy[0], lim) or arm not in arms_for(dest[0], lim):
            return False
        return self.pick_reach(arm, d) and reach(arm, dest, d.h + TOP_CLEAR)

    def claim(self, arm: str) -> Job | None:
        if self.steps_left() < JOB_STEPS:
            return None
        other = "right" if arm == "left" else "left"
        pend = [d for d in self.pending() if d.busy is None]
        cands = []
        why = {}
        nospot = []
        max_relays = MAX_RELAYS + (1 if self.relax else 0)
        for d in pend:
            s = self.slots[d.rank]
            if self.held_off.get((arm, d.rank), -1) >= self.moves:
                why[d.rank] = "held_off"
                continue
            blocking = len([e for e in self.pending() if e is not d and self.blocks_slot(d, e, self.slots[e.rank])])
            blk = self.slot_blockers(d)
            clearing = bool(blk) and all(
                e.busy is not None and e.dest is not None and (not self.blocks_slot(e, d, s, points=[e.dest]))
                for e in blk
            )
            if clearing and (
                self.can_do(arm, d, s)
                or self.can_do(other, d, s)
                or self.can_do(arm, d, s, ignore_recent=True)
                or (d.relays >= 1 and self.can_do(other, d, s, ignore_recent=True))
            ):
                why[d.rank] = "slot_clearing"
                continue
            if not blk or clearing:
                own_fails = d.arm_fails.get(arm, 0)
                if (
                    SELF_RELAY_AFTER is not None
                    and own_fails >= SELF_RELAY_AFTER
                    and (d.relays < max_relays)
                    and (other in arms_for(s[0]))
                    and (arm in arms_for(d.xy[0]))
                    and (not self.can_do(other, d, s, ignore_recent=True))
                    and self.pick_reach(arm, d)
                ):
                    spot = self.find_spot(
                        d, ("left", "right"), arm, takers=(other,), strict=not blocking, slots_ok=True
                    )
                    if spot is not None:
                        cands.append((1, -blocking, 0.0, Job(d, spot, False, "relay", zone=True)))
                        continue
                if self.can_do(arm, d, s):
                    shut = self.shuts_in(d, s)
                    if shut is not None:
                        why[d.rank] = f"access:{shut}"
                        continue
                    took = self.takes_spot(d, s)
                    if took is not None:
                        why[d.rank] = f"spot:{took}"
                        continue
                    cands.append((0, -blocking, float(np.linalg.norm(d.xy - s)), Job(d, s, True, "direct")))
                    continue
                if self.can_do(other, d, s) and (
                    not (SELF_RELAY_AFTER is not None and d.arm_fails.get(other, 0) >= SELF_RELAY_AFTER)
                ):
                    why[d.rank] = "other"
                    continue
                if self.can_do(arm, d, s, ignore_recent=True):
                    why[d.rank] = "retry_later"
                elif d.relays >= 1 and self.can_do(other, d, s, ignore_recent=True):
                    why[d.rank] = "other_later"
                elif (
                    d.relays < max_relays
                    and arm in arms_for(d.xy[0])
                    and self.pick_reach(arm, d)
                    and any(a in arms_for(s[0]) for a in (other,))
                ):
                    spot = self.find_spot(d, ("left", "right"), arm, strict=not blocking, slots_ok=True)
                    if spot is not None:
                        behind = d.xy[1] > s[1] + 0.02 and spot[1] < s[1]
                        cands.append((-1 if behind else 1, -blocking, 0.0, Job(d, spot, False, "relay", zone=True)))
                    else:
                        why[d.rank] = "relay:nospot"
                        nospot.append((d, blocking))
                else:
                    why[d.rank] = f"relay:{d.relays}{int(arm in arms_for(d.xy[0]))}{int(self.pick_reach(arm, d))}"
                if why.get(d.rank, "").startswith("relay:"):
                    aside = self.aside_job(arm, d, blocking)
                    if aside is not None:
                        cands.append((3, -blocking, 0.0, aside))
                        why.pop(d.rank)
                continue
            if (
                blocking
                and d.relays < max_relays
                and (not self.any_direct(d))
                and (arm in arms_for(d.xy[0]))
                and self.pick_reach(arm, d)
            ):
                spot = self.find_spot(d, ("left", "right"), arm, strict=not blocking)
                if spot is not None:
                    cands.append((2, -blocking, 0.0, Job(d, spot, False, "buffer", zone=True)))
                else:
                    why[d.rank] = "buffer:nospot"
            else:
                why[d.rank] = f"slot_taken:{blocking}{d.relays}"
            if d.rank in why and (not self.any_direct(d)):
                aside = self.aside_job(arm, d, blocking)
                if aside is not None:
                    cands.append((3, -blocking, 0.0, aside))
                    why.pop(d.rank)
        if not cands and self.relax:
            for d, blocking in nospot:
                if d.rank in why:
                    spot = self.find_spot(d, ("left", "right"), arm, strict=False, wide=True, slots_ok=True)
                    if spot is not None:
                        cands.append((4, -blocking, 0.0, Job(d, spot, False, "relay", zone=True)))
                        why[d.rank] = "relay:wide"
        if not cands:
            key = arm, self.version
            if key not in self.logged:
                self.logged.add(key)
            return None
        cands.sort(key=lambda c: c[:3])
        for c in cands:
            c[3].zone = c[3].zone or self.zone.inside(c[3].doll.xy) or self.zone.inside(c[3].dest)
        if other in self.zone_want and (not self.zone.holds(arm)):
            cands = [c for c in cands if not c[3].zone]
            if not cands:
                return None
        job = cands[0][3]
        job.doll.busy = arm
        job.doll.dest = job.dest
        self.bump()
        return job

    def forget(self, d) -> None:
        for k in [k for k in self.bad if k[1] == d.rank]:
            del self.bad[k]
        d.plan_fails = 0
        d.arm_fails = {}

    def bump(self) -> None:
        self.version += 1
        self.changed = self.ctx.obs.step

    def retry_marked(self) -> bool:
        temp = [k for k, v in self.bad.items() if v < 10**9]
        for k in temp:
            del self.bad[k]
        temp += list(self.held_off)
        self.held_off.clear()
        if temp:
            self.bump()
        return bool(temp)

    def stalled(self) -> bool:
        if any(d.busy for d in self.dolls.values()):
            return False
        return all(self.no_job_v.get(a, -1) == self.version for a in K.ARMS)

    def shuts_in(self, d, s) -> int | None:
        if self.relax:
            return None
        s = np.asarray(s, float)
        h_top = max(e.h for e in self.dolls.values())
        for e in self.dolls.values():
            if e is d or e.done or e.busy is not None or (e.plan_fails >= ACCESS_FAILS):
                continue
            if e.xy[1] <= float(self.slots[e.rank][1]) + ACCESS_BEHIND:
                continue
            r_up = radius_above(d.model, e.h_g + HOUSING_DZ - 0.01)
            if r_up <= 0 or float(np.linalg.norm(s - e.xy)) >= r_up + HOUSING_R:
                continue
            arms = [a for a in arms_for(e.xy[0]) if self.pick_reach(a, e)]
            if not arms:
                continue
            if any(reach(a, e.xy, h_top + TOP_CLEAR) for a in arms):
                continue
            return int(e.rank)

    def takes_spot(self, d, s) -> int | None:
        if self.relax:
            return None
        s = np.asarray(s, float)
        for e in self.pending():
            if e is d or e.busy is not None or e.plan_fails >= ACCESS_FAILS or (e.relays >= MAX_RELAYS):
                continue
            se = self.slots[e.rank]
            if any(self.can_do(a, e, se, ignore_recent=True) for a in K.ARMS):
                continue
            givers = [a for a in arms_for(e.xy[0]) if self.pick_reach(a, e)]
            givers = [a for a in givers if ("right" if a == "left" else "left") in arms_for(se[0])]
            if not givers:
                continue
            blocking = len([x for x in self.pending() if x is not e and self.blocks_slot(e, x, self.slots[x.rank])])
            g = givers[0]
            if self.find_spot(e, ("left", "right"), g, strict=not blocking, slots_ok=True) is None:
                continue
            xy, dest = d.xy, d.dest
            d.xy, d.dest = s.copy(), None
            try:
                after = self.find_spot(e, ("left", "right"), g, strict=not blocking, slots_ok=True)
            finally:
                d.xy, d.dest = xy, dest
            if after is None:
                return int(e.rank)

    def note_grasp(self, d, h_g: float, wid: float) -> int | None:
        if self.regrouped or d.model.group == DUCK_GROUP:
            return None
        self.grasp_devs[d.rank] = {
            g: wid - grasp_width(D.model(g * D.N_RANKS + d.rank), h_g) for g in D.groups() if g != DUCK_GROUP
        }
        cur = d.model.group
        cands = {}
        for g in self.grasp_devs[d.rank]:
            if g == cur:
                continue
            vs = list(self.grasp_devs.values())
            good = [v for v in vs if abs(v[cur]) >= GROUP_DEV_BAD and abs(v[g]) <= GROUP_DEV_OK]
            votes = sum(1 for v in good)
            if votes >= GROUP_VOTES and all(abs(v[g]) <= abs(v[cur]) + GROUP_DEV_OK for v in vs):
                cands[g] = sum(abs(v[g]) for v in good) / len(good)
        if not cands:
            return None
        g = min(cands, key=lambda k: (cands[k], k))
        self.regroup(g)
        return g

    def regroup(self, g: int) -> None:
        self.regrouped = True
        for e in self.dolls.values():
            m = D.model(g * D.N_RANKS + e.rank)
            if e.busy is not None:
                e.model_next = m
            else:
                e.model = m
        self.bump()

    def finish(self, job: Job, ok: bool, arm: str, status: str = "fail") -> None:
        d = job.doll
        if d.model_next is not None:
            d.model, d.model_next = d.model_next, None
        if ok or job.put is not None:
            self.moves += 1
        elif status == "fail":
            self.held_off[arm, d.rank] = self.moves
        if status == "empty":
            d.lost = True
            d.done = True
            d.fails += 1
            d.busy = None
            d.dest = None
            self.bump()
            return
        if status == "fail" and job.put is None:
            d.arm_fails[arm] = d.arm_fails.get(arm, 0) + 1
        if ok or job.put is not None:
            self.forget(d)
        if ok:
            d.xy = np.asarray(job.dest, float).copy()
            d.done = job.final
            d.yaw = wrap(d.yaw + d.turn)
            d.tilt_spot = not job.final and job.kind in ("relay", "buffer") and self.tilt_only(d, d.xy)
            if job.kind == "aside":
                d.asides += 1
            else:
                d.relays += 0 if job.final else 1
        elif status == "blocked" and job.put is None and (d.blocks < 6):
            d.blocks += 1
        elif status == "fail" and job.put is None and (d.plan_fails < 4):
            d.plan_fails += 1
            self.bad[arm, d.rank, tuple(np.round(job.dest, 2))] = self.version + 1
        else:
            d.fails += 1
            if job.put is not None:
                d.xy = np.asarray(job.put, float).copy()
                d.tilt_spot = False
                d.yaw = wrap(d.yaw + job.turn)
                d.relays += 1
                if job.final and float(np.linalg.norm(d.xy - job.dest)) < DONE_TOL:
                    d.fails -= 1
                    d.done = True
            else:
                self.bad[arm, d.rank, tuple(np.round(job.dest, 2))] = 10**9
            if d.fails >= 3:
                d.done = True
                d.gave_up = True
        d.busy = None
        d.dest = None
        self.bump()


def rebuild_board(ctx, old: Board, scene, obstacles: np.ndarray, obstacle_h: np.ndarray | None = None) -> Board | None:
    dolls = {}
    for s in scene.dolls:
        r = int(s.rank)
        if r not in old.slots or r in dolls:
            continue
        o = old.dolls.get(r)
        d = Doll(r, o.model if o is not None else D.model(s.model_id), np.array([s.x, s.y], float))
        if o is not None:
            d.yaw = o.yaw
        d.done = bool(np.linalg.norm(d.xy - np.asarray(old.slots[r], float)) < VERIFY_TOL)
        dolls[r] = d
    if not any(not d.done for d in dolls.values()):
        return None
    return Board(ctx, dolls, old.slots, obstacles, obstacle_h)


def wrap(a: float) -> float:
    return math.remainder(a, 2 * math.pi)


def tcp_at(xy, z) -> np.ndarray:
    return np.array([float(xy[0]), float(xy[1]), K.TABLE_Z + z])


def pick_psis(board: Board, d: Doll, open_gap: float):
    out = []
    psis = tuple(wrap(d.yaw + a) for a in DUCK_PSIS) if d.model.group == DUCK_GROUP else PICK_PSIS
    for i, psi in enumerate(psis):
        u = np.array([math.cos(psi), math.sin(psi)])
        clear = 1000000000.0
        for o in board.dolls.values():
            if o is d:
                continue
            for sgn in (1, -1):
                f = d.xy + sgn * u * (open_gap / 2 + FINGER_T / 2)
                clear = min(clear, float(np.linalg.norm(o.xy - f)) - o.R - FINGER_W)
        for sgn in (1, -1):
            f = d.xy + sgn * u * (open_gap / 2 + FINGER_T / 2)
            clear = min(clear, min_dist(board.obstacles, f) - FINGER_W)
        out.append((clear >= 0.006, -i, psi, clear))
    out.sort(key=lambda c: (not c[0], -c[1]))
    return [c[2] for c in out[:6]], out[0][3], sum(1 for c in out[:6] if c[0])


def in_way(p, a, b, clear: float) -> bool:
    dmin = seg_dist(p, a, b)
    return dmin < clear and dmin < min(np.linalg.norm(p - a), np.linalg.norm(p - b)) - 0.004


def carry_height(board: Board, d: Doll, a, b) -> float:
    z = LIFT_CLEAR
    a, b = np.asarray(a, float), np.asarray(b, float)
    for e in board.dolls.values():
        if e is d:
            continue
        if in_way(e.xy, a, b, d.R + e.R + 0.012):
            z = max(z, e.h + LIFT_CLEAR)
    if len(board.obstacles):
        near = seg_dists(board.obstacles, a, b) < d.R + 0.01
        if near.any():
            z = max(z, float(board.obstacle_h[near].max()) + LIFT_CLEAR)
    return z


def seg_dists(P: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    ab = b - a
    t = np.clip((P - a) @ ab / max(1e-12, float(ab @ ab)), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


ROUTE_CELL = 0.01
ROUTE_X = -0.46, 0.46
ROUTE_Y = -0.33, 0.02
ROUTE_MARGIN = 0.012
ROUTE_SEARCH = 0.015
ROUTE_REACH_MARGIN = 0.01
REACH_R = (
    (0.02, 0.51),
    (0.03, 0.502),
    (0.04, 0.502),
    (0.05, 0.502),
    (0.06, 0.491),
    (0.07, 0.486),
    (0.08, 0.48),
    (0.09, 0.472),
    (0.1, 0.472),
    (0.11, 0.461),
    (0.12, 0.458),
    (0.13, 0.445),
    (0.14, 0.439),
    (0.15, 0.431),
    (0.16, 0.417),
    (0.17, 0.401),
    (0.18, 0.387),
    (0.19, 0.373),
    (0.2, 0.36),
    (0.21, 0.33),
    (0.22, 0.278),
    (0.23, 0.1),
)


def reach_radius(z: float) -> float | None:
    if not REACH_R:
        return None
    zs = [r[0] for r in REACH_R]
    rs = [r[1] for r in REACH_R]
    return float(np.interp(z, zs, rs))


EDT_PAD = 12


def lattice_distance(cells: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    pad = EDT_PAD
    nx, ny = len(xs) + 2 * pad, len(ys) + 2 * pad
    occ = np.zeros((nx, ny), bool)
    ij = np.round((np.asarray(cells, float) - [xs[0], ys[0]]) / ROUTE_CELL).astype(int) + pad
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    if not ok.any():
        return np.full((len(xs), len(ys)), 1000.0)
    occ[ij[ok, 0], ij[ok, 1]] = True
    return ndimage.distance_transform_edt(~occ)[pad:-pad, pad:-pad] * ROUTE_CELL


def route_low(
    board: Board,
    d: Doll,
    a,
    b,
    arm: str | None = None,
    avoid_zone: bool = False,
    z_c: float | None = None,
    housing_r: float = HOUSING_R,
    clutter: bool = True,
    over: float = 0.0,
):
    import heapq

    xs = np.arange(ROUTE_X[0], ROUTE_X[1] + 1e-09, ROUTE_CELL)
    ys = np.arange(ROUTE_Y[0], ROUTE_Y[1] + 1e-09, ROUTE_CELL)
    pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
    clear = np.full(pts.shape[:2], 1.0)
    for e in board.dolls.values():
        if e is d or e.h + LIFT_CLEAR <= over + 1e-06:
            continue
        need = d.R + e.R
        if z_c is not None:
            r_up = radius_above(e.model, z_c + HOUSING_DZ - 0.01)
            if r_up > 0:
                need = max(need, housing_r + r_up + 0.003)
        for p in [e.xy] if e.dest is None else [e.xy, e.dest]:
            clear = np.minimum(clear, np.linalg.norm(pts - p, axis=-1) - need)
    if clutter and len(board.obstacles):
        tall = board.obstacle_h + LIFT_CLEAR > over + 1e-06
        if tall.any():
            clear = np.minimum(clear, lattice_distance(board.obstacles[tall], xs, ys) - d.R + 0.004)
        else:
            for p in board.obstacles[tall]:
                clear = np.minimum(clear, np.linalg.norm(pts - p, axis=-1) - d.R + 0.004)
    if avoid_zone:
        clear = np.minimum(clear, np.linalg.norm(pts - board.zone.xy, axis=-1) - board.zone.radius)
    a, b = np.asarray(a, float), np.asarray(b, float)
    side = np.zeros(pts.shape[:2], bool)
    lim_l = SIDE_X if SIDE_X_DIRECT is None else max(SIDE_X, a[0] + 0.02, b[0] + 0.02)
    lim_r = SIDE_X if SIDE_X_DIRECT is None else max(SIDE_X, -a[0] + 0.02, -b[0] + 0.02)
    if arm == "left":
        side = pts[..., 0] > lim_l
    elif arm == "right":
        side = pts[..., 0] < -lim_r
    r_max = reach_radius(z_c) if arm in K.BASE_POS and z_c is not None else None
    if r_max is not None:
        side = side | (np.linalg.norm(pts - np.asarray(K.BASE_POS[arm][:2]), axis=-1) > r_max - ROUTE_REACH_MARGIN)
    d_end = np.minimum(np.linalg.norm(pts - a, axis=-1), np.linalg.norm(pts - b, axis=-1))
    near_ends = d_end < 0.02
    blocked = ((clear < ROUTE_MARGIN) | side) & ~near_ends
    search = ((clear < ROUTE_MARGIN + ROUTE_SEARCH * np.clip(d_end / 0.06, 0.0, 1.0)) | side) & ~near_ends

    def cell(p):
        return int(round((p[0] - xs[0]) / ROUTE_CELL)), int(round((p[1] - ys[0]) / ROUTE_CELL))

    nx, ny = blocked.shape
    s, g = cell(a), cell(b)
    if not (0 <= s[0] < nx and 0 <= s[1] < ny and (0 <= g[0] < nx) and (0 <= g[1] < ny)):
        return None
    dist = {s: 0.0}
    prev = {}
    heap = [(0.0, s)]
    nb = [(i, j, math.hypot(i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    while heap:
        _, c = heapq.heappop(heap)
        if c == g:
            break
        dc = dist[c]
        for i, j, w in nb:
            n = c[0] + i, c[1] + j
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or search[n]:
                continue
            nd = dc + w
            if nd < dist.get(n, 1e18):
                dist[n] = nd
                prev[n] = c
                heapq.heappush(heap, (nd + math.hypot(n[0] - g[0], n[1] - g[1]), n))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path = [pts[c] for c in reversed(path)]
    path[0], path[-1] = a, b

    def visible(p, q):
        n = max(2, int(np.linalg.norm(q - p) / 0.005) + 1)
        for t in np.linspace(0.0, 1.0, n):
            c = cell(p + t * (q - p))
            if blocked[min(nx - 1, max(0, c[0])), min(ny - 1, max(0, c[1]))]:
                return False
        return True

    via = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and (not visible(path[i], path[j])):
            j -= 1
        if j < len(path) - 1:
            via.append(np.asarray(path[j], float))
        i = j
    return via


def carry_options_steps(board: Board, d: Doll, a, b, arm: str | None = None, avoid_zone: bool = False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    zb = carry_height(board, d, a, b)
    through_zone = avoid_zone and seg_dist(board.zone.xy, a, b) < board.zone.radius
    z_c = LIFT_CLEAR + d.h_g
    out = []
    if zb <= LIFT_CLEAR + 1e-06 and (not through_zone):
        out.append((zb, []))

    def add(zb_k, via):
        if via is not None and all(
            abs(zb_k - z) > 1e-06 or len(via) != len(v) or any(np.linalg.norm(x - y) > 0.01 for x, y in zip(via, v))
            for z, v in out
        ):
            out.append((zb_k, via))

    for hr in (HOUSING_NARROW, HOUSING_R):
        if over_budget():
            yield
        add(LIFT_CLEAR, route_low(board, d, a, b, arm, avoid_zone, z_c=z_c, housing_r=hr))
    zh = max(zb, high_carry(board, d))
    levels = {round(e.h + LIFT_CLEAR, 4) for e in board.dolls.values() if e is not d}
    if len(board.obstacles):
        step = CLUTTER_LEVEL_STEP
        levels |= {round(math.ceil(h / step - 1e-06) * step + LIFT_CLEAR, 4) for h in np.unique(board.obstacle_h)}
    prev = None
    for zl in sorted(z for z in levels if LIFT_CLEAR + 1e-06 < z < zh - 1e-06):
        if zb <= zl + 1e-06 and (not through_zone):
            add(zl, [])
            break
        if over_budget():
            yield
        via = route_low(board, d, a, b, arm, avoid_zone, z_c=zl + d.h_g, housing_r=HOUSING_NARROW, over=zl)
        if via is not None and (
            prev is None or len(via) != len(prev) or any(np.linalg.norm(x - y) > 0.01 for x, y in zip(via, prev))
        ):
            add(zl, via)
            prev = via
    if zh > LIFT_CLEAR + 1e-06 or not out:
        out.append((zh, []))
    return out


def high_carry(board: Board, d: Doll) -> float:
    return max([LIFT_CLEAR] + [e.h + LIFT_CLEAR for e in board.dolls.values() if e is not d])


def free_height(board: Board, a, b, exclude=None) -> float:
    z = 0.06
    for e in board.dolls.values():
        if e is exclude:
            continue
        if seg_dist(e.xy, a, b) < e.R + 0.06:
            z = max(z, e.h + TOP_CLEAR)
    if len(board.obstacles):
        near = seg_dists(board.obstacles, a, b) < 0.05
        if near.any():
            z = max(z, float(board.obstacle_h[near].max()) + TOP_CLEAR)
    return z


def heading_of(T: np.ndarray) -> float:
    return math.atan2(float(T[1, 1]), float(T[0, 1]))


WRIST_ROLL_FAST = (
    2.5
)


def jmove(q0, q1, joint_step: float = FREE_STEP):
    dense = motion.joint_move(q0, q1, joint_step=joint_step)
    if WRIST_ROLL_FAST is None:
        return dense, dense
    d = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    lim = np.ones(len(d))
    lim[5] = WRIST_ROLL_FAST
    d_max, d_scaled = float(np.max(d)), float(np.max(d / lim))
    if d_scaled < 1e-09 or d_scaled >= d_max:
        return dense, dense
    return motion.joint_move(q0, q1, joint_step=joint_step * d_max / d_scaled), dense


SEG_SPEED = {"acrossJ": 1.3, "acrossL": 1.3, "via": 1.2, "carry": 1.2}


def retime(q0, plan, f: float):
    n = len(plan.q)
    m = math.ceil(n / f) if f > 1.0 else n
    if m >= n or n < 2:
        return plan
    pts = np.vstack([np.asarray(q0, float)] + [np.asarray(q, float) for q in plan.q])
    s_old = np.concatenate([[0.0], 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)])
    s_new = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, m + 1) / m)
    out = []
    for s in s_new:
        k = int(min(n, max(1, np.searchsorted(s_old, s))))
        a = (s - s_old[k - 1]) / max(1e-12, s_old[k] - s_old[k - 1])
        out.append(pts[k - 1] + (pts[k] - pts[k - 1]) * float(min(1.0, max(0.0, a))))
    out[-1] = pts[-1].copy()
    return motion.Plan(out, plan.ok, plan.reason)


def faster(arm, seg: str, plan):
    f = (SEG_SPEED or {}).get(seg)
    return plan if not f else retime(arm.q, plan, f)


def seg_line(name: str, q, p, psi: float, lin_step: float, ang_step: float = 6 * DEG, tilt: float = 0.0):
    pl = motion.line(
        name,
        q,
        tcp_pose(G.grasp_rotation(psi, tilt), p),
        K.TCP_OFFSET,
        lin_step=lin_step,
        ang_step=ang_step,
        joint_step=FREE_STEP,
    )
    return pl if pl.ok and pl.q else None


def path_clear(board: Board, name: str, qs, exclude=None, margin: float = 0.01) -> bool:
    if not qs:
        return True
    for q in list(qs[::2]) + [qs[-1]]:
        p = K.point_of(K.fk(name, q), K.TCP_OFFSET)
        for e in board.dolls.values():
            if e is exclude:
                continue
            for c in [e.xy] if e.dest is None else [e.xy, e.dest]:
                if np.linalg.norm(p[:2] - c) < e.R + 0.05 and p[2] < K.TABLE_Z + e.h + TOP_CLEAR - margin:
                    return False
    return True


WHY: dict = {}


def _why(key: str) -> None:
    WHY[key] = int(WHY.get(key, 0)) + 1


N_BANDS = 5
_BANDS: dict = {}


def doll_bands(m: D.DollModel) -> list:
    if m.model_id not in _BANDS:
        out = []
        for k in range(N_BANDS):
            lo, hi = k * m.height / N_BANDS, (k + 1) * m.height / N_BANDS
            rs = [
                float(np.linalg.norm(p, axis=1).max())
                for z, p in m.slices
                if lo - 0.003 <= z - m.z_bottom <= hi + 0.003
            ]
            out.append((lo, hi, max(rs) if rs else m.max_halfwidth()))
        _BANDS[m.model_id] = out
    return _BANDS[m.model_id]


def radius_above(m: D.DollModel, z: float) -> float:
    return max([r for lo, hi, r in doll_bands(m) if hi > z] + [0.0])


def doll_cylinders(board: Board, exclude=()) -> np.ndarray:
    rows = []
    for e in board.dolls.values():
        if any(e is x for x in exclude):
            continue
        for c in [e.xy] if e.dest is None else [e.xy, e.dest]:
            for lo, hi, r in doll_bands(e.model):
                rows.append((float(c[0]), float(c[1]), r + 0.002, K.TABLE_Z + lo, K.TABLE_Z + hi))
    return np.asarray(rows, float).reshape(-1, 5)


def path_hits(name: str, qs, grip: float, cyls: np.ndarray, clear: float, carried=None) -> bool:
    if not len(qs) or not len(cyls):
        return False
    for i in sorted(set(range(0, len(qs), 2)) | {len(qs) - 1}):
        sph = CO.arm_spheres(name, qs[i], grip)[3:]
        dh = np.maximum(0.0, np.linalg.norm(sph[:, None, :2] - cyls[None, :, :2], axis=2) - cyls[None, :, 2])
        dv = np.maximum(0.0, np.maximum(cyls[None, :, 3] - sph[:, None, 2], sph[:, None, 2] - cyls[None, :, 4]))
        if float((np.hypot(dh, dv) - sph[:, None, 3]).min()) < clear:
            return True
        if carried is not None:
            d, hg = carried
            tcp = K.point_of(K.fk(name, qs[i]), K.TCP_OFFSET)
            zb = tcp[2] - hg
            gh = np.linalg.norm(cyls[:, :2] - tcp[:2], axis=1) - cyls[:, 2] - d.model.max_halfwidth()
            gv = np.maximum(cyls[:, 3] - (zb + d.h), zb - cyls[:, 4])
            g = np.where(gv > 0, np.hypot(np.maximum(gh, 0.0), np.maximum(gv, 0.0)), gh)
            if float(g.min()) < clear:
                return True
    return False


def plan_approach(name: str, board: Board, d: Doll, psi: float, z_a: float, q, tilt: float = 0.0):
    segs = []
    q = np.asarray(q, float)
    T0 = K.fk(name, q)
    p0 = K.point_of(T0, K.TCP_OFFSET)
    if p0[2] < K.TABLE_Z + z_a - 0.005:
        up = T0.copy()
        up[2, 3] += K.TABLE_Z + z_a - p0[2]
        pl = motion.line(name, q, up, 0.0, lin_step=0.014, joint_step=FREE_STEP)
        if pl.ok and pl.q:
            segs.append(("up", pl))
            q = pl.q[-1]
    q_above = ik(name, psi, tcp_at(d.xy, z_a), q, tilt)
    if q_above is None:
        _why("ik_above")
        return None
    g_open = grip_for_gap(min(GAP_MAX, d.w + OPEN_EXTRA))
    every = doll_cylinders(board)
    others = doll_cylinders(board, exclude=(d,))
    jm, jd = jmove(q, q_above)
    if path_clear(board, name, jd.q, exclude=d) and (not path_hits(name, jd.q, g_open, every, ARM_CLEAR)):
        segs.append(("acrossJ", jm))
    else:
        pl = seg_line(name, q, tcp_at(d.xy, z_a), psi, 0.016, tilt=tilt)
        if pl is None:
            _why("across_ik")
            return None
        if not path_clear(board, name, pl.q, exclude=d) or path_hits(name, pl.q, g_open, every, ARM_CLEAR):
            _why("across_hit")
            return None
        segs.append(("acrossL", pl))
    q = segs[-1][1].q[-1] if segs[-1][1].q else q
    pl = seg_line(name, q, tcp_at(d.xy, d.h_g), psi, 0.013, tilt=tilt)
    if pl is None:
        _why("down_ik")
        return None
    if path_hits(name, pl.q, g_open, others, NEAR_CLEAR):
        _why("down_hit")
        return None
    segs.append(("down", pl))
    return segs


CARRY_FAIL: list = [None]


def plan_carry(
    name: str,
    d: Doll,
    dest,
    psi_p: float,
    psi_q: float,
    z_c: float,
    via,
    q,
    board: Board | None = None,
    turn: float | None = None,
    tilt: float = 0.0,
):
    segs = []
    CARRY_FAIL[0] = None
    others = doll_cylinders(board, exclude=(d,)) if board is not None else np.zeros((0, 4))
    g_held = grip_for_gap(d.w)
    g_rel = grip_for_gap(min(GAP_MAX, d.w + 0.016))
    turn = wrap(psi_q - psi_p) if turn is None else turn
    if abs(turn) > math.pi and (not len(via)):
        via = [0.5 * (np.asarray(d.xy, float) + np.asarray(dest, float))]
    moves = [("lift", d.xy, z_c, psi_p, 0.012)]
    moves += [("via", c, z_c, psi_p + 0.5 * turn, CARRY_LIN) for c in via]
    moves += [
        ("carry", dest, z_c, psi_p + turn, CARRY_LIN),
        ("place", dest, d.h_g + PLACE_DROP, psi_p + turn, 0.01),
        ("retreat", dest, max(d.h + TOP_CLEAR, z_c), psi_p + turn, 0.014),
    ]
    for seg, xy, z, psi, step in moves:
        pl = seg_line(name, q, tcp_at(xy, z), psi, step, tilt=tilt)
        if pl is None:
            _why(f"{seg}_ik")
            CARRY_FAIL[0] = f"{seg}_ik"
            return None
        if seg in ("via", "carry"):
            bad = path_hits(name, pl.q, g_held, others, ARM_CLEAR, carried=(d, d.h_g))
        elif seg == "retreat":
            bad = path_hits(name, pl.q, g_rel, others, NEAR_CLEAR)
        else:
            bad = path_hits(name, pl.q, g_held, others, NEAR_CLEAR)
        if bad:
            _why(f"{seg}_hit")
            CARRY_FAIL[0] = f"{seg}_hit"
            return None
        segs.append((seg, pl))
        q = pl.q[-1]
    return segs


MAX_PLAN_TRIES = 60
MAX_LONG_TRIES = 12
LONG_TURN_MAX = 300 * DEG
STEP_BUDGET = 1.0
PREFETCH_WAIT = 2.0
PREFETCH_CALL_WAIT = 40.0
PREFETCH_WORKERS = 8
_STEP_T0: list = [None]
CALL_BUDGET = 8.0
DETECT_CALL_BUDGET = 4.0
_CALL = [None, 0.0]


def call_elapsed(step: int) -> float:
    if _CALL[0] != step:
        _CALL[0] = step
        _CALL[1] = time.perf_counter()
    return time.perf_counter() - _CALL[1]


def over_budget() -> bool:
    t0 = _STEP_T0[0]
    if t0 is None:
        return False
    now = time.perf_counter()
    return now - t0 > STEP_BUDGET or (
        CALL_BUDGET is not None and _CALL[0] is not None and (now - _CALL[1] > CALL_BUDGET)
    )


def timed(arm: Arm, task):
    while True:
        call_elapsed(arm.ctx.obs.step)
        t0 = _STEP_T0[0] = time.perf_counter()
        try:
            next(task)
        except StopIteration:
            return
        finally:
            _STEP_T0[0] = None
        dt = time.perf_counter() - t0
        if dt > getattr(arm, "t_max", (0.0, 0))[0]:
            arm.t_max = dt, arm.ctx.obs.step
        yield


def reach_only(why: dict, behind: bool = False) -> bool:
    ks = [k for k, v in why.items() if isinstance(v, int) and k not in ("long_turn", "tilt_fail")]
    if bool(ks) and all(k.endswith("_ik") for k in ks):
        return True
    if not behind:
        return False
    reach_ks = [k for k in ks if k.endswith("_ik") or k in ("ik_above", "pick_none")]
    return bool(reach_ks) and all(k in reach_ks or k == "across_hit" for k in ks)


def plan_job(arm: Arm, board: Board, job: Job, tilt_retry: bool = True):
    res = yield from _plan_job(arm, board, job)
    if res is not None or not tilt_retry or (not TILT_PICK):
        return res
    why0 = dict(WHY)
    spot = job.doll.tilt_spot or (job.kind in ("relay", "buffer") and board.tilt_only(job.doll, job.dest))
    behind = bool(job.doll.xy[1] > float(board.slots[job.doll.rank][1]) + ACCESS_BEHIND)
    if not (board.needs_tilt(job.doll) or spot or reach_only(why0, behind)):
        return None
    for tilt in TILT_PICK:
        for sgn in (1.0, -1.0):
            res = yield from _plan_job(arm, board, job, sgn * tilt)
            if res is not None:
                return res
    WHY.clear()
    WHY.update(why0)
    WHY["tilt_fail"] = 1


def _plan_job(arm: Arm, board: Board, job: Job, tilt: float = 0.0):
    if over_budget():
        yield
    d = job.doll
    WHY.clear()
    h_g, w = d.h_g, d.w
    open_gap = min(GAP_MAX, w + OPEN_EXTRA)
    psis_p, clear, n_ok = pick_psis(board, d, open_gap)
    carries = yield from carry_options_steps(board, d, d.xy, job.dest, arm.name, avoid_zone=not job.zone)
    WHY["opts"] = [f"{zb_k:.3f}/{len(v)}" for zb_k, v in carries]
    carries = [
        (zb_k, v)
        for zb_k, v in carries
        if all(
            reach(arm.name, p, zb_k + h_g) or (tilt and reach_tilted(arm.name, p, zb_k + h_g) is not None)
            for p in [d.xy, *v, job.dest]
        )
    ] or carries[:1]
    WHY["board"] = [
        (
            e.rank,
            e.model.model_id,
            round(float(e.xy[0]), 3),
            round(float(e.xy[1]), 3),
            None if e.dest is None else [round(float(x), 3) for x in e.dest],
        )
        for e in board.dolls.values()
    ]
    z_full = max(free_height(board, arm.tcp()[:2], d.xy, exclude=d), d.h + TOP_CLEAR)
    tries = 0
    z_list = sorted({round(z_full, 4), round(d.h + TOP_CLEAR, 4)}, reverse=True)
    order = []
    for i, psi_p in enumerate(psis_p):
        for z_a in z_list:
            qa = ik(arm.name, psi_p, tcp_at(d.xy, z_a), arm.q, tilt)
            if qa is not None:
                order.append((i >= n_ok, float(np.max(np.abs(qa - arm.q))), i, psi_p))
                break
    order.sort()
    if not order:
        _why("pick_none")
    long_way = []
    for *_, psi_p in order:
        if tries > MAX_PLAN_TRIES:
            break
        app = None
        for z_a in z_list:
            if over_budget():
                yield
            app = plan_approach(arm.name, board, d, psi_p, z_a, arm.q, tilt)
            if app is not None:
                break
        if app is None:
            continue
        q_g = app[-1][1].q[-1]
        for zb_k, via_k in carries:
            for psi_q in sorted(PLACE_PSIS, key=lambda a: abs(wrap(a - psi_p))):
                tries += 1
                if tries > MAX_PLAN_TRIES:
                    break
                if over_budget():
                    yield
                z_ck = zb_k + h_g
                car = plan_carry(arm.name, d, job.dest, psi_p, psi_q, z_ck, via_k, q_g, board, tilt=tilt)
                if car is None:
                    if CARRY_FAIL[0] in ("via_ik", "carry_ik"):
                        long_way.append((app, q_g, z_a, psi_p, zb_k, via_k, psi_q))
                    continue
                return dict(
                    app=app,
                    car=car,
                    via=via_k,
                    psi_p=psi_p,
                    psi_q=psi_q,
                    turn=wrap(psi_q - psi_p),
                    h_g=h_g,
                    w=w,
                    open=open_gap,
                    clear=clear,
                    z_c=z_ck,
                    z_a=z_a,
                    tilt=round(tilt, 4),
                )
            if tries > MAX_PLAN_TRIES:
                break
    for k, (app, q_g, z_a, psi_p, zb_k, via_k, psi_q) in enumerate(long_way[:MAX_LONG_TRIES]):
        t = wrap(psi_q - psi_p)
        t_long = t - math.copysign(2 * math.pi, t)
        if abs(t_long) > LONG_TURN_MAX:
            continue
        if over_budget():
            yield
        z_ck = zb_k + h_g
        car = plan_carry(arm.name, d, job.dest, psi_p, psi_q, z_ck, via_k, q_g, board, turn=t_long, tilt=tilt)
        if car is not None:
            WHY["long_turn"] = k + 1
            return dict(
                app=app,
                car=car,
                via=via_k,
                psi_p=psi_p,
                psi_q=psi_q,
                turn=t_long,
                h_g=h_g,
                w=w,
                open=open_gap,
                clear=clear,
                z_c=z_ck,
                z_a=z_a,
                tilt=round(tilt, 4),
            )


def put_down(arm: Arm, d: Doll, job: Job):
    T = arm.T.copy()
    p = arm.tcp()
    T[2, 3] += K.TABLE_Z + d.h_g + PLACE_DROP - p[2]
    yield from arm.line_to(T, 0.0, lin_step=0.01, fallback_joint=False)
    yield from arm.set_grip(grip_for_gap(min(GAP_MAX, d.w + 0.016)), n=6)
    job.put = arm.tcp()[:2].copy()
    job.turn = wrap(heading_of(arm.T) - job.psi_p)
    up = arm.T.copy()
    up[2, 3] += max(0.0, K.TABLE_Z + d.h + TOP_CLEAR - arm.tcp()[2])
    yield from arm.line_to(up, 0.0, lin_step=0.014, fallback_joint=False)


def run_job(arm: Arm, board: Board, job: Job):
    try:
        yield from _run_job(arm, board, job)
    finally:
        board.corridors.pop(arm.name, None)


def _corridor(arm: Arm, board: Board, d, pts, pl, live: bool = False) -> None:
    board.corridors[arm.name] = np.asarray([np.asarray(p, float)[:2] for p in pts]), d.R, pl["z_c"] - pl["h_g"], live


def _run_job(arm: Arm, board: Board, job: Job):
    d = job.doll
    d.alt_grip = False
    pl = yield from plan_job(arm, board, job, tilt_retry=job.zone)
    if pl is None and (not job.zone):
        job.zone = True
        pl = yield from plan_job(arm, board, job)
    if pl is None and grasp_heights(d.model)[1] is not None:
        d.alt_grip = True
        pl = yield from plan_job(arm, board, job, tilt_retry=False)
        if pl is None:
            d.alt_grip = False
    if pl is None:
        arm.status = "fail"
        return
    d.turn = pl["turn"]
    job.psi_p = pl["psi_p"]
    vias = [np.asarray(c, float) for c in pl["via"]]
    n_car_via = sum(1 for s, _ in pl["car"] if s == "via")
    if n_car_via > len(vias):
        vias += [0.5 * (np.asarray(d.xy, float) + np.asarray(job.dest, float))] * (n_car_via - len(vias))
    _corridor(arm, board, d, [d.xy, *vias, job.dest], pl)
    g_open = grip_for_gap(pl["open"])
    segs = pl["app"] + pl["car"]
    xys = [seg_xy(arm.name, plan) for _, plan in segs]
    enters = [seg_enters_zone(arm.name, board, plan, xy) for (_, plan), xy in zip(segs, xys)]
    k = 0
    for seg, plan in pl["app"]:
        if not (yield from zone_for(arm, board, enters, k, xys)):
            return
        ok = yield from arm.follow(faster(arm, seg, plan), g_open if seg != "down" else None)
        k += 1
        if not ok:
            arm.status = "blocked"
            return
        if seg.startswith("across"):
            yield from arm.set_grip(g_open, n=2)
    yield from arm.hold(1)
    yield from arm.set_grip(0.0, n=6)
    yield from arm.hold(2)
    wid = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]), 0.0)
    if wid < 0.5 * pl["w"]:
        yield from arm.set_grip(g_open, n=4)
        up = arm.T.copy()
        up[2, 3] += pl["z_a"] - pl["h_g"]
        yield from arm.line_to(up, 0.0, lin_step=0.014, fallback_joint=False)
        arm.status = "empty"
        return
    board.note_grasp(d, pl["h_g"], wid)
    carried = []
    n_via = 0
    for seg, plan in pl["car"]:
        if seg == "via":
            _corridor(arm, board, d, [arm.tcp()[:2], *vias[n_via:], job.dest], pl, live=True)
            n_via += 1
        elif seg == "carry":
            waited = 0
            while waited < CROSS_WAIT and board.crosses(
                arm.name,
                d,
                job.dest,
                now={arm.other.name: arm.other.tcp()[:2]},
                skip=board.cross_waiting | set(board.zone_want),
                live_only=False,
            ):
                board.cross_waiting.add(arm.name)
                yield from arm.hold(1)
                waited += 1
            board.cross_waiting.discard(arm.name)
            _corridor(arm, board, d, [arm.tcp()[:2], job.dest], pl, live=True)
        elif seg == "place":
            _corridor(arm, board, d, [job.dest], pl)
        elif seg == "retreat":
            board.corridors.pop(arm.name, None)
        if seg == "retreat":
            yield from arm.hold(1)
            yield from arm.set_grip(grip_for_gap(min(GAP_MAX, wid + 0.016)), n=6)
            yield from arm.hold(1)
        if not (yield from zone_for(arm, board, enters, k, xys)):
            yield from put_down(arm, d, job)
            arm.status = "blocked"
            return
        if seg in ("via", "carry") and carried_hits(
            board, d, arm.tcp()[:2], vias[n_via - 1] if seg == "via" else job.dest, pl["z_c"] - pl["h_g"]
        ):
            yield from put_down(arm, d, job)
            arm.status = "blocked"
            return
        if seg in ("via", "carry"):
            ok = yield from follow_back_off(arm, faster(arm, seg, plan), carried)
        else:
            ok = yield from arm.follow(plan, None)
        k += 1
        if seg == "lift":
            if ok:
                w_up = held_width(float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0]), 0.0)
                if w_up < 0.5 * pl["w"]:
                    yield from arm.set_grip(g_open, n=4)
                    arm.status = "empty"
                    return
        if not ok:
            if seg == "retreat":
                break
            yield from put_down(arm, d, job)
            arm.status = "blocked"
            return
    arm.status = "ok"


BACK_OFF = 1
BACK_STEPS = 8
BACK_HOLD = 10


def _on_plan(arm: Arm, qs) -> int:
    for i in range(len(qs) - 1, -1, -1):
        if float(np.max(np.abs(qs[i] - arm.q))) < 1e-09:
            return i
    return -1


def follow_back_off(arm: Arm, plan, done: list | None = None):
    done = [] if done is None else done
    todo = [np.asarray(q, dtype=float) for q in plan.q]
    for attempt in range(BACK_OFF + 1):
        ok = yield from arm.follow(motion.Plan(list(todo)), None)
        j = _on_plan(arm, todo)
        done.extend(todo[: j + 1])
        if ok or arm.status != "blocked" or attempt == BACK_OFF:
            return ok
        rest = todo[j + 1 :]
        back = done[-1 - BACK_STEPS : -1][::-1]
        m = -1
        if back:
            yield from arm.follow(motion.Plan(list(back)), None)
            m = _on_plan(arm, back)
        yield from arm.hold(BACK_HOLD)
        if m >= 0:
            todo = back[:m][::-1] + [done[-1]] + rest
            del done[len(done) - (m + 1) :]
        else:
            todo = rest
    return False


ZONE_MARGIN = 0.05
ZONE_SHARE_X = 0.18


def seg_xy(name: str, plan) -> np.ndarray:
    qs = list(plan.q)
    return np.array([K.point_of(K.fk(name, q), K.TCP_OFFSET)[:2] for q in qs]).reshape(-1, 2)


def seg_enters_zone(name: str, board: Board, plan, xy: np.ndarray | None = None) -> bool:
    r = board.zone.radius + ZONE_MARGIN
    xy = seg_xy(name, plan) if xy is None else xy
    return bool(len(xy) and np.any(np.linalg.norm(xy - board.zone.xy, axis=1) < r))


def zone_for(arm: Arm, board: Board, enters, k: int, xys=None):
    z = board.zone
    rest = None
    if xys is not None:
        rest = np.concatenate(
            [np.asarray(arm.tcp(), dtype=float)[None, :2]] + [np.asarray(x).reshape(-1, 2) for x in xys[k:]]
        )
    shared = isinstance(z, PathZone)
    holds = z.holds(arm.name) if shared else z.owner == arm.name
    if enters[k] and (not holds):
        free = (lambda: z.fits(arm.name, rest)) if shared else lambda: z.owner is None
        if not free():
            board.zone_want[arm.name] = arm.ctx.obs.step
            yield from arm.wait_until(free, max_steps=300)
        board.zone_want.pop(arm.name, None)
        if not (z.take(arm.name, rest) if shared else z.take(arm.name)):
            arm.status = "blocked"
            return False
        if shared and len(z.paths) > 1:
            board.shares[arm.name] = board.shares.get(arm.name, 0) + 1
    elif holds and (not any(enters[k:])) and (not near_zone(arm, board, ZONE_MARGIN)):
        z.free(arm.name)
    elif holds and shared and (rest is not None):
        z.update(arm.name, rest)
    return True


def near_zone(arm: Arm, board: Board, margin: float) -> bool:
    return float(np.linalg.norm(arm.tcp()[:2] - board.zone.xy)) < board.zone.radius + margin


def side_q(arm: Arm) -> np.ndarray:
    for psi in (90 * DEG, -90 * DEG, 0.0, math.pi):
        q = ik(arm.name, psi, tcp_at(SIDE_XY[arm.name], SIDE_Z), arm.q)
        if q is not None:
            return q
    return LOOK_Q.copy()


RISE_Z = 0.17
RISE_STEP = 0.01
RISE_BLOCKED = 3


def rise(arm: Arm):
    z = K.TABLE_Z + RISE_Z
    blocked = 0
    while z > arm.tcp()[2] + 0.5 * RISE_STEP:
        up = arm.T.copy()
        up[2, 3] += z - arm.tcp()[2]
        yield from arm.line_to(up, 0.0, lin_step=0.014, fallback_joint=False)
        if arm.status == "ok":
            return
        if arm.status == "blocked":
            blocked += 1
            if blocked > RISE_BLOCKED:
                return
            yield from arm.hold(4)
            continue
        z -= RISE_STEP
        if over_budget():
            yield


def retreat_clear(arm: Arm, board: Board, qs, job=None) -> bool:
    d = None if job is None else job.doll
    return path_clear(board, arm.name, qs, exclude=d) and (
        not path_hits(arm.name, qs, arm.g, doll_cylinders(board, exclude=() if d is None else (d,)), NEAR_CLEAR)
    )


def retreat(arm: Arm, board: Board, job=None, zone_exit: bool = False):
    goal = board.side_q.get(arm.name, LOOK_Q)
    yield from rise(arm)
    jm, jd = jmove(arm.q, goal)
    not retreat_clear(arm, board, jd.q, job)
    k = len(jm.q)
    if zone_exit and board.zone.holds(arm.name):
        r = board.zone.radius + ZONE_MARGIN
        out = np.linalg.norm(seg_xy(arm.name, jm) - board.zone.xy, axis=1) >= r if jm.q else np.zeros(0, bool)
        while k > 0 and out[k - 1]:
            k -= 1
    yield from arm.follow(motion.Plan(list(jm.q[:k])), None, force=True)
    if zone_exit:
        board.zone.free(arm.name)
    yield from arm.follow(motion.Plan(list(jm.q[k:])), None, force=True)
    arm.status = "ok"
    return True


def park(arm: Arm):
    yield from rise(arm)
    ok = yield from arm.follow(jmove(arm.q, LOOK_Q)[0], None, force=True)
    arm.status = "ok" if ok else "blocked"


def worker(arm: Arm, board: Board):
    parked = False
    idle_since = None
    board.side_q[arm.name] = side_q(arm)
    while True:
        if over_budget():
            yield
        v_claim = board.version
        job = board.claim(arm.name)
        if job is None:
            board.no_job_v[arm.name] = v_claim
            if not board.pending() or board.steps_left() < JOB_STEPS:
                break
            if board.stalled() and (not board.retry_marked()) and (not board.loosen()):
                break
            board.idle.add(arm.name)
            if not parked:
                parked = yield from retreat(arm, board)
            if idle_since is None:
                idle_since = arm.ctx.obs.step
            v = board.version
            yield from arm.wait_until(lambda: board.version != v, max_steps=30)
            if arm.ctx.obs.step - idle_since > 700:
                break
            continue
        board.idle.discard(arm.name)
        idle_since = None
        parked = False
        yield from run_job(arm, board, job)
        status = arm.status
        ok = status == "ok"
        board.finish(job, ok, arm.name, status)
        if board.zone.holds(arm.name):
            if near_zone(arm, board, ZONE_MARGIN):
                parked = yield from retreat(arm, board, job, zone_exit=True)
            board.zone.free(arm.name)
        elif status == "blocked" and near_zone(arm, board, 0.06):
            parked = yield from retreat(arm, board, job)
        arm.status = "idle"
        if status == "blocked":
            yield from arm.hold(6)
    board.idle.add(arm.name)
    if not parked:
        yield from retreat(arm, board)


def go_home(arm: Arm):
    T_up = K.fk(arm.name, arm.q_home)
    T_up[2, 3] += 0.06
    r = K.solve_near(arm.name, T_up, arm.q_home, 0.0)
    q_goal = r.q if r.ok else arm.q_home
    ok = yield from arm.follow(jmove(arm.q, q_goal)[0], 1.0, force=True)
    arm.status = "ok" if ok else "blocked"


@register
class SortDolls(Program):
    name = "sort_dolls"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "doll" in s and ("size" in s or "smallest" in s or "largest" in s):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.obs
        seg.prefetch(PREFETCH_WORKERS)
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        yield from run_arms(arms, {a: park(arms[a]) for a in K.ARMS})
        yield from run_arms(arms, {a: arms[a].hold(4) for a in K.ARMS})
        while not seg.prefetched(
            PREFETCH_WAIT if PREFETCH_CALL_WAIT is None else max(0.0, PREFETCH_CALL_WAIT - call_elapsed(ctx.obs.step))
        ) or (CALL_BUDGET is not None and call_elapsed(ctx.obs.step) > DETECT_CALL_BUDGET):
            yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
        cam = C.head()
        scene = D.detect(ctx.obs.images.get("cam_head"), cam)
        yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
        dolls = {d.rank: Doll(d.rank, D.model(d.model_id), np.array([d.x, d.y])) for d in scene.dolls}
        obstacles, obstacle_h = obstacle_cells(scene, cam)
        board = None
        if len(dolls) >= 2:
            slots, _ = plan_row(dolls, obstacles)
            slots = {r: s for r, s in slots.items() if r in dolls}
            for d in dolls.values():
                d.done = bool(np.linalg.norm(d.xy - slots[d.rank]) < DONE_TOL)
            board = Board(ctx, dolls, slots, obstacles, obstacle_h)
            link_arms(arms, board.zone)
            yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
            yield from run_arms(arms, {a: timed(arms[a], worker(arms[a], board)) for a in K.ARMS})
        yield from run_arms(arms, {a: go_home(arms[a]) for a in K.ARMS})
        yield from run_arms(arms, {a: arms[a].hold(4) for a in K.ARMS})
        passes = 0
        while board is not None and passes < VERIFY_PASSES and (STEP_LIMIT - ctx.obs.step >= VERIFY_MIN_STEPS):
            yield from run_arms(arms, {a: arms[a].hold(VERIFY_WAIT) for a in K.ARMS})
            passes += 1
            yield from run_arms(arms, {a: park(arms[a]) for a in K.ARMS})
            yield from run_arms(arms, {a: arms[a].hold(2) for a in K.ARMS})
            scene2 = D.detect(ctx.obs.images.get("cam_head"), cam)
            board2 = rebuild_board(ctx, board, scene2, *obstacle_cells(scene2, cam))
            if board2 is not None:
                board = board2
                link_arms(arms, board.zone)
                yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
                yield from run_arms(arms, {a: timed(arms[a], worker(arms[a], board)) for a in K.ARMS})
            yield from run_arms(arms, {a: go_home(arms[a]) for a in K.ARMS})
            yield from run_arms(arms, {a: arms[a].hold(4) for a in K.ARMS})
            if board2 is None:
                break
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
