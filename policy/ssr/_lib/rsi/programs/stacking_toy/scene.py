from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera

_DATA = json.loads((ASSETS / "stacking_toy.json").read_text())
STAR, SQUARE, RECT, DISC = 0, 1, 2, 3
SYMMETRY = {STAR: 2 * math.pi / 5, SQUARE: math.pi / 2, RECT: math.pi, DISC: 0.0}
THICK = 0.013
OUTLINES = {
    int(k): np.asarray(v["outline"], dtype=float) - np.asarray(v.get("hole_center", (0.0, 0.0)))
    for k, v in _DATA["pieces"].items()
}
_B = _DATA["base"]
BASE_OUTLINE = np.asarray(_B["outline"], dtype=float)
PLATE_TOP = float(_B["plate_top"])
POLES = [(float(p["center"][0]), float(p["center"][1])) for p in _B["poles"]]
POLE_TOP = [float(p["top"]) for p in _B["poles"]]
POLE_R = float(np.mean([p["radius"] for p in _B["poles"]]))
SS = 2


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _prism_polys(cam: Camera, xy: np.ndarray, z0: float, z1: float, top_only: bool = False):
    n = len(xy)
    top, dt = cam.project(np.column_stack([xy, np.full(n, z1)]))
    if np.any(dt <= 0.01):
        return [], []
    if top_only:
        return [top], []
    bot, db = cam.project(np.column_stack([xy, np.full(n, z0)]))
    if np.any(db <= 0.01):
        return [], []
    quads = [np.array([bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]]) for i in range(n)]
    return [top, bot], quads


def _cyl_poly(cam: Camera, x: float, y: float, r: float, z0: float, z1: float, n: int = 16):
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
    pts = np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z1)])])
    uv, d = cam.project(pts)
    if np.any(d <= 0.01):
        return None
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def _draw(shape, origin, polys, quads, ss: int = 1) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    o = np.asarray(origin, dtype=float)
    sc = 16 * ss
    for p in polys:
        cv2.fillPoly(m, [np.round((np.asarray(p) - o) * sc).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    for q in quads:
        cv2.fillConvexPoly(m, np.round((np.asarray(q) - o) * sc).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return m


def piece_outline(cat: int, x: float, y: float, yaw: float) -> np.ndarray:
    return OUTLINES[cat] @ _rot(yaw).T + np.array([x, y])


def piece_polys(cam: Camera, cat: int, x: float, y: float, yaw: float, z0: float = K.TABLE_Z, top_only: bool = False):
    return _prism_polys(cam, piece_outline(cat, x, y, yaw), z0, z0 + THICK, top_only)


def base_polys(
    cam: Camera, x: float, y: float, yaw: float, z0: float = K.TABLE_Z, poles: bool = True, plate_top_only: bool = False
):
    R = _rot(yaw)
    polys, quads = _prism_polys(cam, BASE_OUTLINE @ R.T + np.array([x, y]), z0, z0 + PLATE_TOP, top_only=plate_top_only)
    if poles:
        for (px, py), top in zip(POLES, POLE_TOP):
            c = R @ np.array([px, py]) + np.array([x, y])
            p = _cyl_poly(cam, c[0], c[1], POLE_R, z0 + PLATE_TOP, z0 + top)
            if p is not None:
                quads.append(p)
    return polys, quads


def pole_world(base, k: int) -> np.ndarray:
    c = _rot(base.yaw) @ np.array(POLES[k]) + np.array([base.x, base.y])
    return np.array([c[0], c[1], K.TABLE_Z + POLE_TOP[k]])


BANDS = {
    STAR: ((125, 158), 100, 60),
    SQUARE: ((18, 36), 100, 120),
    RECT: ((104, 130), 90, 35),
    DISC: ((-8, 13), 140, 188),
}
BASE_MAX_S, BASE_MIN_V = 70, 140


def hsv_image(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)


def colour_mask(hsv: np.ndarray, cat: int) -> np.ndarray:
    (h0, h1), smin, vmin = BANDS[cat]
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    if h0 < 0:
        hm = (h >= 180 + h0) | (h <= h1)
    else:
        hm = (h >= h0) & (h <= h1)
    return hm & (s >= smin) & (v >= vmin)


def base_colour_mask(hsv: np.ndarray) -> np.ndarray:
    return (hsv[..., 1] < BASE_MAX_S) & (hsv[..., 2] >= BASE_MIN_V)


def fill_holes(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m8)
    cv2.drawContours(out, cs, -1, 1, thickness=-1)
    return out.astype(bool)


def components(mask: np.ndarray, min_area: int):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


@dataclass
class Piece:
    cat: int
    x: float
    y: float
    yaw: float
    iou: float
    area: int = 0

    def as_dict(self):
        return {
            "cat": self.cat,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(math.degrees(self.yaw), 1),
            "iou": round(self.iou, 3),
        }


@dataclass
class Base:
    x: float
    y: float
    yaw: float
    iou: float

    def as_dict(self):
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(math.degrees(self.yaw), 2),
            "iou": round(self.iou, 3),
        }


@dataclass
class Scene:
    pieces: list = field(default_factory=list)
    base: Base | None = None
    pole_of_cat: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "pieces": [p.as_dict() for p in self.pieces],
            "base": None if self.base is None else self.base.as_dict(),
            "pole_of_cat": self.pole_of_cat,
        }


class _Window:
    def __init__(self, mask: np.ndarray, box, bad: np.ndarray | None = None, ss: int = SS):
        c0, r0, c1, r1 = box
        self.origin = float(c0), float(r0)
        self.ss = ss
        win = mask[r0:r1, c0:c1].astype(np.uint8)
        self.m = np.kron(win, np.ones((ss, ss), np.uint8)).astype(bool)
        self.shape = self.m.shape
        self.valid = None
        if bad is not None:
            b = bad[r0:r1, c0:c1].astype(np.uint8)
            self.valid = ~np.kron(b, np.ones((ss, ss), np.uint8)).astype(bool)

    def iou(self, polys, quads) -> float:
        if not polys:
            return 0.0
        r = _draw(self.shape, self.origin, polys, quads, self.ss).astype(bool)
        m = self.m
        if self.valid is not None:
            r, m = r & self.valid, m & self.valid
        u = np.count_nonzero(r | m)
        return float(np.count_nonzero(r & m)) / u if u else 0.0


def _box(mask: np.ndarray, pad: int):
    rows, cols = np.nonzero(mask)
    h, w = mask.shape
    return (
        max(0, int(cols.min()) - pad),
        max(0, int(rows.min()) - pad),
        min(w, int(cols.max()) + 1 + pad),
        min(h, int(rows.max()) + 1 + pad),
    )


def _search(score, x, y, yaw, steps):
    best = score(x, y, yaw), x, y, yaw
    for dxy, dyaw, n in steps:
        cur = best
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1) if dyaw > 0 else (0,):
                    if i == j == k == 0:
                        continue
                    xx, yy, aa = cur[1] + i * dxy, cur[2] + j * dxy, cur[3] + k * dyaw
                    s = score(xx, yy, aa)
                    if s > best[0]:
                        best = s, xx, yy, aa
    return best


def fit_piece(
    cam: Camera,
    comp: np.ndarray,
    cat: int,
    *,
    bad=None,
    z0: float = K.TABLE_Z,
    init=None,
    yaw_step: float = math.radians(6.0),
) -> Piece | None:
    rows, cols = np.nonzero(comp)
    if len(rows) < 10:
        return None
    win = _Window(comp, _box(comp, 14), bad)
    zc = z0 + THICK / 2

    def score(x, y, a):
        return win.iou(*piece_polys(cam, cat, x, y, a, z0))

    if init is None:
        p = cam.on_plane_z(np.array([[cols.mean() + 0.5, rows.mean() + 0.5]]), zc)[0]
        sym = SYMMETRY[cat]
        yaws = np.arange(0.0, sym, yaw_step) if sym > 0 else np.array([0.0])
        best = None
        for a in yaws:
            x, y = float(p[0]), float(p[1])
            for _ in range(2):
                r = _draw(win.shape, win.origin, *piece_polys(cam, cat, x, y, a, z0), win.ss)
                rr, cc = np.nonzero(r)
                if len(rr) == 0:
                    break
                uv = np.array(
                    [[win.origin[0] + (cc.mean() + 0.5) / win.ss, win.origin[1] + (rr.mean() + 0.5) / win.ss]]
                )
                q = cam.on_plane_z(uv, zc)[0]
                x, y = x + float(p[0] - q[0]), y + float(p[1] - q[1])
            s = score(x, y, a)
            if best is None or s > best[0]:
                best = s, x, y, a
        _, x, y, a = best
    else:
        x, y, a = init
    dyaw = math.radians(2.0) if SYMMETRY[cat] > 0 else 0.0
    s, x, y, a = _search(score, x, y, a, [(0.001, dyaw, 1), (0.0005, dyaw / 2, 1), (0.00025, dyaw / 4, 1)])
    sym = SYMMETRY[cat]
    a = a % sym if sym > 0 else 0.0
    return Piece(cat, x, y, a, s, int(len(rows)))


def fit_base(cam: Camera, comp: np.ndarray, *, bad=None, z0: float = K.TABLE_Z) -> Base | None:
    rows, cols = np.nonzero(comp)
    if len(rows) < 50:
        return None
    pts = cam.on_plane_z(np.column_stack([cols + 0.5, rows + 0.5]), z0 + PLATE_TOP / 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    c = pts[:, :2].mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts[:, :2] - c).T))
    ax = evec[:, int(np.argmax(ev))]
    yaw = math.atan2(ax[1], ax[0])
    if yaw > math.pi / 2:
        yaw -= math.pi
    elif yaw <= -math.pi / 2:
        yaw += math.pi
    win = _Window(comp, _box(comp, 12), bad)

    def score(x, y, a):
        return win.iou(*base_polys(cam, x, y, a, z0, plate_top_only=True))

    s, x, y, a = _search(
        score,
        float(c[0]),
        float(c[1]),
        yaw,
        [
            (0.002, math.radians(1.0), 2),
            (0.001, math.radians(0.5), 1),
            (0.0005, math.radians(0.25), 1),
            (0.00025, math.radians(0.1), 1),
        ],
    )
    return Base(x, y, a, s)


def assign_poles(pieces) -> dict:
    counts = {c: sum(1 for p in pieces if p.cat == c) for c in range(4)}
    order = sorted(range(4), key=lambda c: (-counts[c], c))
    return {c: k for k, c in enumerate(order)}


WORKSPACE = (-0.6, 0.6), (-0.45, 0.15)
_REGION_CACHE: dict = {}


def table_region(cam: Camera, z: float = K.TABLE_Z, region=WORKSPACE, shape=(480, 640)) -> np.ndarray:
    key = tuple(np.round(cam.T.ravel(), 5)), round(z, 4), region, shape
    if key not in _REGION_CACHE:
        if len(_REGION_CACHE) > 8:
            _REGION_CACHE.clear()
        vv, uu = np.mgrid[0 : shape[0], 0 : shape[1]]
        p = cam.on_plane_z(np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5]), z).reshape(shape[0], shape[1], 3)
        (x0, x1), (y0, y1) = region
        with np.errstate(invalid="ignore"):
            _REGION_CACHE[key] = (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1)
    return _REGION_CACHE[key]


def disc_ambiguous(hsv: np.ndarray, strict: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    loose = ((h >= 172) | (h <= 14)) & (s >= 110) & (v >= 90)
    near = cv2.dilate(strict.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return loose & near & ~strict


def detect_head(
    img: np.ndarray,
    cam: Camera,
    bad: np.ndarray | None = None,
    *,
    min_iou: float = 0.55,
    z0: float = K.TABLE_Z,
    base_candidates: int = 3,
) -> Scene:
    hsv = hsv_image(img)
    valid = table_region(cam, z0, shape=img.shape[:2])
    if bad is not None:
        valid = valid & ~bad
    scene = Scene()
    bm = base_colour_mask(hsv) & valid
    bm = cv2.morphologyEx(bm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    comps = components(bm, 200) if base_candidates > 0 else []
    for comp in sorted(comps, key=np.count_nonzero, reverse=True)[:base_candidates]:
        b = fit_base(cam, fill_holes(comp), bad=bad, z0=z0)
        if b is not None and (scene.base is None or b.iou > scene.base.iou):
            scene.base = b
    for cat in range(4):
        m = colour_mask(hsv, cat) & valid
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        ign = bad
        if cat == DISC:
            amb = disc_ambiguous(hsv, m)
            ign = amb if bad is None else amb | bad
        for comp in components(m, 40):
            comp = fill_holes(comp)
            p = fit_piece(cam, comp, cat, bad=ign, z0=z0)
            if p is None or p.iou < min_iou:
                continue
            scene.pieces.append(p)
    scene.pole_of_cat = assign_poles(scene.pieces)
    return scene
