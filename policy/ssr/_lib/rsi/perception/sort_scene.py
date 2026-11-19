from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from . import robot_mask as RM
from . import head_background as TT
from .cameras import Camera

BASKET_SIZE = 0.252, 0.168, 0.077
BASKET_GUESS = (-0.28, 0.05), (0.0, 0.05), (0.28, 0.05)
SEARCH_REGION = (-0.62, 0.62), (-0.42, 0.16)
OBJECT_REGION = (-0.5, 0.5), (-0.4, 0.03)
FOOT_Z = 0.02


@dataclass
class Basket:
    x: float
    y: float
    score: float = 0.0
    size: tuple = BASKET_SIZE

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def contains(self, xy, margin: float = 0.0) -> bool:
        return abs(xy[0] - self.x) <= self.size[0] / 2 + margin and abs(xy[1] - self.y) <= self.size[1] / 2 + margin

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "score": round(self.score, 3)}


def box_corners(x, y, sx, sy, h, yaw=0.0, z0=TABLE_Z, taper=0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dz, t in ((0.0, taper), (h, 0.0)):
        for dx, dy in (
            (-sx / 2 + t, -sy / 2 + t),
            (sx / 2 - t, -sy / 2 + t),
            (sx / 2 - t, sy / 2 - t),
            (-sx / 2 + t, sy / 2 - t),
        ):
            pts.append((x + c * dx - s * dy, y + s * dx + c * dy, z0 + dz))
    return np.asarray(pts)


def box_mask(cam: Camera, shape, x, y, sx, sy, h, yaw=0.0, taper=0.0) -> np.ndarray:
    import cv2

    uv, d = cam.project(box_corners(x, y, sx, sy, h, yaw, taper=taper))
    m = np.zeros(shape[:2], np.uint8)
    if np.all(d > 0.01):
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m > 0


BASKET_TAPER = 0.0075


def detect_baskets(
    img: np.ndarray, cam: Camera, guess=BASKET_GUESS, taper: float | None = None, refine: bool = True
) -> list[Basket]:
    import cv2

    taper = BASKET_TAPER if taper is None else float(taper)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b, L = lab[..., 1] - 128.0, lab[..., 2] - 128.0, lab[..., 0]
    tab = (a > 8) & (a < 30) & (b > 8) & (b < 32) & (L < 175) & (L > 40)
    non = ~tab
    out = []
    sx, sy, h = BASKET_SIZE

    def score(x, y) -> float:
        m = box_mask(cam, img.shape, x, y, sx, sy, h, taper=taper)
        inter = np.count_nonzero(m & non)
        iou = inter / max(1, np.count_nonzero(m))
        ring = box_mask(cam, img.shape, x, y, sx + 0.03, sy + 0.03, h + 0.01, taper=taper) & ~m
        outside = np.count_nonzero(ring & non) / max(1, np.count_nonzero(ring))
        return iou - outside

    for gx, gy in guess:
        best = -1.0, gx, gy
        for dx in np.arange(-0.04, 0.0401, 0.01):
            for dy in np.arange(-0.03, 0.0301, 0.01):
                sc = score(gx + dx, gy + dy)
                if sc > best[0]:
                    best = sc, gx + dx, gy + dy
        if refine:
            _, bx, by = best
            for dx in np.arange(-0.0075, 0.00751, 0.0025):
                for dy in np.arange(-0.0075, 0.00751, 0.0025):
                    sc = score(bx + dx, by + dy)
                    if sc > best[0]:
                        best = sc, bx + dx, by + dy
        out.append(Basket(float(best[1]), float(best[2]), float(best[0])))
    return out


@dataclass
class SceneObject:
    region: TT.Region
    scores: dict = field(default_factory=dict)
    category: str | None = None
    group: int | None = None
    id: int = 0

    @property
    def xy(self) -> np.ndarray:
        return self.region.xy

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "cat": self.category,
            "group": self.group,
            "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
        }
        d.update(self.region.as_dict())
        return d


CATEGORY_HEIGHT = {"pen": 0.018, "watch": 0.04, "camera": 0.05, "car": 0.04, "figure": 0.065}


def ring_centre(cam: Camera, mask: np.ndarray, z: float, r_range=(0.018, 0.036)) -> np.ndarray | None:
    rows, cols = np.nonzero(mask)
    if len(rows) < 30:
        return None
    uv = np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)
    p = cam.on_plane_z(uv, z)[:, :2]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 30:
        return None
    m = p.mean(axis=0)
    q = p - m
    A = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))])
    b = -(q[:, 0] ** 2 + q[:, 1] ** 2)
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = np.array([-D / 2, -E / 2])
    r2 = c @ c - F
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    if not r_range[0] <= r <= r_range[1]:
        return None
    res = np.abs(np.linalg.norm(q - c, axis=1) - r)
    if float(np.median(res)) > 0.006:
        return None
    return c + m


def refine_footprint(cam: Camera, ob: "SceneObject") -> None:
    import cv2

    h = CATEGORY_HEIGHT.get(ob.category or "", 2 * FOOT_Z)
    filled = TT.fill_holes(ob.region.mask)
    rect = TT.footprint_rect(cam, filled, TABLE_Z + h / 2)
    if rect is None:
        return
    xy, (L, W), yaw = rect
    ob.region.extra["xy_foot2cm"] = [round(float(v), 4) for v in ob.region.xy]
    ob.region.xy = np.asarray(xy, dtype=float)
    ob.region.yaw = yaw
    ob.region.height = h
    if ob.category == "watch":
        partial = W < 0.045 or ob.region.area < 250
        c = ring_centre(cam, ob.region.mask, TABLE_Z + h / 2) if partial else None
        if c is not None and float(np.linalg.norm(c - ob.region.xy)) < 0.03:
            ob.region.extra["xy_rect"] = [round(float(v), 4) for v in ob.region.xy]
            ob.region.xy = c
        return
    if h < 0.035:
        return
    rows, cols = np.nonzero(filled)
    uv = np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)
    p = cam.on_plane_z(uv, TABLE_Z + h / 2)[:, :2]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 10:
        return
    c = p.mean(axis=0)
    v = c - np.asarray(cam.pos[:2])
    dist = float(np.linalg.norm(v))
    if dist < 1e-06:
        return
    v /= dist
    tan_t = dist / max(0.1, float(cam.pos[2]) - (TABLE_Z + h / 2))
    s = (p - c) @ v
    e_v = float(s.max() - s.min())
    shrink = max(0.25, (e_v - h * tan_t) / max(e_v, 1e-06))
    q = p - np.outer(s * (1.0 - shrink), v)
    (_cx, _cy), (a, b), ang = cv2.minAreaRect(q.astype(np.float32))
    ang = math.radians(ang)
    if a < b:
        a, b = b, a
        ang += math.pi / 2
    ob.region.extra["axis_raw"] = round(math.degrees(yaw), 1)
    ob.region.extra["LW_comp"] = [round(float(a), 4), round(float(b), 4)]
    if a > 1.15 * b and abs(math.remainder(ang - yaw, math.pi)) > math.radians(45):
        ob.region.yaw = math.remainder(ang, math.pi)


def robot_valid(cam: Camera, shape, q_left=None, q_right=None, g_left=1.0, g_right=1.0) -> np.ndarray:
    valid = TT.table_polygon_mask(cam, shape, SEARCH_REGION)
    if q_left is not None or q_right is not None:
        valid &= ~RM.mask(cam, q_left, q_right, g_left, g_right, dilate=4)
    return valid


RIM_STRIP = 0.03
SPLIT_MIN_AREA = 600
STREAK_MIN_LEN = 0
STREAK_MAX_H = 4
SPLIT_MIN_PIECE = 120


def detect_objects(
    img: np.ndarray,
    cam: Camera,
    q_left=None,
    q_right=None,
    g_left=1.0,
    g_right=1.0,
    baskets: list[Basket] | None = None,
    fg_out: list | None = None,
    split=None,
    alts: list | None = None,
    instances: list | None = None,
    seg_drop_unseen: bool = False,
    seg_clip: bool = False,
    region=None,
) -> list[TT.Region]:
    import cv2

    valid = robot_valid(cam, img.shape, q_left, q_right, g_left, g_right)
    bm, ring = None, None
    if baskets:
        sx, sy, h = BASKET_SIZE
        k = np.ones((5, 5), np.uint8)
        bm = [
            cv2.dilate(box_mask(cam, img.shape, b.x, b.y, sx, sy, h, taper=BASKET_TAPER).astype(np.uint8), k) > 0
            for b in baskets
        ]
        ring = np.zeros(img.shape[:2], bool)
        for b, m in zip(baskets, bm):
            ring |= box_mask(cam, img.shape, b.x, b.y, sx + RIM_STRIP, sy + RIM_STRIP, h, taper=0.0) & ~m
    f = TT.foreground(img, valid, bm, ignore=ring)
    if f is None:
        return []
    raw = f.mask
    fg = TT.clean(raw, open_px=1, close_px=1)
    fg = TT.drop_row_streaks(fg, STREAK_MAX_H, min_len=STREAK_MIN_LEN)
    if fg_out is not None:
        fg_out.append(fg)
    near_robot = ~cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    comps = TT.components(fg, min_area=25, link_px=3)
    big = [c for c in comps if np.count_nonzero(c) >= 200]
    small = [c for c in comps if np.count_nonzero(c) < 200]
    if big:
        dist_maps = [cv2.distanceTransform((~c).astype(np.uint8), cv2.DIST_L2, 3) for c in big]
        for c in small:
            rr, cc = np.nonzero(c)
            d = [float(dm[rr, cc].min()) for dm in dist_maps]
            j = int(np.argmin(d))
            if d[j] <= 12.0:
                big[j] = big[j] | c
            else:
                big.append(c)
        comps = big
    seg_src = [None] * len(comps)
    if instances:
        comps, seg_src = seg_components(comps, instances, valid, bm, drop_unseen=seg_drop_unseen, clip_inside=seg_clip)
    out = []
    for comp, src in zip(comps, seg_src):
        reg = make_region(img, cam, comp, raw, near_robot, baskets, region)
        if reg is None:
            continue
        if src is not None:
            reg.extra[src[0]] = src[1]
            if src[0] != "seg_one":
                out.append(reg)
                continue
        if split is not None:
            n, lab, st, _ = cv2.connectedComponentsWithStats(comp.astype(np.uint8), connectivity=8)
            keep = np.nonzero(st[1:, cv2.CC_STAT_AREA] >= 200)[0] + 1
            if n > 2 and 0 < len(keep) < n - 1:
                core = np.isin(lab, keep)
                rc = make_region(img, cam, core, raw, near_robot, baskets, region)
                if rc is not None and split(reg, [rc]):
                    rc.extra["unjoined"] = int(reg.area - rc.area)
                    reg, comp = rc, core
        if split is not None and reg.area >= SPLIT_MIN_AREA and (src is None):
            pieces = split_region(comp)
            if len(pieces) >= 2:
                regs = [make_region(img, cam, pc, raw, near_robot, baskets, region) for pc in pieces]
                regs = [r for r in regs if r is not None and r.area >= SPLIT_MIN_PIECE]
                if len(regs) >= 2 and split(reg, regs):
                    for r in regs:
                        r.extra["split_from"] = reg.area
                    out.extend(regs)
                    continue
                if len(regs) >= 2 and alts is not None:
                    alts.append((reg, regs))
        out.append(reg)
    out.sort(key=lambda r: r.xy[0])
    return out


SEG_OWN_FRAC = 0.4
SEG_SEEN_FRAC = 0.2
SEG_NEAR_PX = 6.0
SEG_MIN_PX = 150
SEG_INSIDE_FRAC = 0.6


def seg_components(
    comps: list,
    instances: list,
    valid: np.ndarray,
    bm: list | None = None,
    drop_unseen: bool = False,
    clip_inside: bool = False,
):
    import cv2

    masks = [np.asarray(i.mask, bool) for i in instances]
    areas = [max(1, int(np.count_nonzero(m))) for m in masks]
    grown = [cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for m in masks]
    seen = [False] * len(masks)
    out, src = [], []
    joined: dict = {}
    for comp in comps:
        n_c = max(1, int(np.count_nonzero(comp)))
        inter = [int(np.count_nonzero(m & comp)) for m in masks]
        for j, n in enumerate(inter):
            if n >= SEG_SEEN_FRAC * areas[j]:
                seen[j] = True
        mine = [j for j, n in enumerate(inter) if n >= SEG_OWN_FRAC * areas[j]]
        if masks and len(mine) <= 1:
            inside = [np.count_nonzero(g & comp) / n_c for g in grown]
            j = int(np.argmax(inside))
            if inside[j] >= SEG_INSIDE_FRAC and (not mine or mine[0] == j):
                joined.setdefault(j, []).append(comp & grown[j] if drop_unseen and clip_inside else comp)
                continue
            if drop_unseen and mine:
                joined.setdefault(mine[0], []).append(comp & grown[mine[0]])
                continue
            if drop_unseen and inside[j] < SEG_SEEN_FRAC:
                continue
        pieces = []
        if len(mine) >= 2:
            dist = np.stack([cv2.distanceTransform((~masks[j]).astype(np.uint8), cv2.DIST_L2, 3) for j in mine])
            lab = np.argmin(dist, axis=0)
            near = dist.min(axis=0) <= SEG_NEAR_PX
            pieces = [comp & near & (lab == k) for k in range(len(mine))]
            pieces = [p for p in pieces if np.count_nonzero(p) >= SEG_MIN_PX]
        if len(pieces) >= 2:
            out.extend(pieces)
            src.extend([("seg_split", len(pieces))] * len(pieces))
        else:
            out.append(comp)
            src.append(None)
    for j, cs in joined.items():
        if len(cs) == 1:
            out.append(cs[0])
            src.append(("seg_one", 1))
        else:
            u = cs[0].copy()
            for c in cs[1:]:
                u |= c
            out.append(u)
            src.append(("seg_joined", len(cs)))
    free = valid.copy()
    for m in bm or []:
        free &= ~m
    for j, m in enumerate(masks):
        if not seen[j]:
            p = m & free
            if np.count_nonzero(p) >= SEG_MIN_PX:
                out.append(p)
                src.append(("seg_added", 1))
    return out, src


def split_region(comp: np.ndarray, max_r: int = 7, min_seed: int = 40) -> list[np.ndarray]:
    import cv2

    m = comp.astype(np.uint8)
    for r in range(1, max_r + 1):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        er = cv2.erode(m, k)
        n, lab = cv2.connectedComponents(er)
        cores = [lab == i for i in range(1, n) if np.count_nonzero(lab == i) >= min_seed]
        if len(cores) >= 2:
            dists = np.stack([cv2.distanceTransform((~c).astype(np.uint8), cv2.DIST_L2, 3) for c in cores])
            idx = np.argmin(dists, axis=0)
            near = dists.min(axis=0) <= r + 1.5
            return [comp & (idx == i) & near for i in range(len(cores))]
        if n <= 1:
            break
    return [comp]


def make_region(
    img, cam: Camera, comp: np.ndarray, raw: np.ndarray, near_robot: np.ndarray, baskets=None, region=None
) -> "TT.Region | None":
    import cv2

    area = int(np.count_nonzero(comp))
    if area < 60:
        return None
    filled = TT.fill_holes(comp)
    rect = TT.footprint_rect(cam, filled, TABLE_Z + FOOT_Z)
    if rect is None:
        return None
    xy, (L, W), yaw = rect
    (x0, x1), (y0, y1) = OBJECT_REGION if region is None else region
    if not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
        return None
    if baskets and any(b.contains(xy, 0.005) for b in baskets):
        return None
    if baskets and area < 300 and any(b.contains(xy, 0.015) for b in baskets):
        return None
    col = TT.colour_stats(img, comp)
    if (
        area < 400
        and 10.0 <= col.get("hue", 0.0) <= 26.0
        and (15.0 <= col.get("chroma", 0.0) <= 36.0)
        and (col.get("dark", 1.0) < 0.05)
        and (col.get("bright", 1.0) < 0.05)
    ):
        return None
    rows, cols = np.nonzero(comp)
    closed = (
        cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        > 0
    )
    outline = TT.fill_holes(closed)
    hole = 1.0 - float(np.count_nonzero(raw & outline)) / max(1, int(np.count_nonzero(outline)))
    hull = cv2.convexHull(np.column_stack([cols, rows]).astype(np.int32))
    hm = np.zeros(comp.shape, np.uint8)
    cv2.fillConvexPoly(hm, hull, 1)
    hull_def = 1.0 - float(np.count_nonzero(raw & (hm > 0))) / max(1, int(np.count_nonzero(hm)))
    touches = []
    if np.any(near_robot & cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)):
        touches.append("robot")
    reg = TT.Region(
        mask=comp,
        area=area,
        bbox=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())),
        uv=(float(cols.mean()), float(rows.mean())),
        xy=np.asarray(xy, dtype=float),
        length=L,
        width=W,
        yaw=yaw,
        hole=hole,
        color=col,
        touches=tuple(touches),
    )
    reg.extra["filled_area"] = int(np.count_nonzero(filled))
    reg.extra["hull_def"] = round(hull_def, 3)
    _c, (rw, rh), _a = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
    reg.extra["fill"] = round(area / max(1.0, rw * rh), 3)
    return reg
