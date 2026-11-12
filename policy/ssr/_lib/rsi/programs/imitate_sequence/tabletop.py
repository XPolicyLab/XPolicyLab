from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ...planning.kinematics import TABLE_Z
from ...perception.cameras import Camera

TABLE_AREA = (-0.62, 0.62), (-0.42, 0.46)
TABLE_AB = 18.0
DARK_L = 22.0
WHITE_L = 72.0
WHITE_AB = 12.0
MIN_BOX_PX = 1500


def lab_image(img: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def plane_mask(cam: Camera, shape, region=TABLE_AREA, z: float = TABLE_Z) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1) = region
    poly = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    uv, d = cam.project(poly)
    m = np.zeros(shape[:2], np.uint8)
    if np.all(d > 0):
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m > 0


def region_with_height(cam: Camera, shape, region, h: float, z: float = TABLE_Z) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1) = region
    pts = []
    for zz in (z, z + h):
        pts += [[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz]]
    uv, d = cam.project(np.array(pts))
    m = np.zeros(shape[:2], np.uint8)
    if np.all(d > 0):
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    return m > 0


def table_colour(lab: np.ndarray, area: np.ndarray) -> np.ndarray:
    return np.median(lab[area], axis=0) if area.any() else np.median(lab.reshape(-1, 3), axis=0)


BG_CHUNK = 128
BG_STEP = 32
BG_AB = 12.0
BG_LIGHT = 18.0
BG_DARK = 25.0


def table_background(lab: np.ndarray, valid: np.ndarray, chunk: int = BG_CHUNK, step: int = BG_STEP) -> np.ndarray:
    import warnings

    H, W, _ = lab.shape
    starts = list(range(0, max(1, W - chunk + 1), step))
    centers = np.array([s + chunk / 2.0 for s in starts])
    x = np.where(valid[..., None], lab, np.nan).astype(np.float32)
    meds = np.full((H, len(starts), 3), np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, s in enumerate(starts):
            seg = x[:, s : s + chunk]
            cnt = np.sum(valid[:, s : s + chunk], axis=1)
            m = np.nanmedian(seg, axis=1)
            m[cnt < chunk * 0.3] = np.nan
            meds[:, k] = m
        pad = np.pad(meds, ((2, 2), (0, 0), (0, 0)), mode="edge")
        meds = np.nanmedian(np.stack([pad[i : i + H] for i in range(5)], 0), axis=0)
    out = np.full((H, W, 3), np.nan, np.float32)
    cols = np.arange(W)
    for v in range(H):
        ok = np.isfinite(meds[v, :, 0])
        if not ok.any():
            continue
        for c in range(3):
            out[v, :, c] = np.interp(cols, centers[ok], meds[v, ok, c])
    return out


def not_table(img: np.ndarray, cam: Camera, area: np.ndarray | None = None, lab=None) -> np.ndarray:
    lab = lab_image(img) if lab is None else lab
    if area is None:
        area = plane_mask(cam, img.shape)
    bg = table_background(lab, area)
    with np.errstate(invalid="ignore"):
        dl = lab[..., 0] - bg[..., 0]
        dab = np.hypot(lab[..., 1] - bg[..., 1], lab[..., 2] - bg[..., 2])
        m = (dab > BG_AB) | (dl > BG_LIGHT) | (dl < -BG_DARK) | (lab[..., 0] < DARK_L)
    return m & area & np.isfinite(bg[..., 0])


@dataclass
class WhiteBox:
    mask: np.ndarray
    area: int
    centroid: tuple
    xy: np.ndarray
    bottom_v: float

    def as_dict(self):
        return {
            "area": self.area,
            "centroid": [round(float(c), 1) for c in self.centroid],
            "xy": [round(float(v), 3) for v in self.xy],
        }


def white_boxes(
    img: np.ndarray, cam: Camera, lab=None, min_px: int = MIN_BOX_PX, rim_h: float = 0.04
) -> list[WhiteBox]:
    import cv2

    lab = lab_image(img) if lab is None else lab
    m = ((lab[..., 0] > WHITE_L) & (np.hypot(lab[..., 1], lab[..., 2]) < WHITE_AB)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_px or stats[k, cv2.CC_STAT_TOP] <= 1:
            continue
        mk = lbl == k
        hull = cv2.convexHull(np.argwhere(mk)[:, ::-1].astype(np.int32))
        full = np.zeros(mk.shape, np.uint8)
        cv2.fillConvexPoly(full, hull, 1)
        top = cam.on_plane_z(hull.reshape(-1, 2).astype(float) + 0.5, TABLE_Z + rim_h * 2.0)
        top = top[np.isfinite(top[:, 0])]
        if len(top):
            base = top.copy()
            base[:, 2] = TABLE_Z
            uv, d = cam.project(np.vstack([top, base]))
            if np.all(d > 0):
                cv2.fillConvexPoly(full, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        vs, us = np.nonzero(full)
        p = cam.on_plane_z(np.stack([us, vs], axis=1).astype(float) + 0.5, TABLE_Z + rim_h)
        p = p[np.isfinite(p[:, 0])]
        xy = np.median(p[:, :2], axis=0) if len(p) else np.array([np.nan, np.nan])
        out.append(
            WhiteBox(
                full > 0,
                area,
                (float(cents[k][0]), float(cents[k][1])),
                xy,
                float(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT]),
            )
        )
    out.sort(key=lambda b: -b.area)
    return out


@dataclass
class Region:
    mask: np.ndarray
    area: int
    centroid: tuple
    front_xy: np.ndarray
    fit: "Footprint | None" = None
    desc: dict = field(default_factory=dict)

    def as_dict(self):
        d = {
            "area": self.area,
            "centroid": [round(float(c), 1) for c in self.centroid],
            "front_xy": [round(float(v), 3) for v in self.front_xy],
        }
        if self.fit is not None:
            d["fit"] = self.fit.as_dict()
        return d


def segment(fg: np.ndarray, cam: Camera, min_px: int = 60, join_px: int = 4, fill_holes: bool = False) -> list[Region]:
    import cv2

    m = fg.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    joined = cv2.dilate(m, np.ones((2 * join_px + 1, 2 * join_px + 1), np.uint8))
    n, lbl = cv2.connectedComponents(joined, connectivity=8)
    out = []
    for k in range(1, n):
        mk = (lbl == k) & (m > 0)
        if fill_holes and mk.sum() >= 10:
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled = np.zeros(mk.shape, np.uint8)
            cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
            mk = filled > 0
        area = int(mk.sum())
        if area < min_px:
            continue
        vs, us = np.nonzero(mk)
        vmax = vs.max()
        sel = vs >= vmax - 1
        front = cam.on_plane_z(np.array([[float(np.mean(us[sel])) + 0.5, float(vmax) + 1.0]]), TABLE_Z)[0]
        out.append(Region(mk, area, (float(us.mean()), float(vs.mean())), front[:2]))
    out.sort(key=lambda r: -r.area)
    return out


def split_region(
    r: Region, cam: Camera, lab: np.ndarray, min_px: int = 60, accept=None, tries: int = 5
) -> list[Region]:
    import cv2

    vs, us = np.nonzero(r.mask)
    if len(vs) < 2 * min_px:
        return [r]
    px = lab[vs, us]
    feat = np.column_stack([us / 4.0, vs / 4.0, px[:, 0] / 6.0, px[:, 1] / 4.0, px[:, 2] / 4.0]).astype(np.float32)
    crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1
    for seed in range(tries):
        cv2.setRNGSeed(seed)
        _, lbl, _ = cv2.kmeans(feat, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        out = []
        for k in range(2):
            m = np.zeros(r.mask.shape, bool)
            m[vs[lbl.ravel() == k], us[lbl.ravel() == k]] = True
            n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            if n <= 1:
                continue
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            mk = cc == j
            if mk.sum() < min_px:
                continue
            vv, uu = np.nonzero(mk)
            vmax = vv.max()
            sel = vv >= vmax - 1
            front = cam.on_plane_z(np.array([[float(np.mean(uu[sel])) + 0.5, float(vmax) + 1.0]]), TABLE_Z)[0]
            out.append(Region(mk, int(mk.sum()), (float(uu.mean()), float(vv.mean())), front[:2]))
        if len(out) == 2 and (accept is None or all(accept(p) for p in out)):
            return out
    return [r]


@dataclass
class Footprint:
    xy: np.ndarray
    yaw: float
    length: float
    width: float
    height: float
    iou: float

    def as_dict(self):
        return {
            "xy": [round(float(v), 4) for v in self.xy],
            "yaw_deg": round(math.degrees(self.yaw), 1),
            "length": round(self.length, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "iou": round(self.iou, 3),
        }


HEIGHTS = tuple(np.round(np.arange(0.01, 0.0951, 0.005), 4))
H_PRIOR = 0.035


def fit_footprint(
    mask: np.ndarray,
    cam: Camera,
    cell: float = 0.003,
    heights=HEIGHTS,
    z0: float = TABLE_Z,
    pad: float = 0.03,
    h_prior: float = H_PRIOR,
    prior_slack: float = 0.08,
    prior_min_iou: float = 0.0,
) -> Footprint | None:
    import cv2

    prior = None
    h_prior = float(min(heights, key=lambda hh: abs(hh - h_prior)))
    vs, us = np.nonzero(mask)
    if len(vs) < 10:
        return None
    H, W = mask.shape
    p = cam.on_plane_z(np.stack([us, vs], axis=1).astype(float) + 0.5, z0)
    p = p[np.isfinite(p[:, 0])]
    if len(p) < 10:
        return None
    lo = p[:, :2].min(axis=0) - pad
    hi = p[:, :2].max(axis=0) + pad
    nx = int(math.ceil((hi[0] - lo[0]) / cell))
    ny = int(math.ceil((hi[1] - lo[1]) / cell))
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    zs = np.arange(0.0, max(heights) + 0.0025 + 1e-09, 0.0025)
    bound = np.full(len(pts), -1.0)
    alive = np.ones(len(pts), bool)
    for z in zs:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        uv, d = cam.project(np.column_stack([pts[idx], np.full(idx.size, z0 + z)]))
        u = np.floor(uv[:, 0]).astype(int)
        v = np.floor(uv[:, 1]).astype(int)
        ok = (d > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        inside = np.zeros(idx.size, bool)
        inside[ok] = mask[v[ok], u[ok]]
        bound[idx[inside]] = z
        alive[idx[~inside]] = False
    u0, v0 = int(us.min()) - 3, int(vs.min()) - 3
    sub = mask[max(0, v0) : vs.max() + 4, max(0, u0) : us.max() + 4]
    off = np.array([max(0, u0), max(0, v0)])
    best = None
    for h in heights:
        cells = pts[bound >= h - 1e-09]
        if len(cells) < 4:
            continue
        samples = [np.column_stack([cells, np.full(len(cells), z0 + zz)]) for zz in np.linspace(0, h, 4)]
        uv, _ = cam.project(np.vstack(samples))
        r = np.zeros_like(sub, np.uint8)
        ij = np.floor(uv - off).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < r.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < r.shape[0])
        r[ij[ok, 1], ij[ok, 0]] = 1
        r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        inter = float(np.sum((r > 0) & sub))
        union = float(np.sum((r > 0) | sub))
        iou = inter / max(union, 1.0)
        if best is None or iou > best[0] + 1e-06:
            best = iou, h, cells
        if abs(h - h_prior) < 1e-06:
            prior = iou, h, cells
    if best is None:
        return None
    if prior is not None and prior[0] >= best[0] - prior_slack and (prior[0] >= prior_min_iou):
        best = prior
    iou, h, cells = best
    rect = cv2.minAreaRect(cells.astype(np.float32))
    (cx, cy), (s0, s1), ang = rect
    yaw = math.radians(ang)
    if s1 > s0:
        l, w = s1, s0
        yaw += math.pi / 2
    else:
        l, w = s0, s1
    yaw = math.remainder(yaw, math.pi)
    centre = cells.mean(axis=0)
    return Footprint(np.asarray(centre), yaw, float(l) + cell, float(w) + cell, float(h), float(iou))


DIM_HEIGHTS = 0.015, 0.025, 0.035, 0.045, 0.055, 0.065


def dims_by_height(
    mask: np.ndarray, cam: Camera, heights=DIM_HEIGHTS, cell: float = 0.003, z0: float = TABLE_Z, pad: float = 0.03
) -> dict:
    import cv2

    vs, us = np.nonzero(mask)
    if len(vs) < 10:
        return {}
    H, W = mask.shape
    p = cam.on_plane_z(np.stack([us, vs], axis=1).astype(float) + 0.5, z0)
    p = p[np.isfinite(p[:, 0])]
    if len(p) < 10:
        return {}
    lo = p[:, :2].min(axis=0) - pad
    hi = p[:, :2].max(axis=0) + pad
    nx = int(math.ceil((hi[0] - lo[0]) / cell))
    ny = int(math.ceil((hi[1] - lo[1]) / cell))
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    bound = np.full(len(pts), -1.0)
    alive = np.ones(len(pts), bool)
    for z in np.arange(0.0, max(heights) + 0.0026, 0.0025):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        uv, d = cam.project(np.column_stack([pts[idx], np.full(idx.size, z0 + z)]))
        u = np.floor(uv[:, 0]).astype(int)
        v = np.floor(uv[:, 1]).astype(int)
        ok = (d > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        inside = np.zeros(idx.size, bool)
        inside[ok] = mask[v[ok], u[ok]]
        bound[idx[inside]] = z
        alive[idx[~inside]] = False
    out = {}
    for h in heights:
        cells = pts[bound >= h - 1e-09]
        if len(cells) < 4:
            continue
        (_, _), (w, l), _ = cv2.minAreaRect(cells.astype(np.float32))
        out[h] = max(w, l) + cell, min(w, l) + cell
    return out


HIST_BINS = 5, 7, 7
HIST_RANGE = (0.0, 120.0), (-45.0, 45.0), (-45.0, 65.0)
HUE_BINS = 12
SAT_CHROMA = 8.0
SHADOW_TOL = 0.12
DIM_WEIGHT = 0.5


def shadow_mask(px: np.ndarray, table: np.ndarray) -> np.ndarray:
    L = np.maximum(px[:, 0], 1.0)
    Lt = np.maximum(table[:, 0], 1.0)
    c = px[:, 1:] / L[:, None]
    ct = table[:, 1:] / Lt[:, None]
    r = L / Lt
    ok = np.isfinite(table[:, 0])
    sh = (np.linalg.norm(c - ct, axis=1) < SHADOW_TOL) & (r < 0.97) & (r > 0.2) | (
        np.linalg.norm(px - table, axis=1) < 6
    )
    return sh & ok


def appearance(mask: np.ndarray, px: np.ndarray, table: np.ndarray, cam: Camera | None = None) -> dict:
    px = np.asarray(px, np.float32)
    table = np.asarray(table, np.float32)
    sh = shadow_mask(px, table)
    keep = ~sh
    if keep.sum() < 10:
        keep = np.ones(len(px), bool)
    m2 = mask.copy()
    vs, us = np.nonzero(mask)
    m2[vs[~keep], us[~keep]] = False
    p = px[keep]
    t = table[np.isfinite(table[:, 0])]
    lt = float(np.median(t[:, 0])) if len(t) else 50.0
    q = p * (50.0 / max(lt, 5.0))
    h1, _ = np.histogramdd(q, bins=HIST_BINS, range=HIST_RANGE)
    h1 = h1.ravel().astype(np.float64)
    h1 /= max(h1.sum(), 1.0)
    ch = np.hypot(p[:, 1], p[:, 2])
    hue = np.degrees(np.arctan2(p[:, 2], p[:, 1])) % 360.0
    sat = ch > SAT_CHROMA
    hh = np.histogram(hue[sat], bins=HUE_BINS, range=(0.0, 360.0), weights=ch[sat])[0].astype(np.float64)
    hh = np.append(hh, float(np.sum(~sat)) * (float(np.mean(ch[sat])) if sat.any() else 1.0))
    hh /= max(hh.sum(), 1e-09)
    dims = dims_by_height(m2, cam) if cam is not None and m2.sum() >= 10 else {}
    return {"h1": h1, "hh": hh, "dims": dims, "n": int(keep.sum()), "shadow": float(np.mean(sh))}


def _hell(a, b) -> float:
    return math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b)))))


def appearance_cost(a: dict, b: dict) -> float:
    c = 0.5 * _hell(a["h1"], b["h1"]) + 0.5 * _hell(a["hh"], b["hh"])
    da, db = a.get("dims") or {}, b.get("dims") or {}
    common = [h for h in da if h in db]
    if common:
        c += DIM_WEIGHT * min(
            0.5 * (abs(math.log(da[h][0] / db[h][0])) + abs(math.log(da[h][1] / db[h][1]))) for h in common
        )
    return c


def match(costs: np.ndarray) -> tuple[list, float, float]:
    n, m = costs.shape
    if n > m:
        cols, total, margin = match(costs.T)
        out = [None] * n
        for j, i in enumerate(cols):
            out[i] = j
        return out, total, margin
    best = math.inf, None
    second = math.inf
    for perm in itertools.permutations(range(m), n):
        c = float(sum(costs[i, j] for i, j in enumerate(perm)))
        if c < best[0]:
            second = best[0]
            best = c, list(perm)
        elif c < second:
            second = c
    return best[1], best[0], second - best[0]
