from __future__ import annotations

import math

import numpy as np

from ...perception.cameras import Camera

TABLE_Z = 0.765
BOTTLE_PROFILE = (
    (0.0, 0.0382),
    (0.03, 0.0382),
    (0.07, 0.0395),
    (0.1, 0.0415),
    (0.105, 0.04),
    (0.111, 0.036),
    (0.117, 0.032),
    (0.122, 0.0253),
    (0.127, 0.016),
    (0.218, 0.0155),
)
BOTTLE_H = 0.218
NECK_R_IN = 0.0121
BOWL_PROFILE = (
    (0.0, 0.025),
    (0.0045, 0.0266),
    (0.0105, 0.0386),
    (0.0195, 0.0461),
    (0.0295, 0.0517),
    (0.034, 0.0537),
    (0.038, 0.055),
    (0.04, 0.0548),
)
BOWL_H = 0.04
BOWL_R = 0.055
BOTTLE_COLOURS = "red", "turquoise", "violet"
SS = 4
N_RING = 40


def hsv_image(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def bottle_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if colour == "red":
        return ((h <= 4) | (h >= 172)) & (s >= 170) & (v >= 110)
    if colour == "turquoise":
        return (h >= 78) & (h <= 98) & (s >= 120) & (v >= 80)
    if colour == "violet":
        return (h >= 118) & (h <= 148) & (s >= 12) & (v >= 120)
    raise ValueError(colour)


def table_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 4) & (h <= 14) & (s >= 75) & (v >= 95)


def _rings(profile):
    a = np.linspace(0.0, 2 * math.pi, N_RING, endpoint=False)
    zs = np.array([p[0] for p in profile])
    rs = np.array([p[1] for p in profile])
    return zs, rs, np.cos(a), np.sin(a)


def silhouette_polys(cam: Camera, x: float, y: float, profile, z0: float = TABLE_Z) -> list[np.ndarray]:
    import cv2

    zs, rs, ca, sa = _rings(profile)
    pts = np.concatenate([np.stack([x + r * ca, y + r * sa, np.full_like(ca, z0 + z)], axis=1) for z, r in zip(zs, rs)])
    uv, d = cam.project(pts)
    if np.any(d <= 0.01):
        return []
    uv = uv.reshape(len(zs), N_RING, 2).astype(np.float32)
    return [cv2.convexHull(np.vstack([uv[i], uv[i + 1]])).reshape(-1, 2) for i in range(len(zs) - 1)]


def render(cam: Camera, x: float, y: float, profile, shape, z0: float = TABLE_Z) -> np.ndarray:
    import cv2

    m = np.zeros(shape, np.uint8)
    for p in silhouette_polys(cam, x, y, profile, z0):
        cv2.fillConvexPoly(m, np.round(p * 4).astype(np.int32), 1, shift=2)
    return m.astype(bool)


def fit_profile(
    cam: Camera,
    region: np.ndarray,
    profile,
    *,
    init,
    bad: np.ndarray | None = None,
    span: float = 0.02,
    coarse: float = 0.003,
    z0: float = TABLE_Z,
    pad: int = 8,
):
    import cv2

    rows, cols = np.nonzero(region)
    if len(rows) < 10:
        return None
    polys = []
    for dx in (-span, span):
        for dy in (-span, span):
            polys += silhouette_polys(cam, init[0] + dx, init[1] + dy, profile, z0)
    allp = np.vstack([np.column_stack([cols, rows])] + [p for p in polys if len(p)])
    c0, r0 = (np.floor(allp.min(0)) - pad).astype(int)
    c1, r1 = (np.ceil(allp.max(0)) + pad).astype(int)
    h, w = region.shape
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(w, c1), min(h, r1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    m_ss = np.kron(win, np.ones((SS, SS), np.uint8))
    keep = None
    if bad is not None:
        keep = np.kron((~bad[r0:r1, c0:c1]).astype(np.uint8), np.ones((SS, SS), np.uint8))
        m_ss &= keep
    am = float(np.count_nonzero(m_ss))
    buf = np.zeros(m_ss.shape, np.uint8)
    shift = np.array([c0, r0], dtype=np.float64)

    def iou(x, y) -> float:
        ps = silhouette_polys(cam, x, y, profile, z0)
        if not ps:
            return -1.0
        buf[:] = 0
        for poly in ps:
            p = (poly - shift) * SS - 0.5
            cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
        if keep is not None:
            buf[:] &= keep
        inter = float(np.count_nonzero(buf & m_ss))
        union = am + float(np.count_nonzero(buf)) - inter
        return inter / union if union > 0 else 0.0

    best = -1.0, float(init[0]), float(init[1])
    g = np.arange(-span, span + 1e-09, coarse)
    for dx in g:
        for dy in g:
            v = iou(init[0] + dx, init[1] + dy)
            if v > best[0]:
                best = v, init[0] + dx, init[1] + dy
    for sp, st in ((coarse, 0.001), (0.001, 0.0005)):
        _, bx, by = best
        offs = np.arange(-sp, sp + 1e-09, st)
        for dx in offs:
            for dy in offs:
                v = iou(bx + dx, by + dy)
                if v > best[0]:
                    best = v, bx + dx, by + dy
    return best


def _largest(mask: np.ndarray, min_area: int):
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    return lab == i


def _init_from_region(cam: Camera, region: np.ndarray, z: float):
    rows, cols = np.nonzero(region)
    p = cam.on_plane_z(np.array([[float(cols.mean()) + 0.5, float(rows.mean()) + 0.5]]), z)[0]
    return (float(p[0]), float(p[1])) if np.all(np.isfinite(p)) else None


class Fit:
    def __init__(self, kind: str, colour: str, x: float, y: float, iou: float, n_px: int):
        self.kind, self.colour, self.x, self.y, self.iou, self.n_px = kind, colour, x, y, iou, n_px

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return dict(
            kind=self.kind,
            colour=self.colour,
            x=round(self.x, 4),
            y=round(self.y, 4),
            iou=round(self.iou, 3),
            n_px=self.n_px,
        )


def detect_bottles(
    img: np.ndarray, cam: Camera, bad: np.ndarray | None = None, min_area: int = 300, min_iou: float = 0.6
) -> dict[str, Fit]:
    import cv2

    hsv = hsv_image(img)
    out = {}
    for c in BOTTLE_COLOURS:
        m = bottle_mask(hsv, c)
        if bad is not None:
            m &= ~bad
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        reg = _largest(m, min_area)
        if reg is None:
            continue
        init = _init_from_region(cam, reg, TABLE_Z + 0.09)
        if init is None:
            continue
        f = fit_profile(cam, reg, BOTTLE_PROFILE, init=init, bad=bad, span=0.016, coarse=0.004)
        if f is None or f[0] < min_iou:
            continue
        out[c] = Fit("bottle", c, f[1], f[2], f[0], int(reg.sum()))
    return out


def detect_bowls(
    img: np.ndarray,
    cam: Camera,
    bad: np.ndarray | None = None,
    min_area: int = 500,
    min_iou: float = 0.55,
    region_y=(-0.4, 0.1),
    region_x=(-0.55, 0.55),
) -> dict[str, Fit]:
    import cv2

    hsv = hsv_image(img)
    nt = ~table_mask(hsv)
    for c in BOTTLE_COLOURS:
        nt &= ~cv2.dilate(bottle_mask(hsv, c).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if bad is not None:
        nt &= ~bad
    H, W = nt.shape
    vv, uu = np.mgrid[0:H, 0:W]
    pts = cam.on_plane_z(np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5]).astype(float), TABLE_Z + 0.02)
    inside = (
        np.isfinite(pts[:, 0])
        & (pts[:, 0] > region_x[0])
        & (pts[:, 0] < region_x[1])
        & (pts[:, 1] > region_y[0])
        & (pts[:, 1] < region_y[1])
    ).reshape(H, W)
    nt &= inside
    nt = cv2.morphologyEx(nt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nt = cv2.morphologyEx(nt, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nt.astype(np.uint8), connectivity=8)
    cands = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        reg = lab == i
        init = _init_from_region(cam, reg, TABLE_Z + 0.02)
        if init is None:
            continue
        f = fit_profile(cam, reg, BOWL_PROFILE, init=init, bad=bad, span=0.02, coarse=0.004)
        if f is None or f[0] < min_iou:
            continue
        sil = render(cam, f[1], f[2], BOWL_PROFILE, nt.shape)
        if bad is not None:
            sil &= ~bad
        px = hsv[sil]
        v_med, s_med = float(np.median(px[:, 2])), float(np.median(px[:, 1]))
        cands.append(dict(x=f[1], y=f[2], iou=f[0], v=v_med, s=s_med, n=int(sil.sum())))
    cands.sort(key=lambda c: -c["iou"])
    cands = cands[:3]
    out = {}
    if not cands:
        return out
    by_v = sorted(cands, key=lambda c: c["v"])
    names = {}
    if len(by_v) == 3:
        names = {id(by_v[0]): "black", id(by_v[1]): "brown", id(by_v[2]): "white"}
    else:
        for c in by_v:
            names[id(c)] = "white" if c["v"] > 150 else "black" if c["v"] < 60 else "brown"
    for c in cands:
        name = names[id(c)]
        f = Fit("bowl", name, c["x"], c["y"], c["iou"], c["n"])
        f.v, f.s = c["v"], c["s"]
        out[name] = f
    return out
