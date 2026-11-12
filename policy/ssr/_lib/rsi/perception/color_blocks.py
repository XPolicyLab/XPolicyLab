from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from . import blocks as B
from .cameras import Camera

CUBE_SIZE = 0.035
BANDS: dict[str, list[tuple[int, int, int, int]]] = {
    "red": [(0, 3, 130, 140), (170, 179, 130, 140), (0, 3, 70, 210), (170, 179, 70, 210)],
    "orange": [(12, 25, 140, 130), (8, 11, 205, 150), (12, 25, 80, 210)],
    "yellow": [(26, 36, 100, 120), (26, 36, 60, 210)],
    "green": [(45, 75, 80, 100), (45, 75, 50, 210)],
    "cyan": [(80, 100, 50, 120), (80, 100, 40, 210)],
    "blue": [(105, 135, 80, 90), (105, 135, 50, 200)],
    "purple": [(136, 155, 80, 90)],
    "pink": [(156, 169, 80, 120)],
}
SYNONYMS = {"magenta": "pink", "violet": "purple", "teal": "cyan", "turquoise": "cyan", "aqua": "cyan"}
COLOR_NAMES = tuple(BANDS)
MIN_AREA = 25


@dataclass
class ColorCube:
    color: str
    x: float
    y: float
    s: float
    yaw: float
    z0: float = TABLE_Z
    iou: float = 0.0
    area: int = 0
    uv: tuple = (0.0, 0.0)
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z0 + self.s / 2])

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def top(self) -> float:
        return self.z0 + self.s

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "s": self.s,
            "yaw": round(math.degrees(self.yaw), 1),
            "z0": round(self.z0, 4),
            "iou": round(self.iou, 3),
            "area": self.area,
            "uv": [round(float(v), 1) for v in self.uv],
        }


def canonical(name: str) -> str | None:
    n = name.lower().strip()
    n = SYNONYMS.get(n, n)
    return n if n in BANDS else None


def hsv_image(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2HSV)


def color_mask(hsv: np.ndarray, name: str) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(hsv.shape[:2], bool)
    for h0, h1, s0, v0 in BANDS[name]:
        m |= (h >= h0) & (h <= h1) & (s >= s0) & (v >= v0)
    return m


def wrap_yaw(yaw: float) -> float:
    q = math.pi / 2
    return (yaw + q / 2) % q - q / 2


def _components(mask: np.ndarray, min_area: int = MIN_AREA):
    import cv2

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((int(stats[i, cv2.CC_STAT_AREA]), lab == i, (float(cents[i][0]), float(cents[i][1]))))
    out.sort(key=lambda t: -t[0])
    return out


def fit_cube(
    cam: Camera,
    region: np.ndarray,
    s: float = CUBE_SIZE,
    z0: float = TABLE_Z,
    *,
    init=None,
    bad: np.ndarray | None = None,
    pad: int = 8,
    ds: int = 1,
    span: float = 0.012,
):
    import cv2

    rows, cols = np.nonzero(region)
    if len(rows) < 6:
        return None
    r0, r1 = max(0, rows.min() - pad * ds), min(region.shape[0], rows.max() + pad * ds + 1)
    c0, c1 = max(0, cols.min() - pad * ds), min(region.shape[1], cols.max() + pad * ds + 1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    keep = None
    if bad is not None:
        keep = (~bad[r0:r1, c0:c1]).astype(np.uint8)
    if ds > 1:
        win = cv2.resize(win, (max(1, win.shape[1] // ds), max(1, win.shape[0] // ds)), interpolation=cv2.INTER_AREA)
        win = (win > 0).astype(np.uint8) if win.dtype != np.uint8 else (win >= 1).astype(np.uint8)
        if keep is not None:
            keep = cv2.resize(keep, (win.shape[1], win.shape[0]), interpolation=cv2.INTER_NEAREST)
    ss = 4 if ds == 1 else 2
    m_ss = np.kron(win, np.ones((ss, ss), np.uint8))
    k_ss = None if keep is None else np.kron(keep, np.ones((ss, ss), np.uint8))
    if k_ss is not None:
        m_ss &= k_ss
    am = float(np.count_nonzero(m_ss))
    buf = np.zeros(m_ss.shape, np.uint8)
    shift = np.array([c0, r0], dtype=np.float64)

    def iou(x, y, yaw) -> float:
        poly = B.silhouette(cam, x, y, s, yaw, z0)
        if len(poly) < 3:
            return -1.0
        buf[:] = 0
        p = (poly - shift) / ds * ss - 0.5
        cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
        if k_ss is not None:
            buf[:] &= k_ss
        inter = float(np.count_nonzero(buf & m_ss))
        union = am + float(np.count_nonzero(buf)) - inter
        return inter / union if union > 0 else 0.0

    if init is None:
        uc, vc = float(cols.mean()), float(rows.mean())
        p = cam.on_plane_z(np.array([[uc, vc]]), z0 + s / 2)[0]
        if not np.all(np.isfinite(p)):
            return None
        init = p[0], p[1]
    best = -1.0, float(init[0]), float(init[1]), 0.0
    g = np.arange(-span, span + 1e-05, 0.003)
    for yaw in np.radians(np.arange(-45.0, 45.0, 5.0)):
        for dx in g:
            for dy in g:
                v = iou(init[0] + dx, init[1] + dy, yaw)
                if v > best[0]:
                    best = v, init[0] + dx, init[1] + dy, float(yaw)
    for span, step, yspan, ystep in ((0.003, 0.001, 5.0, 1.0), (0.0008, 0.0004, 1.0, 0.5)):
        _, bx, by, byaw = best
        offs = np.arange(-span, span + 1e-09, step)
        for dyaw in np.radians(np.arange(-yspan, yspan + 1e-09, ystep)):
            for dx in offs:
                for dy in offs:
                    v = iou(bx + dx, by + dy, byaw + dyaw)
                    if v > best[0]:
                        best = v, bx + dx, by + dy, byaw + float(dyaw)
    return best[0], best[1], best[2], wrap_yaw(best[3])


def table_region_mask(cam: Camera, shape, region) -> np.ndarray:
    return B.workspace_mask(cam, shape, region=region)


def detect(
    img: np.ndarray,
    cam: Camera,
    colors,
    *,
    region=((-0.5, 0.5), (-0.33, 0.06)),
    s: float = CUBE_SIZE,
    z0: float = TABLE_Z,
    bad: np.ndarray | None = None,
    hsv: np.ndarray | None = None,
) -> dict:
    if hsv is None:
        hsv = hsv_image(img)
    roi = table_region_mask(cam, img.shape, region)
    out = {}
    for name in colors:
        c = canonical(name)
        if c is None:
            out[name] = None
            continue
        m = color_mask(hsv, c) & roi
        if bad is not None:
            m &= ~bad
        comps = _components(m)
        if not comps:
            out[name] = None
            continue
        area, reg, uv = comps[0]
        f = fit_cube(cam, reg, s, z0, bad=bad)
        if f is None:
            out[name] = None
            continue
        out[name] = ColorCube(
            c,
            f[1],
            f[2],
            s,
            f[3],
            z0,
            f[0],
            area,
            uv,
            extra={"n_regions": len(comps), "second_area": comps[1][0] if len(comps) > 1 else 0},
        )
    return out


def fit_at(
    img: np.ndarray,
    cam: Camera,
    color: str,
    xy,
    level: int,
    *,
    others=(),
    s: float = CUBE_SIZE,
    bad: np.ndarray | None = None,
    hsv: np.ndarray | None = None,
    span: float = 0.006,
):
    import cv2

    c = canonical(color)
    if c is None:
        return None
    if hsv is None:
        hsv = hsv_image(img)
    z0 = TABLE_Z + level * s
    uv, _ = cam.project(np.array([[xy[0], xy[1], z0 + s / 2]]))
    u, v = float(uv[0, 0]), float(uv[0, 1])
    m = color_mask(hsv, c)
    if bad is not None:
        m &= ~bad
    comps = _components(m, min_area=8)
    best = None
    for area, reg, cuv in comps:
        rows, cols = np.nonzero(reg)
        d = float(np.min(np.hypot(cols + 0.5 - u, rows + 0.5 - v)))
        if d < 12 and (best is None or d < best[0]):
            best = d, area, reg, cuv
    if best is None:
        return None
    _, area, reg, cuv = best
    unknown = np.zeros(img.shape[:2], bool)
    for o in others:
        co = canonical(o)
        if co is not None and co != c:
            unknown |= color_mask(hsv, co)
    unknown = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if bad is not None:
        unknown |= bad
    f = fit_cube(cam, reg, s, z0, init=(float(xy[0]), float(xy[1])), bad=unknown, span=span)
    if f is None:
        return None
    return ColorCube(c, f[1], f[2], s, f[3], z0, f[0], area, cuv)
