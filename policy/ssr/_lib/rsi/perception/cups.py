from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from . import blocks as B
from .cameras import Camera

CUP_MODEL = 0.036, 0.027, 0.072
TABLE_AB = 20.0
TABLE_L = 25.0
SS = 3
HUES = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "cyan": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 300.0,
    "pink": 330.0,
}
HUE_TOL = 25.0
TABLE_HUE_TOL = 8.0
MIN_SAT = 110
MIN_VAL = 70
MIN_BLOCK_PX = 12
MAX_BLOCK_REGIONS = 6
MIN_BLOCK_IOU = 0.6
MIN_CUP_PX = 300


@dataclass
class CupFit:
    x: float
    y: float
    iou: float
    model: tuple = CUP_MODEL
    area: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def top_z(self) -> float:
        return TABLE_Z + self.model[2]

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "iou": round(self.iou, 3), "area": self.area}


@dataclass
class BlockFit:
    colour: str
    x: float
    y: float
    s: float
    yaw: float
    iou: float
    area: int = 0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return {
            "colour": self.colour,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "s": self.s,
            "yaw": round(math.degrees(self.yaw), 1),
            "iou": round(self.iou, 3),
            "area": self.area,
        }


def colour_words(text: str) -> list[str]:
    return [w for w in re.findall("[a-z]+", text.lower()) if w in HUES]


def _hsv(img):
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)


def colour_mask(img: np.ndarray, colour: str, hsv=None) -> np.ndarray:
    hsv = _hsv(img) if hsv is None else hsv
    h = hsv[..., 0].astype(np.float32) * 2.0
    d = np.abs((h - HUES[colour] + 180.0) % 360.0 - 180.0)
    return (d <= HUE_TOL) & (hsv[..., 1] >= MIN_SAT) & (hsv[..., 2] >= MIN_VAL)


def table_region(cam: Camera, shape, region=((-0.55, 0.55), (-0.35, 0.1))) -> np.ndarray:
    return B.workspace_mask(cam, shape, region)


def not_table_mask(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    reg = table_region(cam, img.shape)
    if robot is not None:
        reg &= ~robot
    t = np.median(lab[reg], axis=0) if reg.any() else np.median(lab.reshape(-1, 3), axis=0)
    dab = np.hypot(lab[..., 1] - t[1], lab[..., 2] - t[2])
    m = (dab > TABLE_AB) | (lab[..., 0] - t[0] > TABLE_L)
    return m & reg


def table_hue(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None, hsv=None) -> float:
    hsv = _hsv(img) if hsv is None else hsv
    reg = table_region(cam, img.shape)
    if robot is not None:
        reg &= ~robot
    h = hsv[..., 0][reg].astype(np.float32) * 2.0 if reg.any() else hsv[..., 0].reshape(-1).astype(np.float32) * 2.0
    a = np.deg2rad(h)
    mean = math.degrees(math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))))
    d = (h - mean + 180.0) % 360.0 - 180.0
    return float((mean + np.median(d)) % 360.0)


def not_table_hue(img: np.ndarray, t_hue: float, hsv=None) -> np.ndarray:
    hsv = _hsv(img) if hsv is None else hsv
    h = hsv[..., 0].astype(np.float32) * 2.0
    return np.abs((h - t_hue + 180.0) % 360.0 - 180.0) > TABLE_HUE_TOL


def cup_points(x: float, y: float, model=CUP_MODEL, z0: float = TABLE_Z, n: int = 24) -> np.ndarray:
    r_bot, r_top, h = model
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    bot = np.stack([x + r_bot * c, y + r_bot * s, np.full(n, z0)], axis=1)
    top = np.stack([x + r_top * c, y + r_top * s, np.full(n, z0 + h)], axis=1)
    return np.vstack([bot, top])


def cup_silhouette(cam: Camera, x: float, y: float, model=CUP_MODEL, z0: float = TABLE_Z) -> np.ndarray:
    import cv2

    uv, d = cam.project(cup_points(x, y, model, z0))
    if np.any(d <= 0.01):
        return np.zeros((0, 2))
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def _iou_grid(cam, m_ss, off, xs, ys, model, z0):
    import cv2

    am = float(np.count_nonzero(m_ss))
    buf = np.zeros(m_ss.shape, np.uint8)
    best = -1.0, float(xs[0]), float(ys[0])
    shift = np.array([off[1], off[0]], dtype=np.float64)
    for x in xs:
        for y in ys:
            poly = cup_silhouette(cam, x, y, model, z0)
            if len(poly) < 3:
                continue
            buf[:] = 0
            p = (poly - shift) * SS - 0.5
            cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
            inter = float(np.count_nonzero(buf & m_ss))
            union = am + float(np.count_nonzero(buf)) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best[0]:
                best = iou, float(x), float(y)
    return best


def fit_cup(cam: Camera, region: np.ndarray, model=CUP_MODEL, z0: float = TABLE_Z, pad: int = 12) -> CupFit | None:
    rows, cols = np.nonzero(region)
    if len(rows) < 20:
        return None
    r0, r1 = max(0, rows.min() - pad), min(region.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(region.shape[1], cols.max() + pad + 1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    m_ss = np.kron(win, np.ones((SS, SS), np.uint8))
    p = cam.on_plane_z(np.array([[cols.mean(), rows.mean()]]), z0 + model[2] / 2)[0]
    if not np.all(np.isfinite(p)):
        return None
    best = -1.0, p[0], p[1]
    for span, step in ((0.024, 0.006), (0.006, 0.0015), (0.0015, 0.0005)):
        g = np.arange(-span, span + 1e-09, step)
        best = _iou_grid(cam, m_ss, (r0, c0), best[1] + g, best[2] + g, model, z0)
    return CupFit(best[1], best[2], best[0], model, area=int(len(rows)))


def detect_cups(
    img: np.ndarray,
    cam: Camera,
    robot: np.ndarray | None = None,
    n: int = 3,
    model=CUP_MODEL,
    min_iou: float = 0.6,
    exclude: np.ndarray | None = None,
) -> list[CupFit]:
    import cv2

    m = not_table_mask(img, cam, robot)
    if robot is not None:
        m &= ~robot
    if exclude is not None:
        m &= ~exclude
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab = cv2.connectedComponents(m)
    out = []
    for i in range(1, k):
        reg = lab == i
        if np.count_nonzero(reg) < MIN_CUP_PX:
            continue
        f = fit_cup(cam, reg, model)
        if f is not None and f.iou >= min_iou:
            out.append(f)
    out.sort(key=lambda c: -c.iou)
    return out[:n]


def block_pixels(img: np.ndarray, cam: Camera, colours, robot: np.ndarray | None = None, dilate: int = 2) -> np.ndarray:
    import cv2

    hsv = _hsv(img)
    m = np.zeros(img.shape[:2], bool)
    for col in colours:
        if col in HUES:
            m |= colour_mask(img, col, hsv)
    m &= not_table_mask(img, cam, robot) & not_table_hue(img, table_hue(img, cam, robot, hsv), hsv)
    if dilate > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    return m


def detect_blocks(
    img: np.ndarray, cam: Camera, colours, robot: np.ndarray | None = None, size: float = 0.025, z0: float = TABLE_Z
) -> dict[str, BlockFit]:
    import cv2

    hsv = _hsv(img)
    reg_all = table_region(cam, img.shape)
    nt = not_table_mask(img, cam, robot) & not_table_hue(img, table_hue(img, cam, robot, hsv), hsv)
    out = {}
    for col in colours:
        if col not in HUES:
            continue
        m = colour_mask(img, col, hsv) & reg_all & nt
        if robot is not None:
            m &= ~robot
        k, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if k <= 1:
            continue
        order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:MAX_BLOCK_REGIONS]
        best = None
        for i in order:
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_BLOCK_PX:
                break
            fits = B.fit_region(cam, lab == i, sizes=(size,), z0=z0, fit_yaw=True)
            if not fits:
                continue
            iou, x, y, s, yaw = fits[0]
            if best is None or iou > best.iou:
                best = BlockFit(col, float(x), float(y), float(s), float(yaw), float(iou), area)
        if best is not None and best.iou >= MIN_BLOCK_IOU:
            out[col] = best
    return out
