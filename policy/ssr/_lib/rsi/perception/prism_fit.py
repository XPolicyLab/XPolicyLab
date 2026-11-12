from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from .cameras import Camera


@dataclass(frozen=True)
class PrismModel:
    boxes: tuple
    height: float
    origin_z: float = 0.0

    def footprint(self, x: float, y: float, yaw: float) -> list[np.ndarray]:
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for x0, x1, y0, y1 in self.boxes:
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            out.append(np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1))
        return out

    def corners(self, x: float, y: float, yaw: float, z0: float = TABLE_Z) -> list[np.ndarray]:
        out = []
        for fp in self.footprint(x, y, yaw):
            lo = np.hstack([fp, np.full((4, 1), z0)])
            hi = np.hstack([fp, np.full((4, 1), z0 + self.height)])
            out.append(np.vstack([lo, hi]))
        return out

    def points(self, x: float, y: float, yaw: float, z: float) -> np.ndarray:
        return np.array([x, y, z])


T_BLOCK = PrismModel(boxes=((-0.04, -0.02, -0.03, 0.03), (-0.02, 0.04, -0.01, 0.01)), height=0.015, origin_z=0.0075)


def silhouette_polys(
    cam: Camera, model: PrismModel, x: float, y: float, yaw: float, z0: float = TABLE_Z
) -> list[np.ndarray]:
    import cv2

    polys = []
    for cs in model.corners(x, y, yaw, z0):
        uv, d = cam.project(cs)
        if np.any(d <= 0.01):
            return []
        polys.append(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2))
    return polys


def render(
    cam: Camera,
    model: PrismModel,
    x,
    y,
    yaw,
    shape,
    z0: float = TABLE_Z,
    off=(0, 0),
    ss: int = 1,
    buf: np.ndarray | None = None,
) -> np.ndarray:
    import cv2

    if buf is None:
        buf = np.zeros((shape[0] * ss, shape[1] * ss), np.uint8)
    else:
        buf[:] = 0
    shift = np.array([off[1], off[0]], dtype=np.float64)
    for poly in silhouette_polys(cam, model, x, y, yaw, z0):
        p = (poly - shift) * ss - 0.5
        cv2.fillConvexPoly(buf, np.round(p * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return buf


@dataclass
class PrismFit:
    x: float
    y: float
    yaw: float
    iou: float
    area: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "yaw_deg": round(math.degrees(self.yaw), 2),
            "iou": round(self.iou, 4),
            "area": self.area,
        }


class _Scorer:
    def __init__(
        self,
        cam: Camera,
        model: PrismModel,
        region: np.ndarray,
        z0: float,
        pad: int,
        ss: int,
        ignore: np.ndarray | None = None,
    ):
        rows, cols = np.nonzero(region)
        self.ok = len(rows) >= 6
        if not self.ok:
            return
        h, w = region.shape
        r0, r1 = max(0, rows.min() - pad), min(h, rows.max() + pad + 1)
        c0, c1 = max(0, cols.min() - pad), min(w, cols.max() + pad + 1)
        self.off = int(r0), int(c0)
        win = region[r0:r1, c0:c1].astype(np.uint8)
        self.shape = win.shape
        self.ss = ss
        self.m = np.kron(win, np.ones((ss, ss), np.uint8)) if ss > 1 else win
        self.v = None
        if ignore is not None:
            v = (~ignore[r0:r1, c0:c1]).astype(np.uint8)
            self.v = np.kron(v, np.ones((ss, ss), np.uint8)) if ss > 1 else v
            self.m &= self.v
        self.am = float(np.count_nonzero(self.m))
        self.buf = np.zeros(self.m.shape, np.uint8)
        self.cam, self.model, self.z0 = cam, model, z0
        self.area = int(len(rows))
        self.uv = float(cols.mean()), float(rows.mean())

    def iou(self, x, y, yaw) -> float:
        b = render(self.cam, self.model, x, y, yaw, self.shape, self.z0, self.off, self.ss, self.buf)
        if self.v is not None:
            b &= self.v
        inter = float(np.count_nonzero(b & self.m))
        union = self.am + float(np.count_nonzero(b)) - inter
        return inter / union if union > 0 else 0.0

    def grid(self, xs, ys, yaws):
        best = -1.0, xs[0], ys[0], yaws[0]
        for yaw in yaws:
            for x in xs:
                for y in ys:
                    v = self.iou(x, y, yaw)
                    if v > best[0]:
                        best = v, x, y, yaw
        return best


def fit(
    cam: Camera,
    region: np.ndarray,
    model: PrismModel = T_BLOCK,
    z0: float = TABLE_Z,
    init=None,
    yaw_init: float | None = None,
    yaw_span: float = math.pi,
    pad: int = 6,
    ss_fine: int = 4,
    coarse_step: float | None = None,
    ignore: np.ndarray | None = None,
    coarse_yaw: float = math.radians(10.0),
) -> PrismFit | None:
    sc = _Scorer(cam, model, region, z0, pad, 1, ignore)
    if not sc.ok:
        return None
    if init is None:
        p = cam.on_plane_z(np.array([sc.uv]), z0 + model.height / 2)[0]
        if not np.all(np.isfinite(p)):
            return None
        init = p[:2]
    x0, y0 = float(init[0]), float(init[1])
    pa = cam.on_plane_z(np.array([sc.uv, (sc.uv[0] + 1.0, sc.uv[1])]), z0 + model.height)
    px = float(np.linalg.norm(pa[1] - pa[0])) if np.all(np.isfinite(pa)) else 0.002
    step = coarse_step if coarse_step is not None else max(0.002, 2.0 * px)
    nc = max(3, math.ceil(0.012 / step))
    g = np.arange(-nc, nc + 1) * step
    yc = 0.0 if yaw_init is None else yaw_init
    n_yaw = max(1, int(round(2 * yaw_span / coarse_yaw)))
    yaws = (
        yc + np.linspace(-yaw_span, yaw_span, n_yaw, endpoint=False)
        if yaw_span < math.pi
        else np.arange(0.0, 2 * math.pi - 1e-09, coarse_yaw)
    )
    best = sc.grid(x0 + g, y0 + g, yaws)
    for dpos, dyaw, n in ((step / 2, math.radians(2.0), 3), (step / 4, math.radians(0.7), 3)):
        gp = np.arange(-n, n + 1) * dpos
        gy = np.arange(-n, n + 1) * dyaw
        best = sc.grid(best[1] + gp, best[2] + gp, best[3] + gy)
    if ss_fine > 1:
        sf = _Scorer(cam, model, region, z0, pad, ss_fine, ignore)
        for dpos, dyaw, n in ((step / 8, math.radians(0.3), 2), (step / 16, math.radians(0.12), 2)):
            gp = np.arange(-n, n + 1) * dpos
            gy = np.arange(-n, n + 1) * dyaw
            best = sf.grid(best[1] + gp, best[2] + gp, best[3] + gy)
    iou, x, y, yaw = best
    return PrismFit(float(x), float(y), float(math.remainder(yaw, 2 * math.pi)), float(iou), sc.area)


MIN_REGION_PX = 40
MIN_FIT_IOU = 0.6
MAX_REGIONS = 4


def _table_stats(img: np.ndarray, hsv: np.ndarray, ref: np.ndarray):
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    t = np.median(lab[ref], axis=0)
    h = hsv[..., 0][ref].astype(np.float32) * 2.0
    a = np.deg2rad(h)
    mean = math.degrees(math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))))
    d = (h - mean + 180.0) % 360.0 - 180.0
    return lab, t, float((mean + np.median(d)) % 360.0)


def colour_regions(
    img: np.ndarray, cam: Camera, robot: np.ndarray | None = None, region: np.ndarray | None = None, colours=None
):
    import cv2
    from . import cups as CU

    hsv = CU._hsv(img)
    treg = CU.table_region(cam, img.shape)
    if robot is not None:
        treg &= ~robot
    if treg.mean() < 0.2:
        treg = np.ones(img.shape[:2], bool) if robot is None else ~robot
    if region is None:
        region = CU.table_region(cam, img.shape)
    lab, t, t_hue = _table_stats(img, hsv, treg)
    dab = np.hypot(lab[..., 1] - t[1], lab[..., 2] - t[2])
    nt = (dab > CU.TABLE_AB) | (lab[..., 0] - t[0] > CU.TABLE_L)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    nt &= np.abs((hue - t_hue + 180.0) % 360.0 - 180.0) > CU.TABLE_HUE_TOL
    out = []
    for col in colours or CU.HUES:
        m = CU.colour_mask(img, col, hsv) & region & nt
        if robot is not None:
            m &= ~robot
        _, lab_, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:MAX_REGIONS]
        for i in order:
            if int(stats[i, cv2.CC_STAT_AREA]) < MIN_REGION_PX:
                break
            out.append((col, lab_ == i))
    return out


def detect(
    img: np.ndarray,
    cam: Camera,
    model: PrismModel = T_BLOCK,
    robot: np.ndarray | None = None,
    z0: float = TABLE_Z,
    n: int = 2,
    min_iou: float = MIN_FIT_IOU,
    min_dist: float = 0.04,
    colours=None,
    region: np.ndarray | None = None,
) -> list[tuple[str, PrismFit]]:
    fits = []
    done = np.zeros(img.shape[:2], bool)
    for col, reg in sorted(
        colour_regions(img, cam, robot, region=region, colours=colours), key=lambda cr: -int(cr[1].sum())
    ):
        n = int(reg.sum())
        if int((reg & done).sum()) > 0.5 * n:
            continue
        done |= reg
        f = fit(cam, reg, model, z0, ignore=robot)
        if f is not None and f.iou >= min_iou:
            fits.append((col, f))
    fits.sort(key=lambda cf: -cf[1].iou)
    out = []
    for col, f in fits:
        if all(float(np.linalg.norm(f.xy - g.xy)) >= min_dist for _, g in out):
            out.append((col, f))
        if len(out) >= n:
            break
    return out
