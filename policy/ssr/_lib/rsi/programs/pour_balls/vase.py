from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...planning.kinematics import TABLE_Z
from ...perception import cups as CU
from ...perception.cameras import Camera

UPRIGHT_CUP = 0.0275, 0.0361, 0.072
VASE_TOP_Z = TABLE_Z + 0.1465
VASE_R_RANGE = 0.018, 0.07
MIN_VASE_PX = 400
TABLE_TOP = (-0.68, 0.68), (-0.42, 0.48)
CUP_MIN_IOU = 0.6


@dataclass
class VaseFit:
    x: float
    y: float
    z_top: float
    r_top: float
    score: float
    area: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def r_open(self) -> float:
        return float(self.extra.get("r_open", OPEN_FRAC * self.r_top))

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z_top": round(self.z_top, 4),
            "r_top": round(self.r_top, 4),
            "score": round(self.score, 3),
            "area": self.area,
            **self.extra,
        }


def not_table(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    reg = CU.table_region(cam, img.shape)
    if robot is not None:
        reg &= ~robot
    t = np.median(lab[reg], axis=0) if reg.any() else np.median(lab.reshape(-1, 3), axis=0)
    dab = np.hypot(lab[..., 1] - t[1], lab[..., 2] - t[2])
    m = (dab > CU.TABLE_AB) | (lab[..., 0] - t[0] > CU.TABLE_L)
    return m & CU.table_region(cam, img.shape, TABLE_TOP)


def components(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None, min_px: int = 150):
    import cv2

    m = not_table(img, cam, robot)
    if robot is not None:
        m &= ~robot
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    regs = [(int(stats[i, cv2.CC_STAT_AREA]), lab == i) for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    regs.sort(key=lambda t: -t[0])
    return [r for _, r in regs]


def top_boundary(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = np.nonzero(region.any(axis=0))[0]
    tops = np.argmax(region[:, cols], axis=0)
    return cols, tops.astype(float)


_RIM_N = 360
_RIM_A = np.linspace(0, 2 * np.pi, _RIM_N, endpoint=False)


def _ellipse_top(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ui = np.round(uv[:, 0]).astype(int)
    u0 = int(ui.min())
    top = np.full(int(ui.max()) - u0 + 1, np.inf)
    np.minimum.at(top, ui - u0, uv[:, 1])
    us = np.arange(u0, u0 + len(top))
    ok = np.isfinite(top)
    return us[ok], top[ok]


def rim_score(
    cam: Camera,
    cols,
    tops,
    x: float,
    y: float,
    r: float,
    z: float,
    tol: float = 1.5,
    b: float | None = None,
    yaw: float = 0.0,
) -> float:
    if b is None:
        pts = np.stack([x + r * np.cos(_RIM_A), y + r * np.sin(_RIM_A), np.full(_RIM_N, z)], axis=1)
    else:
        ca, sa = math.cos(yaw), math.sin(yaw)
        ex, ey = r * np.cos(_RIM_A), b * np.sin(_RIM_A)
        pts = np.stack([x + ca * ex - sa * ey, y + sa * ex + ca * ey, np.full(_RIM_N, z)], axis=1)
    uv, _ = cam.project(pts)
    us, ve = _ellipse_top(uv)
    if len(us) == 0:
        return -1000000000.0
    idx = np.clip(np.searchsorted(cols, us), 0, len(cols) - 1)
    have = cols[idx] == us
    err = np.where(have, np.abs(tops[idx] - ve), 99.0)
    s = np.maximum(0.0, 1.0 - err / (2 * tol))
    return float(s.sum() - 0.5 * np.count_nonzero(err > 3 * tol))


def fit_rim(cam: Camera, region: np.ndarray, z: float = VASE_TOP_Z) -> VaseFit | None:
    cols, tops = top_boundary(region)
    if len(cols) < 5:
        return None
    i = int(np.argmin(tops))
    pb = cam.on_plane_z(np.array([[cols[i], tops[i]]]), z)[0]
    if not np.all(np.isfinite(pb)):
        return None
    away = pb[:2] - cam.pos[:2]
    away /= np.linalg.norm(away)
    best = -1000000000.0, 0.0, 0.0, 0.0
    g = np.arange(-0.02, 0.0201, 0.002)
    for r in np.arange(VASE_R_RANGE[0], VASE_R_RANGE[1] + 1e-09, 0.002):
        c = pb[:2] - r * away
        for dx in g:
            for dy in g:
                s = rim_score(cam, cols, tops, c[0] + dx, c[1] + dy, r, z)
                if s > best[0]:
                    best = s, c[0] + dx, c[1] + dy, r
    for span, step in ((0.002, 0.0005),):
        _, x0, y0, r0 = best
        best = rim_score(cam, cols, tops, x0, y0, r0, z), x0, y0, r0
        for r in np.arange(r0 - span, r0 + span + 1e-09, step):
            if r < 0.01:
                continue
            for dx in np.arange(-span, span + 1e-09, step):
                for dy in np.arange(-span, span + 1e-09, step):
                    s = rim_score(cam, cols, tops, x0 + dx, y0 + dy, r, z)
                    if s > best[0]:
                        best = s, x0 + dx, y0 + dy, r
    s, x, y, r = best
    return VaseFit(float(x), float(y), float(z), float(r), float(s), area=int(np.count_nonzero(region)))


VASE_MODELS = {
    0: (
        0.142,
        0.0521,
        (
            0.0495,
            0.0535,
            0.0538,
            0.053,
            0.0491,
            0.0359,
            0.0281,
            0.0305,
            0.0329,
            0.0361,
            0.0395,
            0.0438,
            0.0486,
            0.0545,
            0.0592,
        ),
    ),
    1: (
        0.1479,
        0.0258,
        (
            0.0377,
            0.044,
            0.0485,
            0.0498,
            0.05,
            0.0497,
            0.0478,
            0.0451,
            0.0411,
            0.0362,
            0.031,
            0.0289,
            0.0315,
            0.0356,
            0.0399,
        ),
    ),
    2: (
        0.148,
        0.0302,
        (
            0.0529,
            0.0707,
            0.0839,
            0.0897,
            0.0901,
            0.0875,
            0.0767,
            0.0613,
            0.0463,
            0.0397,
            0.0383,
            0.0371,
            0.036,
            0.0347,
            0.0333,
        ),
    ),
    3: (
        0.1511,
        0.0307,
        (
            0.0256,
            0.0169,
            0.0191,
            0.0214,
            0.0225,
            0.0227,
            0.0222,
            0.0186,
            0.0189,
            0.0201,
            0.0215,
            0.0232,
            0.0256,
            0.0306,
            0.0399,
        ),
    ),
    4: (
        0.1488,
        0.0222,
        (
            0.0353,
            0.0408,
            0.0452,
            0.0494,
            0.053,
            0.0548,
            0.0569,
            0.057,
            0.0567,
            0.0554,
            0.0518,
            0.047,
            0.04,
            0.0264,
            0.0267,
        ),
    ),
}
MODEL_SS = 2
RIM_ELLIPSE = {
    1: 0.0313
}
MODEL_MIN_IOU = 0.72
RIM_DZ = -0.004
RIM_DZ_MODEL = {1: 0.0}
OPEN_FRAC = 0.75
_RING_A = np.linspace(0, 2 * np.pi, 32, endpoint=False)


def model_rings(model: int, x: float, y: float, z0: float = TABLE_Z) -> np.ndarray:
    h, _, radii = VASE_MODELS[model]
    n = len(radii)
    zs = [0.0] + [(k + 0.5) * h / n for k in range(n)] + [h]
    rs = [radii[0]] + list(radii) + [radii[-1]]
    c, s = np.cos(_RING_A), np.sin(_RING_A)
    return np.stack([np.stack([x + r * c, y + r * s, np.full_like(c, z0 + z)], axis=1) for z, r in zip(zs, rs)])


def _model_iou(cam, m_ss, area, off, model, x, y, buf):
    import cv2

    rings = model_rings(model, x, y)
    uv, d = cam.project(rings.reshape(-1, 3))
    if np.any(d <= 0.01):
        return -1.0
    uv = ((uv - [off[1], off[0]]) * MODEL_SS - 0.5).reshape(rings.shape[0], -1, 2)
    buf[:] = 0
    for k in range(len(uv) - 1):
        hull = cv2.convexHull(np.round(np.vstack([uv[k], uv[k + 1]]) * 4).astype(np.int32))
        cv2.fillConvexPoly(buf, hull, 1, shift=2)
    inter = float(np.count_nonzero(buf & m_ss))
    union = area + float(np.count_nonzero(buf)) - inter
    return inter / union if union > 0 else 0.0


def base_guess(cam: Camera, region: np.ndarray, model: int):
    rows, cols = np.nonzero(region)
    if len(rows) == 0:
        return None
    v = rows.max()
    u = float(np.mean(cols[rows >= v - 1]))
    b = cam.on_plane_z(np.array([[u + 0.5, v + 1.0]]), TABLE_Z)[0]
    if not np.all(np.isfinite(b)):
        return None
    toward = cam.pos[:2] - b[:2]
    toward = toward / max(1e-09, float(np.linalg.norm(toward)))
    return b[:2] - VASE_MODELS[model][2][0] * toward


def fit_models(cam: Camera, region: np.ndarray, guess_xy, models=None, pad: int = 25):
    rows, cols = np.nonzero(region)
    if len(rows) < 30:
        return None
    r0, r1 = max(0, rows.min() - pad), min(region.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(region.shape[1], cols.max() + pad + 1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    m_ss = np.kron(win, np.ones((MODEL_SS, MODEL_SS), np.uint8))
    area = float(np.count_nonzero(m_ss))
    buf = np.zeros_like(m_ss)
    best = None
    for model in models if models is not None else sorted(VASE_MODELS):
        starts = [(float(guess_xy[0]), float(guess_xy[1]))]
        bg = base_guess(cam, region, model)
        if bg is not None and np.hypot(bg[0] - starts[0][0], bg[1] - starts[0][1]) > 0.02:
            starts.append((float(bg[0]), float(bg[1])))
        for sx, sy in starts:
            cur = -1.0, sx, sy
            for span, step in ((0.03, 0.006), (0.006, 0.0015), (0.0015, 0.0005)):
                g = np.arange(-span, span + 1e-09, step)
                x0, y0 = cur[1], cur[2]
                for dx in g:
                    for dy in g:
                        iou = _model_iou(cam, m_ss, area, (r0, c0), model, x0 + dx, y0 + dy, buf)
                        if iou > cur[0]:
                            cur = iou, x0 + dx, y0 + dy
            if best is None or cur[0] > best[0]:
                best = cur[0], model, cur[1], cur[2]
    return best


def fit_rim_fixed(cam: Camera, region: np.ndarray, z: float, r: float, guess_xy, span: float = 0.012):
    cols, tops = top_boundary(region)
    best = -1000000000.0, float(guess_xy[0]), float(guess_xy[1])
    for sp, st in ((span, 0.001), (0.001, 0.00025)):
        g = np.arange(-sp, sp + 1e-09, st)
        x0, y0 = best[1], best[2]
        for dx in g:
            for dy in g:
                sc = rim_score(cam, cols, tops, x0 + dx, y0 + dy, r, z)
                if sc > best[0]:
                    best = sc, x0 + dx, y0 + dy
    return best


def fit_rim_ellipse(cam: Camera, region: np.ndarray, z: float, a: float, b: float, guess_xy, span: float = 0.014):
    cols, tops = top_boundary(region)
    best = -1000000000.0, float(guess_xy[0]), float(guess_xy[1]), 0.0
    g = np.arange(-span, span + 1e-09, 0.002)
    for yaw in np.radians(np.arange(-90.0, 90.0, 15.0)):
        for dx in g:
            for dy in g:
                sc = rim_score(cam, cols, tops, guess_xy[0] + dx, guess_xy[1] + dy, a, z, b=b, yaw=yaw)
                if sc > best[0]:
                    best = sc, guess_xy[0] + dx, guess_xy[1] + dy, float(yaw)
    for sp, st, ys in ((0.002, 0.0005, math.radians(5.0)), (0.0005, 0.00025, math.radians(2.0))):
        _, x0, y0, w0 = best
        for yaw in (w0 - 2 * ys, w0 - ys, w0, w0 + ys, w0 + 2 * ys):
            for dx in np.arange(-sp, sp + 1e-09, st):
                for dy in np.arange(-sp, sp + 1e-09, st):
                    sc = rim_score(cam, cols, tops, x0 + dx, y0 + dy, a, z, b=b, yaw=yaw)
                    if sc > best[0]:
                        best = sc, x0 + dx, y0 + dy, float(yaw)
    return best


def refine_with_model(cam: Camera, region: np.ndarray, vase: VaseFit) -> VaseFit:
    vase.extra["r_open"] = OPEN_FRAC * vase.r_top
    fm = fit_models(cam, region, vase.xy)
    if fm is None:
        return vase
    iou, m, mx, my = fm
    vase.extra.update(
        model=int(m),
        model_iou=round(float(iou), 3),
        model_xy=[round(mx, 4), round(my, 4)],
        generic_xy=[round(vase.x, 4), round(vase.y, 4)],
        generic_r=round(vase.r_top, 4),
    )
    if iou < MODEL_MIN_IOU:
        return vase
    h, r_open, radii = VASE_MODELS[m]
    z, r = TABLE_Z + h, radii[-1]
    if m in RIM_ELLIPSE:
        sc, x, y, yaw = fit_rim_ellipse(cam, region, z + RIM_DZ_MODEL.get(m, RIM_DZ), r, RIM_ELLIPSE[m], (mx, my))
        vase.extra["rim_yaw"] = round(math.degrees(yaw), 1)
    else:
        sc, x, y = fit_rim_fixed(cam, region, z + RIM_DZ_MODEL.get(m, RIM_DZ), r, (mx, my))
    vase.x, vase.y, vase.z_top, vase.r_top, vase.score = float(x), float(y), float(z), float(r), float(sc)
    vase.extra["r_open"] = r_open
    return vase


def cup_mask(cam: Camera, cup, shape, dilate: int = 3) -> np.ndarray:
    import cv2

    m = np.zeros(shape[:2], np.uint8)
    poly = CU.cup_silhouette(cam, cup.x, cup.y, UPRIGHT_CUP)
    if len(poly) >= 3:
        cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 1)
        if dilate > 0:
            m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return m > 0


def split_cup(img: np.ndarray, region: np.ndarray, cam: Camera):
    import cv2

    n = int(np.count_nonzero(region))
    if n < CU.MIN_CUP_PX + MIN_VASE_PX // 2:
        return None
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    rows, cols = np.nonzero(region)
    px = lab[rows, cols]
    best = None
    crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5
    for k in (2, 3):
        _, lbl, _ = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.reshape(-1)
        for j in range(k):
            m = np.zeros(region.shape, np.uint8)
            m[rows[lbl == j], cols[lbl == j]] = 1
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            kk, cc, st, _ = cv2.connectedComponentsWithStats(m)
            if kk <= 1:
                continue
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            part = cc == big
            if np.count_nonzero(part) < CU.MIN_CUP_PX:
                continue
            f = CU.fit_cup(cam, part, UPRIGHT_CUP)
            if f is not None and f.iou >= CUP_MIN_IOU and (best is None or f.iou > best.iou):
                best = f
    return best


PART_GAP = 10
PART_MIN_OVERLAP = 0.5
PART_MAX_ASPECT = 3.0


def _is_part(vase_reg: np.ndarray, r: np.ndarray) -> bool:
    vr, vc = np.nonzero(vase_reg)
    rr, rc = np.nonzero(r)
    w, h = rc.max() - rc.min() + 1, rr.max() - rr.min() + 1
    ov = max(0, min(rc.max(), vc.max()) - max(rc.min(), vc.min()) + 1)
    return bool(rr.mean() > vr.min() and ov >= PART_MIN_OVERLAP * w and (w <= PART_MAX_ASPECT * h))


def with_near_parts(regs, i: int, cup_i: int):
    import cv2

    near = cv2.dilate(regs[i].astype(np.uint8), np.ones((2 * PART_GAP + 1, 2 * PART_GAP + 1), np.uint8)) > 0
    parts = [r for j, r in enumerate(regs) if j not in (i, cup_i) and np.any(r & near) and _is_part(regs[i], r)]
    if not parts:
        return None
    out = regs[i].copy()
    for r in parts:
        out |= r
    return out


def detect(img: np.ndarray, cam: Camera, robot: np.ndarray | None = None):
    regs = components(img, cam, robot)
    cup, cup_i = None, -1
    for i, reg in enumerate(regs[:6]):
        if np.count_nonzero(reg) < CU.MIN_CUP_PX:
            continue
        f = CU.fit_cup(cam, reg, UPRIGHT_CUP)
        if f is not None and f.iou >= CUP_MIN_IOU and (cup is None or f.iou > cup.iou):
            cup, cup_i = f, i
    if cup is None:
        for i, reg in enumerate(regs[:3]):
            f = split_cup(img, reg, cam)
            if f is not None and (cup is None or f.iou > cup.iou):
                cup, cup_i = f, i
        if cup is not None:
            regs = list(regs)
            regs[cup_i] = regs[cup_i] & ~cup_mask(cam, cup, img.shape)
            cup_i = -1
    vase = None
    for i, reg in enumerate(regs):
        if i == cup_i or np.count_nonzero(reg) < MIN_VASE_PX:
            continue
        vase = fit_rim(cam, reg)
        if vase is not None:
            vase = refine_with_model(cam, reg, vase)
            merged = with_near_parts(regs, i, cup_i)
            if merged is not None:
                v2 = fit_rim(cam, merged)
                if v2 is not None:
                    v2 = refine_with_model(cam, merged, v2)
                    if v2.extra.get("model_iou", 0.0) > vase.extra.get("model_iou", 0.0):
                        v2.extra["merged"] = True
                        vase = v2
            break
    return cup, vase
