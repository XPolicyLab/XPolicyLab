from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from . import robot_mask as RM
from .cameras import Camera

BLOCK_SIZES = 0.03, 0.035, 0.04
ALT_SIZES = 0.024, 0.05, 0.065
ALT_MARGIN = 0.03
SPLIT_BELOW = 0.7
SPLIT_UNION_MIN = 0.8
WORKSPACE = (-0.58, 0.58), (-0.42, 0.12)
FG_L = 28.0
FG_AB = 9.0
YAW_EVIDENCE = 0.04


@dataclass
class Cube:
    x: float
    y: float
    s: float
    yaw: float = 0.0
    z0: float = TABLE_Z
    score: float = 0.0
    area: int = 0
    color: tuple = (0, 0, 0)
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
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "s": self.s,
            "yaw": round(math.degrees(self.yaw), 1),
            "z0": round(self.z0, 4),
            "score": round(self.score, 3),
            "area": self.area,
            "color": [int(c) for c in self.color],
            "uv": [round(float(v), 1) for v in self.uv],
            "touch_robot": bool(self.extra.get("touch_robot", False)),
        }


def corners(x: float, y: float, s: float, yaw: float, z0: float = TABLE_Z, height: float | None = None) -> np.ndarray:
    c, sn = math.cos(yaw), math.sin(yaw)
    h = s / 2
    pts = []
    for dz in (0.0, s if height is None else height):
        for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h)):
            pts.append((x + c * dx - sn * dy, y + sn * dx + c * dy, z0 + dz))
    return np.asarray(pts)


def silhouette(cam: Camera, x, y, s, yaw, z0=TABLE_Z, height: float | None = None) -> np.ndarray:
    import cv2

    uv, d = cam.project(corners(x, y, s, yaw, z0, height))
    if np.any(d <= 0.01):
        return np.zeros((0, 2))
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def workspace_mask(cam: Camera, shape, region=WORKSPACE, z=TABLE_Z) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1) = region
    poly = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    uv, d = cam.project(poly)
    m = np.zeros(shape[:2], np.uint8)
    if np.all(d > 0):
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m > 0


def foreground_score(img: np.ndarray, robot: np.ndarray | None = None, ksize: int = 41) -> np.ndarray:
    import cv2

    src = img
    if robot is not None and robot.any():
        src = cv2.inpaint(img, robot.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    bg = cv2.medianBlur(src, ksize)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    labb = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = lab - labb
    return np.maximum(np.abs(d[..., 0]) / FG_L, np.hypot(d[..., 1], d[..., 2]) / FG_AB)


SS = 4


def _iou_grid(
    cam: Camera, m: np.ndarray, off, xs, ys, s, yaws, z0, m_ss: np.ndarray | None = None, height: float | None = None
):
    import cv2

    if m_ss is None:
        m_ss = np.kron(m, np.ones((SS, SS), np.uint8))
    am = float(np.count_nonzero(m_ss))
    buf = np.zeros(m_ss.shape, np.uint8)
    best = -1.0, xs[0], ys[0], yaws[0]
    shift = np.array([off[1], off[0]], dtype=np.float64)
    for yaw in yaws:
        for x in xs:
            for y in ys:
                poly = silhouette(cam, x, y, s, yaw, z0, height)
                if len(poly) < 3:
                    continue
                buf[:] = 0
                p = (poly - shift) * SS - 0.5
                cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
                inter = float(np.count_nonzero(buf & m_ss))
                union = am + float(np.count_nonzero(buf)) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best[0]:
                    best = iou, x, y, yaw
    return best


def fit_region(
    cam: Camera, region: np.ndarray, sizes=BLOCK_SIZES, z0: float = TABLE_Z, fit_yaw: bool = True, pad: int = 10
) -> list[tuple]:
    rows, cols = np.nonzero(region)
    if len(rows) < 6:
        return []
    r0, r1 = max(0, rows.min() - pad), min(region.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(region.shape[1], cols.max() + pad + 1)
    win = region[r0:r1, c0:c1].astype(np.uint8)
    uc, vc = float(cols.mean()), float(rows.mean())
    out = []
    for s in sizes:
        p = cam.on_plane_z(np.array([[uc, vc]]), z0 + s / 2)[0]
        if not np.all(np.isfinite(p)):
            continue
        m_ss = np.kron(win, np.ones((SS, SS), np.uint8))
        g = np.arange(-0.012, 0.0121, 0.003)
        f = np.arange(-0.0024, 0.00241, 0.0008)
        b = _iou_grid(cam, win, (r0, c0), p[0] + g, p[1] + g, s, (0.0,), z0, m_ss)
        b0 = _iou_grid(cam, win, (r0, c0), b[1] + f, b[2] + f, s, (0.0,), z0, m_ss)
        if fit_yaw:
            yaws = np.radians(np.arange(-40.0, 45.0, 10.0))
            b1 = _iou_grid(cam, win, (r0, c0), [b[1]], [b[2]], s, yaws, z0, m_ss)
            if abs(b1[3]) > 1e-06:
                yf = b1[3] + np.radians(np.array([-4.0, 0.0, 4.0]))
                b1 = _iou_grid(cam, win, (r0, c0), b1[1] + f, b1[2] + f, s, yf, z0, m_ss)
                if b1[0] > b0[0] + YAW_EVIDENCE:
                    b0 = b1
        out.append((b0[0], b0[1], b0[2], s, b0[3]))
    return out


def foreground_mask(
    img: np.ndarray,
    cam: Camera,
    q_left=None,
    q_right=None,
    g_left: float = 1.0,
    g_right: float = 1.0,
    region=WORKSPACE,
    z0: float = TABLE_Z,
):
    import cv2

    robot = None
    valid = workspace_mask(cam, img.shape, region, z0)
    if q_left is not None or q_right is not None:
        robot = RM.mask(cam, q_left, q_right, g_left, g_right, dilate=3)
        valid &= ~robot
    score = foreground_score(img, robot)
    fg = ((score > 1.0) & valid).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return fg, robot


def detect(
    img: np.ndarray,
    cam: Camera,
    q_left=None,
    q_right=None,
    g_left: float = 1.0,
    g_right: float = 1.0,
    sizes=BLOCK_SIZES,
    region=WORKSPACE,
    min_iou: float = 0.45,
    z0: float = TABLE_Z,
    max_regions: int = 40,
    fg_out: list | None = None,
    alt_sizes=ALT_SIZES,
    erase=(),
) -> list[Cube]:
    import cv2

    fg, robot = foreground_mask(img, cam, q_left, q_right, g_left, g_right, region, z0)
    if erase:
        fg = erase_boxes(fg, cam, erase, grow_px=2).astype(np.uint8)
    if fg_out is not None:
        fg_out.append(fg)
    _, lab, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    near_robot = None
    if robot is not None:
        near_robot = cv2.dilate(robot.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out = []
    for i in order[:max_regions]:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 15:
            continue
        reg = lab == i
        touch = bool(near_robot is not None and np.any(near_robot & reg))
        all_sizes = tuple(sizes) + tuple(alt_sizes)
        fits = fit_region(cam, reg, all_sizes, z0)
        main0 = [f for f in fits if f[3] in sizes]
        if main0 and max(f[0] for f in main0) < SPLIT_BELOW and (not touch):
            sp = split_side_by_side(cam, reg, sizes, z0)
            if sp is not None:
                col = tuple(int(v) for v in img[reg].mean(0))
                for f in sp:
                    out.append(
                        Cube(
                            float(f[1]),
                            float(f[2]),
                            float(f[3]),
                            float(f[4]),
                            z0=z0,
                            score=float(f[0]),
                            area=area,
                            color=col,
                            uv=(float(cents[i][0]), float(cents[i][1])),
                            extra={
                                "fits": {f[3]: (f[0], f[1], f[2], f[4])},
                                "touch_robot": touch,
                                "alt_better": False,
                                "alt": None,
                                "split": len(sp),
                            },
                        )
                    )
                continue
        parts = [(reg, fits)]
        for part, fits in parts:
            if not fits:
                continue
            main = [f for f in fits if f[3] in sizes]
            if not main:
                continue
            best = max(main, key=lambda f: f[0])
            alt = max((f for f in fits if f[3] not in sizes), key=lambda f: f[0], default=None)
            if best[0] < min_iou:
                continue
            col = tuple(int(v) for v in img[part].mean(0))
            rr, cc = np.nonzero(part)
            out.append(
                Cube(
                    float(best[1]),
                    float(best[2]),
                    float(best[3]),
                    float(best[4]),
                    z0=z0,
                    score=float(best[0]),
                    area=int(part.sum()),
                    color=col,
                    uv=(float(cc.mean()), float(rr.mean())),
                    extra={
                        "fits": {f[3]: (f[0], f[1], f[2], f[4]) for f in main},
                        "touch_robot": touch,
                        "alt_better": bool(alt is not None and alt[0] > best[0] + ALT_MARGIN and (not touch)),
                        "alt": None if alt is None else (alt[3], round(alt[0], 3)),
                        "split": len(parts) > 1,
                    },
                )
            )
    out.sort(key=lambda c: -c.score)
    return out


def _union_iou(cam: Camera, m_ss: np.ndarray, off, cubes, z0=TABLE_Z) -> float:
    import cv2

    buf = np.zeros(m_ss.shape, np.uint8)
    shift = np.array([off[1], off[0]], dtype=np.float64)
    for x, y, s, yaw in cubes:
        poly = silhouette(cam, x, y, s, yaw, z0)
        if len(poly) < 3:
            continue
        p = (poly - shift) * SS - 0.5
        cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
    inter = float(np.count_nonzero(buf & m_ss))
    union = float(np.count_nonzero(buf | m_ss))
    return inter / union if union > 0 else 0.0


def split_side_by_side(cam: Camera, reg: np.ndarray, sizes, z0: float = TABLE_Z):
    import cv2

    rows, cols = np.nonzero(reg)
    if len(rows) < 40:
        return None
    r0, c0 = max(0, rows.min() - 10), max(0, cols.min() - 10)
    win = reg[r0 : min(reg.shape[0], rows.max() + 10 + 1), c0 : min(reg.shape[1], cols.max() + 10 + 1)].astype(np.uint8)
    m_ss = np.kron(win, np.ones((SS, SS), np.uint8))
    pts = np.column_stack([cols, rows]).astype(np.float32)
    crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2
    best = None
    grid = np.arange(-0.006, 0.0061, 0.0015)
    for k in (2, 3):
        if len(rows) < 25 * k:
            break
        _, labels, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        parts = []
        for j in range(k):
            m = np.zeros_like(reg)
            sel = labels.ravel() == j
            m[rows[sel], cols[sel]] = True
            parts.append(m)
        for s in sizes:
            cubes = []
            for m in parts:
                f = fit_region(cam, m, (s,), z0, fit_yaw=False)
                if f:
                    cubes.append([f[0][1], f[0][2], s, 0.0])
            if len(cubes) != k:
                continue
            u = _union_iou(cam, m_ss, (r0, c0), cubes, z0)
            for _ in range(2):
                for i in range(k):
                    x0, y0 = cubes[i][0], cubes[i][1]
                    best_xy = x0, y0
                    for dx in grid:
                        for dy in grid:
                            cubes[i][0], cubes[i][1] = x0 + dx, y0 + dy
                            v = _union_iou(cam, m_ss, (r0, c0), cubes, z0)
                            if v > u:
                                u, best_xy = v, (x0 + dx, y0 + dy)
                    cubes[i][0], cubes[i][1] = best_xy
            apart = all(
                math.hypot(a[0] - b[0], a[1] - b[1]) >= 0.95 * s for i, a in enumerate(cubes) for b in cubes[i + 1 :]
            )
            if apart and u >= SPLIT_UNION_MIN and (best is None or u > best[0] + 0.02):
                best = u, [(u, c[0], c[1], s, 0.0) for c in cubes]
    return None if best is None else best[1]


def with_size(c: Cube, s: float) -> Cube | None:
    fits = c.extra.get("fits", {})
    f = fits.get(s)
    if f is None:
        if not fits:
            return None
        k = min(fits, key=lambda t: abs(t - s))
        if abs(k - s) > 0.003:
            return None
        f = fits[k]
    return Cube(
        float(f[1]),
        float(f[2]),
        float(s),
        float(f[3]),
        z0=c.z0,
        score=float(f[0]),
        area=c.area,
        color=c.color,
        uv=c.uv,
        extra=c.extra,
    )


def select_same_size(cands: list[Cube], n: int = 3, sizes=BLOCK_SIZES, min_iou: float = 0.5) -> list[Cube]:
    best, best_score = [], -math.inf
    cands = [c for c in cands if not c.extra.get("alt_better")]
    for s in sizes:
        cs = [c2 for c2 in (with_size(c, s) for c in cands) if c2 is not None and c2.score >= min_iou]
        cs.sort(key=lambda c: -c.score)
        top = cs[:n]
        sc = sum(c.score for c in top) - (n - len(top)) * 1.0
        if sc > best_score:
            best, best_score = top, sc
    return best


def _cube_edges(x, y, s, yaw, z0=TABLE_Z):
    c = corners(x, y, s, yaw, z0)
    top = [(c[4 + i], c[4 + (i + 1) % 4]) for i in range(4)]
    vert = [(c[i], c[4 + i]) for i in range(4)]
    return top, vert


def edge_score(
    grad: tuple[np.ndarray, np.ndarray], cam: Camera, x, y, s, yaw, z0=TABLE_Z, step_px: float = 1.0, trim: float = 0.15
) -> float:
    gx, gy = grad
    h, w = gx.shape
    top, _ = _cube_edges(x, y, s, yaw, z0)
    tot, n = 0.0, 0
    for a, b in top:
        uv, d = cam.project(np.array([a, b]))
        if np.any(d <= 0):
            continue
        p0, p1 = uv[0] - 0.5, uv[1] - 0.5
        L = float(np.linalg.norm(p1 - p0))
        if L < 3:
            continue
        nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / L
        ts = np.linspace(trim, 1 - trim, max(3, int(L * (1 - 2 * trim) / step_px)))
        pts = p0[None] + ts[:, None] * (p1 - p0)[None]
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        g = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1])
        tot += float(g.sum())
        n += len(g)
    return tot / n if n else 0.0


def erase_boxes(fg: np.ndarray, cam: Camera, boxes, grow_px: int = 2) -> np.ndarray:
    import cv2

    m = np.zeros(fg.shape, np.uint8)
    for x, y, s, yaw, z0, height in boxes:
        poly = silhouette(cam, x, y, s, yaw, z0, height)
        if len(poly) >= 3:
            cv2.fillConvexPoly(m, np.round(poly * 4).astype(np.int32), 1, shift=2)
    if grow_px > 0:
        m = cv2.dilate(m, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    out = fg.copy().astype(bool)
    out[m > 0] = False
    return out
