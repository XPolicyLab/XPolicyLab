from __future__ import annotations

import math

import numpy as np

from .cameras import Camera

SS = 4
N_CIRCLE = 48


def silhouette(cam: Camera, x: float, y: float, r: float, z_lo: float, z_hi: float) -> np.ndarray:
    import cv2

    a = np.linspace(0.0, 2 * math.pi, N_CIRCLE, endpoint=False)
    ring = np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
    pts = np.vstack([np.column_stack([ring, np.full(len(a), z_lo)]), np.column_stack([ring, np.full(len(a), z_hi)])])
    uv, d = cam.project(pts)
    if np.any(d <= 0.01):
        return np.zeros((0, 2))
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def fit_cylinder(
    cam: Camera,
    region: np.ndarray,
    r: float,
    z_lo: float,
    z_hi: float,
    *,
    init=None,
    bad: np.ndarray | None = None,
    span: float = 0.015,
    pad: int = 8,
):
    import cv2

    rows, cols = np.nonzero(region)
    if len(rows) < 10:
        return None
    if init is None:
        p = cam.on_plane_z(np.array([[float(cols.mean()) + 0.5, float(rows.mean()) + 0.5]]), z_hi)[0]
        if not np.all(np.isfinite(p)):
            return None
        init = float(p[0]), float(p[1])
    polys = [silhouette(cam, init[0] + dx, init[1] + dy, r, z_lo, z_hi) for dx in (-span, span) for dy in (-span, span)]
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
        poly = silhouette(cam, x, y, r, z_lo, z_hi)
        if len(poly) < 3:
            return -1.0
        buf[:] = 0
        p = (poly - shift) * SS - 0.5
        cv2.fillConvexPoly(buf, np.round(p * 4).astype(np.int32), 1, shift=2)
        if keep is not None:
            buf[:] &= keep
        inter = float(np.count_nonzero(buf & m_ss))
        union = am + float(np.count_nonzero(buf)) - inter
        return inter / union if union > 0 else 0.0

    best = -1.0, float(init[0]), float(init[1])
    g = np.arange(-span, span + 1e-09, 0.002)
    for dx in g:
        for dy in g:
            v = iou(init[0] + dx, init[1] + dy)
            if v > best[0]:
                best = v, init[0] + dx, init[1] + dy
    for sp, st in ((0.002, 0.0005), (0.0005, 0.0002)):
        _, bx, by = best
        offs = np.arange(-sp, sp + 1e-09, st)
        for dx in offs:
            for dy in offs:
                v = iou(bx + dx, by + dy)
                if v > best[0]:
                    best = v, bx + dx, by + dy
    return best


def components(mask: np.ndarray, min_area: int = 30):
    import cv2

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((int(stats[i, cv2.CC_STAT_AREA]), lab == i, (float(cents[i][0]), float(cents[i][1]))))
    out.sort(key=lambda t: -t[0])
    return out
