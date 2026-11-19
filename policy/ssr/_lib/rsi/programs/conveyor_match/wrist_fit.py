from __future__ import annotations

import math

import numpy as np

from . import objects as O
from ...perception.cameras import Camera, H, W
from . import shape_grasp as SG

SCALE = 0.5


def finger_mask(cam_w: Camera, T6: np.ndarray, grip: float) -> np.ndarray:
    import cv2

    m = np.zeros((H, W), np.uint8)
    for x0, x1, y0, y1, z0, z1 in SG.gripper_boxes(grip):
        c = np.array([[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        w = (c @ np.asarray(T6, dtype=float).T)[:, :3]
        uv, d = cam_w.project(w)
        if (d <= 0.005).any():
            continue
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    return m.astype(bool)


def _segment(img, cam_w, shape, x, y, yaw, z0, fingers, others) -> np.ndarray | None:
    import cv2

    pred = O.render(cam_w, shape.posed(x, y, yaw, z0), scale=1.0, close=9)
    if pred.sum() < 200:
        return None
    mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    mask[cv2.dilate(pred.astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)] = cv2.GC_PR_BGD
    mask[pred] = cv2.GC_PR_FGD
    core = cv2.erode(pred.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    mask[core] = cv2.GC_FGD
    mask[fingers] = cv2.GC_BGD
    for pts in others:
        mo = O.render(cam_w, pts, scale=1.0, close=9)
        mask[mo & ~core] = cv2.GC_BGD
    if not (mask == cv2.GC_FGD).any() and (not (mask == cv2.GC_PR_FGD).any()):
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    im = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(im, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def _fit(cam_w, shape, seg, fingers_s, x, y, yaw, z0, symmetric, grid: bool):
    import cv2

    h, w = int(round(H * SCALE)), int(round(W * SCALE))
    ev = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0

    def score(xx, yy, aa):
        m = O.render(cam_w, shape.posed(xx, yy, aa, z0), scale=SCALE, close=7) & ~fingers_s
        union = np.count_nonzero((m | ev) & ~fingers_s)
        return np.count_nonzero(m & ev) / union if union else 0.0

    best = score(x, y, yaw), x, y, yaw
    if grid:
        for dx in np.arange(-0.03, 0.0301, 0.01):
            for dy in np.arange(-0.03, 0.0301, 0.01):
                for da in (0.0,) if symmetric else np.radians((-20.0, -10.0, 0.0, 10.0, 20.0)):
                    sc = score(x + dx, y + dy, yaw + da)
                    if sc > best[0]:
                        best = sc, x + dx, y + dy, yaw + da
    for st in (0.01, 0.005, 0.0025):
        for _ in range(25):
            moves = [(best[1] + dx, best[2] + dy, best[3]) for dx, dy in ((st, 0), (-st, 0), (0, st), (0, -st))]
            if not symmetric:
                moves += [(best[1], best[2], best[3] + da) for da in (math.radians(st * 500), -math.radians(st * 500))]
            top = max(((score(*mv),) + mv for mv in moves), key=lambda r: r[0])
            if top[0] <= best[0] + 1e-06:
                break
            best = top
    return best


def refine(
    img,
    T6,
    grip: float,
    shape: O.Shape,
    x: float,
    y: float,
    yaw: float,
    z0: float,
    *,
    symmetric: bool = False,
    others=(),
) -> tuple[float, float, float, float] | None:
    import cv2
    from ...perception.cameras import wrist_from_link6

    cam_w = wrist_from_link6(T6)
    fingers = finger_mask(cam_w, T6, grip)
    h, w = int(round(H * SCALE)), int(round(W * SCALE))
    fingers_s = cv2.resize(fingers.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
    seg = _segment(img, cam_w, shape, x, y, yaw, z0, fingers, others)
    if seg is None:
        return None
    best = _fit(cam_w, shape, seg, fingers_s, x, y, yaw, z0, symmetric, grid=True)
    seg2 = _segment(img, cam_w, shape, best[1], best[2], best[3], z0, fingers, others)
    if seg2 is not None:
        best = _fit(cam_w, shape, seg2, fingers_s, best[1], best[2], best[3], z0, symmetric, grid=False)
    return best


DARK_LUM = 30.0
SEG_MIN_PX = 1500


def _rail_row(img) -> int:
    im = np.asarray(img).astype(np.int32)
    blue = (im[..., 2] > im[..., 0] + 50) & (im[..., 2] > im[..., 1] + 20)
    frac = blue[: H // 2].mean(axis=1)
    rows = np.nonzero(frac > 0.4)[0]
    return int(rows.max()) if len(rows) else 0


def object_segment(img, exclude: np.ndarray) -> np.ndarray:
    import cv2

    im = np.asarray(img, np.float32)
    vv, uu = np.mgrid[0:H, 0:W]
    belt = ~exclude
    belt[: _rail_row(img) + 5] = False
    sub = np.zeros_like(belt)
    sub[::4, ::4] = True
    sel = belt & sub
    if sel.sum() < 100:
        return np.zeros((H, W), bool)
    u, v = uu[sel] / W, vv[sel] / H
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    Y = im[sel]
    keep = np.ones(len(u), bool)
    coef = None
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        res = np.abs(Y - A @ coef).max(axis=1)
        keep = res < max(12.0, 3.0 * float(np.median(res[keep])))
        if keep.sum() < 50:
            break
    U, V = uu / W, vv / H
    bg = np.stack([np.ones_like(U), U, V, U * U, U * V, V * V], -1).astype(np.float32) @ coef.astype(np.float32)
    lum = im.mean(axis=2) + 1.0
    lum_bg = np.maximum(bg.mean(axis=2), 1.0) + 1.0
    chroma = np.linalg.norm(im - lum[..., None] - (bg - lum_bg[..., None]), axis=2)
    fg = (lum - lum_bg > 35.0) | (chroma > 30.0) | (lum / lum_bg < 0.55)
    fg &= belt
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.zeros((H, W), bool)
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def object_darkness(img, T6, grip: float) -> tuple[float, int] | None:
    from ...perception.cameras import wrist_from_link6

    cam_w = wrist_from_link6(T6)
    seg = object_segment(img, finger_mask(cam_w, T6, grip))
    n = int(seg.sum())
    if n < SEG_MIN_PX:
        return None
    lum = np.asarray(img, np.float32).mean(axis=2)
    return float((lum[seg] < DARK_LUM).mean()), n
