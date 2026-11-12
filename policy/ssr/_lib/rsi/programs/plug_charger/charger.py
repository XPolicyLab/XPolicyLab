from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...planning import kinematics as K
from ...perception import cameras as CAM

CHARGER_Z = K.TABLE_Z + 0.0135
CHARGER_TOP = K.TABLE_Z + 0.027
SOCKET_LO = np.array([-0.0786, -0.0182, 0.0008])
SOCKET_HI = np.array([0.0786, 0.0182, 0.0202])
SOCKET_Z = K.TABLE_Z - 0.0008
SOCKET_TOP = K.TABLE_Z + 0.0194
OUTLETS = (0.04357, 0.0064), (0.00703, 0.0064), (-0.02952, 0.0064)
STRIP_TOP_LOCAL = 0.0202
SLOT_Z = 0.01864
SLOTS = np.array(
    [
        (0.037635, 0.0064),
        (0.049505, 0.0064),
        (0.043575, 0.005525),
        (0.03701, -0.00732),
        (0.05012, -0.00734),
        (0.001095, 0.0064),
        (0.012965, 0.0064),
        (0.007025, 0.005525),
        (0.00049, -0.00735),
        (0.01359, -0.00732),
        (-0.035455, 0.0064),
        (-0.023585, 0.0064),
        (-0.029515, 0.005525),
        (-0.03606, -0.00733),
        (-0.02296, -0.00732),
    ]
)
S_MAX = 60
V_MIN = 110
REGION = (-0.6, 0.6), (-0.4, 0.15)


@dataclass
class BoxFit:
    kind: str
    x: float
    y: float
    yaw: float
    iou: float
    extra: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


def white_mask(img: np.ndarray, s_max: int = S_MAX, v_min: int = V_MIN) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)


def box_pose(x: float, y: float, yaw: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = K.rot_z(yaw)
    t[:3, 3] = [x, y, z]
    return t


def box_corners(T: np.ndarray, lo, hi) -> np.ndarray:
    c = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])])
    return c @ T[:3, :3].T + T[:3, 3]


def silhouette(cam, T: np.ndarray, lo, hi, shape) -> np.ndarray:
    uv, d = cam.project(box_corners(T, lo, hi))
    m = np.zeros(shape[:2], np.uint8)
    if np.any(d <= 0.01):
        return m.astype(bool)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2)
    return m.astype(bool)


def iou(sil: np.ndarray, fg: np.ndarray, bad: np.ndarray | None) -> float:
    if bad is not None:
        ok = ~bad
        sil, fg = sil & ok, fg & ok
    u = np.count_nonzero(sil | fg)
    return 0.0 if u == 0 else np.count_nonzero(sil & fg) / u


def fit_box(
    cam,
    fg: np.ndarray,
    bad,
    x0: float,
    y0: float,
    yaw0: float,
    lo,
    hi,
    z: float,
    *,
    span=(0.012, 0.012, math.radians(20)),
    steps=(0.004, 0.004, math.radians(5)),
    levels: int = 3,
):
    best = iou(silhouette(cam, box_pose(x0, y0, yaw0, z), lo, hi, fg.shape), fg, bad), x0, y0, yaw0
    sx, sy, sa = span
    dx, dy, da = steps
    for _ in range(levels):
        _, bx, by, ba = best
        for a in np.arange(ba - sa, ba + sa + 1e-09, da):
            for xx in np.arange(bx - sx, bx + sx + 1e-09, dx):
                for yy in np.arange(by - sy, by + sy + 1e-09, dy):
                    s = iou(silhouette(cam, box_pose(xx, yy, a, z), lo, hi, fg.shape), fg, bad)
                    if s > best[0]:
                        best = s, float(xx), float(yy), float(a)
        sx, sy, sa = 1.5 * dx, 1.5 * dy, 1.5 * da
        dx, dy, da = dx / 3, dy / 3, da / 3
    return best


def regions(cam, img: np.ndarray, robot: np.ndarray | None, region=REGION):
    fg = white_mask(img)
    if robot is not None:
        fg &= ~robot
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 25:
            continue
        m = lab == k
        vs, us = np.nonzero(m)
        uv = np.stack([us, vs], axis=1).astype(float) + 0.5
        rects = {}
        for name, z in (("charger", CHARGER_TOP), ("socket", SOCKET_TOP)):
            p = cam.on_plane_z(uv, z)[:, :2]
            p = p[np.all(np.isfinite(p), axis=1)]
            if len(p) < 10:
                continue
            (cx, cy), (w, h), ang = cv2.minAreaRect(p.astype(np.float32))
            lng, sht = max(w, h), min(w, h)
            a = math.radians(ang) + (0.0 if w >= h else math.pi / 2)
            rects[name] = float(cx), float(cy), float(lng), float(sht), a
        if "socket" not in rects:
            continue
        cx, cy = rects["socket"][:2]
        if not (region[0][0] <= cx <= region[0][1] and region[1][0] <= cy <= region[1][1]):
            continue
        out.append((m, rects))
    return out, fg


def detect_head(img: np.ndarray, cam, robot: np.ndarray | None = None, charger_max_yaw: float | None = None):
    regs, _ = regions(cam, img, robot)
    bad = robot
    sock_c, chg_c = [], []
    for m, rects in regs:
        lng, sht = rects["socket"][2:4]
        if 0.1 <= lng <= 0.22 and 0.025 <= sht <= 0.075:
            sock_c.append((abs(lng - 0.16) + abs(sht - 0.045), m, rects["socket"]))
        lng, sht = rects["charger"][2:4]
        if 0.03 <= lng <= 0.1 and 0.025 <= sht <= 0.08 and (np.count_nonzero(m) >= 60):
            chg_c.append((abs(lng - 0.065) + abs(sht - 0.05), m, rects["charger"]))
    socket = charger = None
    if sock_c:
        _, m, (cx, cy, lng, sht, a) = min(sock_c, key=lambda t: t[0])
        a = math.remainder(a, math.pi)
        s, x, y, yaw = fit_box(cam, m, bad, cx, cy + 0.004, a, SOCKET_LO, SOCKET_HI, SOCKET_Z)
        socket = BoxFit("socket", x, y, math.remainder(yaw, math.pi), s, {"rect": (lng, sht)})
    if chg_c:
        cands = [c for c in chg_c if socket is None or np.count_nonzero(c[1] & sil_of(cam, socket, img.shape)) == 0]
        if cands:
            _, m, (cx, cy, lng, sht, a) = min(cands, key=lambda t: t[0])
            lo = np.array([-0.021, -0.024, -0.0135])
            hi = np.array([0.021, 0.024, 0.0135])
            a0 = math.remainder(a - math.pi / 2, math.pi)
            s, x, y, yaw = fit_box(cam, m, bad, cx, cy + 0.006, a0, lo, hi, CHARGER_Z)
            if charger_max_yaw is not None and abs(math.remainder(yaw, math.pi)) > charger_max_yaw:
                s2, x2, y2, yaw2 = fit_box(cam, m, bad, cx, cy + 0.006, a0 + math.pi / 2, lo, hi, CHARGER_Z)
                if abs(math.remainder(yaw2, math.pi)) <= charger_max_yaw or s2 > s:
                    s, x, y, yaw = s2, x2, y2, yaw2
            charger = BoxFit(
                "charger", x, y, math.remainder(yaw, math.pi), s, {"rect": (lng, sht), "body_centre": True}
            )
    return socket, charger


def sil_of(cam, fit: BoxFit, shape) -> np.ndarray:
    if fit.kind == "socket":
        return silhouette(cam, box_pose(fit.x, fit.y, fit.yaw, SOCKET_Z), SOCKET_LO, SOCKET_HI, shape)
    lo = np.array([-0.021, -0.024, -0.0135])
    hi = np.array([0.021, 0.024, 0.0135])
    return silhouette(cam, box_pose(fit.x, fit.y, fit.yaw, CHARGER_Z), lo, hi, shape)


@dataclass
class StripFit:
    x: float
    y: float
    yaw: float
    rms_px: float
    n_matched: int
    n_blobs: int
    extra: dict = field(default_factory=dict)

    def pose(self) -> np.ndarray:
        return box_pose(self.x, self.y, self.yaw, SOCKET_Z)

    def outlet(self, k: int) -> np.ndarray:
        T = self.pose()
        ox, oy = OUTLETS[k]
        return (T[:3, :3] @ [ox, oy, 0.0] + T[:3, 3])[:2]


def _slot_pixels(cam, x, y, yaw):
    T = box_pose(x, y, yaw, SOCKET_Z)
    p = np.c_[SLOTS, np.full(len(SLOTS), SLOT_Z)] @ T[:3, :3].T + T[:3, 3]
    uv, d = cam.project(p)
    return uv, d


def slot_blobs(img: np.ndarray, cam, x0: float, y0: float, yaw0: float, margin: float = 0.008):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lo = SOCKET_LO - [margin, margin, 0.0]
    hi = SOCKET_HI + [margin, margin, 0.0]
    T = box_pose(x0, y0, yaw0, SOCKET_Z)
    roi = silhouette(cam, T, lo, hi, img.shape)
    if np.count_nonzero(roi) < 100:
        return [], None
    white = float(np.percentile(gray[roi], 90))
    bright = (gray > 0.75 * white) & roi
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n < 2:
        return [], None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    strip = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(strip)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    holes = (filled > 0) & (strip == 0) & (gray < 0.6 * white)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    blobs = []
    for j in range(1, n2):
        a = st2[j, cv2.CC_STAT_AREA]
        if a < 6:
            continue
        m = lab2 == j
        vs, us = np.nonzero(m)
        w = (0.6 * white - gray[m]).clip(1.0, None)
        blobs.append((float(np.sum(w * (us + 0.5)) / np.sum(w)), float(np.sum(w * (vs + 0.5)) / np.sum(w)), int(a)))
    return blobs, filled > 0


def fit_strip_wrist(
    img: np.ndarray,
    cam,
    x0: float,
    y0: float,
    yaw0: float,
    *,
    search: float = 0.006,
    search_deg: float = 4.0,
    both_ways: bool = True,
    max_px: float = 6.0,
) -> StripFit | None:
    blobs, _ = slot_blobs(img, cam, x0, y0, yaw0)
    if len(blobs) < 4:
        return None
    B = np.array([(u, v) for u, v, _ in blobs])
    best = None
    yaws = [yaw0, yaw0 + math.pi] if both_ways else [yaw0]
    for yb in yaws:
        for dyaw in np.radians(np.arange(-search_deg, search_deg + 1e-09, 1.0)):
            for dx in np.arange(-search, search + 1e-09, 0.001):
                for dy in np.arange(-search, search + 1e-09, 0.001):
                    uv, d = _slot_pixels(cam, x0 + dx, y0 + dy, yb + dyaw)
                    dist = np.linalg.norm(uv[:, None, :] - B[None, :, :], axis=2)
                    dmin = dist.min(axis=1)
                    ok = (dmin < max_px) & (d > 0)
                    score = int(np.count_nonzero(ok)) - 0.02 * float(np.sum(np.minimum(dmin, max_px)))
                    if best is None or score > best[0]:
                        best = score, x0 + dx, y0 + dy, yb + dyaw
    _, x, y, yaw = best
    p = np.array([x, y, yaw])
    for it in range(8):
        uv, d = _slot_pixels(cam, *p)
        dist = np.linalg.norm(uv[:, None, :] - B[None, :, :], axis=2)
        j = dist.argmin(axis=1)
        ok = dist[np.arange(len(uv)), j] < (max_px if it < 2 else 3.0)
        if np.count_nonzero(ok) < 4:
            return None
        tgt = B[j[ok]]
        r0 = (uv[ok] - tgt).ravel()
        J = np.zeros((len(r0), 3))
        for q, h in enumerate((1e-05, 1e-05, 0.0001)):
            pp = p.copy()
            pp[q] += h
            uv2, _ = _slot_pixels(cam, *pp)
            J[:, q] = ((uv2[ok] - tgt).ravel() - r0) / h
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        p = p + step
        if np.abs(step[:2]).max() < 1e-06 and abs(step[2]) < 1e-05:
            break
    uv, d = _slot_pixels(cam, *p)
    dist = np.linalg.norm(uv[:, None, :] - B[None, :, :], axis=2)
    dmin = dist.min(axis=1)
    ok = dmin < 3.0
    rms = float(np.sqrt(np.mean(dmin[ok] ** 2))) if np.any(ok) else 99.0
    return StripFit(
        float(p[0]),
        float(p[1]),
        float(math.remainder(p[2], 2 * math.pi)),
        rms,
        int(np.count_nonzero(ok)),
        len(blobs),
        {"matched": ok.tolist()},
    )


HELD_EDGE_X = np.linspace(-0.01, 0.01, 21)
HELD_S = np.arange(-0.025, 0.03 + 1e-09, 0.0005)


@dataclass
class HeldFit:
    slide: float
    v_edge: float
    n_cols: int
    spread_px: float


def held_slide(
    img: np.ndarray,
    R_c: np.ndarray,
    o_c: np.ndarray,
    *,
    s_max: int = 35,
    v_min: int = 190,
    min_cols: int = 12,
    max_spread: float = 2.5,
) -> HeldFit | None:
    if img is None:
        return None
    c = CAM.Camera(CAM.WRIST_MOUNT, CAM.WRIST_F)
    edge = np.stack([HELD_EDGE_X, np.full_like(HELD_EDGE_X, 0.0479), np.full_like(HELD_EDGE_X, -0.0135)], axis=1)
    pts = o_c[None, None, :] + HELD_S[:, None, None] * R_c[:, 1][None, None, :] + (edge @ R_c.T)[None, :, :]
    uv, _ = c.project(pts.reshape(-1, 3))
    uv = uv.reshape(len(HELD_S), len(HELD_EDGE_X), 2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(float)
    bright = (hsv[:, :, 1] <= s_max) & (hsv[:, :, 2] >= v_min)
    H, W = bright.shape
    k0 = int(np.argmin(np.abs(HELD_S)))
    rows, used = [], []
    for j in range(len(HELD_EDGE_X)):
        u = int(round(uv[k0, j, 0]))
        if not 0 <= u < W:
            continue
        col = bright[:, u]
        run = col[:-2] & col[1:-1] & col[2:]
        idx = np.nonzero(run)[0]
        if len(idx) == 0 or idx[0] == 0:
            continue
        v = int(idx[0])
        a, b = S[max(0, v - 3), u], S[min(H - 1, v + 2), u]
        mid = 0.5 * (a + b)
        sub = float(v)
        if a - b > 40:
            for vv in range(max(1, v - 3), min(H, v + 3)):
                s0, s1 = S[vv - 1, u], S[vv, u]
                if s0 >= mid > s1:
                    sub = vv - 1 + (s0 - mid) / max(1e-06, s0 - s1) + 0.5
                    break
        rows.append(sub)
        used.append(j)
    if len(rows) < min_cols:
        return None
    rows = np.asarray(rows)
    vpred = uv[:, used, 1]
    err = np.median(np.abs(vpred - rows[None, :]), axis=1)
    k = int(np.argmin(err))
    off = float(np.median(rows - vpred[k]))
    jc = int(np.argmin(np.abs(HELD_EDGE_X[used])))
    vc = uv[:, used[jc], 1]
    dv = np.gradient(vc, HELD_S)
    slide = float(HELD_S[k] + off / dv[k]) if abs(dv[k]) > 1e-06 else float(HELD_S[k])
    spread = float(np.median(np.abs(rows - (vpred[k] + off))))
    if spread > max_spread or not HELD_S[0] < slide < HELD_S[-1]:
        return None
    return HeldFit(slide, float(vc[k] + off), len(rows), spread)
