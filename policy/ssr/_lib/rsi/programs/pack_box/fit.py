from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera

DATA = ASSETS / "pack_assets.npz"
COM_DATA = ASSETS / "pack_com.json"
COARSE_YAW = math.radians(7.5)
FINE_YAWS = tuple(math.radians(d) for d in (-4, -2, 0, 2, 4))
COARSE_PTS = 1500
SHIFT_PX = 20
TOP_K = 4
W_CHROMA = 3.0
W_APP = 0.3


def quat_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Asset:
    key: str
    category: str
    points: np.ndarray
    colors: np.ndarray
    rest: np.ndarray
    h0: float
    target: np.ndarray
    extents: np.ndarray
    normals: np.ndarray = None
    com: np.ndarray = (
        None
    )


@lru_cache(maxsize=1)
def assets() -> dict[str, Asset]:
    z = np.load(DATA)
    index = json.loads(bytes(z["index"]).decode())
    coms = json.loads(COM_DATA.read_text())
    out = {}
    for a in index:
        P = z[a["key"] + "/points"].astype(float)
        places = a["places"]
        if places:
            R = quat_mat(next(iter(places.values())))
        else:
            ax = int(np.argmin(P.max(0) - P.min(0)))
            R = {0: _rot_y(math.pi / 2), 1: _rot_x(math.pi / 2), 2: np.eye(3)}[ax]
        Pr = P @ R.T
        cp = a["functional"].get("checkpoint")
        if a["category"] == "shoe" or not cp:
            t = np.array([-1.0, 0.0, 0.0])
        else:
            t = quat_mat(cp[0][3:7]) @ np.array([1.0, 0.0, 0.0])
        N = z[a["key"] + "/normals"].astype(float) / 127.0
        N /= np.maximum(1e-09, np.linalg.norm(N, axis=1, keepdims=True))
        com = np.asarray(coms[a["key"]]["com"], float) if a["category"] != "box" else np.zeros(3)
        out[a["key"]] = Asset(
            a["key"],
            a["category"],
            P,
            z[a["key"] + "/colors"],
            R,
            float(-Pr[:, 2].min()),
            t,
            np.asarray(a["extents"], float),
            N,
            com,
        )
    return out


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def of_category(category: str) -> list[Asset]:
    return [a for a in assets().values() if a.category == category]


@dataclass
class ObjFit:
    category: str
    key: str
    x: float
    y: float
    z: float
    yaw: float
    iou: float
    alt_iou: float = 0.0
    R: np.ndarray = field(default=None, repr=False)
    target_dir: np.ndarray = field(default=None, repr=False)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def points(self) -> np.ndarray:
        a = assets()[self.key]
        return a.points @ self.R.T + np.array([self.x, self.y, self.z])

    def normals(self) -> np.ndarray:
        return assets()[self.key].normals @ self.R.T

    def com(self) -> np.ndarray:
        return self.R @ assets()[self.key].com + np.array([self.x, self.y, self.z])


def world_points(a: Asset, x: float, y: float, yaw: float, z0: float = K.TABLE_Z, pts=None) -> np.ndarray:
    P = a.points if pts is None else pts
    R = _rz(yaw) @ a.rest
    return P @ R.T + np.array([x, y, z0 + a.h0])


def _silhouette(cam: Camera, W: np.ndarray):
    import cv2
    from ...perception.pickup_find import silhouette

    S, off = silhouette(cam, W)
    if S is None:
        return S, off
    pad = np.zeros((S.shape[0] + 2, S.shape[1] + 2), np.uint8)
    pad[1:-1, 1:-1] = S
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 2)
    return (ff[1:-1, 1:-1] != 2).astype(np.uint8), off


def _best_shift(S: np.ndarray, off, M: np.ndarray, m_area: int, span: int):
    import cv2

    h, w = S.shape
    u0, v0 = off
    H, W = M.shape
    pu0, pv0 = u0 - span, v0 - span
    pad = np.zeros((h + 2 * span, w + 2 * span), np.float32)
    a0, a1 = max(0, pv0), min(H, pv0 + h + 2 * span)
    b0, b1 = max(0, pu0), min(W, pu0 + w + 2 * span)
    if a1 <= a0 or b1 <= b0:
        return 0.0, 0, 0
    pad[a0 - pv0 : a1 - pv0, b0 - pu0 : b1 - pu0] = M[a0:a1, b0:b1]
    inter = cv2.matchTemplate(pad, S.astype(np.float32), cv2.TM_CCORR)
    s_area = float(S.sum())
    iou = inter / np.maximum(1.0, s_area + m_area - inter)
    j = np.unravel_index(int(np.argmax(iou)), iou.shape)
    return float(iou[j]), int(j[1]) - span, int(j[0]) - span


def _shift_xy(cam: Camera, W: np.ndarray, du: float, dv: float, z: float) -> np.ndarray:
    uv, _ = cam.project(W.mean(0, keepdims=True))
    p0 = cam.on_plane_z(uv, z)[0]
    p1 = cam.on_plane_z(uv + np.array([[du, dv]]), z)[0]
    d = p1 - p0
    return d[:2] if np.all(np.isfinite(d)) else np.zeros(2)


def _init_xy(cam: Camera, mask: np.ndarray, z: float):
    vs, us = np.nonzero(mask)
    p = cam.on_plane_z(np.array([[float(us.mean()), float(vs.mean())]]), z)[0]
    return p[:2] if np.all(np.isfinite(p)) else None


def fit(
    cam: Camera, mask: np.ndarray, category: str, z0: float = K.TABLE_Z, *, keys=None, img=None, w_app: float = None
) -> ObjFit | None:
    w_app = W_APP if w_app is None else w_app
    M = mask.astype(np.float32)
    m_area = int(mask.sum())
    if m_area < 20:
        return None
    cands = []
    rng = np.random.default_rng(0)
    for a in of_category(category):
        if keys is not None and a.key not in keys:
            continue
        sub = a.points[rng.choice(len(a.points), min(COARSE_PTS, len(a.points)), replace=False)]
        zc = z0 + a.h0 + float((a.points @ a.rest.T)[:, 2].mean())
        xy0 = _init_xy(cam, mask, zc)
        if xy0 is None:
            continue
        for yaw in np.arange(0.0, 2 * math.pi, COARSE_YAW):
            W = world_points(a, xy0[0], xy0[1], yaw, z0, sub)
            S, off = _silhouette(cam, W)
            if S is None:
                continue
            iou, du, dv = _best_shift(S, off, M, m_area, SHIFT_PX)
            d = _shift_xy(cam, W, du, dv, zc)
            cands.append((iou, a.key, float(xy0[0] + d[0]), float(xy0[1] + d[1]), float(yaw)))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    best = None
    refined = []
    for iou0, key, x, y, yaw in cands[:TOP_K] + _opposite(cands, TOP_K):
        r = _refine(cam, M, m_area, assets()[key], x, y, yaw, z0)
        refined.append(r)
    if img is not None and w_app > 0:
        lum = _luma(img)
        app = {
            i: appearance(cam, lum, mask, assets()[r[1]], r[2], r[3], r[4], z0, rgb=img) for i, r in enumerate(refined)
        }
        order = sorted(range(len(refined)), key=lambda i: -(refined[i][0] + w_app * app[i]))
        scores = [refined[i][0] + w_app * app[i] for i in order]
        refined = [refined[i] for i in order]
    else:
        refined.sort(key=lambda r: -r[0])
        scores = [r[0] for r in refined]
    iou, key, x, y, yaw = refined[0]
    alt = max([sc for r, sc in zip(refined, scores) if abs(_wrap(r[4] - yaw)) > math.pi / 2] + [0.0])
    alt = alt - (scores[0] - iou)
    a = assets()[key]
    R = _rz(yaw) @ a.rest
    t = R @ a.target
    best = ObjFit(category, key, x, y, z0 + a.h0, yaw, iou, alt, R, t[:2] / max(1e-09, np.linalg.norm(t[:2])))
    return best


def _luma(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def appearance(
    cam: Camera,
    lum: np.ndarray,
    mask: np.ndarray,
    a: Asset,
    x: float,
    y: float,
    yaw: float,
    z0: float = K.TABLE_Z,
    rgb: np.ndarray | None = None,
) -> float:
    R = _rz(yaw) @ a.rest
    W = a.points @ R.T + np.array([x, y, z0 + a.h0])
    Nw = a.normals @ R.T
    uv, d = cam.project(W)
    facing = np.einsum("ij,ij->i", Nw, cam.pos[None, :] - W) > 0
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    h, w = lum.shape
    ok = facing & (d > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.nonzero(ok)[0]
    if len(idx) < 30:
        return 0.0
    idx = idx[np.argsort(d[idx])]
    pix = v[idx] * w + u[idx]
    _, first = np.unique(pix, return_index=True)
    idx = idx[first]
    idx = idx[mask[v[idx], u[idx]]]
    if len(idx) < 30:
        return 0.0
    col = a.colors[idx].astype(np.float32)
    lr = 0.299 * col[:, 0] + 0.587 * col[:, 1] + 0.114 * col[:, 2]
    lo = lum[v[idx], u[idx]]
    corr = 0.0 if lr.std() < 0.001 or lo.std() < 0.001 else float(np.corrcoef(lr, lo)[0, 1])
    if rgb is None:
        return corr
    ob = rgb[v[idx], u[idx]].astype(np.float32)
    cr = col / np.maximum(1.0, col.sum(1, keepdims=True))
    co = ob / np.maximum(1.0, ob.sum(1, keepdims=True))
    return corr - W_CHROMA * float(np.abs(cr - co).sum(1).mean())


def _opposite(cands, k):
    yaw0 = cands[0][4]
    opp = [c for c in cands if abs(_wrap(c[4] - yaw0)) > math.pi / 2]
    return opp[: max(1, k // 2)]


def _refine(cam, M, m_area, a: Asset, x, y, yaw, z0):
    zc = z0 + a.h0 + float((a.points @ a.rest.T)[:, 2].mean())
    best = -1.0, a.key, x, y, yaw
    for _ in range(2):
        _, _, x, y, yaw = best
        for dy in FINE_YAWS:
            W = world_points(a, x, y, yaw + dy, z0)
            S, off = _silhouette(cam, W)
            if S is None:
                continue
            iou, du, dv = _best_shift(S, off, M, m_area, 4)
            d = _shift_xy(cam, W, du, dv, zc)
            if iou > best[0]:
                best = iou, a.key, float(x + d[0]), float(y + d[1]), float(_wrap(yaw + dy))
    return best


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def heading(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


FRONT_STEPS = (0.005, math.radians(6.0)), (0.0025, math.radians(3.0))
FRONT_TOP = 3
FRONT_YIELD_S = 1.0


def front_check_iter(
    cam: Camera, rgb: np.ndarray, free: np.ndarray, fit: ObjFit, z0: float = K.TABLE_Z, with_fit: bool = False
):
    import time

    t_y = time.monotonic()
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    keys = [k for k, a in assets().items() if a.category == fit.category]

    def search(b, sxy, sa):
        nonlocal t_y
        _, k, x0, y0, a0 = b
        a = assets()[k]
        for dx in (-sxy, 0.0, sxy):
            for dy in (-sxy, 0.0, sxy):
                for da in (-sa, 0.0, sa):
                    v = appearance(cam, lum, free, a, x0 + dx, y0 + dy, a0 + da, z0=z0)
                    if v > b[0]:
                        b = v, k, x0 + dx, y0 + dy, a0 + da
        return b

    best = {}
    own = {}
    for flip in (0, 1):
        level = []
        for k in keys:
            level.append(search((-9.0, k, fit.x, fit.y, fit.yaw + math.pi * flip), *FRONT_STEPS[0]))
            if time.monotonic() - t_y > FRONT_YIELD_S:
                yield None
                t_y = time.monotonic()
        if with_fit:
            mine = [b for b in level if b[1] == fit.key]
            own[flip] = float(search(mine[0], *FRONT_STEPS[1])[0]) if mine else 0.0
        level.sort(key=lambda b: -b[0])
        top = level[0]
        for b in level[:FRONT_TOP]:
            b = search(b, *FRONT_STEPS[1])
            if b[0] > top[0]:
                top = b
            if time.monotonic() - t_y > FRONT_YIELD_S:
                yield None
                t_y = time.monotonic()
        best[flip] = top
    k1 = best[1][1]
    a1 = assets()[k1]
    yaw1 = _wrap(fit.yaw + math.pi)
    R = _rz(yaw1) @ a1.rest
    t = R @ a1.target
    f1 = ObjFit(
        fit.category,
        k1,
        fit.x,
        fit.y,
        z0 + a1.h0,
        yaw1,
        fit.iou,
        fit.alt_iou,
        R,
        t[:2] / max(1e-09, float(np.linalg.norm(t[:2]))),
    )
    if with_fit:
        return float(best[0][0]), float(best[1][0]), f1, own[0], own[1]
    return float(best[0][0]), float(best[1][0]), f1
