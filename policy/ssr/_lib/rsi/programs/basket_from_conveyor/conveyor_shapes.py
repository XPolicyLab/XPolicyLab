from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...planning import kinematics as K
from ...perception.cameras import Camera, H, W

DATA = ASSETS / "conveyor_objects.json"
BELT_Z = 0.762
SCALE = 0.5


@dataclass(frozen=True)
class Shape:
    asset: str
    points: np.ndarray
    extent: tuple
    mass: float
    friction: float


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(DATA.read_text())


def assets() -> list[str]:
    return list(_raw()["assets"])


@lru_cache(maxsize=None)
def shape(asset: str, place: str = "up") -> Shape:
    d = _raw()["assets"][asset]
    p = np.asarray(d["points"], dtype=float) * float(_raw()["unit"])
    R = K.quat_to_mat(np.asarray(d["places"][place], dtype=float))
    q = p @ R.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    q = q - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    ph = d.get("physics", {})
    return Shape(
        asset, q, tuple(float(v) for v in hi - lo), float(min(ph.get("mass", 0.5), 0.5)), float(ph.get("friction", 0.5))
    )


def posed(sh: Shape, x: float, y: float, yaw: float, z0: float = BELT_Z) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    p = sh.points
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1], z0 + p[:, 2]], axis=1)


def render(cam: Camera, pts: np.ndarray, box=None, scale: float = SCALE, close: int = 3) -> np.ndarray:
    import cv2

    hs, ws = int(H * scale), int(W * scale)
    u0, v0, u1, v1 = box if box is not None else (0, 0, ws, hs)
    uv, d = cam.project(pts)
    uv = np.floor(uv * scale).astype(int)
    ok = (d > 0) & (uv[:, 0] >= u0) & (uv[:, 0] < u1) & (uv[:, 1] >= v0) & (uv[:, 1] < v1)
    m = np.zeros((v1 - v0, u1 - u0), np.uint8)
    m[uv[ok, 1] - v0, uv[ok, 0] - u0] = 1
    if close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return m > 0


def downsample(mask: np.ndarray, scale: float = SCALE) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) > 0


@dataclass
class Fit:
    asset: str
    x: float
    y: float
    yaw: float
    score: float


def fit(
    cam: Camera,
    strong: np.ndarray,
    weak: np.ndarray,
    asset: str,
    xy0,
    *,
    span: float = 0.04,
    yaws=None,
    scale: float = SCALE,
    step: float = 0.01,
) -> Fit:
    S = downsample(strong, scale)
    Wk = downsample(weak, scale) | S
    sh = shape(asset)
    hs, ws = S.shape
    c, _ = cam.project(np.array([[xy0[0], xy0[1], BELT_Z + sh.extent[2] / 2]]))
    cu, cv = c[0] * scale
    r = int(0.2 * 288.13 * scale / 0.6) + 10
    box = max(0, int(cu) - r), max(0, int(cv) - r), min(ws, int(cu) + r), min(hs, int(cv) + r)
    u0, v0, u1, v1 = box
    s_w, w_w = S[v0:v1, u0:u1], Wk[v0:v1, u0:u1]
    if yaws is None:
        yaws = np.radians(np.arange(0, 180, 15))

    def score(x, y, yw):
        m = render(cam, posed(sh, x, y, yw), box, scale)
        inter = np.count_nonzero(m & w_w)
        union = np.count_nonzero(m | s_w)
        return inter / max(union, 1)

    best = Fit(asset, float(xy0[0]), float(xy0[1]), 0.0, -1.0)
    for yw in yaws:
        for dx in np.arange(-span, span + 1e-09, step):
            for dy in np.arange(-span, span + 1e-09, step):
                s = score(xy0[0] + dx, xy0[1] + dy, yw)
                if s > best.score:
                    best = Fit(asset, float(xy0[0] + dx), float(xy0[1] + dy), float(yw), s)
    b0 = best
    fine = max(0.008, 0.6 * step)
    for dyaw in np.radians([-10, -5, 0, 5, 10]):
        for dx in np.arange(-fine, fine + 0.0001, 0.004):
            for dy in np.arange(-fine, fine + 0.0001, 0.004):
                s = score(b0.x + dx, b0.y + dy, b0.yaw + dyaw)
                if s > best.score:
                    best = Fit(asset, b0.x + dx, b0.y + dy, b0.yaw + dyaw, s)
    return best


def blob_xy(cam: Camera, blob_mask: np.ndarray, height: float = 0.05) -> tuple[float, float]:
    vs, us = np.nonzero(blob_mask)
    p = cam.on_plane_z(np.array([[us.mean(), vs.mean()]]), BELT_Z + height / 2)[0]
    return float(p[0]), float(p[1])
