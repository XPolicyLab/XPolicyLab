from __future__ import annotations

from ... import ASSETS

import json
import math
from dataclasses import dataclass

import numpy as np

DATA = ASSETS / "basket_model.json"
BELT_Z = 0.7625
HANDLE_TOP = 0.109
HANDLE_THICK = 0.0126
FLOOR_IN = -0.101
WALL_TOP_SIDE = 0.053
HALF_LONG, HALF_SHORT = 0.18, 0.12
B_MIN = 90
SAT_MIN, SAT_MAX = 0.05, 0.36
CLOSE_PX = 7


@dataclass
class BasketFit:
    x: float
    y: float
    z: float
    yaw: float
    iou: float
    n_px: int


_MODEL = None


def model_points() -> np.ndarray:
    global _MODEL
    if _MODEL is None:
        _MODEL = np.asarray(json.loads(DATA.read_text())["points"], dtype=float)
    return _MODEL


def colour_mask(img: np.ndarray) -> np.ndarray:
    im = np.asarray(img, dtype=np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    s = (b - r) / np.maximum(b, 1.0)
    return (b > B_MIN) & (s > SAT_MIN) & (s < SAT_MAX) & (g > r)


def largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    k = 1 + int(np.argmax(st[1:, 4]))
    return lab == k


def world_points(x: float, y: float, yaw: float, z: float | None = None) -> np.ndarray:
    z = BELT_Z + 0.1095 if z is None else z
    c, s = math.cos(yaw), math.sin(yaw)
    p = model_points()
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1], z + p[:, 2]], axis=1)


def _silhouette(cam, x, y, yaw, box, dil: int = 2) -> np.ndarray:
    import cv2

    u0, v0, u1, v1 = box
    uv, d = cam.project(world_points(x, y, yaw))
    uv = np.round(uv).astype(int)
    ok = (d > 0) & (uv[:, 0] >= u0) & (uv[:, 0] < u1) & (uv[:, 1] >= v0) & (uv[:, 1] < v1)
    sil = np.zeros((v1 - v0, u1 - u0), np.uint8)
    sil[uv[ok, 1] - v0, uv[ok, 0] - u0] = 1
    k = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)
    sil = cv2.dilate(sil, k)
    return cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((CLOSE_PX, CLOSE_PX), np.uint8)) > 0


def fit(
    img: np.ndarray, cam, yaw: float = math.pi / 2, bad: np.ndarray | None = None, yaws: tuple = (0.0,)
) -> BasketFit | None:
    import cv2

    m = largest_component(colour_mask(img))
    n_px = int(m.sum())
    if n_px < 500:
        return None
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((CLOSE_PX, CLOSE_PX), np.uint8)) > 0
    vs, us = np.nonzero(m)
    H, W = m.shape
    box = max(0, us.min() - 40), max(0, vs.min() - 40), min(W, us.max() + 41), min(H, vs.max() + 41)
    u0, v0, u1, v1 = box
    obs = m[v0:v1, u0:u1]
    valid = np.ones_like(obs) if bad is None else ~np.asarray(bad)[v0:v1, u0:u1]
    c = cam.on_plane_z(np.array([[us.mean(), vs.mean()]]), BELT_Z + 0.11)[0]
    if not np.all(np.isfinite(c)):
        return None

    def score(x, y, yw):
        s = _silhouette(cam, x, y, yw, box)
        inter = np.count_nonzero(s & obs & valid)
        union = np.count_nonzero((s | obs) & valid)
        return inter / max(union, 1)

    best = -1.0, c[0], c[1], yaw
    for dyaw in yaws:
        for step, span in ((0.01, 0.06), (0.002, 0.01)):
            x0, y0 = (c[0], c[1]) if step == 0.01 else (best[1], best[2])
            if step != 0.01 and abs(best[3] - (yaw + dyaw)) > 1e-09:
                continue
            for dx in np.arange(-span, span + 1e-09, step):
                for dy in np.arange(-span, span + 1e-09, step):
                    s = score(x0 + dx, y0 + dy, yaw + dyaw)
                    if s > best[0]:
                        best = s, x0 + dx, y0 + dy, yaw + dyaw
    return BasketFit(
        x=float(best[1]), y=float(best[2]), z=BELT_Z + 0.1095, yaw=float(best[3]), iou=float(best[0]), n_px=n_px
    )
