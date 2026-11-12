from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...perception.cameras import Camera

BELT_Z = 0.762
Y_FAR, Y_NEAR = 0.3, -0.26
DIFF = 30
MIN_AREA = 60
HUE_BINS, SAT_SPLIT = 12, 0.5
SAT_MIN, VAL_MIN = 0.22, 0.12


@dataclass
class Blob:
    label: int
    area: int
    bbox: tuple
    centroid: tuple
    bottom: tuple
    mask: np.ndarray


def belt_rows(cam: Camera) -> tuple[int, int]:
    uv, _ = cam.project(np.array([[0.0, Y_FAR, BELT_Z], [0.0, Y_NEAR, BELT_Z]]))
    return int(round(uv[0, 1])), int(round(uv[1, 1]))


SHADOW_MIN = 0.3
SHADOW_MAX = 0.93
SHADOW_CHROMA = 0.07
HISTORY = 240
EVERY = 6


class BeltBackground:
    def __init__(self, every: int = EVERY, history: int = HISTORY):
        self.every, self.history = int(every), int(history)
        self.frames: list[tuple[int, np.ndarray]] = []
        self._bg = None
        self._bg_step = -(10**9)

    def add(self, step: int, img: np.ndarray) -> None:
        if self.frames and step - self.frames[-1][0] < self.every:
            return
        self.frames.append((int(step), np.asarray(img, dtype=np.uint8).copy()))
        self.frames = [(t, f) for t, f in self.frames if t >= step - self.history]

    def ready(self, min_frames: int = 6) -> bool:
        return len(self.frames) >= min_frames

    def image(self, step: int, max_age: int = 18) -> np.ndarray:
        if self._bg is None or step - self._bg_step > max_age:
            self._bg = np.median(np.stack([f for _, f in self.frames]), axis=0).astype(np.float32)
            self._bg_step = step
        return self._bg


def foreground(
    img: np.ndarray, bg: np.ndarray, bad: np.ndarray | None = None, cam: Camera | None = None
) -> tuple[np.ndarray, np.ndarray]:
    im = np.asarray(img, dtype=np.float32)
    B = np.asarray(bg, dtype=np.float32)
    d = np.abs(im - B).max(axis=2)
    weak = d > DIFF
    r = (im + 4.0) / (B + 4.0)
    rm = r.mean(axis=2)
    chroma = np.abs(r - rm[..., None]).max(axis=2)
    shadow = (rm > SHADOW_MIN) & (rm < SHADOW_MAX) & (chroma < SHADOW_CHROMA)
    strong = weak & ~shadow
    if cam is not None:
        v0, v1 = belt_rows(cam)
        band = np.zeros(d.shape, bool)
        band[max(0, v0) : max(0, v1)] = True
        strong &= band
        weak &= band
    if bad is not None:
        strong &= ~bad
        weak &= ~bad
    return strong, weak


def blobs(fg: np.ndarray, min_area: int = MIN_AREA) -> list[Blob]:
    import cv2

    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = (int(t) for t in st[k])
        if a < min_area:
            continue
        mk = lab == k
        vs, us = np.nonzero(mk)
        vb = vs.max()
        ub = float(us[vs >= vb - 2].mean())
        out.append(Blob(k, a, (x, y, x + w - 1, y + h - 1), (float(cen[k][0]), float(cen[k][1])), (ub, float(vb)), mk))
    return out


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    mx, mn = a.max(axis=1), a.min(axis=1)
    d = mx - mn
    s = np.where(mx > 1e-06, d / np.maximum(mx, 1e-06), 0.0)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros(len(a))
    nz = d > 1e-06
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (g - b)[rm] / d[rm] % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    return h / 6.0, s, mx


def signature(pixels: np.ndarray) -> np.ndarray:
    h, s, v = _hsv(pixels)
    col = (s > SAT_MIN) & (v > VAL_MIN)
    hb = np.minimum((h[col] * HUE_BINS).astype(int), HUE_BINS - 1)
    sb = (s[col] > SAT_SPLIT).astype(int)
    hist = np.zeros(HUE_BINS * 2 + 3)
    np.add.at(hist, hb * 2 + sb, 1.0)
    grey = ~col
    vb = np.digitize(v[grey], [0.3, 0.6])
    np.add.at(hist, HUE_BINS * 2 + vb, 1.0)
    return hist / max(hist.sum(), 1.0)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())
