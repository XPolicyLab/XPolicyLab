from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import objects as O
from ...perception.cameras import Camera, H, W

BELT_Z = 0.762
DIFF_STRONG = 16
DIFF_WEAK = 9
SCALE = 0.5


def _f(img) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


@dataclass
class Foreground:
    strong: np.ndarray
    weak: np.ndarray


def foreground(img, bg, valid=None) -> Foreground:
    I, B = _f(img), _f(bg)
    d = np.abs(I - B).max(axis=2)
    weak = d > DIFF_WEAK
    r = (I + 4.0) / (B + 4.0)
    rm = r.mean(axis=2)
    chroma = np.abs(r - rm[..., None]).max(axis=2)
    shadow_like = (rm > 0.3) & (rm < 0.97) & (chroma < 0.06)
    strong = (d > DIFF_STRONG) & ~shadow_like
    if valid is not None:
        weak &= valid
        strong &= valid
    return Foreground(strong=strong, weak=weak | strong)


def downsample(mask: np.ndarray, scale: float = SCALE) -> np.ndarray:
    import cv2

    h, w = int(round(H * scale)), int(round(W * scale))
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0


@dataclass
class Blob:
    label: int
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    mask: np.ndarray
    support: np.ndarray
    touches_border: bool = False
    touches_robot: bool = False


SUPPORT_PX = 15


def blobs(
    fg: Foreground, robot: np.ndarray | None = None, roi: np.ndarray | None = None, min_area: int = 150
) -> list[Blob]:
    import cv2

    weak = fg.weak.copy()
    strong = fg.strong.copy()
    if roi is not None:
        weak &= roi
        strong &= roi
    k3 = np.ones((3, 3), np.uint8)
    strong_c = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, k3)
    strong_c = cv2.morphologyEx(strong_c, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(strong_c, connectivity=8)
    keep = [i for i in range(1, n) if st[i][4] >= min_area]
    near_robot = None
    if robot is not None:
        near_robot = cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    big = np.isin(lab, keep)
    dist, nearest = cv2.distanceTransformWithLabels(
        (~big).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    vs, us = np.nonzero(big)
    pix_lab = np.zeros(nearest.max() + 1, dtype=np.int32)
    pix_lab[nearest[vs, us]] = lab[vs, us]
    owner = pix_lab[nearest]
    owner[dist > SUPPORT_PX] = 0
    out = []
    for i in keep:
        x, y, w, h, a = st[i]
        m = lab == i
        sup = weak & (owner == i) | m
        b = Blob(i, (int(x), int(y), int(x + w), int(y + h)), int(a), (float(cen[i][0]), float(cen[i][1])), m, sup)
        b.touches_border = x <= 1 or x + w >= W - 1 or y <= 1 or (y + h >= H - 1)
        if near_robot is not None:
            b.touches_robot = bool((near_robot & sup).any())
        out.append(b)
    return out


def merge_blobs(bl: list) -> Blob:
    m = np.zeros_like(bl[0].mask)
    sup = np.zeros_like(bl[0].support)
    for b in bl:
        m |= b.mask
        sup |= b.support
    vs, us = np.nonzero(m)
    out = Blob(
        bl[0].label,
        (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1),
        int(m.sum()),
        (float(us.mean()), float(vs.mean())),
        m,
        sup,
    )
    out.touches_border = any(b.touches_border for b in bl)
    out.touches_robot = any(b.touches_robot for b in bl)
    return out


@dataclass
class Fit:
    shape: O.Shape
    x: float
    y: float
    yaw: float
    score: float
    evals: int = 0

    @property
    def category(self) -> str:
        return self.shape.category


def _score(model: np.ndarray, strong: np.ndarray, weak: np.ndarray) -> float:
    inter = np.count_nonzero(model & weak)
    union = np.count_nonzero(model | strong)
    return inter / union if union else 0.0


class _Window:
    def __init__(self, strong_s, weak_s, bbox_s, pad: int):
        h, w = strong_s.shape
        u0, v0, u1, v1 = bbox_s
        self.u0, self.v0 = max(0, u0 - pad), max(0, v0 - pad)
        self.u1, self.v1 = min(w, u1 + pad), min(h, v1 + pad)
        self.strong = strong_s[self.v0 : self.v1, self.u0 : self.u1]
        self.weak = weak_s[self.v0 : self.v1, self.u0 : self.u1]

    def cut(self, m):
        return m[self.v0 : self.v1, self.u0 : self.u1]


def initial_xy(cam: Camera, blob: Blob, shape: O.Shape) -> tuple[float, float]:
    u0, v0, u1, v1 = blob.bbox
    vs, us = np.nonzero(blob.support[v0:v1, u0:u1])
    vb = v0 + np.percentile(vs, 97)
    ub = u0 + float(np.median(us[vs >= np.percentile(vs, 90)]))
    p = cam.on_plane_z([[ub, vb]], BELT_Z)[0]
    cx = cam.on_plane_z([[blob.centroid[0], blob.centroid[1]]], BELT_Z + shape.extent[2] / 2)[0]
    x = float(cx[0]) if np.isfinite(cx[0]) else float(p[0])
    return x, float(p[1]) + float(min(shape.extent[0], shape.extent[1])) / 2


def fit_shape(
    cam: Camera,
    fg: Foreground,
    blob: Blob,
    shape: O.Shape,
    *,
    yaw_range=math.radians(45),
    symmetric: bool = False,
    n_points: int = 1500,
    rounds: int = 3,
) -> Fit:
    s = SCALE
    strong_s = downsample(fg.strong & blob.mask)
    weak_s = downsample(fg.weak & blob.support)
    u0, v0, u1, v1 = blob.bbox
    win = _Window(strong_s, weak_s, (int(u0 * s), int(v0 * s), int(u1 * s) + 1, int(v1 * s) + 1), pad=12)
    x, y = initial_xy(cam, blob, shape)
    yaw = 0.0
    evals = 0

    def score(xx, yy, aa, n=n_points):
        nonlocal evals
        evals += 1
        m = win.cut(O.render(cam, shape.posed(xx, yy, aa, BELT_Z, n=n), scale=s))
        return _score(m, win.strong, win.weak)

    best = score(x, y, yaw)
    yaws = [0.0] if symmetric else list(np.linspace(-yaw_range, yaw_range, 11))
    steps = 0.02, 0.008, 0.003
    for r in range(rounds):
        st = steps[min(r, len(steps) - 1)]
        if not symmetric:
            cand = yaws if r == 0 else [yaw + d for d in np.radians([-8, -4, -2, 2, 4, 8]) / (r if r else 1)]
            for a in cand:
                sc = score(x, y, a)
                if sc > best:
                    best, yaw = sc, a
        improved = True
        while improved:
            improved = False
            for dx, dy in ((st, 0), (-st, 0), (0, st), (0, -st), (st, st), (-st, -st), (st, -st), (-st, st)):
                sc = score(x + dx, y + dy, yaw)
                if sc > best + 1e-06:
                    best, x, y = sc, x + dx, y + dy
                    improved = True
            if evals > 400:
                break
    best = score(x, y, yaw, n=None)
    return Fit(shape, x, y, yaw, best, evals)


SYMMETRIC = {"ice_cream/up", "plate/up", "bottle/up"}


def fit_all(cam: Camera, fg: Foreground, blob: Blob, shapes=None) -> list[Fit]:
    shapes = list(shapes) if shapes is not None else O.all_shapes()
    out = []
    for sh in shapes:
        out.append(fit_shape(cam, fg, blob, sh, symmetric=sh.key in SYMMETRIC))
    out.sort(key=lambda f: -f.score)
    return out


def colour_hist(img, mask, bins=(12, 4, 4)) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    px = hsv[np.asarray(mask, dtype=bool)]
    if len(px) == 0:
        return np.zeros(bins[0] + bins[1] * bins[2])
    h, s, v = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    sat = s > 60
    hh = np.bincount((h[sat] * bins[0] // 180).clip(0, bins[0] - 1), minlength=bins[0])
    vv = (v[~sat] * bins[1] // 256).clip(0, bins[1] - 1)
    ss = (s[~sat] * bins[2] // 61).clip(0, bins[2] - 1)
    gg = np.bincount(vv * bins[2] + ss, minlength=bins[1] * bins[2])
    hist = np.concatenate([hh, gg]).astype(float)
    return hist / max(1.0, hist.sum())


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.maximum(a, 0) * np.maximum(b, 0))))


def colour_bins(img, mask, bins=(12, 4, 4)) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    px = hsv[np.asarray(mask, dtype=bool)].astype(int)
    h, s, v = px[:, 0], px[:, 1], px[:, 2]
    sat = s > 60
    out = np.empty(len(px), dtype=int)
    out[sat] = (h[sat] * bins[0] // 180).clip(0, bins[0] - 1)
    vv = (v[~sat] * bins[1] // 256).clip(0, bins[1] - 1)
    ss = (s[~sat] * bins[2] // 61).clip(0, bins[2] - 1)
    out[~sat] = bins[0] + vv * bins[2] + ss
    return out


def colour_part(img, blob: Blob, hist_ref: np.ndarray, ratio: float = 0.8, min_area: int = 150) -> Blob | None:
    import cv2

    vs, us = np.nonzero(blob.mask)
    if len(vs) < min_area:
        return None
    b = colour_bins(img, blob.mask)
    h_blob = np.bincount(b, minlength=len(hist_ref)).astype(float)
    h_blob /= max(1.0, h_blob.sum())
    keep = hist_ref[b] >= ratio * np.maximum(h_blob[b], 1e-06)
    m = np.zeros_like(blob.mask)
    m[vs[keep], us[keep]] = True
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i][4] < min_area:
        return None
    part = (lab == i) & (blob.mask | (m > 0))
    near = cv2.dilate(part.astype(np.uint8), np.ones((2 * SUPPORT_PX + 1,) * 2, np.uint8)).astype(bool)
    other = blob.mask & ~cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sup = blob.support & near & ~other | part
    pv, pu = np.nonzero(part)
    out = Blob(
        blob.label,
        (int(pu.min()), int(pv.min()), int(pu.max()) + 1, int(pv.max()) + 1),
        int(part.sum()),
        (float(pu.mean()), float(pv.mean())),
        part,
        sup,
    )
    out.touches_border = blob.touches_border
    out.touches_robot = blob.touches_robot
    return out


@dataclass
class Track:
    tid: int
    first_step: int
    obs: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    hists: list = field(default_factory=list)

    def last(self):
        return self.obs[-1]
