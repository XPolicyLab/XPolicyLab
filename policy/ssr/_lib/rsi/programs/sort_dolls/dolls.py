from __future__ import annotations

from ... import ASSETS

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from ...planning import kinematics as K

DATA = os.path.join(ASSETS, "doll_models.json")
N_RANKS = 5
REGION = (-0.58, 0.58), (-0.36, 0.12)


@dataclass
class DollModel:
    model_id: int
    rank: int
    group: int
    height: float
    z_bottom: float
    slices: list
    mass: float = 0.05
    friction: float = 0.45

    def width_at(self, z_above_bottom: float, axis_yaw: float = 0.0) -> float:
        z = self.z_bottom + z_above_bottom
        _, poly = min(self.slices, key=lambda s: abs(s[0] - z))
        d = np.array([math.cos(axis_yaw), math.sin(axis_yaw)])
        p = poly @ d
        return float(p.max() - p.min())

    def max_halfwidth(self) -> float:
        r = self.__dict__.get("_max_halfwidth")
        if r is None:
            r = self.__dict__["_max_halfwidth"] = max(float(np.linalg.norm(p, axis=1).max()) for _, p in self.slices)
        return r


@lru_cache(maxsize=1)
def models() -> tuple[DollModel, ...]:
    with open(DATA) as f:
        d = json.load(f)
    out = []
    for m in d["models"]:
        out.append(
            DollModel(
                model_id=m["model_id"],
                rank=m["rank"],
                group=m["group"],
                height=m["height"],
                z_bottom=m["z_bottom"],
                slices=[(float(z), np.asarray(p, dtype=float)) for z, p in m["slices"]],
                mass=m.get("mass") or 0.05,
                friction=m.get("friction") or 0.45,
            )
        )
    return tuple(out)


def model(model_id: int) -> DollModel:
    return models()[model_id]


def groups() -> list[int]:
    return sorted({m.group for m in models()})


@lru_cache(maxsize=128)
def _stack(model_id: int):
    m = model(model_id)
    pts = np.concatenate([p for _, p in m.slices])
    zs = np.concatenate([np.full(len(p), z - m.z_bottom) for z, p in m.slices])
    bounds = np.cumsum([0] + [len(p) for _, p in m.slices])
    return pts, zs, bounds


def silhouette_polys(
    cam, m: DollModel, x: float, y: float, yaw: float = 0.0, z0: float = K.TABLE_Z, scale: float = 1.0
) -> list[np.ndarray]:
    pts, zs, b = _stack(m.model_id)
    if yaw:
        c, s_ = math.cos(yaw), math.sin(yaw)
        pts = pts @ np.array([[c, s_], [-s_, c]])
    w = np.c_[pts + [x, y], zs + z0]
    uv = (cam.project(w)[0] * scale).astype(np.float32)
    return [cv2.convexHull(uv[b[i] : b[i + 2]]).reshape(-1, 2) for i in range(len(b) - 2)]


def render(
    cam,
    m: DollModel,
    x: float,
    y: float,
    yaw: float = 0.0,
    z0: float = K.TABLE_Z,
    shape=(480, 640),
    ss: int = 1,
    offset=(0, 0),
) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    off = np.asarray(offset, dtype=float)
    for p in silhouette_polys(cam, m, x, y, yaw, z0):
        q = np.round((p - off) * ss * 16).astype(np.int32)
        cv2.fillConvexPoly(mask, q, 1, lineType=cv2.LINE_8, shift=4)
    return mask


def lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def background(img: np.ndarray, robot: np.ndarray | None = None, ds: int = 4, k: int = 41) -> np.ndarray:
    h, w = img.shape[:2]
    im = img.copy()
    if robot is not None and robot.any():
        im = cv2.inpaint(im, robot.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    small = cv2.resize(im, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, k)
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def foreground(
    img: np.ndarray,
    robot: np.ndarray | None = None,
    *,
    thresh: float = 22.0,
    l_weight: float = 0.5,
    bg: np.ndarray | None = None,
) -> np.ndarray:
    if bg is None:
        bg = background(img, robot)
    a, b = lab(img), lab(bg)
    d = (a - b) * [l_weight, 1.0, 1.0]
    dist = np.sqrt((d**2).sum(-1))
    fg = (dist > thresh).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if robot is not None:
        fg[robot.astype(bool)] = 0
    return fg


@dataclass
class DollFit:
    x: float
    y: float
    model_id: int
    iou: float
    yaw: float = 0.0
    region: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return self.model_id % N_RANKS

    @property
    def group(self) -> int:
        return self.model_id // N_RANKS

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "model": self.model_id,
            "rank": self.rank,
            "iou": round(self.iou, 3),
            "yaw": round(self.yaw, 3),
            "region": self.region,
        }


def _iou_roi(cam, m, x, y, yaw, fg_roi, valid_roi, off, z0) -> float:
    sil = render(cam, m, x, y, yaw, z0, shape=fg_roi.shape, offset=off)
    sil &= valid_roi
    inter = float((sil & fg_roi).sum())
    union = float((sil | fg_roi).sum())
    return inter / union if union > 0 else 0.0


def fit_region(
    cam,
    comp: np.ndarray,
    m: DollModel,
    *,
    init=None,
    valid: np.ndarray | None = None,
    z0: float = K.TABLE_Z,
    yaw: float = 0.0,
    span: float = 0.03,
    pad: int = 12,
) -> tuple[float, float, float]:
    vs, us = np.nonzero(comp)
    if len(us) == 0:
        return 0.0, 0.0, 0.0
    u0, u1 = max(0, us.min() - pad), min(comp.shape[1], us.max() + pad + 1)
    v0, v1 = max(0, vs.min() - pad - 40), min(comp.shape[0], vs.max() + pad + 1)
    fg_roi = comp[v0:v1, u0:u1].astype(np.uint8)
    valid_roi = np.ones_like(fg_roi) if valid is None else valid[v0:v1, u0:u1].astype(np.uint8)
    off = u0, v0
    if init is None:
        c = np.array([[float(us.mean()), float(vs.mean())]])
        p = cam.on_plane_z(c, z0 + 0.45 * m.height)[0]
        init = float(p[0]), float(p[1])
    best = -1.0, init[0], init[1]
    for step, half in ((0.006, span), (0.002, 0.006), (0.001, 0.002)):
        cx, cy = best[1], best[2]
        n = int(round(half / step))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x, y = cx + i * step, cy + j * step
                s = _iou_roi(cam, m, x, y, yaw, fg_roi, valid_roi, off, z0)
                if s > best[0]:
                    best = s, x, y
    return best


def components(fg: np.ndarray, min_area: int = 60) -> list[np.ndarray]:
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((lab_img == i).astype(np.uint8))
    return out


def _bbox(polys) -> tuple[int, int, int, int]:
    a = np.concatenate(polys)
    return (
        int(math.floor(a[:, 0].min())),
        int(math.floor(a[:, 1].min())),
        int(math.ceil(a[:, 0].max())) + 1,
        int(math.ceil(a[:, 1].max())) + 1,
    )


def _render_box(polys, box) -> np.ndarray:
    u0, v0, u1, v1 = box
    t = np.zeros((v1 - v0, u1 - u0), np.uint8)
    for p in polys:
        q = np.round((p - [u0, v0]) * 16).astype(np.int32)
        cv2.fillConvexPoly(t, q, 1, lineType=cv2.LINE_8, shift=4)
    return t


def _crop(a: np.ndarray, u0: int, v0: int, u1: int, v1: int) -> np.ndarray:
    h, w = a.shape[:2]
    out = np.zeros((v1 - v0, u1 - u0), np.float32)
    cu0, cv0, cu1, cv1 = max(0, u0), max(0, v0), min(w, u1), min(h, v1)
    if cu1 > cu0 and cv1 > cv0:
        out[cv0 - v0 : cv1 - v0, cu0 - u0 : cu1 - u0] = a[cv0:cv1, cu0:cu1]
    return out


def _jac(cam, x, y, zc) -> np.ndarray:
    p0 = cam.project(np.array([[x, y, zc], [x + 0.005, y, zc], [x, y + 0.005, zc]]))[0]
    return np.stack([(p0[1] - p0[0]) / 0.005, (p0[2] - p0[0]) / 0.005], axis=1)


def shift_fit(
    cam,
    mask: np.ndarray,
    m: DollModel,
    x: float,
    y: float,
    *,
    valid: np.ndarray | None = None,
    search: int = 24,
    z0: float = K.TABLE_Z,
) -> tuple[float, float, float]:
    polys = silhouette_polys(cam, m, x, y, 0.0, z0)
    box = _bbox(polys)
    t = _render_box(polys, box).astype(np.float32)
    u0, v0, u1, v1 = box
    big = u0 - search, v0 - search, u1 + search, v1 + search
    mk = _crop(mask, *big)
    n_mask = float(mask.sum())
    inter = cv2.matchTemplate(mk, t, cv2.TM_CCORR)
    if valid is not None:
        vt = cv2.matchTemplate(_crop(valid, *big), t, cv2.TM_CCORR)
    else:
        vt = np.full_like(inter, float(t.sum()))
    iou = inter / np.maximum(1.0, vt + n_mask - inter)
    j, i = np.unravel_index(int(np.argmax(iou)), iou.shape)
    du, dv = i - search, j - search
    if du == 0 and dv == 0:
        return float(iou[j, i]), x, y
    J = _jac(cam, x, y, z0 + 0.5 * m.height)
    dxy = np.linalg.solve(J, np.array([du, dv], dtype=float))
    return float(iou[j, i]), float(x + dxy[0]), float(y + dxy[1])


def mask_init(cam, mask: np.ndarray, m: DollModel, z0: float = K.TABLE_Z) -> tuple[float, float]:
    vs, us = np.nonzero(mask)
    p = cam.on_plane_z(np.array([[float(us.mean()), float(vs.mean())]]), z0 + 0.45 * m.height)[0]
    return float(p[0]), float(p[1])


def fit_instance(
    cam,
    mask: np.ndarray,
    m: DollModel,
    *,
    init=None,
    valid: np.ndarray | None = None,
    z0: float = K.TABLE_Z,
    fine: bool = True,
) -> tuple[float, float, float]:
    x, y = mask_init(cam, mask, m, z0) if init is None else init
    s, x, y = shift_fit(cam, mask, m, x, y, valid=valid, z0=z0)
    s, x, y = shift_fit(cam, mask, m, x, y, valid=valid, search=6, z0=z0)
    if not fine:
        return s, x, y
    vs, us = np.nonzero(mask)
    u0, v0 = max(0, us.min() - 40), max(0, vs.min() - 40)
    u1, v1 = min(mask.shape[1], us.max() + 40), min(mask.shape[0], vs.max() + 40)
    mk = mask[v0:v1, u0:u1].astype(bool)
    vd = None if valid is None else valid[v0:v1, u0:u1].astype(bool)
    best = -1.0, x, y
    for step, n in ((0.0015, 2), (0.0006, 1)):
        cx, cy = best[1], best[2]
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                xx, yy = cx + i * step, cy + j * step
                sil = render(cam, m, xx, yy, 0.0, z0, shape=mk.shape, offset=(u0, v0)).astype(bool)
                if vd is not None:
                    sil &= vd
                inter = float((sil & mk).sum())
                sc = inter / max(1.0, float((sil | mk).sum()))
                if sc > best[0]:
                    best = sc, xx, yy
    return best


@dataclass
class Scene:
    dolls: list
    group: int
    others: list
    extra: dict = field(default_factory=dict)
    others_cat: list = field(default_factory=list)


DOLL_CATEGORY = "matryoshka_dolls"


def detect(img: np.ndarray, cam, *, insts=None, z0: float = K.TABLE_Z, min_iou: float = 0.45) -> Scene:
    from scipy.optimize import linear_sum_assignment

    if insts is None:
        from ...perception import seg

        insts = seg.segment(img, camera="head")
    cands = [i for i in insts if i.category == DOLL_CATEGORY]
    others = [np.asarray(i.mask, bool) for i in insts if i.category != DOLL_CATEGORY]
    others_cat = [str(i.category) for i in insts if i.category != DOLL_CATEGORY]
    n = len(cands)
    if n == 0:
        return Scene([], -1, others, {"n_cands": 0}, others_cat)
    masks = [np.asarray(c.mask, bool) for c in cands]
    allm = np.zeros(masks[0].shape, bool)
    for mk in masks:
        allm |= mk
    ms = models()
    S = np.zeros((n, len(ms)))
    XY = np.zeros((n, len(ms), 2))
    valids = []
    for ci, mk in enumerate(masks):
        valid = (~(allm & ~mk)).astype(np.uint8)
        valids.append(valid)
        m8 = mk.astype(np.uint8)
        for m in ms:
            s, x, y = fit_instance(cam, m8, m, valid=valid, z0=z0, fine=False)
            S[ci, m.model_id], XY[ci, m.model_id] = s, (x, y)
    best = None
    for g in groups():
        sub = S[:, g * N_RANKS : (g + 1) * N_RANKS]
        rows, cols = linear_sum_assignment(-sub)
        tot = float(sub[rows, cols].sum())
        if best is None or tot > best[0]:
            best = tot, g, rows, cols
    _, g, rows, cols = best
    dolls, used = [], set()
    for ci, r in zip(rows, cols):
        mid = g * N_RANKS + int(r)
        if S[ci, mid] < min_iou:
            continue
        s, x, y = fit_instance(
            cam, masks[ci].astype(np.uint8), ms[mid], init=tuple(XY[ci, mid]), valid=valids[ci], z0=z0
        )
        dolls.append(
            DollFit(
                x=x, y=y, model_id=mid, iou=s, region=int(ci), extra={"score": float(getattr(cands[ci], "score", 0.0))}
            )
        )
        used.add(int(ci))
    others += [masks[ci] for ci in range(n) if ci not in used]
    others_cat += [DOLL_CATEGORY for ci in range(n) if ci not in used]
    dolls.sort(key=lambda d: d.rank)
    return Scene(
        dolls,
        g,
        others,
        {"n_cands": n, "group_scores": {int(gg): float(S[:, gg * 5 : (gg + 1) * 5].max(1).sum()) for gg in groups()}},
        others_cat,
    )
