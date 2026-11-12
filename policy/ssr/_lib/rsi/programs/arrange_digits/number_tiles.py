from __future__ import annotations

from ... import ASSETS

import base64
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from ...perception.cameras import Camera

_DATA = ASSETS / "number_tiles.json"
TABLE_Z = 0.765
PAD_TOP = TABLE_Z + 0.005
PAD_R = 0.04
TILE_T = 0.0155
RES = 0.001
TPL = 72
CANVAS = 112


@dataclass
class Tile:
    digit: int
    xy: np.ndarray
    yaw: float
    score: float
    margin: float
    scores: dict
    z_support: float
    uv: tuple = (0.0, 0.0)
    area: float = 0.0
    clipped: bool = False
    src: str = "head"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "digit": self.digit,
            "xy": [round(float(v), 4) for v in self.xy],
            "yaw_deg": round(math.degrees(self.yaw), 1),
            "score": round(self.score, 3),
            "margin": round(self.margin, 3),
            "z": round(self.z_support, 4),
            "clipped": self.clipped,
            "src": self.src,
            "area_cm2": round(self.area * 10000.0, 2),
        }


@dataclass
class Pad:
    xy: np.ndarray
    score: float = 0.0

    def as_dict(self) -> dict:
        return {"xy": [round(float(v), 4) for v in self.xy], "score": round(self.score, 3)}


def _proj_matrix(cam: Camera) -> np.ndarray:
    k = np.array([[cam.f, 0.0, -cam.cx], [0.0, -cam.f, -cam.cy], [0.0, 0.0, -1.0]])
    rt = np.hstack([cam.R.T, (-cam.R.T @ cam.pos)[:, None]])
    return k @ rt


def plane_homography(cam: Camera, x0: float, y0: float, z: float, n: int, res: float = RES) -> np.ndarray:
    c = n / 2.0
    m = np.array([[res, 0.0, x0 - c * res], [0.0, -res, y0 + c * res], [0.0, 0.0, z], [0.0, 0.0, 1.0]])
    return _proj_matrix(cam) @ m


def to_top(
    img: np.ndarray, cam: Camera, x0: float, y0: float, z: float, n: int = CANVAS, res: float = RES
) -> np.ndarray:
    h = plane_homography(cam, x0, y0, z, n, res)
    return cv2.warpPerspective(
        img, h, (n, n), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def canvas_to_world(col: float, row: float, x0: float, y0: float, n: int = CANVAS, res: float = RES) -> np.ndarray:
    c = n / 2.0
    return np.array([x0 + (col - c) * res, y0 - (row - c) * res])


def table_view(m: np.ndarray, cam: Camera, z: float, region, cell: float) -> np.ndarray:
    (x0, x1), (y0, y1) = region
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    mm = np.array([[cell, 0.0, x0], [0.0, -cell, y1], [0.0, 0.0, z], [0.0, 0.0, 1.0]])
    h = _proj_matrix(cam) @ mm
    return cv2.warpPerspective(
        m.astype(np.float32),
        h,
        (nx, ny),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@lru_cache(maxsize=1)
def digit_shapes() -> tuple:
    d = json.loads(_DATA.read_text())
    n = int(d["shape"][0])
    out = []
    for k in range(10):
        rec = d["pieces"][f"number_{k:05d}"]
        bits = np.frombuffer(base64.b64decode(rec["mask"]), dtype=np.uint8).reshape(n, -1)
        m = np.unpackbits(bits, axis=-1)[:, :n].astype(np.float32)
        out.append(cv2.resize(m, (TPL, TPL), interpolation=cv2.INTER_AREA))
    return tuple(out)


def _rotate(t: np.ndarray, yaw: float) -> np.ndarray:
    n = t.shape[0]
    m = cv2.getRotationMatrix2D((n / 2.0 - 0.5, n / 2.0 - 0.5), math.degrees(yaw), 1.0)
    return cv2.warpAffine(t, m, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)


def _shift(t: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    m = np.float32([[1, 0, dx_px], [0, 1, dy_px]])
    return cv2.warpAffine(t, m, (t.shape[1], t.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def swept(t: np.ndarray, sweep_xy: np.ndarray, round_r: float = 0.005) -> np.ndarray:
    out = np.zeros_like(t)
    for s in (-0.85, -0.45, 0.0, 0.45, 0.85):
        e = round_r * (1.0 - math.sqrt(1.0 - s * s)) / RES
        sl = t
        if e >= 0.75:
            k = int(round(2 * e)) | 1
            sl = cv2.erode(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sl = _shift(sl, s * sweep_xy[0] / RES, -s * sweep_xy[1] / RES)
        out = np.maximum(out, sl)
    return out


def sweep_vector(cam: Camera, xy, z_mid: float, half_t: float) -> np.ndarray:
    d = np.asarray([xy[0] - cam.pos[0], xy[1] - cam.pos[1]], dtype=float)
    return d * half_t / max(1e-06, cam.pos[2] - z_mid)


SLICES = -0.85, -0.45, 0.0, 0.45, 0.85
ROUND_R = 0.005
YAW_Q = math.radians(0.25)
COARSE = 2
COARSE_YAW = math.radians(8)
REFINE_WITHIN = 0.12
FINE_YAWS = tuple(math.radians(v) for v in (-6, -4, -2, 0, 2, 4, 6))


@lru_cache(maxsize=None)
def _coarse_shapes() -> tuple:
    n = TPL // COARSE
    return tuple(cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA) for t in digit_shapes())


def _erode_k(s: float, res: float) -> int:
    e = ROUND_R * (1.0 - math.sqrt(1.0 - s * s)) / res
    return int(round(2 * e)) | 1 if e >= 0.75 else 0


@lru_cache(maxsize=2000)
def _rotated(d: int, yaw_q: int, coarse: bool) -> np.ndarray:
    return _rotate((_coarse_shapes() if coarse else digit_shapes())[d], yaw_q * YAW_Q)


@lru_cache(maxsize=6000)
def _eroded(d: int, yaw_q: int, coarse: bool, k: int) -> np.ndarray:
    r = _rotated(d, yaw_q, coarse)
    return cv2.erode(r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) if k else r


def _ishift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return a
    h, w = a.shape
    out = np.zeros_like(a)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = a[
        max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)
    ]
    return out


def template(d: int, yaw: float, sweep_xy, coarse: bool = False, slices=SLICES) -> np.ndarray:
    res = RES * (COARSE if coarse else 1)
    yq = int(round(yaw / YAW_Q))
    out = None
    for s_ in slices:
        a = _ishift(
            _eroded(int(d), yq, bool(coarse), _erode_k(s_, res)),
            int(round(s_ * sweep_xy[0] / res)),
            int(round(-s_ * sweep_xy[1] / res)),
        )
        out = a if out is None else np.maximum(out, a)
    return out


def _match_direct(m: np.ndarray, sweep_xy, yaws, digits, coarse: bool = False, slices=SLICES):
    sm = float(m.sum())
    best = -1.0, None, 0.0, 0.0, 0.0
    per = {}
    scale = COARSE if coarse else 1
    for d in digits:
        for yaw in yaws:
            t = template(d, yaw, sweep_xy, coarse, slices)
            st = float(t.sum())
            r = cv2.matchTemplate(m, t, cv2.TM_CCORR)
            _, mx, _, loc = cv2.minMaxLoc(r)
            sc = 2.0 * mx / max(1e-06, st + sm)
            if sc > per.get(d, (-1.0,))[0]:
                per[d] = sc, yaw
            if sc > best[0]:
                n = t.shape[0]
                best = sc, d, yaw, (loc[0] + n / 2.0) * scale, (loc[1] + n / 2.0) * scale
    return best, per


def match_tile(soft_top: np.ndarray, sweep_xy: np.ndarray, yaws, digits=range(10), slices=SLICES):
    m = soft_top.astype(np.float32)
    digits = list(digits)
    yaws = list(yaws)
    if len(digits) == 1 or len(yaws) <= 12:
        best, per = _match_direct(m, sweep_xy, yaws, digits, slices=slices)
        return best, {d: v[0] for d, v in per.items()}
    n = m.shape[0] // COARSE
    mc = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA)
    lo, hi = min(yaws), max(yaws)
    cy = list(np.arange(lo, hi + 1e-09, COARSE_YAW))
    _, coarse = _match_direct(mc, sweep_xy, cy, digits, coarse=True, slices=slices)
    top = max(v[0] for v in coarse.values())
    per = {d: v[0] for d, v in coarse.items()}
    best = -1.0, None, 0.0, 0.0, 0.0
    one = math.radians(1)
    for d in digits:
        if coarse[d][0] < top - REFINE_WITHIN:
            continue
        y0 = coarse[d][1]
        b1, _ = _match_direct(m, sweep_xy, [min(hi, max(lo, y0 + dy)) for dy in FINE_YAWS], [d], slices=slices)
        b2, _ = _match_direct(m, sweep_xy, [b1[2] - one, b1[2] + one], [d], slices=slices)
        bd = b2 if b2[0] > b1[0] else b1
        per[d] = bd[0]
        if bd[0] > best[0]:
            best = bd
    return best, per


TILE_REGION = (-0.47, 0.47), (-0.33, 0.01)
FG_AB = 12.0
FG_L = 999.0
MIN_TILE_AREA = 0.0003
MAX_TILE_AREA = 0.0032
MIN_SCORE = 0.55


def lab_image(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def local_background(img: np.ndarray, robot: np.ndarray | None = None, k: int = 21) -> np.ndarray:
    src = img
    if robot is not None and robot.any():
        src = img.copy()
        src[robot] = np.median(img[~robot].reshape(-1, 3), axis=0).astype(np.uint8)
    small = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.medianBlur(small, k), (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)


def foreground(
    img: np.ndarray,
    robot: np.ndarray | None = None,
    bg: np.ndarray | None = None,
    fg_ab: float | None = None,
    fg_l: float | None = None,
) -> np.ndarray:
    if bg is None:
        bg = local_background(img, robot)
    lab, labb = lab_image(img), lab_image(bg)
    d_ab = np.hypot(lab[..., 1] - labb[..., 1], lab[..., 2] - labb[..., 2])
    d_l = np.abs(lab[..., 0] - labb[..., 0])
    m = (d_ab >= (FG_AB if fg_ab is None else fg_ab)) | (d_l >= (FG_L if fg_l is None else fg_l))
    if robot is not None:
        m &= ~robot
    return m


def fit_region(
    reg: np.ndarray,
    cam: Camera,
    z_support: float = TABLE_Z,
    digits=range(10),
    yaw_range=math.radians(60),
    yaw_step=math.radians(4),
    soft: np.ndarray | None = None,
):
    ys, xs = np.nonzero(reg)
    if len(xs) == 0:
        return None
    z_mid = z_support + TILE_T / 2
    p = cam.on_plane_z([[float(xs.mean()), float(ys.mean())]], z_mid)[0]
    if not np.all(np.isfinite(p)):
        return None
    reg_d = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    src = reg_d.astype(np.float32) if soft is None else np.where(reg_d, soft, 0.0).astype(np.float32)
    top = to_top(src, cam, p[0], p[1], z_mid)
    area = float(top.sum()) * RES * RES
    if area < MIN_TILE_AREA or area > MAX_TILE_AREA:
        return None
    sw = sweep_vector(cam, p[:2], z_mid, TILE_T / 2)
    yaws = np.arange(-yaw_range, yaw_range + 1e-09, yaw_step)
    (sc, d, yaw, col, row), per = match_tile(top, sw, yaws, digits)
    if d is None:
        return None
    fine = np.arange(yaw - yaw_step, yaw + yaw_step + 1e-09, yaw_step / 4)
    (sc2, _, yaw2, col2, row2), _ = match_tile(top, sw, fine, digits=[d])
    if sc2 >= sc:
        sc, yaw, col, row = sc2, yaw2, col2, row2
        per[d] = max(per[d], sc2)
    others = [s for k, s in per.items() if k != d]
    return Tile(
        digit=int(d),
        xy=canvas_to_world(col, row, p[0], p[1]),
        yaw=float(yaw),
        score=float(sc),
        margin=float(sc - max(others)) if others else 1.0,
        scores={int(k): round(float(s), 3) for k, s in per.items()},
        z_support=z_support,
        uv=(float(xs.mean()), float(ys.mean())),
        area=area,
    )


@dataclass
class TopView:
    region: tuple
    lab: np.ndarray
    bg: np.ndarray
    valid: np.ndarray
    sweep_cam: np.ndarray
    z_support: float = TABLE_Z
    top: np.ndarray | None = None

    def world(self, col: float, row: float) -> np.ndarray:
        (x0, _), (_, y1) = self.region
        return np.array([x0 + col * RES, y1 - row * RES])


def top_view(
    img: np.ndarray,
    cam: Camera,
    robot: np.ndarray | None = None,
    region=TILE_REGION,
    exclude: np.ndarray | None = None,
    bg_kernel: float = 0.061,
    z_support: float = TABLE_Z,
) -> TopView | None:
    z_mid = z_support + TILE_T / 2
    valid_img = np.ones(img.shape[:2], np.float32)
    if robot is not None:
        valid_img[robot] = 0
    if exclude is not None:
        valid_img[exclude] = 0
    top = np.stack([table_view(img[..., i], cam, z_mid, region, RES) for i in range(3)], axis=-1)
    valid = table_view(valid_img, cam, z_mid, region, RES) > 0.99
    if valid.sum() < 100:
        return None
    top = np.clip(top, 0, 255).astype(np.uint8)
    top[~valid] = np.median(top[valid], axis=0).astype(np.uint8)
    k = min(255, int(round(bg_kernel / RES)) | 1)
    bg = cv2.medianBlur(top, k)
    return TopView(
        region=region,
        lab=lab_image(top),
        bg=lab_image(bg),
        valid=valid,
        sweep_cam=cam.pos.copy(),
        z_support=z_support,
        top=top,
    )


def _sweep(tv: TopView, xy) -> np.ndarray:
    z_mid = tv.z_support + TILE_T / 2
    d = np.asarray([xy[0] - tv.sweep_cam[0], xy[1] - tv.sweep_cam[1]], dtype=float)
    return d * (TILE_T / 2) / max(1e-06, tv.sweep_cam[2] - z_mid)


def match_regions(tv: TopView, m: np.ndarray, min_score: float = MIN_SCORE) -> list[Tile]:
    edge = cv2.dilate((~tv.valid).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = []
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    half = CANVAS // 2
    for i in range(1, n_lab):
        area = float(stats[i, cv2.CC_STAT_AREA]) * RES * RES
        if area < MIN_TILE_AREA or area > MAX_TILE_AREA:
            continue
        cu, cv_ = cents[i]
        reg = labels == i
        r0, c0 = int(round(cv_)), int(round(cu))
        win = np.pad(reg, half).astype(np.float32)[r0 : r0 + CANVAS, c0 : c0 + CANVAS]
        p = tv.world(cu, cv_)
        sw = _sweep(tv, p)
        yaws = np.arange(-math.radians(60), math.radians(60) + 1e-09, math.radians(4))
        (sc, d, yaw, col, row), per = match_tile(win, sw, yaws)
        if d is None or sc < min_score:
            continue
        fine = np.arange(yaw - math.radians(4), yaw + math.radians(4) + 1e-09, math.radians(1))
        (sc2, _, yaw2, col2, row2), _ = match_tile(win, sw, fine, digits=[d])
        if sc2 >= sc:
            sc, yaw, col, row = sc2, yaw2, col2, row2
            per[d] = max(per[d], sc2)
        others = [s for kk, s in per.items() if kk != d]
        c = tv.world(c0, r0)
        xy = canvas_to_world(col, row, c[0], c[1])
        t = Tile(
            digit=int(d),
            xy=xy,
            yaw=float(yaw),
            score=float(sc),
            margin=float(sc - max(others)) if others else 1.0,
            scores={int(kk): round(float(s), 3) for kk, s in per.items()},
            z_support=tv.z_support,
            area=area,
        )
        t.clipped = bool((reg & edge).any())
        t.extra["lab"] = [round(float(v), 1) for v in np.median(tv.lab[reg], axis=0)]
        t.extra["region"] = reg
        out.append(t)
    return out


def tile_pixels(tv: TopView, t: Tile, erode_px: int = 2) -> np.ndarray:
    s = _rotate(digit_shapes()[t.digit], t.yaw) > 0.5
    if erode_px:
        s = cv2.erode(s.astype(np.uint8), np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)).astype(bool)
    (x0, _), (_, y1) = tv.region
    col = (t.xy[0] - x0) / RES
    row = (y1 - t.xy[1]) / RES
    m = np.zeros(tv.valid.shape, bool)
    r0, c0 = int(round(row - TPL / 2)), int(round(col - TPL / 2))
    rr, cc = np.nonzero(s)
    rr, cc = rr + r0, cc + c0
    ok = (rr >= 0) & (rr < m.shape[0]) & (cc >= 0) & (cc < m.shape[1])
    m[rr[ok], cc[ok]] = True
    return m & tv.valid


SEED_SCORE = 0.8
COLOUR_TOL = 8.0, 24.0
COLOUR_TOL_FRAC = 0.5


def colour_mask(tv: TopView, colour: np.ndarray, tol: float) -> np.ndarray:
    d_t = np.linalg.norm(tv.lab - colour[None, None, :], axis=-1)
    d_b = np.linalg.norm(tv.lab - tv.bg, axis=-1)
    m = (d_t <= tol) & (d_t < d_b) & tv.valid
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)


MULTI_CUES = (
    ("ab", 12.0),
    ("ab", 20.0),
    ("ab", 32.0),
    ("ab", 48.0),
    ("dark", 16.0),
    ("dark", 28.0),
    ("dark", 42.0),
    ("light", 12.0),
    ("light", 20.0),
    ("light", 30.0),
    ("absl", 12.0),
    ("absl", 20.0),
    ("lab", 20.0),
    ("lab", 30.0),
    ("chromw", 25.0),
    ("chromw", 40.0),
    ("chromw", 60.0),
)
DUP_IOU = 0.9
NMS_OVERLAP = 0.3
NMS_R = 0.015
MERGE_R = 0.012
TOP_SLICES = (0.95,)
TOP_CUES = ("light",)
BG_WIDE = 0.121


def cue_maps(tv: TopView, kinds=None) -> dict:
    want = set(kinds) if kinds is not None else {"ab", "dark", "light", "absl", "lab", "maha", "chrom", "chromw"}
    lab = tv.lab
    d_l = lab[..., 0] - tv.bg[..., 0]
    d_ab = np.hypot(lab[..., 1] - tv.bg[..., 1], lab[..., 2] - tv.bg[..., 2])
    out = {"ab": d_ab, "dark": -d_l, "light": d_l, "absl": np.abs(d_l), "lab": np.hypot(d_ab, d_l)}
    if "maha" in want:
        out["maha"] = mahalanobis(lab - tv.bg, tv.valid)
    if "chrom" in want:
        out["chrom"] = chromaticity_diff(lab, tv.bg)
    if (
        "chromw" in want and tv.top is not None
    ):
        k = min(255, int(round(BG_WIDE / RES)) | 1)
        out["chromw"] = chromaticity_diff(lab, lab_image(cv2.medianBlur(tv.top, k)))
    return out


def chromaticity_diff(lab: np.ndarray, bg: np.ndarray) -> np.ndarray:
    def chrom(x):
        rgb = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        s = rgb.sum(axis=-1, keepdims=True) + 30.0
        return rgb[..., :2] / s

    d = chrom(lab) - chrom(bg)
    return 1000.0 * np.hypot(d[..., 0], d[..., 1])


def mahalanobis(diff: np.ndarray, valid: np.ndarray, iters: int = 2, keep: float = 9.0) -> np.ndarray:
    d = diff[valid].reshape(-1, 3).astype(np.float64)
    if len(d) < 50:
        return np.zeros(diff.shape[:2], np.float32)
    sel = d
    inv = None
    for _ in range(iters + 1):
        cov = np.cov(sel.T) + np.eye(3) * 1.0
        inv = np.linalg.inv(cov)
        q = np.einsum("ij,jk,ik->i", d, inv, d)
        sel = d[q < keep]
        if len(sel) < 50:
            break
    flat = diff.reshape(-1, 3).astype(np.float64)
    q = np.einsum("ij,jk,ik->i", flat, inv, flat)
    return np.sqrt(np.maximum(q, 0.0)).reshape(diff.shape[:2]).astype(np.float32)


def _components(m: np.ndarray):
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    for i in range(1, n_lab):
        area = float(stats[i, cv2.CC_STAT_AREA]) * RES * RES
        if area < MIN_TILE_AREA or area > MAX_TILE_AREA:
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        yield (y, x, labels[y : y + h, x : x + w] == i, (float(cents[i][0]), float(cents[i][1])), area)


def _overlap(a, b) -> int:
    (ya, xa, ca), (yb, xb, cb) = a, b
    y0, x0 = max(ya, yb), max(xa, xb)
    y1, x1 = min(ya + ca.shape[0], yb + cb.shape[0]), min(xa + ca.shape[1], xb + cb.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.count_nonzero(ca[y0 - ya : y1 - ya, x0 - xa : x1 - xa] & cb[y0 - yb : y1 - yb, x0 - xb : x1 - xb]))


def _match_component(tv: TopView, comp, edge: np.ndarray, min_score: float, cue: str, slices=SLICES) -> Tile | None:
    y, x, crop, (cu, cv_), area = comp
    half = CANVAS // 2
    r0, c0 = int(round(cv_)), int(round(cu))
    win = np.zeros((CANVAS, CANVAS), np.float32)
    ys, xs = y - r0 + half, x - c0 + half
    h, w = crop.shape
    wy0, wx0, wy1, wx1 = max(ys, 0), max(xs, 0), min(ys + h, CANVAS), min(xs + w, CANVAS)
    if wy1 <= wy0 or wx1 <= wx0:
        return None
    win[wy0:wy1, wx0:wx1] = crop[wy0 - ys : wy1 - ys, wx0 - xs : wx1 - xs]
    p = tv.world(cu, cv_)
    sw = _sweep(tv, p)
    yaws = np.arange(-math.radians(60), math.radians(60) + 1e-09, math.radians(4))
    (sc, d, yaw, col, row), per = match_tile(win, sw, yaws, slices=slices)
    if d is None or sc < min_score:
        return None
    c = tv.world(c0, r0)
    xy = canvas_to_world(col, row, c[0], c[1])
    others = [v for kk, v in per.items() if kk != d]
    t = Tile(
        digit=int(d),
        xy=xy,
        yaw=float(yaw),
        score=float(sc),
        margin=float(sc - max(others)) if others else 1.0,
        scores={int(kk): round(float(v), 3) for kk, v in per.items()},
        z_support=tv.z_support,
        area=area,
    )
    t.clipped = bool((edge[y : y + h, x : x + w] & crop).any())
    t.extra["lab"] = [round(float(v), 1) for v in np.median(tv.lab[y : y + h, x : x + w][crop], axis=0)]
    t.extra["crop"] = y, x, crop
    t.extra["cue"] = cue
    return t


def detect_tiles_multi(
    tv: TopView, min_score: float = MIN_SCORE, cues=MULTI_CUES, info: dict | None = None
) -> list[Tile]:
    maps = cue_maps(tv, {k for k, _ in cues})
    edge = cv2.dilate((~tv.valid).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    seen, dets = [], []
    kernel = np.ones((3, 3), np.uint8)
    for kind, thr in cues:
        m = ((maps[kind] >= thr) & tv.valid).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        for comp in _components(m):
            key = comp[0], comp[1], comp[2]
            a = int(comp[2].sum())
            if any(_overlap(key, k2) >= DUP_IOU * max(a, a2) for k2, a2 in seen):
                continue
            seen.append((key, a))
            t = _match_component(
                tv, comp, edge, min_score, f"{kind}{thr:g}", TOP_SLICES if kind in TOP_CUES else SLICES
            )
            if t is not None:
                dets.append(t)
    dets.sort(key=lambda t: -t.score)
    kept = []
    for t in dets:
        ct = t.extra["crop"]
        at = int(ct[2].sum())
        same = [
            k
            for k in kept
            if np.linalg.norm(np.asarray(t.xy) - np.asarray(k.xy)) < NMS_R
            or _overlap(ct, k.extra["crop"]) > NMS_OVERLAP * min(at, int(k.extra["crop"][2].sum()))
        ]
        if same:
            if np.linalg.norm(np.asarray(t.xy) - np.asarray(same[0].xy)) < MERGE_R:
                k = same[0]
                for d, v in t.scores.items():
                    if v > k.scores.get(d, -1.0):
                        k.scores[d] = v
            continue
        kept.append(t)
    if info is not None:
        info.update(n_regions=len(seen), n_dets=len(dets), n_kept=len(kept))
    return kept


def detect_tiles(
    img: np.ndarray,
    cam: Camera,
    robot: np.ndarray | None = None,
    region=TILE_REGION,
    exclude: np.ndarray | None = None,
    min_score: float = MIN_SCORE,
    colour=None,
    info: dict | None = None,
    two_pass: bool = False,
    z_support: float = TABLE_Z,
    cues: str = "ab",
) -> list[Tile]:
    tv = top_view(img, cam, robot, region, exclude, z_support=z_support)
    if tv is None:
        return []
    if cues == "multi":
        return detect_tiles_multi(tv, min_score, info=info)
    d_ab = np.hypot(tv.lab[..., 1] - tv.bg[..., 1], tv.lab[..., 2] - tv.bg[..., 2])
    d_l = np.abs(tv.lab[..., 0] - tv.bg[..., 0])
    m = ((d_ab >= FG_AB) | (d_l >= FG_L)) & tv.valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    first = match_regions(tv, m, min_score)
    if info is not None:
        info.update(n_first=len(first))
    if not two_pass:
        return first
    if colour is None:
        seeds = [t for t in first if t.score >= SEED_SCORE and (not t.clipped)]
        if not seeds:
            return first
        px = np.concatenate([tv.lab[tile_pixels(tv, t)] for t in seeds])
        cols = np.array([np.median(tv.lab[tile_pixels(tv, t)], axis=0) for t in seeds])
        best = max(range(len(cols)), key=lambda i: (np.linalg.norm(cols - cols[i], axis=1) < 15).sum())
        near = np.linalg.norm(cols - cols[best], axis=1) < 15
        px = np.concatenate([tv.lab[tile_pixels(tv, t)] for t, k in zip(seeds, near) if k])
        colour = np.median(px, axis=0)
    colour = np.asarray(colour, dtype=np.float32)
    sep = float(np.median(np.linalg.norm(tv.bg[tv.valid] - colour, axis=-1)))
    tol = float(np.clip(COLOUR_TOL_FRAC * sep, *COLOUR_TOL))
    second = match_regions(tv, colour_mask(tv, colour, tol), min_score)
    if info is not None:
        info.update(
            colour=[round(float(v), 1) for v in colour], tol=round(tol, 1), n_first=len(first), n_second=len(second)
        )
    for t in second:
        t.extra["pass"] = 2
    return second


PAD_REGION = (-0.4, 0.4), (-0.22, 0.07)
PAD_BG_KERNEL = 0.201
PAD_MIN_SCORE = 0.55
PAD_TEX_BOX = 5


def detect_pads(
    img: np.ndarray,
    cam: Camera,
    robot: np.ndarray | None = None,
    region=PAD_REGION,
    max_n: int = 6,
    info: dict | None = None,
    min_score: float = PAD_MIN_SCORE,
    soft: float | None = None,
    smooth: float | None = None,
) -> list[Pad]:
    (x0, x1), (y0, y1) = region
    valid_img = np.ones(img.shape[:2], np.float32)
    if robot is not None:
        valid_img[robot] = 0
    top = np.stack([table_view(img[..., i], cam, PAD_TOP, region, 0.002) for i in range(3)], axis=-1)
    valid = table_view(valid_img, cam, PAD_TOP, region, 0.002) > 0.99
    if valid.sum() < 100:
        return []
    top = np.clip(top, 0, 255).astype(np.uint8)
    top[~valid] = np.median(top[valid], axis=0).astype(np.uint8)
    k = min(255, int(round(PAD_BG_KERNEL / 0.002)) | 1)
    bg = cv2.medianBlur(top, k)
    lab, labb = lab_image(top), lab_image(bg)
    diff = np.linalg.norm(lab - labb, axis=-1)
    if smooth is not None:
        lum = lab[..., 0]
        k = PAD_TEX_BOX, PAD_TEX_BOX
        mu = cv2.blur(lum, k)
        sd = np.sqrt(np.maximum(cv2.blur(lum * lum, k) - mu * mu, 0.0))
        thr = smooth * float(np.median(sd[valid]))
        m = ((sd <= thr) & valid).astype(np.float32)
    elif soft is None:
        m = ((diff >= 20.0) & valid).astype(np.float32)
    else:
        m = (np.clip(diff / soft, 0.0, 1.0) * valid).astype(np.float32)
    r = PAD_R / 0.002
    n = int(2 * 1.4 * r) | 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    rr = np.hypot(xx, yy)
    inner = (rr <= 0.9 * r).astype(np.float32)
    ring = ((rr >= 1.15 * r) & (rr <= 1.4 * r)).astype(np.float32)
    f_in = cv2.filter2D(m, -1, inner / inner.sum(), borderType=cv2.BORDER_CONSTANT)
    f_ring = cv2.filter2D(m, -1, ring / ring.sum(), borderType=cv2.BORDER_CONSTANT)
    score = f_in - 0.5 * f_ring
    out = []
    sc = score.copy()
    for _ in range(max_n):
        _, mx, _, loc = cv2.minMaxLoc(sc)
        if mx < min_score:
            break
        col, row = loc
        out.append(Pad(xy=np.array([x0 + col * 0.002, y1 - row * 0.002]), score=float(mx)))
        cv2.circle(sc, (int(col), int(row)), int(round(1.9 * r)), -1.0, -1)
    if info is not None:
        info["pad_scores"] = [round(p.score, 3) for p in out]
        info["score_map"] = score, x0, y1, 0.002
    return sorted(out, key=lambda p: p.xy[0])


PAD_SPACING = 0.085
ROW_Y = -0.115, -0.035
ROW_DX = 0.012


def pad_row_model(
    score_map, sizes=(4, 5), spacing: float = PAD_SPACING, y_range=ROW_Y, dx_max: float = ROW_DX, loc_frac: float = 0.8
):
    score, x0, y1, cell = score_map
    h, w = score.shape
    sm = cv2.dilate(score, np.ones((5, 5), np.uint8))
    rows = np.array([r for r in range(h) if y_range[0] <= y1 - r * cell <= y_range[1]], dtype=int)
    best_by_n, cand = {}, {}
    for n in sizes:
        xs = (np.arange(n) - (n - 1) / 2.0) * spacing
        recs = []
        for dx in np.arange(-dx_max, dx_max + 1e-09, cell):
            cols = np.round((xs + dx - x0) / cell).astype(int)
            if len(rows) == 0 or cols.min() < 0 or cols.max() >= w:
                continue
            vmin = sm[np.ix_(rows, cols)].min(axis=1)
            vmean = score[np.ix_(rows, cols)].mean(axis=1)
            for r, a, b in zip(rows, vmin, vmean):
                recs.append((float(a), float(b), float(dx), int(r), cols))
        if recs:
            best_by_n[n] = round(max(r[0] for r in recs), 3)
            cand[n] = recs, xs
    if not best_by_n:
        return [], best_by_n
    n = max(best_by_n, key=best_by_n.get)
    recs, xs = cand[n]
    top = max(r[0] for r in recs)
    _, _, dx, r, cols = max((rc for rc in recs if rc[0] >= loc_frac * top), key=lambda rc: rc[1])
    y = y1 - r * cell
    pads = [Pad(xy=np.array([float(x + dx), float(y)]), score=float(score[r, c])) for x, c in zip(xs, cols)]
    return pads, best_by_n


ROW_DY = 0.015
ROW_GAP = 2 * PAD_R - 0.004, 2 * PAD_R + 0.025


def pad_row(pads: list[Pad], sizes=(4, 5)) -> list[Pad]:
    best = None
    ps = sorted(pads, key=lambda p: p.xy[0])
    for i, p0 in enumerate(ps):
        run = [p0]
        for q in ps[i + 1 :]:
            last = run[-1]
            if abs(q.xy[1] - p0.xy[1]) > ROW_DY:
                continue
            d = q.xy[0] - last.xy[0]
            if ROW_GAP[0] <= d <= ROW_GAP[1]:
                run.append(q)
            elif d > ROW_GAP[1]:
                break
        for n in sorted(sizes, reverse=True):
            if len(run) >= n:
                for j in range(len(run) - n + 1):
                    sub = run[j : j + n]
                    key = n, sum(p.score for p in sub)
                    if best is None or key > best[0]:
                        best = key, sub
                break
    return [] if best is None else list(best[1])


EDGE_RES = 0.001
EDGE_ANGLES = 72


def pad_row_edges(
    img: np.ndarray,
    cam: Camera,
    robot: np.ndarray | None = None,
    region=PAD_REGION,
    sizes=(4, 5),
    spacing: float = PAD_SPACING,
    y_range=ROW_Y,
    dx_max: float = ROW_DX,
    radial: bool = False,
):
    (x0, x1), (y0, y1) = region
    res = EDGE_RES
    top = np.stack([table_view(img[..., i], cam, PAD_TOP, region, res) for i in range(3)], axis=-1)
    valid_img = np.ones(img.shape[:2], np.float32)
    if robot is not None:
        valid_img[robot] = 0
    valid = table_view(valid_img, cam, PAD_TOP, region, res) > 0.99
    lab = lab_image(np.clip(top, 0, 255).astype(np.uint8))
    lab = cv2.GaussianBlur(lab, (0, 0), 1.0)
    g2 = np.zeros(lab.shape[:2], np.float32)
    for i in range(3):
        gx = cv2.Sobel(lab[..., i], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[..., i], cv2.CV_32F, 0, 1, ksize=3)
        g2 += gx * gx + gy * gy
    g = np.sqrt(g2)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    g[~inner] = 0.0
    med = float(np.median(g[valid])) if valid.any() else 1.0
    h, w = g.shape
    th = np.linspace(0, 2 * np.pi, EDGE_ANGLES, endpoint=False)
    r_px = PAD_R / res
    ox, oy = np.round(r_px * np.cos(th)).astype(int), np.round(r_px * np.sin(th)).astype(int)
    if radial:
        gxs, gys = [], []
        for i in range(3):
            gxs.append(cv2.Sobel(lab[..., i], cv2.CV_32F, 1, 0, ksize=3))
            gys.append(cv2.Sobel(lab[..., i], cv2.CV_32F, 0, 1, ksize=3))
        gx, gy = np.stack(gxs, -1), np.stack(gys, -1)
        maps = []
        for c_, s_ in zip(np.cos(th), np.sin(th)):
            rad = np.sqrt(((gx * c_ + gy * s_) ** 2).sum(-1))
            tan = np.sqrt(((-gx * s_ + gy * c_) ** 2).sum(-1))
            d = (rad - tan) / max(med, 1e-06)
            d[~inner] = 0.0
            maps.append(d)
        maps = np.stack(maps, 0)
        offs = [
            (np.round((r_px + dr) * np.cos(th)).astype(int), np.round((r_px + dr) * np.sin(th)).astype(int))
            for dr in (-1.0, 0.0, 1.0)
        ]
        kk = np.arange(EDGE_ANGLES)

        def rim(rows_, c):
            vals = [maps[kk[None, :], rows_[:, None] + oy_[None, :], c + ox_[None, :]] for ox_, oy_ in offs]
            return np.max(np.stack(vals, 0), axis=0).mean(axis=1)
    else:
        g = cv2.dilate(g / max(med, 1e-06), np.ones((3, 3), np.uint8))

        def rim(rows_, c):
            return g[rows_[:, None] + oy[None, :], c + ox[None, :]].mean(axis=1)

    ox, oy = np.round((r_px + 1) * np.cos(th)).astype(int), np.round((r_px + 1) * np.sin(th)).astype(int)
    rows = np.array(
        [r for r in range(h) if y_range[0] <= y1 - r * res <= y_range[1] and r + oy.min() >= 0 and (r + oy.max() < h)],
        dtype=int,
    )
    best_by_n, best = {}, None
    for n in sizes:
        xs = (np.arange(n) - (n - 1) / 2.0) * spacing
        bn = None
        for dx in np.arange(-dx_max, dx_max + 1e-09, res):
            cols = np.round((xs + dx - x0) / res).astype(int)
            if len(rows) == 0 or (cols + ox.min()).min() < 0 or (cols + ox.max()).max() >= w:
                continue
            per = np.stack([rim(rows, c) for c in cols], axis=1)
            vmin = per.min(axis=1)
            k = int(np.argmax(vmin))
            if bn is None or vmin[k] > bn[0]:
                bn = float(vmin[k]), float(dx), int(rows[k]), cols, per[k]
        if bn is None:
            continue
        best_by_n[n] = round(bn[0], 2)
        if best is None or bn[0] > best[0][0]:
            best = bn, xs
    if best is None:
        return [], best_by_n
    (v, dx, r, cols, per), xs = best
    y = y1 - r * res
    return [Pad(xy=np.array([float(x + dx), float(y)]), score=float(sv)) for x, sv in zip(xs, per)], best_by_n


PAD_HS = (
    0.004,
    0.007,
    0.01,
)
PAD_H = 0.01
MAX_CHORD = 0.06
MAX_GAP = 0.086
OPEN_MARGIN = 0.022
FOOT_H = 0.014
FINGER_T = 0.012
FOOT_CLEAR = 0.003


@lru_cache(maxsize=None)
def _boundary_cached(digit: int) -> np.ndarray:
    t = (digit_shapes()[digit] > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.concatenate([x.reshape(-1, 2) for x in cnts]).astype(float)
    return np.stack([(c[:, 0] - TPL / 2 + 0.5) * RES, (TPL / 2 - 0.5 - c[:, 1]) * RES], axis=1)


def boundary(digit: int) -> np.ndarray:
    return _boundary_cached(digit)


def jaw_opening(pts: np.ndarray, q, phi: float, pad_h: float = PAD_H):
    u = np.array([math.cos(phi), math.sin(phi)])
    v = np.array([-u[1], u[0]])
    d = pts - np.asarray(q)
    sel = np.abs(d @ v) <= pad_h
    if sel.sum() < 4:
        return None
    a = d[sel] @ u
    return float(a.max() - a.min()), float(0.5 * (a.max() + a.min()))


@lru_cache(maxsize=None)
def tile_centroid(digit: int) -> np.ndarray:
    rows, cols = np.nonzero(digit_shapes()[digit] > 0.5)
    return np.array([(cols.mean() - TPL / 2 + 0.5) * RES, (TPL / 2 - 0.5 - rows.mean()) * RES])


def lever_arm(digit: int, phi: float, q) -> float:
    v = np.array([-math.sin(phi), math.cos(phi)])
    return abs(float(v @ (tile_centroid(digit) - np.asarray(q, dtype=float))))


def contacts_inside(digit: int, phi: float, q, pad_h: float = 0.004, margin: float = 0.001) -> bool:
    u = np.array([math.cos(phi), math.sin(phi)])
    v = np.array([-u[1], u[0]])
    d = boundary(digit) - np.asarray(q, dtype=float)
    sel = np.abs(d @ v) <= pad_h
    if sel.sum() < 4:
        return False
    a, b = d[sel] @ u, np.abs(d[sel] @ v)
    lo = b[a <= a.min() + 0.0003]
    hi = b[a >= a.max() - 0.0003]
    return bool(lo.min() <= pad_h - margin and hi.min() <= pad_h - margin)


def contact_offset(digit: int, phi: float, q, pad_h: float = 0.004) -> float:
    u = np.array([math.cos(phi), math.sin(phi)])
    v = np.array([-u[1], u[0]])
    d = boundary(digit) - np.asarray(q, dtype=float)
    sel = np.abs(d @ v) <= pad_h
    if sel.sum() < 4:
        return pad_h
    a, b = d[sel] @ u, np.abs(d[sel] @ v)
    lo = b[a <= a.min() + 0.0003]
    hi = b[a >= a.max() - 0.0003]
    return float(max(lo.min(), hi.min()))


def open_gap(pts: np.ndarray, q, phi: float, chord: float) -> float | None:
    u = np.array([math.cos(phi), math.sin(phi)])
    v = np.array([-u[1], u[0]])
    d = pts - np.asarray(q)
    a, b = d @ u, d @ v
    under = np.abs(b) <= FOOT_H
    half = 0.5 * (chord + OPEN_MARGIN)
    for _ in range(40):
        hit = under & (np.abs(a) >= half - FOOT_CLEAR) & (np.abs(a) <= half + FINGER_T)
        if not hit.any():
            return 2.0 * half if 2.0 * half <= MAX_GAP else None
        half = float(np.abs(a[hit]).max()) + FOOT_CLEAR + 0.0005
        if 2.0 * half > MAX_GAP:
            return None


@lru_cache(maxsize=None)
def tile_grasps(digit: int) -> tuple:
    pts = boundary(digit)
    cands = []
    dphis = np.radians(np.arange(-10.0, 10.01, 1.0))
    for phi0 in np.radians(np.arange(0.0, 180.0, 3.0)):
        u0 = np.array([math.cos(phi0), math.sin(phi0)])
        v0 = np.array([-u0[1], u0[0]])
        for c in np.arange(-0.012, 0.0121, 0.002):
            r = jaw_opening(pts, c * v0, phi0, PAD_HS[1])
            if r is None:
                continue
            q = c * v0 + r[1] * u0
            depth, w_mid, ok = np.inf, None, True
            for ph in PAD_HS:
                ws = []
                for dp in dphis:
                    rr = jaw_opening(pts, q, phi0 + dp, ph)
                    ws.append(np.inf if rr is None else rr[0])
                ws = np.asarray(ws)
                k0 = len(ws) // 2
                w0 = ws[k0]
                if not np.isfinite(ws).all() or w0 > MAX_CHORD:
                    ok = False
                    break
                left, right = ws[: k0 + 1][::-1], ws[k0:]
                if np.any(np.diff(left) < -0.0002) or np.any(np.diff(right) < -0.0002):
                    ok = False
                    break
                depth = min(depth, float(min(left[-1], right[-1]) - w0))
                if ph == PAD_HS[-1]:
                    w_mid = float(w0)
            if not ok or depth < 0.0005:
                continue
            gap = open_gap(pts, q, phi0, w_mid)
            if gap is None:
                continue
            cost = -depth + 0.15 * float(np.linalg.norm(q))
            cands.append((cost, float(phi0), q, w_mid, depth, gap))
    cands.sort(key=lambda r: r[0])
    if not cands:
        for phi in (0.0, math.pi / 2):
            r = jaw_opening(pts, np.zeros(2), phi)
            if r is not None and r[0] <= MAX_CHORD:
                u = np.array([math.cos(phi), math.sin(phi)])
                q = r[1] * u
                gap = open_gap(pts, q, phi, r[0])
                if gap is not None:
                    cands.append((r[0], phi, q, r[0], 0.0, gap))
        cands.sort(key=lambda r: r[0])
    out = []
    for cost, phi, q, w, depth, gap in cands:
        if all(abs(math.remainder(phi - o[0], math.pi)) > math.radians(20) for o in out):
            out.append((phi, q, w, depth, gap))
        if len(out) >= 4:
            break
    return tuple(out)
