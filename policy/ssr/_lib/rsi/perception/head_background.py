from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning.kinematics import TABLE_Z
from .cameras import Camera

DEV_T = 0.3
BRIGHT_T = 0.35
DARK_T = -2.2
BG_HALF = 64
BG_STEP = 8
BG_MIN_N = 12
SMOOTH_HALF = 4, 24
NEAR_PX = 15
SHADE_K = -0.25
WOOD_A_MAX = 32
WOOD_B_MAX = 34
WIDE_A_MAX = 34
WIDE_B_MAX = 40
SCARCE_FRAC = 0.0
CORR_SRC = "wood"
CHECK_ROWS = 215, 400


def _lin(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, 255.0) / 255.0, 2.2) + 0.002


def _lab(img: np.ndarray):
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0


def wood_mask(img: np.ndarray, a_max: float | None = None, b_max: float | None = None) -> np.ndarray:
    L, a, b = _lab(img)
    a_max = WOOD_A_MAX if a_max is None else a_max
    b_max = WOOD_B_MAX if b_max is None else b_max
    return (a > 4) & (a < a_max) & (b > 4) & (b < b_max) & (L > 20) & (L < 175)


def shaded_wood(img: np.ndarray) -> np.ndarray:
    L, a, b = _lab(img)
    chroma = np.hypot(a, b)
    return (a > 2.0) & (b > 1.0) & (chroma > 3.0) & (np.abs(b - a) < 12.0) & (L < 175.0)


def row_background(
    lin: np.ndarray, use: np.ndarray, half: int = BG_HALF, step: int = BG_STEP, min_n: int = BG_MIN_N
) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    H, W, _ = lin.shape
    out = np.full(lin.shape, np.nan, np.float32)
    rows = np.nonzero(use.any(axis=1))[0]
    if len(rows) == 0:
        return out
    cols = np.arange(step // 2, W, step)
    pad = np.full((len(rows), W + 2 * half, 3), np.inf, np.float32)
    pad[:, half : half + W] = np.where(use[rows][..., None], lin[rows], np.inf)
    win = sliding_window_view(pad, 2 * half + 1, axis=1)[:, cols]
    n = np.count_nonzero(np.isfinite(win[:, :, 0, :]), axis=-1)
    srt = np.sort(win, axis=-1)
    idx = np.clip((n - 1) // 2, 0, None)[:, :, None, None]
    med = np.take_along_axis(srt, np.broadcast_to(idx, srt.shape[:3] + (1,)), axis=-1)[..., 0]
    ok = n >= min_n
    xs = np.arange(W)
    have = np.zeros(H, bool)
    for i, r in enumerate(rows):
        if not ok[i].any():
            continue
        for ch in range(3):
            out[r, :, ch] = np.interp(xs, cols[ok[i]], med[i, ok[i], ch])
        have[r] = True
    if not have.any():
        return out
    hr = np.nonzero(have)[0]
    for r in rows:
        if not have[r]:
            out[r] = out[hr[np.argmin(np.abs(hr - r))]]
    return out


def basket_colour_mask(img: np.ndarray, masks: list) -> np.ndarray:
    L, a, b = _lab(img)
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    out = np.zeros(img.shape[:2], bool)
    for m in masks:
        if not np.any(m):
            continue
        cm = float(np.median(chroma[m]))
        if cm > 20.0:
            hm = math.atan2(float(np.median(b[m])), float(np.median(a[m])))
            dh = np.abs(np.angle(np.exp(1j * (hue - hm))))
            like = (dh < math.radians(35.0)) & (chroma > 0.35 * cm)
        else:
            like = (chroma < 14.0) & (L > 115.0)
        out |= m & like
    shade = shaded_wood(img)
    for m in masks:
        out |= m & shade
    return out


@dataclass
class Foreground:
    mask: np.ndarray
    dev: np.ndarray
    kk: np.ndarray
    match: float
    background: np.ndarray


def shadow_shift(li: np.ndarray, lb: np.ndarray, use: np.ndarray, min_n: int = 400) -> np.ndarray | None:
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(li / lb)
    k = lr.mean(axis=2)
    sel = use & np.isfinite(k) & (k < -0.15) & (k > -1.5)
    if np.count_nonzero(sel) < min_n:
        return None
    kk = -k[sel]
    d = lr[sel] - k[sel][:, None]
    return (d * kk[:, None]).sum(axis=0) / float((kk * kk).sum())


def _difference(
    li: np.ndarray, lb: np.ndarray, dev_t: float, bright_t: float, dark_t: float, shift: np.ndarray | None = None
):
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(li / lb)
    k = lr.mean(axis=2)
    d = lr - k[..., None]
    if shift is not None:
        d -= np.clip(-k, 0.0, None)[..., None] * shift
    dev = np.sqrt((d * d).sum(axis=2))
    known = np.isfinite(k)
    fg = known & ((dev > dev_t) | (k > bright_t) | (k < dark_t))
    return fg, np.where(known, dev, 0.0), np.where(known, k, 0.0)


def smooth_correction(
    li: np.ndarray,
    lb: np.ndarray,
    use: np.ndarray,
    half: tuple = SMOOTH_HALF,
    step: int = BG_STEP // 2,
    min_n: int = 40,
) -> np.ndarray:
    import cv2
    from numpy.lib.stride_tricks import sliding_window_view

    H, W, _ = li.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(li / lb)
    ok = use & np.all(np.isfinite(lr), axis=2)
    hr, hc = half
    pad = np.full((H + 2 * hr, W + 2 * hc, 3), np.inf, np.float32)
    pad[hr : hr + H, hc : hc + W] = np.where(ok[..., None], lr, np.inf)
    rs = np.arange(step // 2, H, step)
    cs = np.arange(step // 2, W, 2 * step)
    win = sliding_window_view(pad, (2 * hr + 1, 2 * hc + 1), axis=(0, 1))[step // 2 :: step, step // 2 :: 2 * step]
    win = win.copy().reshape(len(rs), len(cs), 3, -1)
    m = win.shape[-1]
    unknown = ~np.isfinite(win[:, :, 0, :])
    n = m - np.count_nonzero(unknown, axis=-1)
    c = (m - 1) // 2
    lo = unknown & (np.cumsum(unknown, axis=-1) <= (c - (n - 1) // 2)[..., None])
    np.copyto(win, np.float32(-np.inf), where=lo[:, :, None, :])
    win.partition(c, axis=-1)
    med = np.where((n >= min_n)[..., None], win[..., c], 0.0).astype(np.float32)
    out = cv2.resize(med, (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def foreground(
    img: np.ndarray,
    valid: np.ndarray | None = None,
    basket_masks: list | None = None,
    ignore: np.ndarray | None = None,
    dev_t: float = DEV_T,
    bright_t: float = BRIGHT_T,
    dark_t: float = DARK_T,
) -> Foreground | None:
    import cv2

    if valid is None:
        valid = np.ones(img.shape[:2], bool)
    bmask = np.zeros(img.shape[:2], bool)
    for m in basket_masks or []:
        bmask |= m
    valid0 = valid
    if ignore is not None:
        valid = valid & ~ignore
    src = img
    li = _lin(src.astype(np.float32))
    wood = wood_mask(img) & valid & ~bmask
    table = valid & ~bmask
    wide = wood_mask(img, WIDE_A_MAX, WIDE_B_MAX) & table
    k = np.ones((1, 2 * BG_HALF + 1), np.float32)
    n_tab = cv2.filter2D(table.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    scarce = cv2.filter2D(wood.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT) < SCARCE_FRAC * n_tab

    def row_bg(use_n, use_w):
        lb_n = row_background(li, use_n)
        if not np.any(scarce):
            return lb_n
        lb_w = row_background(li, use_w)
        return np.where(scarce[..., None] & np.isfinite(lb_w), lb_w, lb_n)

    lb = row_bg(wood, wide)
    fg, dev, kk = _difference(li, lb, dev_t, bright_t, dark_t)
    near = cv2.dilate(fg.astype(np.uint8), np.ones((NEAR_PX, NEAR_PX), np.uint8)) > 0
    far = wood & ~near
    lb_row = row_bg(far, wide & ~near)
    shift = shadow_shift(li, lb_row, far)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_row = np.log(li / lb_row).mean(axis=2)
    lit = far & (k_row > SHADE_K)
    corr_src = {"wood": wood, "far": far, "lit": lit, "woodlit": wood & (k_row > SHADE_K)}[CORR_SRC]
    corr = smooth_correction(li, lb_row, corr_src, half=SMOOTH_HALF)
    lb = lb_row * np.exp(corr)
    fg, dev, kk = _difference(li, lb, dev_t, bright_t, dark_t, shift)
    fg_row, _, _ = _difference(li, lb_row, dev_t, bright_t, dark_t, shift)
    fg &= fg_row
    if np.any(bmask):
        fg = np.where(bmask, fg & ~basket_colour_mask(img, basket_masks or []), fg)
    if ignore is not None:
        strip = kk < dark_t
        fg = np.where(ignore, strip, fg)
    fg &= valid0
    r0, r1 = CHECK_ROWS
    chk = valid[r0:r1] & ~bmask[r0:r1]
    tab = ~fg[r0:r1] & chk
    close = (dev[r0:r1] < 0.15) & (np.abs(kk[r0:r1]) < 0.5)
    match = float(np.count_nonzero(close & tab) / max(1, np.count_nonzero(chk)))
    return Foreground(fg, dev, kk, match, lb)


def clean(mask: np.ndarray, open_px: int = 1, close_px: int = 2) -> np.ndarray:
    import cv2

    m = mask.astype(np.uint8)
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2))
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2))
    return m > 0


def drop_row_streaks(mask: np.ndarray, max_h: int = 4, min_ratio: float = 3.0, min_len: int = 0) -> np.ndarray:
    import cv2

    out = mask.astype(bool)
    for attached in (False, True):
        if attached and (not min_len):
            break
        if attached:
            thick = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max_h + 1, 1), np.uint8)) > 0
            cand = out & ~thick
        else:
            cand = out
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        w, h = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
        drop = (w >= min_ratio * h) & (w >= min_len if attached else h <= max_h)
        drop[0] = False
        out = out & ~(drop[lab] & cand)
    return out


def table_polygon_mask(cam: Camera, shape, region, z: float = TABLE_Z) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1) = region
    poly = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    uv, d = cam.project(poly)
    m = np.zeros(shape[:2], np.uint8)
    if np.all(d > 0):
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m > 0


@dataclass
class Region:
    mask: np.ndarray
    area: int
    bbox: tuple
    uv: tuple
    xy: np.ndarray
    length: float = 0.0
    width: float = 0.0
    yaw: float = 0.0
    height: float = 0.0
    hole: float = 0.0
    color: dict = field(default_factory=dict)
    touches: tuple = ()
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "area": self.area,
            "uv": [round(float(v), 1) for v in self.uv],
            "xy": [round(float(v), 4) for v in self.xy],
            "L": round(self.length, 4),
            "W": round(self.width, 4),
            "yaw": round(math.degrees(self.yaw), 1),
            "H": round(self.height, 4),
            "hole": round(self.hole, 3),
            "color": {k: round(v, 3) if isinstance(v, float) else v for k, v in self.color.items()},
            "touches": list(self.touches),
            **{k: v for k, v in self.extra.items() if not isinstance(v, np.ndarray)},
        }


def components(mask: np.ndarray, min_area: int = 40, link_px: int = 2):
    import cv2

    m = mask.astype(np.uint8)
    if link_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_px + 1,) * 2)
        joined = cv2.dilate(m, k)
    else:
        joined = m
    n, lab, _, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for i in range(1, n):
        comp = (lab == i) & (m > 0)
        a = int(np.count_nonzero(comp))
        if a >= min_area:
            out.append(comp)
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out > 0


def footprint_rect(cam: Camera, mask: np.ndarray, z: float = TABLE_Z):
    import cv2

    rows, cols = np.nonzero(mask)
    if len(rows) < 5:
        return None
    uv = np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)
    p = cam.on_plane_z(uv, z)[:, :2]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 5:
        return None
    (cx, cy), (a, b), ang = cv2.minAreaRect(p.astype(np.float32))
    ang = math.radians(ang)
    if a < b:
        a, b = b, a
        ang += math.pi / 2
    return np.array([cx, cy]), (float(a), float(b)), math.remainder(ang, math.pi)


def colour_stats(img: np.ndarray, mask: np.ndarray) -> dict:
    import cv2

    px = img[mask]
    if len(px) == 0:
        return {}
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0] * 100.0 / 255.0
    a = lab[:, 1] - 128.0
    b = lab[:, 2] - 128.0
    chroma = np.hypot(a, b)
    hue_ab = np.degrees(np.arctan2(b, a)) % 360.0
    return {
        "blue": float(np.mean((chroma > 8.0) & (hue_ab > 230.0) & (hue_ab < 320.0))),
        "L50": float(np.median(L)),
        "L10": float(np.percentile(L, 10)),
        "L90": float(np.percentile(L, 90)),
        "a": float(np.median(a)),
        "b": float(np.median(b)),
        "chroma": float(np.median(chroma)),
        "chroma90": float(np.percentile(chroma, 90)),
        "dark": float(np.mean(L < 25.0)),
        "bright": float(np.mean(L > 70.0)),
        "sat": float(np.median(hsv[:, 1])) / 255.0,
        "hue": float(np.median(hsv[:, 0])) * 2.0,
    }
