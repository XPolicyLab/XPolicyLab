from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..planning import kinematics as K
from .cameras import Camera

YAW_SPAN = math.radians(25.0)
YAW_STEP = math.radians(5.0)
FINE_YAW_DEG = -3, -2, -1, 1, 2, 3
REGION = (-0.56, 0.56), (-0.32, 0.16)
SHADES = 0.3, 0.45, 0.6, 0.8, 1.0, 1.2
CH_BINS = 24
V_BINS = 8
MIN_AREA = 40
MAX_PER_COMP = 3
REST_FRAC = 0.5
CAT_MIN_SCORE = 0.5
CAT_MIN_OVERLAP = 0.3
LLR_CLIP = 3.0
RING_PX = 4
W_COVER, W_IOU, W_RING = (
    0.5,
    0.5,
    0.5,
)
W_HIST = 2.0
W_CONT = 0.0
BG_KSIZE = 31
FG_CHROMA = 0.035
FG_BRIGHT = 1.5
FG_DARK = 0.35


@dataclass
class Detection:
    x: float
    y: float
    yaw: float
    tag: str
    score: float
    colour: float
    iou: float
    uv: tuple
    extra: dict = field(default_factory=dict)


def _srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def features(rgb: np.ndarray):
    im = np.asarray(rgb, dtype=np.float32)
    s = im.sum(axis=-1) + 3.0
    return (im[..., 0] + 1.0) / s, (im[..., 1] + 1.0) / s, im.max(axis=-1)


def _bins(r, g, v):
    br = np.clip((r * CH_BINS).astype(int), 0, CH_BINS - 1)
    bg = np.clip((g * CH_BINS).astype(int), 0, CH_BINS - 1)
    bv = np.clip((v / 256.0 * V_BINS).astype(int), 0, V_BINS - 1)
    return (br * CH_BINS + bg) * V_BINS + bv


def _smooth(h: np.ndarray) -> np.ndarray:
    import cv2

    h3 = h.reshape(CH_BINS, CH_BINS, V_BINS).astype(np.float32)
    out = np.empty_like(h3)
    for k in range(V_BINS):
        out[:, :, k] = cv2.GaussianBlur(h3[:, :, k], (3, 3), 0.8)
    out = (
        0.25 * np.roll(out, 1, axis=2) * (np.arange(V_BINS) > 0)
        + 0.5 * out
        + 0.25 * np.roll(out, -1, axis=2) * (np.arange(V_BINS) < V_BINS - 1)
    )
    return out.reshape(-1)


def colour_hist(colors: np.ndarray, shades=SHADES) -> np.ndarray:
    lin = _srgb_to_lin(colors)
    h = np.zeros(CH_BINS * CH_BINS * V_BINS)
    for k in shades:
        c = _lin_to_srgb(lin * k)
        r, g, v = features(c)
        np.add.at(h, _bins(r, g, v), 1.0)
    h = _smooth(h)
    return h / h.sum()


def image_hist(r, g, v, valid) -> np.ndarray:
    h = np.bincount(_bins(r[valid], g[valid], v[valid]), minlength=CH_BINS * CH_BINS * V_BINS).astype(float)
    h = _smooth(h)
    return h / max(h.sum(), 1.0)


def region_mask(cam: Camera, region=REGION, z: float = K.TABLE_Z, shape=(480, 640)) -> np.ndarray:
    import cv2

    (x0, x1), (y0, y1) = region
    uv, _ = cam.project(np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]))
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
    return m > 0


def foreground(rgb: np.ndarray, valid: np.ndarray, robot: np.ndarray | None = None):
    import cv2

    src = rgb
    if robot is not None and robot.any():
        src = cv2.inpaint(rgb, robot.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    H, W = rgb.shape[:2]
    small = cv2.resize(src, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, BG_KSIZE), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    im = rgb.astype(np.float32)
    ch = im / (im.sum(axis=2, keepdims=True) + 3.0)
    chb = bg / (bg.sum(axis=2, keepdims=True) + 3.0)
    dc = np.linalg.norm(ch - chb, axis=2)
    ratio = (im.sum(axis=2) + 3.0) / (bg.sum(axis=2) + 3.0)
    dark = im.max(axis=2) < 20.0
    fg = (dc > FG_CHROMA) & ~dark | (ratio > FG_BRIGHT) | (ratio < FG_DARK)
    fg &= valid
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return fg > 0, {"dc_med": float(np.median(dc[valid])) if valid.any() else None}


BG_DIFF = 14.0
BG_CHROMA = 0.03
BG_BRIGHT = 1.15
BG_DARK = 0.55


def shadow_mask(rgb: np.ndarray, bg: np.ndarray, dark: float = FG_DARK) -> np.ndarray:
    im = rgb.astype(np.float32)
    b = bg.astype(np.float32)
    s_im = im.sum(axis=2) + 3.0
    s_b = b.sum(axis=2) + 3.0
    ratio = s_im / s_b
    dc = np.linalg.norm(im / s_im[..., None] - b / s_b[..., None], axis=2)
    return (dc <= BG_CHROMA) & (ratio < 1.0) & (ratio > dark)


def foreground_bg(rgb: np.ndarray, bg: np.ndarray, valid: np.ndarray, drop_shadows: bool = False):
    import cv2

    im = rgb.astype(np.float32)
    b = bg.astype(np.float32)
    s_im = im.sum(axis=2) + 3.0
    s_b = b.sum(axis=2) + 3.0
    ratio = s_im / s_b
    dc = np.linalg.norm(im / s_im[..., None] - b / s_b[..., None], axis=2)
    diff = np.abs(im - b).max(axis=2)
    fg = (diff > BG_DIFF) & ((dc > BG_CHROMA) | (ratio > BG_BRIGHT) | (ratio < BG_DARK)) & valid
    if drop_shadows:
        fg &= ~shadow_mask(rgb, bg)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return fg > 0, {"bg": True, "fg_px": int(fg.sum())}


def obstacle_mask(
    rgb: np.ndarray,
    cam: Camera,
    robot,
    bg: np.ndarray,
    target_pts: np.ndarray,
    *,
    region=REGION,
    z0: float = K.TABLE_Z,
    excl_px: int = 5,
    shadow_reach: float = 0.0,
    shadow_ratio: float = 0.85,
) -> np.ndarray:
    import cv2

    H, W = rgb.shape[:2]
    valid = region_mask(cam, region, z0, (H, W))
    if robot is not None:
        valid &= ~robot
    fg, _ = foreground_bg(rgb, bg, valid)
    fg &= ~shadow_mask(rgb, bg)
    sil, (u0, v0) = silhouette(cam, target_pts)
    if sil is not None:
        ex = np.zeros((H, W), np.uint8)
        th, tw = sil.shape
        uu, vv = max(0, u0), max(0, v0)
        ex[vv : v0 + th, uu : u0 + tw] = sil[vv - v0 :, uu - u0 :][: H - vv, : W - uu] > 0
        ex = cv2.dilate(ex, np.ones((2 * excl_px + 1, 2 * excl_px + 1), np.uint8))
        fg &= ex == 0
    if shadow_reach > 0.0 and fg.any():
        from scipy.spatial import cKDTree

        vs, us = np.nonzero(fg)
        im = rgb[vs, us].astype(np.float32).sum(axis=1) + 3.0
        b = bg[vs, us].astype(np.float32).sum(axis=1) + 3.0
        dark = im / b < shadow_ratio
        if dark.any():
            g = cam.on_plane_z(np.stack([us[dark], vs[dark]], axis=1).astype(float), z0)[:, :2]
            fin = np.all(np.isfinite(g), axis=1)
            dd = np.full(len(g), np.inf)
            dd[fin] = cKDTree(target_pts[:, :2]).query(g[fin])[0]
            own = dd <= shadow_reach
            fg[vs[dark][own], us[dark][own]] = False
    return fg


def obstacle_points(
    cam: Camera,
    mask: np.ndarray,
    target_pts: np.ndarray,
    *,
    z_obj: float = 0.03,
    z0: float = K.TABLE_Z,
    step: int = 2,
    excl_m: float = 0.02,
) -> np.ndarray:
    vs, us = np.nonzero(mask[::step, ::step])
    if len(us) == 0:
        return np.zeros((0, 2))
    g = cam.on_plane_z(np.stack([us * step, vs * step], axis=1).astype(float), z0 + z_obj)[:, :2]
    g = g[np.all(np.isfinite(g), axis=1)]
    tp = target_pts[:: max(1, len(target_pts) // 300), :2]
    if len(g):
        d = np.sqrt(((g[:, None, :] - tp[None, :, :]) ** 2).sum(-1)).min(axis=1)
        g = g[d > excl_m]
    return g


def xy_blocker(points: np.ndarray, clear: float = 0.01):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)

    def blocked(w: np.ndarray) -> bool:
        if len(w) == 0 or len(pts) == 0:
            return False
        lo = w[:, :2].min(axis=0) - clear
        hi = w[:, :2].max(axis=0) + clear
        near = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
        if len(near) == 0:
            return False
        d2 = ((w[:, None, :2] - near[None, :, :]) ** 2).sum(-1)
        return bool(d2.min() < clear * clear)

    return blocked


def silhouette(cam: Camera, pts: np.ndarray, dilate: int = 0):
    import cv2

    uv, depth = cam.project(pts)
    uv = uv[depth > 0]
    if len(uv) == 0:
        return None, None
    u0, v0 = np.floor(uv.min(axis=0)).astype(int) - 3
    u1, v1 = np.ceil(uv.max(axis=0)).astype(int) + 3
    m = np.zeros(((v1 - v0 + 1) * 2, (u1 - u0 + 1) * 2), np.uint8)
    iu = np.clip(np.round((uv[:, 0] - u0) * 2).astype(int), 0, m.shape[1] - 1)
    iv = np.clip(np.round((uv[:, 1] - v0) * 2).astype(int), 0, m.shape[0] - 1)
    m[iv, iu] = 1
    area = max(1.0, float(cv2.contourArea(cv2.convexHull(np.stack([iu, iv], 1).astype(np.int32)))))
    spacing = math.sqrt(area / max(1.0, 0.5 * len(uv)))
    k = int(np.clip(round(1.6 * spacing), 2, 15)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if dilate > 0:
        m = cv2.dilate(m, np.ones((2 * dilate * 2 + 1, 2 * dilate * 2 + 1), np.uint8))
    m = cv2.resize(m, (u1 - u0 + 1, v1 - v0 + 1), interpolation=cv2.INTER_AREA)
    return (m > 0).astype(np.uint8), (int(u0), int(v0))


def on_category(
    cam: Camera,
    asset,
    dets,
    instances,
    category: str,
    *,
    min_score: float = CAT_MIN_SCORE,
    min_overlap: float = CAT_MIN_OVERLAP,
    shape=(480, 640),
) -> list[bool]:
    import cv2

    masks = [
        cv2.dilate(np.asarray(i.mask, np.uint8), np.ones((5, 5), np.uint8)) > 0
        for i in instances
        if i.category == category and i.score >= min_score
    ]
    out = []
    for d in dets:
        pts = asset.posed(d.tag, d.x, d.y, d.yaw, n=800)
        sil, corner = silhouette(cam, pts)
        ok = False
        if sil is not None and masks:
            u0, v0 = corner
            H, W = shape
            m = np.zeros((H, W), bool)
            th, tw = sil.shape
            uu, vv = max(0, u0), max(0, v0)
            m[vv : v0 + th, uu : u0 + tw] = sil[vv - v0 :, uu - u0 :][: H - vv, : W - uu] > 0
            a = float(m.sum())
            ok = a > 0 and any(float((m & im).sum()) / a >= min_overlap for im in masks)
        out.append(ok)
    return out


def prefer_category(cam: Camera, asset, dets, instances, category: str, refind=None, refine: bool = False, **kw):
    on = on_category(cam, asset, dets, instances, category, **kw)
    extra = []
    if refind is not None:
        min_score = kw.get("min_score", CAT_MIN_SCORE)
        for inst in instances:
            if inst.category != category or inst.score < min_score:
                continue
            if not refine and any(on_category(cam, asset, [d], [inst], category, **kw)[0] for d in dets):
                continue
            new = refind(np.asarray(inst.mask, bool))
            if new and on_category(cam, asset, new[:1], [inst], category, **kw)[0]:
                extra.append(new[0])
    first = sorted([d for d, o in zip(dets, on) if o] + extra, key=lambda d: -d.score)
    out = first + [d for d, o in zip(dets, on) if not o]
    return out


def _components(mask: np.ndarray, min_area: int):
    import cv2

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y, w, h = stats[i, :4]
            out.append((i, (int(x), int(y), int(w), int(h)), int(stats[i, cv2.CC_STAT_AREA]), cents[i]))
    return lab, out


def _match(win: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    import cv2

    if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
        return None
    return cv2.matchTemplate(win.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCORR)


import cv2 as _cv2

_RING_K = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))


def find(
    rgb: np.ndarray,
    cam: Camera,
    asset,
    robot: np.ndarray | None = None,
    *,
    region=REGION,
    z0: float = K.TABLE_Z,
    yaw_span: float = YAW_SPAN,
    yaw_step: float = YAW_STEP,
    colour_w: float = 1.0,
    n_best: int = 5,
    bg: np.ndarray | None = None,
    hist=None,
    generic=None,
    w_hist: float | None = None,
    w_cont: float | None = None,
    fg_mask: np.ndarray | None = None,
) -> list[Detection]:
    w_hist = W_HIST if w_hist is None else w_hist
    w_cont = W_CONT if w_cont is None else w_cont
    H, W = rgb.shape[:2]
    valid = region_mask(cam, region, z0, (H, W))
    if robot is not None:
        valid &= ~robot
    feats = features(rgb)
    r, g, v = feats
    if fg_mask is not None:
        fg, tm = np.asarray(fg_mask, bool) & valid, {"bg": "mask"}
    elif bg is not None:
        fg, tm = foreground_bg(rgb, bg, valid, drop_shadows=False)
    else:
        fg, tm = foreground(rgb, valid, robot)
    if hist is not None and generic is not None:
        llr = np.clip(np.log((hist + 0.0001) / (generic + 0.0001)), -LLR_CLIP, LLR_CLIP).astype(np.float32)
        tm["colour"] = "rendered"
    else:
        h_t = colour_hist(asset.colors)
        h_i = image_hist(r, g, v, valid)
        llr = np.clip(np.log((h_t + 0.0001) / (h_i + 0.0001)), -LLR_CLIP, LLR_CLIP).astype(np.float32)
        tm["colour"] = "albedo"
    llr_map = llr[_bins(r, g, v)]
    llr_map[~valid] = 0.0
    lab, comps = _components(fg, MIN_AREA)
    yaws = np.arange(-yaw_span, yaw_span + 1e-09, yaw_step)
    dets = []
    fg_all = fg
    jobs = [(lab == ci, (bx, by, bw, bh), area, cent, fg_all, 0) for ci, (bx, by, bw, bh), area, cent in comps]
    while jobs:
        comp, (bx, by, bw, bh), area, cent, fg, depth = jobs.pop(0)
        best = None
        for pass_ in (0, 1):
            pairs = (
                [(tg, yw) for tg in asset.places for yw in yaws]
                if pass_ == 0
                else []
                if best is None
                else [(best[4], best[3] + math.radians(dd)) for dd in FINE_YAW_DEG]
            )
            for tag, yaw in pairs:
                hgt = asset.height(tag)
                c3 = cam.on_plane_z(np.array([cent]), z0 + 0.5 * hgt)[0]
                if not np.all(np.isfinite(c3)):
                    continue
                pts0 = asset.posed(tag, 0.0, 0.0, float(yaw), z0, n=1500)
                off = pts0[:, :2].mean(axis=0)
                pts = pts0 + np.array([c3[0] - off[0], c3[1] - off[1], 0.0])
                tmpl, (tu0, tv0) = silhouette(cam, pts)
                if tmpl is None:
                    continue
                tmpl = np.pad(tmpl, RING_PX)
                tu0, tv0 = tu0 - RING_PX, tv0 - RING_PX
                ring = _cv2.dilate(tmpl, _RING_K) - tmpl
                a_r = float(ring.sum())
                th, tw = tmpl.shape
                u0 = max(0, min(bx, tu0) - 24)
                v0 = max(0, min(by, tv0) - 24)
                u1 = min(W, max(bx + bw, tu0 + tw) + 24)
                v1 = min(H, max(by + bh, tv0 + th) + 24)
                fgw = fg[v0:v1, u0:u1].astype(np.float32)
                cw = comp[v0:v1, u0:u1].astype(np.float32)
                lw = llr_map[v0:v1, u0:u1]
                a_t = float(tmpl.sum())
                s_fg = _match(fgw, tmpl)
                if s_fg is None:
                    continue
                s_c = _match(cw, tmpl)
                s_l = _match(lw, tmpl)
                inter = s_fg
                iou = s_c / (a_t + float(cw.sum()) - s_c + 1e-06)
                cover = inter / a_t
                col = s_l / a_t
                s_r = _match(fgw, ring)
                ring_bg = 1.0 - s_r / max(a_r, 1.0)
                score = colour_w * col / LLR_CLIP + W_COVER * cover + W_IOU * iou + W_RING * ring_bg
                k = int(np.argmax(score))
                dv, du = np.unravel_index(k, score.shape)
                if best is None or score[dv, du] > best[0]:
                    su = u0 + du - tu0
                    sv = v0 + dv - tv0
                    c_img, _ = cam.project(np.array([[c3[0], c3[1], z0 + 0.5 * hgt]]))
                    c_new = cam.on_plane_z(c_img + np.array([[su, sv]]), z0 + 0.5 * hgt)[0]
                    x = float(c_new[0] - off[0])
                    y = float(c_new[1] - off[1])
                    best = (
                        float(score[dv, du]),
                        x,
                        y,
                        float(yaw),
                        tag,
                        float(col[dv, du]),
                        float(iou[dv, du]),
                        float(cover[dv, du]),
                        (tmpl, u0 + du, v0 + dv),
                    )
        if best is not None:
            s, x, y, yaw, tag, col, iou, cover, (tm_b, uu, vv) = best
            bc = 0.0
            cont = 0.0
            if w_hist and hist is not None:
                th, tw = tm_b.shape
                win = slice(vv, vv + th), slice(uu, uu + tw)
                crop = slice(0, max(0, min(th, H - vv))), slice(0, max(0, min(tw, W - uu)))
                m = np.zeros((H, W), bool)
                m[win] = tm_b[crop] > 0
                m &= fg & valid
                if m.sum() >= 10:
                    h_c = image_hist(r, g, v, m)
                    bc = float(np.sqrt(h_c * hist).sum())
                    if w_cont:
                        ring_b = np.zeros((H, W), bool)
                        ring_b[win] = (_cv2.dilate(tm_b, _RING_K) - tm_b)[crop] > 0
                        ring_b &= valid
                        mr = ring_b & fg
                        if mr.sum() >= 10:
                            h_r = image_hist(r, g, v, mr)
                            cont = float(mr.sum()) / float(ring_b.sum()) * float(np.sqrt(h_r * h_c).sum())
                s += w_hist * bc - w_cont * cont
            dets.append(
                Detection(
                    x,
                    y,
                    yaw,
                    tag,
                    s,
                    col,
                    iou,
                    (float(cent[0]), float(cent[1])),
                    {"cover": cover, "area": area, "bbox": (bx, by, bw, bh), "bc": bc, "cont": cont, "depth": depth},
                )
            )
            if depth + 1 < MAX_PER_COMP:
                th, tw = tm_b.shape
                cut = np.zeros((H, W), np.uint8)
                cut[vv : vv + th, uu : uu + tw] = tm_b[: max(0, min(th, H - vv)), : max(0, min(tw, W - uu))]
                cut = _cv2.dilate(cut, _RING_K) > 0
                rest = comp & ~cut
                n_rest = int(rest.sum())
                if n_rest >= max(MIN_AREA, REST_FRAC * float(tm_b.sum())):
                    vs_, us_ = np.nonzero(rest)
                    bb = int(us_.min()), int(vs_.min()), int(us_.max() - us_.min() + 1), int(vs_.max() - vs_.min() + 1)
                    jobs.insert(0, (rest, bb, n_rest, (float(us_.mean()), float(vs_.mean())), fg & ~cut, depth + 1))
    dets.sort(key=lambda d: -d.score)
    return dets[:n_best] if n_best else dets
