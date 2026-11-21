
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ...lib.perception import colour as V
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import HeadCamera as C

_image_lo, _image_hi = 70, 466

BIAS_BOTTLE = np.array([-0.0044, +0.0036])
BIAS_CUP = np.array([-0.0069, +0.0047])

BIAS_R_BOTTLE = np.array([+0.0040, -0.0076])

CUP_MODELS = {"cup_a": (0.0366, 0.0822), "cup_b": (0.0488, 0.0731), "cup_c": (0.0351, 0.0774)}

BOTTLE_MIN_PX = 250

BOTTLE_MIN_H = 55
CUP_BAND_X = 0.0756
CUP_MAX_W = 75

TABLE_FIT_MIN = 0.80
SCENE_MAX_BLOBS = 5

def wood(rgb):
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, Vv = h[..., 0].astype(np.int16), h[..., 1].astype(np.int16), h[..., 2].astype(np.int16)
    return ((H <= 25) | (H >= 168)) & (S >= 40) & (S <= 220) & (Vv >= 45) & (Vv <= 200)

_ARM_MASKS = {}

def arm_mask(arm_qs, shape=(480, 640)):
    key = tuple(np.round(np.concatenate([arm_qs[a] for a in sorted(arm_qs)]), 3))
    if key not in _ARM_MASKS:
        from .pb_see import arm_mask as _arm_mask

        _ARM_MASKS[key] = _arm_mask(shape, arm_qs, dilate=7)
    return _ARM_MASKS[key]

def components(rgb, min_px=BOTTLE_MIN_PX, arm_qs=None):
    fg = (~wood(rgb)).astype(np.uint8)
    if arm_qs is not None:
        fg[arm_mask(arm_qs, rgb.shape[:2])] = 0
    else:
        fg = V.blank_zones(fg)
    fg[:_image_lo] = 0
    fg[_image_hi:] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lab, n = ndimage.label(fg)
    out = []
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(vs) < min_px:
            continue
        u0, u1, v0, v1 = int(us.min()), int(us.max()), int(vs.min()), int(vs.max())
        if u0 <= 1 or u1 >= 638 or v0 <= _image_lo + 2:
            continue
        if (u1 - u0) > 200 or (v1 - v0) > 200:
            continue
        out.append(
            {
                "n": int(len(vs)),
                "bbox": (u0, v0, u1, v1),
                "uv": (float(us.mean()), float(vs.mean())),
                "w": u1 - u0,
                "h": v1 - v0,
            }
        )
    out.sort(key=lambda b: -b["n"])
    return out

def axis_xy(blob):
    u0, v0, u1, v1 = blob["bbox"]
    ub = blob["uv"][0]
    p = C.to_plane(ub, v1, TABLE_Z)[:2]
    height = 0.0
    for _ in range(5):
        c = C.R.T @ (np.array([p[0], p[1], TABLE_Z + 0.05]) - C.POS)
        depth = float(-c[2])
        height = blob["h"] * depth / C.FX
        zc = TABLE_Z + 0.5 * height
        p2 = C.to_plane(blob["uv"][0], blob["uv"][1], zc)[:2]
        if np.linalg.norm(p2 - p) < 1e-5:
            p = p2
            break
        p = p2
    return p, float(height)

def table_fit(rgb, arm_qs=None):
    valid = _table_quad(rgb.shape[:2]).copy()
    valid[:_image_lo] = False
    valid[_image_hi:] = False
    if arm_qs is not None:
        valid &= ~arm_mask(arm_qs, rgb.shape[:2])
    return float(wood(rgb)[valid].mean())

def measure(rgb, arm_qs=None):
    bottle = cup = None
    cs = components(rgb, arm_qs=arm_qs)
    for b in cs:
        p, h = axis_xy(b)


        if b["h"] >= BOTTLE_MIN_H and b["w"] >= 25:
            cand = dict(xy=p - BIAS_BOTTLE, height=h, n=b["n"], w=b["w"], px_h=b["h"])
            if bottle is None or cand["n"] > bottle["n"]:
                bottle = cand
        elif abs(p[0]) < CUP_BAND_X and b["w"] < CUP_MAX_W:
            cand = dict(xy=p - BIAS_CUP, height=h, n=b["n"], w=b["w"], px_h=b["h"])
            if cup is None or cand["n"] > cup["n"]:
                cup = cand
    return {"bottle": bottle, "cup": cup, "n_components": len(cs)}

_REL_OBJ = {
    "bottle": dict(xw=((-0.3105, -0.0445), (0.0445, 0.3107)), yw=(-0.2106, 0.0606)),
    "cup": dict(xw=((-0.0555, 0.0555),), yw=(-0.1107, 0.0106)),
}
_TABLE_QUAD = {}

def _table_quad(shape, xmax=0.58, ymin=-0.42, ymax=0.42):
    if shape not in _TABLE_QUAD:
        from ...lib.geometry.transforms import TABLE_Z as _TZ

        pts = [C.project([x, y, _TZ]) for x, y in ((-xmax, ymin), (xmax, ymin), (xmax, ymax), (-xmax, ymax))]
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        _TABLE_QUAD[shape] = m.astype(bool)
    return _TABLE_QUAD[shape]

def rel_foreground(rgb, arm_qs=None):
    tm = _table_quad(rgb.shape[:2])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    valid = tm & ~arm_mask(arm_qs, rgb.shape[:2]) if arm_qs is not None else tm
    small = cv2.resize(lab, (160, 120), interpolation=cv2.INTER_AREA)
    vsm = cv2.resize(valid.astype(np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST).astype(bool)
    bg = np.empty_like(small)
    for k in range(3):
        ch = small[..., k].copy()
        ch[~vsm] = float(np.median(ch[vsm]))
        bg[..., k] = ndimage.median_filter(ch, size=25)
    bg = cv2.resize(bg, (640, 480), interpolation=cv2.INTER_LINEAR)
    d = np.linalg.norm(lab - bg, axis=-1)
    f = np.clip((d - 6.0) / 18.0, 0.0, 1.0)
    f[~valid] = 0.0
    return cv2.GaussianBlur(f, (5, 5), 0.0)

_SIL = {}

def _sil(x, y, r, h, shape=(480, 640)):
    key = (round(x, 3), round(y, 3), r, h)
    if key not in _SIL:


        m = np.zeros(shape, np.uint8)
        ts = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
        for z0, z1 in ((TABLE_Z, TABLE_Z + h),):
            lo = [C.project([x + r * np.cos(t), y + r * np.sin(t), z0]) for t in ts]
            hi = [C.project([x + r * np.cos(t), y + r * np.sin(t), z1]) for t in ts]
            cv2.fillPoly(m, [np.round(np.array(lo)).astype(np.int32)], 1)
            cv2.fillPoly(m, [np.round(np.array(hi)).astype(np.int32)], 1)
            for t0, t1 in zip(ts[:-1], ts[1:]):
                quad = [C.project([x + r * np.cos(t0), y + r * np.sin(t0), z0]),
                        C.project([x + r * np.cos(t0), y + r * np.sin(t0), z1]),
                        C.project([x + r * np.cos(t1), y + r * np.sin(t1), z1]),
                        C.project([x + r * np.cos(t1), y + r * np.sin(t1), z0])]
                cv2.fillPoly(m, [np.round(np.array(quad)).astype(np.int32)], 1)
        _SIL[key] = (m.astype(bool), (cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0) & ~m.astype(bool))
    return _SIL[key]

def _bottle_masks(x, y, shape=(480, 640)):
    from .plc_plan import BOTTLE_H, bottle_radius

    key = ("b", round(x, 3), round(y, 3))
    if key not in _SIL:
        body = np.zeros(shape, np.uint8)
        neck = np.zeros(shape, np.uint8)
        outer = np.zeros(shape, np.uint8)
        for s in np.linspace(0.0, BOTTLE_H, 40):
            r = bottle_radius(s)
            c = C.project([x, y, TABLE_Z + s])
            e = C.project([x + r, y, TABLE_Z + s])
            rp = max(1, int(round(abs(e[0] - c[0]))))
            cv2.circle(neck if s >= 0.110 else body, (int(round(c[0])), int(round(c[1]))), rp, 1, -1)
            if s >= 0.120:
                e2 = C.project([x + 0.034, y, TABLE_Z + s])
                cv2.circle(outer, (int(round(c[0])), int(round(c[1]))), max(1, int(round(abs(e2[0] - c[0])))), 1, -1)
        bodyb = body.astype(bool)
        neckb = neck.astype(bool) & ~bodyb
        _SIL[key] = (bodyb, neckb, outer.astype(bool) & ~neckb & ~bodyb)
    return _SIL[key]

def candidates(fg, name, step=0.010, n=4, min_sep=0.06):
    fgb = (np.asarray(fg, float) > 0.5) if getattr(fg, "dtype", None) is not bool else fg
    o = _REL_OBJ[name]

    def sc(x, y):
        b, nn, r = _bottle_masks(x, y)
        return float(fgb[b].mean() + fgb[nn].mean() - fgb[r].mean())

    scored = []
    for (x0, x1) in o["xw"]:
        for x in np.arange(x0, x1 + 1e-9, step):
            for y in np.arange(o["yw"][0] - 0.02, o["yw"][1] + 0.02 + 1e-9, step):
                scored.append((sc(x, y), x, y))
    scored.sort(reverse=True)
    out = []
    for s_, x, y in scored:
        if all(np.hypot(x - px, y - py) > min_sep for _, px, py, _ in out):
            out.append((s_, x, y, None))
        if len(out) >= n:
            break
    return [(s_, np.array([x, y])) for s_, x, y, _ in out]

def _search(fg, name, step=0.010):
    fgb = (np.asarray(fg, float) > 0.5) if fg.dtype != bool else fg
    o = _REL_OBJ[name]

    def score(x, y):
        b, n, r = _bottle_masks(x, y)
        return float(fgb[b].mean() + fgb[n].mean() - fgb[r].mean())

    best = (-9.0, 0.0, 0.0)
    for (x0, x1) in o["xw"]:
        for x in np.arange(x0, x1 + 1e-9, step):
            for y in np.arange(o["yw"][0] - 0.02, o["yw"][1] + 0.02 + 1e-9, step):
                s = score(x, y)
                if s > best[0]:
                    best = (s, x, y)
    _, x0, y0 = best
    for dx in np.arange(-0.008, 0.0081, 0.002):
        for dy in np.arange(-0.008, 0.0081, 0.002):
            s = score(x0 + dx, y0 + dy)
            if s > best[0]:
                best = (s, x0 + dx, y0 + dy)
    return best

def measure_template(rgb, arm_qs=None):
    from .plc_plan import BOTTLE_H

    fg = rel_foreground(rgb, arm_qs)
    out = {"bottle": None, "cup": None, "bottle_alts": [], "n_components": int((fg > 0.5).sum() > 0)}

    for s_, xy in candidates(fg, "bottle", n=6)[1:]:
        out["bottle_alts"].append(xy - BIAS_R_BOTTLE)
    s, x, y = _search(fg, "bottle")
    if s > 0.05:
        out["bottle"] = dict(xy=np.array([x, y]) - BIAS_R_BOTTLE, height=BOTTLE_H,
                             n=int((fg > 0.2).sum()), w=0, px_h=0, score=s)
    s, mdl, x, y = _search_cup_models(fg)
    if s > 0.05:
        out["cup"] = dict(xy=np.array([x, y]), height=CUP_MODELS[mdl][1], n=int((fg > 0.2).sum()),
                          w=0, px_h=0, score=s, model=mdl)
    return out

def _search_cup_models(fg):
    fgb = (np.asarray(fg, float) > 0.5) if fg.dtype != bool else fg

    def sc(x, y, r, h):
        m, ring = _sil(x, y, r, h)
        return float(fgb[m].mean() - 0.6 * fgb[ring].mean())

    best = (-9.0, None, 0.0, 0.0)
    for mdl, (r, h) in CUP_MODELS.items():

        for x in np.arange(-0.0657, 0.0658, 0.010):
            for y in np.arange(-0.1157, 0.0157, 0.010):
                v = sc(x, y, r, h)
                if v > best[0]:
                    best = (v, mdl, x, y)
    for half, st in ((0.006, 0.002), (0.002, 0.001)):
        _, mdl, x0, y0 = best
        r, h = CUP_MODELS[mdl]
        for dx in np.arange(-half, half + 1e-9, st):
            for dy in np.arange(-half, half + 1e-9, st):
                v = sc(x0 + dx, y0 + dy, r, h)
                if v > best[0]:
                    best = (v, mdl, x0 + dx, y0 + dy)
    return best

WRIST_NECK_MIN = 20.0
_R_NECK = 0.0146

def wrist_neck_score(labs, Ts, xy):
    from ...lib.geometry.camera import WRIST

    best = -1.0
    for arm in labs:
        rows = []
        for z in np.arange(0.135, 0.2051, 0.005):
            p = np.array([xy[0], xy[1], TABLE_Z + z])
            c = WRIST.project(Ts[arm], p)
            if c is None:
                rows = None
                break
            depth = float(-(Ts[arm][:3, :3].T @ (p - Ts[arm][:3, 3]))[2])
            w = WRIST.FX * _R_NECK / max(depth, 1e-6)
            u, v = c
            if depth < 0.08 or not (3 * w < u < 640 - 3 * w and 5 < v < 475):
                rows = None
                break
            vi, lab = int(round(v)), labs[arm]
            ctr = lab[vi, int(u - 0.5 * w):int(u + 0.5 * w) + 1].reshape(-1, 3)
            lf = lab[vi, int(u - 2.6 * w):int(u - 1.6 * w) + 1].reshape(-1, 3)
            rf = lab[vi, int(u + 1.6 * w):int(u + 2.6 * w) + 1].reshape(-1, 3)
            if not (len(ctr) and len(lf) and len(rf)):
                rows = None
                break
            mc = np.median(ctr, 0)
            rows.append(min(np.linalg.norm(mc - np.median(lf, 0)), np.linalg.norm(mc - np.median(rf, 0))))
        if rows is not None and len(rows) >= 8:
            best = max(best, float(np.median(rows)))
    return best

def verify_bottle(meas, wrist_rgbs, wrist_Ts):
    if meas.get("bottle") is None:
        return None
    labs = {a: cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            for a, rgb in wrist_rgbs.items()}
    prim = np.asarray(meas["bottle"]["xy"], float)
    alts = [np.asarray(a, float) for a in meas.get("bottle_alts", [])]
    s0 = wrist_neck_score(labs, wrist_Ts, prim)
    sa = [wrist_neck_score(labs, wrist_Ts, a) for a in alts]
    chosen = s0
    if 0.0 <= s0 < WRIST_NECK_MIN and sa and max(sa) >= WRIST_NECK_MIN:
        k = int(np.argmax(sa))
        meas["bottle"] = dict(meas["bottle"], xy=alts[k], wrist_overruled=True)
        chosen = sa[k]
        alts[k], sa[k] = prim, s0
    order = sorted(range(len(alts)), key=lambda i: (0 if sa[i] >= WRIST_NECK_MIN else (1 if sa[i] < 0 else 2), -sa[i]))
    meas["bottle_alts"] = [alts[i] for i in order]
    return {"head_first": s0, "chosen": chosen, "alts": [sa[i] for i in order]}


HIDDEN_MIN = 0.15

def hiding_arms(meas, arm_qs, confirmed=False, shape=(480, 640)):
    from .pb_see import arm_mask as _arm_mask

    hyps = ([meas["bottle"]["xy"]] if meas.get("bottle") is not None else [])
    if not confirmed:
        hyps += list(meas.get("bottle_alts", []))
    out = []
    for a in sorted(arm_qs):
        am = _arm_mask(shape, {a: arm_qs[a]}, dilate=7)
        for xy in hyps:
            body, neck, _ = _bottle_masks(float(xy[0]), float(xy[1]), shape)
            sil = body | neck
            if sil.any() and float((sil & am).sum()) / float(sil.sum()) >= HIDDEN_MIN:
                out.append(a)
                break
    return out

def measure_scene(rgb, arm_qs=None):
    fit = table_fit(rgb, arm_qs)
    if fit >= TABLE_FIT_MIN:
        meas = measure(rgb, arm_qs=arm_qs)
        if meas["n_components"] <= SCENE_MAX_BLOBS:
            return dict(meas, path="components", table_fit=fit)
    return dict(measure_template(rgb, arm_qs=arm_qs), path="template", table_fit=fit)
