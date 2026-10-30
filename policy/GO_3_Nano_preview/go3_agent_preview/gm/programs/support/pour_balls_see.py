
from __future__ import annotations

import cv2
import numpy as np

from ...lib.perception import colour as V
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import HeadCamera, WRIST, _project_head

YELLOW = ((20, 40), (70, 255), (190, 255))
WRIST_YELLOW = ((18, 42), (60, 255), (110, 255))

CUP_Z = TABLE_Z + 0.0515
CUP_RINGS = ((0.0, 0.0268), (0.0356, 0.0316), (0.0712, 0.0361))

CUP_MIN_PX = 120
WRIST_MIN_PX = 4000
VASE_MIN_PX = 350
TABLE_MIN_PX = 5000
TABLE_DH = 24.0

TABLE_TOP = (-0.70, 0.70, -0.60, 0.50)
CUP_SPAN = (-0.50, 0.50, -0.30, 0.15)

def _stand_zone(pad=8):
    us, vs = [], []
    for x in (-0.11, 0.11):
        for y in (-0.65, -0.32):
            for z in (TABLE_Z, 1.10):
                uv = HeadCamera.project([x, y, z])
                if uv is not None:
                    us.append(uv[0])
                    vs.append(uv[1])
    if not us:
        return None
    return (int(min(us)) - pad, int(min(vs)) - pad, int(max(us)) + pad, int(max(vs)) + pad)

STAND_ZONE = _stand_zone()

VASE_REGION = (-0.47, 0.47, -0.24, 0.17)
VASE_TOP_Z = TABLE_Z + 0.1599

def _region_mask(shape, region, z0, z1):
    x0, x1, y0, y1 = region
    pts = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                uv = HeadCamera.project([x, y, z])
                if uv is not None:
                    pts.append(uv)
    mask = np.zeros(shape, np.uint8)
    if len(pts) >= 3:
        _fill_hull(mask, pts)
    return mask

def _xy_of(pix, z):
    return V.pixels_to_plane(pix, z)[:, :2].mean(axis=0)

def _fill_hull(mask, pts_uv, value=255):
    hull = cv2.convexHull(np.round(np.asarray(pts_uv, np.float32)).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, value)

def _blank_stand(mask):
    if STAND_ZONE is not None:
        u0, v0, u1, v1 = STAND_ZONE
        mask[max(0, v0):v1, max(0, u0):u1] = 0
    return mask

def arm_mask(shape, qs, dilate=7):
    from ...lib.kinematics import chain as kin

    mask = np.zeros(shape, np.uint8)
    for arm, q in qs.items():
        for corners in kin.link_box_corners(arm, q, kin.GRIP_MAX, pad=0.006).values():
            uv = [HeadCamera.project(c) for c in corners]
            uv = [p for p in uv if p is not None]
            if len(uv) >= 3:
                _fill_hull(mask, uv)
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return mask

def _home_arms():
    return {"left": np.zeros(6), "right": np.zeros(6)}

def table_model(hsv, region):
    px = hsv[region > 0].astype(np.int32)
    if len(px) < TABLE_MIN_PX:
        return None
    hist = np.bincount(px[:, 0], minlength=180)[:180].astype(float)
    hist = np.convolve(np.r_[hist[-4:], hist, hist[:4]], np.ones(9), "valid")
    h0 = int(np.argmax(hist))
    dh = np.abs((px[:, 0] - h0 + 90) % 180 - 90)
    core = px[dh <= 10]
    if len(core) < TABLE_MIN_PX:
        return None
    s = core[:, 1].astype(float)
    v = core[:, 2].astype(float)
    return {"h0": h0, "dh": TABLE_DH,
            "s_lo": float(max(12.0, 0.8 * np.quantile(s, 0.05))),
            "v_lo": float(max(10.0, 0.35 * np.quantile(v, 0.05))),
            "v_hi": float(min(255.0, np.quantile(v, 0.995) + 20.0))}

def table_pixels(hsv, model):
    h = hsv[..., 0].astype(np.int32)
    dh = np.abs((h - model["h0"] + 90) % 180 - 90)
    s, v = hsv[..., 1], hsv[..., 2]
    m = (dh <= model["dh"]) & (s >= model["s_lo"]) & (v >= model["v_lo"]) & (v <= model["v_hi"])
    return m.astype(np.uint8) * 255

def _plane_mask(shape, box, z):
    x0, x1, y0, y1 = box
    uv = [HeadCamera.project([x, y, z]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    uv = [p for p in uv if p is not None]
    mask = np.zeros(shape, np.uint8)
    if len(uv) >= 3:
        _fill_hull(mask, uv)
    return mask

def table_mask(img, hsv=None):
    h = V.hsv(img) if hsv is None else hsv
    region = _plane_mask(h.shape[:2], TABLE_TOP, TABLE_Z)
    region[arm_mask(region.shape, _home_arms()) > 0] = 0
    _blank_stand(region)
    model = table_model(h, region)
    if model is None:
        return None, np.zeros(h.shape[:2], np.uint8), None
    wood = V.clean(table_pixels(h, model), 5, 9)
    model["frac"] = float(np.count_nonzero(wood[region > 0]) / max(1, np.count_nonzero(region)))
    b = V.blobs(wood, TABLE_MIN_PX)
    if not b:
        return None, wood, model
    mask = np.zeros(wood.shape, np.uint8)
    _fill_hull(mask, b[0]["pix"])
    return mask, wood, model

def cup_silhouette(xy, project, shape):
    pts = []
    ang = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    for dz, r in CUP_RINGS:
        for a in ang:
            uv = project([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), TABLE_Z + dz])
            if uv is None:
                return None
            pts.append(uv)
    m = np.zeros(shape, np.uint8)
    _fill_hull(m, pts)
    return m

def _cup_shape(b, z):
    xy = _xy_of(b["pix"], z)
    sil = cup_silhouette(xy, HeadCamera.project, (480, 640))
    hull = np.zeros((480, 640), np.uint8)
    _fill_hull(hull, b["pix"])
    exp = 0 if sil is None else int(np.count_nonzero(sil))
    ratio = b["area"] / max(1, exp)
    solid = b["area"] / max(1, np.count_nonzero(hull))
    return xy, ratio, solid

def yellow_blob(img, z=CUP_Z):
    h = V.hsv(img)
    m = V.clean(V.mask_hsv(h, *YELLOW), 3, 5)
    m[_plane_mask(m.shape, CUP_SPAN, z) == 0] = 0
    bs = V.blobs(m, CUP_MIN_PX)
    if not bs:
        return None, 0
    best = None
    for b in bs:
        xy, ratio, solid = _cup_shape(b, z)
        b["shape"] = (round(float(ratio), 3), round(float(solid), 3))
        ok = 0.45 <= ratio <= 1.6 and solid >= 0.7
        key = (ok, b["area"])
        if best is None or key > best[0]:
            best = (key, b)
    best[1]["shape_ok"] = bool(best[0][0])
    return best[1], len(bs)

def find_cup(img, z=None):
    b, n = yellow_blob(img, CUP_Z if z is None else z)
    if b is None:
        return None
    return {"xy": _xy_of(b["pix"], CUP_Z if z is None else z), "uv": b["uv"], "area": b["area"], "n_yellow": n,
            "pix": b["pix"], "shape": b["shape"], "shape_ok": b["shape_ok"]}

def vase_blob(img, cup=None, arm_qs=None):
    tmask, wood, model = table_mask(img)
    if tmask is None:
        return None, 0, model
    keep = np.where((wood == 0) & (tmask > 0), 255, 0).astype(np.uint8)
    keep[_region_mask(keep.shape, VASE_REGION, TABLE_Z, VASE_TOP_Z) == 0] = 0
    qs = arm_qs if arm_qs is not None else _home_arms()
    keep[arm_mask(keep.shape, qs) > 0] = 0
    _blank_stand(keep)
    if cup is not None:
        cm = np.zeros(keep.shape, np.uint8)
        _fill_hull(cm, cup["pix"])
        keep[cv2.dilate(cm, np.ones((9, 9), np.uint8)) > 0] = 0
    bs = V.blobs(V.clean(keep, 3, 5), VASE_MIN_PX)
    if not bs:
        return None, 0, model
    return max(bs, key=lambda x: x["area"]), len(bs), model

_RING = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)

def render_vase(xy, prof, box):
    u0, v0, u1, v1 = box
    m = np.zeros((v1 - v0, u1 - u0), np.uint8)
    prev = None
    for z, r in [(0.0, prof[0][1])] + list(prof):
        P = np.stack([xy[0] + r * np.cos(_RING), xy[1] + r * np.sin(_RING), np.full(len(_RING), TABLE_Z + z)], 1)
        uv = _project_head(P) - np.array([u0, v0])
        if prev is not None:
            cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.r_[prev, uv]).astype(np.int32)), 1)
        prev = uv
    return m > 0

def _sil_iou(obs, known, xy, prof, box):
    sil = render_vase(xy, prof, box)
    return np.count_nonzero(sil & obs & known) / max(1, np.count_nonzero((sil | obs) & known))

def fit_profile(obs, known, prof, xy0, box, span=0.03, step=0.005):
    best = (_sil_iou(obs, known, xy0, prof, box), np.asarray(xy0, float))
    for d, k in ((step, int(round(span / step))), (step / 5, 5)):
        c = best[1]
        for dx in np.arange(-k, k + 1) * d:
            for dy in np.arange(-k, k + 1) * d:
                xy = c + np.array([dx, dy])
                sc = _sil_iou(obs, known, xy, prof, box)
                if sc > best[0]:
                    best = (sc, xy)
    return best

def vase_fits(shape, b, cup, arm_qs, cands):
    u, v, w, h = b["bbox"]
    box = (max(0, u - 40), max(0, v - 60), min(shape[1], u + w + 40), min(shape[0], v + h + 40))
    obs = np.zeros((box[3] - box[1], box[2] - box[0]), bool)
    obs[b["pix"][:, 1] - box[1], b["pix"][:, 0] - box[0]] = True
    hide = arm_mask(shape, arm_qs if arm_qs is not None else _home_arms())
    if STAND_ZONE is not None:
        u0, v0, u1, v1 = STAND_ZONE
        hide[max(0, v0):v1, max(0, u0):u1] = 255
    if cup is not None:
        cm = np.zeros(shape, np.uint8)
        _fill_hull(cm, cup["pix"])
        hide[cv2.dilate(cm, np.ones((9, 9), np.uint8)) > 0] = 255
    known = hide[box[1]:box[3], box[0]:box[2]] == 0
    return {k: float(fit_profile(obs, known, VASE_PROFILE[k], _xy_of(b["pix"], TABLE_Z + VASE_PIX_H[k]), box)[0])
            for k in cands}

PALE = ("bright", "grey", "mid")

def find_vase(img, cup=None, z=None, arm_qs=None):
    b, n, model = vase_blob(img, cup, arm_qs)
    if b is None:
        return None
    kind, feat = classify_vase(img, b["pix"])
    fits = None
    if kind in PALE:
        fits = vase_fits(img.shape[:2], b, cup, arm_qs, PALE)
        kind = max(fits, key=fits.get)
    zz = (TABLE_Z + VASE_PIX_H[kind]) if z is None else z
    return {"xy": _xy_of(b["pix"], zz), "uv": b["uv"], "area": b["area"], "kind": kind, "feat": feat, "fits": fits,
            "bbox": b["bbox"], "n_other": n, "pix": b["pix"], "table": model}

VASE_PIX_H = {"bright": 0.100, "teal": 0.090, "grey": 0.056, "dark": 0.110, "mid": 0.0788}

VASE_PROFILE = {
    "bright": ((0.00995, 0.05009), (0.02002, 0.05367), (0.03003, 0.05382), (0.04105, 0.05252), (0.05098, 0.04721), (0.06100, 0.02708),
        (0.07103, 0.02929), (0.08105, 0.03190), (0.09100, 0.03489), (0.10099, 0.03848), (0.11200, 0.04277), (0.12201, 0.04791),
        (0.13201, 0.05410), (0.142, 0.05919)),
    "teal": ((0.01101, 0.03814), (0.02098, 0.04479), (0.03201, 0.04929), (0.04198, 0.04981), (0.05304, 0.04998), (0.06302, 0.04941),
        (0.07397, 0.04677), (0.08500, 0.04330), (0.09501, 0.03830), (0.10605, 0.03308), (0.11601, 0.02901), (0.12696, 0.03101),
        (0.13701, 0.03522), (0.148, 0.03989)),
    "grey": ((0.01098, 0.05429), (0.02101, 0.07312), (0.03200, 0.08593), (0.04199, 0.09012), (0.05298, 0.08993), (0.06300, 0.08450),
        (0.07403, 0.07019), (0.08500, 0.05340), (0.09497, 0.04139), (0.10598, 0.03872), (0.11601, 0.03747), (0.12700, 0.03619),
        (0.13696, 0.03489), (0.148, 0.03342)),
    "dark": ((0.01100, 0.02559), (0.02203, 0.01618), (0.03199, 0.01972), (0.04299, 0.02180), (0.05405, 0.02264), (0.06501, 0.02270),
        (0.07597, 0.02090), (0.08601, 0.01841), (0.09701, 0.01972), (0.10799, 0.02110), (0.11901, 0.02280), (0.12900, 0.02518),
        (0.14000, 0.03002), (0.151, 0.03989)),
    "mid": ((0.01099, 0.03572), (0.02103, 0.04152), (0.03198, 0.04628), (0.04297, 0.05042), (0.05300, 0.05370), (0.06398, 0.05609),
        (0.07401, 0.05699), (0.08500, 0.05695), (0.09599, 0.05602), (0.10602, 0.05342), (0.11702, 0.05261), (0.12802, 0.05282),
        (0.13800, 0.04459), (0.149, 0.02668)),
}
VASE_H = {k: v[-1][0] for k, v in VASE_PROFILE.items()}

def vase_radius(kind, h, band=0.012):
    prof = VASE_PROFILE[kind]
    tops = np.array([t for t, _ in prof])
    bots = np.r_[0.0, tops[:-1]]
    rs = np.array([r for _, r in prof])
    h = np.atleast_1d(np.asarray(h, float))
    out = np.zeros(len(h))
    for i, hh in enumerate(h):
        sel = (tops >= hh - band) & (bots <= hh + band)
        if np.any(sel) and hh <= tops[-1] + band:
            out[i] = rs[sel].max()
    return out

def classify_vase(img, pix):
    hsv = V.hsv(img)
    us, vs = pix[:, 0], pix[:, 1]
    h, s, v = (hsv[vs, us, k].astype(float) for k in range(3))
    rgb = img[vs, us].astype(float)
    feat = {"dark": float(np.mean(v < 60)), "teal": float(np.mean((h >= 70) & (h <= 105) & (s >= 70) & (v >= 60))),
            "bg": float(np.mean(rgb[:, 2] - rgb[:, 1])), "s": float(np.mean(s)), "v": float(np.mean(v)),
            "v_std": float(np.std(v))}
    if feat["dark"] >= 0.5:
        return "dark", feat
    if feat["teal"] >= 0.5:
        return "teal", feat
    if feat["bg"] > -2.5 and feat["s"] < 8.0:
        return "grey", feat
    return ("bright" if feat["v"] >= 177.0 else "mid"), feat

def cut_by_arms(vase, arm_qs=None, ring=3):
    qs = arm_qs if arm_qs is not None else _home_arms()
    m = arm_mask((480, 640), qs)
    m = cv2.dilate(m, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8))
    u, v = vase["pix"][:, 0], vase["pix"][:, 1]
    ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
    return int(np.count_nonzero(m[v[ok], u[ok]]))

def scene_fg(img, cup, vase, arm_qs=None):
    tmask, wood, _ = table_mask(img)
    if tmask is None:
        return None
    fg = np.where((wood == 0) & (tmask > 0), 255, 0).astype(np.uint8)
    fg[arm_mask(fg.shape, arm_qs if arm_qs is not None else _home_arms()) > 0] = 0
    _blank_stand(fg)
    for ob in (cup, vase):
        if ob is not None:
            m = np.zeros(fg.shape, np.uint8)
            _fill_hull(m, ob["pix"])
            fg[cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0] = 0
    return V.clean(fg, 3, 5)

def occupied(fg, xy, r=0.045, frac=0.3):
    if fg is None:
        return False
    ang = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    uv = [HeadCamera.project([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), TABLE_Z]) for a in ang]
    uv = [p for p in uv if p is not None]
    if len(uv) < 3:
        return False
    m = np.zeros(fg.shape, np.uint8)
    _fill_hull(m, uv)
    n = np.count_nonzero(m)
    return n > 0 and np.count_nonzero(fg[m > 0]) / n >= frac

def wrist_cup(img, T_cam, prior_xy, span=0.024, step=0.004):
    h = V.hsv(img)
    m = V.clean(V.mask_hsv(h, *WRIST_YELLOW), 3, 5)
    bs = V.blobs(m, 400)
    if not bs:
        return None
    b = bs[0]
    u0, v0, w, hh = b["bbox"]
    edge = u0 <= 1 or v0 <= 1 or u0 + w >= img.shape[1] - 2 or v0 + hh >= img.shape[0] - 2
    if edge:
        return {"ok": False, "why": "edge", "area": b["area"], "edge": True}
    if b["area"] < WRIST_MIN_PX:
        return {"ok": False, "why": "small", "area": b["area"], "edge": False}
    obs = np.zeros(m.shape, np.uint8)
    _fill_hull(obs, b["pix"])
    obs = obs > 0
    proj = lambda p: WRIST.project(T_cam, p)

    def iou(xy):
        sil = cup_silhouette(xy, proj, m.shape)
        if sil is None:
            return 0.0
        sil = sil > 0
        return float(np.count_nonzero(sil & obs) / max(1, np.count_nonzero(sil | obs)))

    prior = np.asarray(prior_xy, float)
    best = (iou(prior), prior)
    for d in (step, step / 4):
        c = best[1]
        k = int(round(span / d)) if d == step else 4
        for dx in np.arange(-k, k + 1) * d:
            for dy in np.arange(-k, k + 1) * d:
                xy = c + np.array([dx, dy])
                s = iou(xy)
                if s > best[0]:
                    best = (s, xy)
    return {"ok": True, "xy": best[1], "iou": best[0], "iou0": iou(prior), "area": b["area"], "edge": edge}

def measure(img, arm_qs=None):
    cup = find_cup(img)
    return cup, find_vase(img, cup, arm_qs=arm_qs)
