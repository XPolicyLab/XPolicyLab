
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from ...lib.perception import carve as carve
from ...lib.kinematics import chain as kin
from ...lib.perception import colour as V
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import WRIST
from ...lib.geometry.camera import HeadCamera as C

TABLE = (-0.41, 0.59, -0.6, 0.5)
REGION = (-0.42, 0.56, -0.34, 0.12)
NADIR = C.project(C.POS + np.array([0.0, 0.0, -1.0]))
R_DEF = 0.031

FINGER_POLYS = (
    ((562, 284), (534, 350), (535, 365), (593, 477), (639, 479), (639, 325), (574, 286)),
    ((77, 284), (65, 286), (0, 325), (0, 479), (46, 477), (104, 365), (104, 353)),
)
_FINGERS = None

def finger_mask(shape=(480, 640)):
    global _FINGERS
    if _FINGERS is None or _FINGERS.shape != shape:
        m = np.zeros(shape, np.uint8)
        for poly in FINGER_POLYS:
            cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
        _FINGERS = m > 0
    return _FINGERS

def wood(rgb):
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = h[..., 0], h[..., 1], h[..., 2]
    return (hue >= 5) & (hue <= 13) & (sat >= 40) & (sat <= 230) & (val >= 30) & (val <= 190)

def _table_poly_mask(shape, project, T=None):
    pts = []
    for x, y in ((TABLE[0], TABLE[2]), (TABLE[1], TABLE[2]), (TABLE[1], TABLE[3]), (TABLE[0], TABLE[3])):
        p = np.array([x, y, TABLE_Z])
        uv = project(p) if T is None else project(T, p)
        if uv is None:
            return None
        pts.append(uv)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(pts)).astype(np.int32)], 1)
    return mask > 0

_HEAD_TABLE = None

def head_table_mask(shape=(480, 640)):
    global _HEAD_TABLE
    if _HEAD_TABLE is None:
        m = _table_poly_mask(shape, C.project)

        stand = np.zeros(shape, bool)
        stand[392:, 205:435] = True
        _HEAD_TABLE = m & ~stand
    return _HEAD_TABLE

def _fill_hull(mask, P):
    uv = [C.project(p) for p in P]
    uv = np.array([x for x in uv if x is not None])
    if len(uv) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)

def arm_mask(shape, arm_qs, grips=None, held=(), dilate=5, held_segs=None, extra_qs=()):
    mask = np.zeros(shape, np.uint8)
    for qs in (arm_qs, *extra_qs):
        for arm, q in qs.items():
            g = kin.GRIP_MAX if grips is None else grips.get(arm, kin.GRIP_MAX)
            for corners in kin.link_box_corners(arm, q, g, pad=0.004).values():
                _fill_hull(mask, corners)
    for arm in held:
        seg = (held_segs or {}).get(arm)
        if seg is not None:
            a, b, r = seg
            from ...lib.scene.silhouette import cylinder_points
            _fill_hull(mask, cylinder_points(np.asarray(a, float), np.asarray(b, float), r + 0.012))
            continue
        T = kin.tcp_from_link6(kin.fk(arm, arm_qs[arm]))
        p = T[:3, 3]
        for dz in (0.12, 0.06, 0.0, -0.06, -0.12):
            uv = C.project(p + np.array([0, 0, dz]))
            if uv is not None:
                cv2.circle(mask, (int(uv[0]), int(uv[1])), 55, 1, -1)
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return mask > 0

def head_blobs(rgb, unknown):
    h = V.hsv(rgb)
    fg = ~wood(rgb) & head_table_mask(rgb.shape[:2]) & (h[..., 2] >= 0) & ~unknown
    fg = ndimage.binary_opening(fg, iterations=2)
    lab, n = ndimage.label(fg)
    out = []
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(vs) < 150:
            continue
        out.append(np.stack([us, vs], axis=1).astype(float))
    return out

def _hyps(pix):
    foot = V.pixels_to_plane(pix, TABLE_Z + R_DEF)[:, :2]
    c, u, (lo, hi) = V.line_fit(foot)
    n = np.array([-u[1], u[0]])
    w = (foot - c) @ n
    lying = {"c": c + u * (lo + hi) / 2, "u": u, "len": float(hi - lo),
             "wid": float(np.percentile(w, 97) - np.percentile(w, 3))}
    vb = pix[:, 1].max()
    ub = pix[pix[:, 1] >= vb - 2, 0].mean()
    front = V.to_plane([ub, vb], TABLE_Z)[:2]
    away = front - C.POS[:2]
    away /= np.linalg.norm(away)
    center = front + away * R_DEF
    vt = pix[:, 1].min()
    ut = pix[pix[:, 1] <= vt + 2, 0].mean()
    d = C.ray(ut, vt)
    t = np.dot(center - C.POS[:2], d[:2]) / np.dot(d[:2], d[:2])
    hgt = C.POS[2] + t * d[2] - TABLE_Z
    standing = {"c": center, "h": float(hgt)}
    cov = np.cov((pix - pix.mean(0)).T)
    ev, evec = np.linalg.eigh(cov)
    axis = evec[:, 1]
    radial = NADIR - pix.mean(0)
    radial /= np.linalg.norm(radial)
    off = float(np.degrees(np.arccos(min(1.0, abs(axis @ radial)))))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    return lying, standing, off, elong

def _split2(pix):
    c = pix.mean(0)
    _, _, vt = np.linalg.svd(pix - c, full_matrices=False)
    a = (pix - c) @ vt[0]
    centers = np.percentile(a, [25, 75])
    lab = None
    for _ in range(12):
        lab = np.abs(a[:, None] - centers[None]).argmin(1)
        centers = np.array([a[lab == k].mean() if np.any(lab == k) else centers[k] for k in (0, 1)])
    return [pix[lab == k] for k in (0, 1) if np.sum(lab == k) >= 150]

def _kind(lying, standing, off, elong):
    if off > 12.0 and elong > 1.5:
        return "lying"
    if standing["h"] < 0.12 or standing["h"] > 0.30:
        return "lying"
    return "standing"

def _single(lying, standing, off, elong):
    if lying["wid"] > 0.115:
        return False
    if _kind(lying, standing, off, elong) == "standing":
        return True
    return lying["len"] <= 0.31

def _cand(pix, depth=0):
    lying, standing, off, elong = _hyps(pix)
    single = _single(lying, standing, off, elong)
    if not single and depth == 0:
        parts = _split2(pix)
        if len(parts) == 2:
            out = []
            for p in parts:
                out.extend(_cand(p, 1))
            return out
    kind = _kind(lying, standing, off, elong)
    xy = lying["c"] if kind == "lying" else standing["c"]
    return [{"kind": kind, "xy": xy, "lying": lying, "standing": standing, "off": round(off, 1),
             "elong": round(elong, 2), "n": len(pix), "merged": not single}]

def in_region(xy):
    return REGION[0] <= xy[0] <= REGION[1] and REGION[2] <= xy[1] <= REGION[3]

def head_candidates(rgb, unknown):
    out = []
    for pix in head_blobs(rgb, unknown):
        for c in _cand(pix):
            if in_region(c["lying"]["c"]) or in_region(c["standing"]["c"]):
                out.append(c)
    return out

def _cyl_points(p0, p1, r, nring=20, nlen=5):
    a = p1 - p0
    L = np.linalg.norm(a)
    a = a / max(L, 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ang = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    ring = r * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
    t = np.linspace(0, 1, nlen)
    return (p0 + t[:, None, None] * (p1 - p0)[None, None, :] + ring[None]).reshape(-1, 3)

def _project_wrist(T, P):
    c = (P - T[:3, 3]) @ T[:3, :3]
    z = -c[:, 2]
    z = np.maximum(z, 1e-4)
    return np.stack([WRIST.CX + WRIST.FX * c[:, 0] / z, WRIST.CY - WRIST.FX * c[:, 1] / z], axis=1)

class WristFit:

    def __init__(self, T_cam, blob, unknown, pad=30):
        vs, us = np.nonzero(blob)
        H, W = blob.shape
        self.box = (max(0, us.min() - pad), max(0, vs.min() - pad), min(W, us.max() + pad), min(H, vs.max() + pad))
        u0, v0, u1, v1 = self.box
        self.obs = blob[v0:v1, u0:u1]
        self.known = ~unknown[v0:v1, u0:u1]
        self.T = T_cam

    def render(self, P):
        u0, v0, u1, v1 = self.box
        uv = _project_wrist(self.T, P) - np.array([u0, v0])
        uv = np.clip(uv, -4000, 4000)
        mask = np.zeros((v1 - v0, u1 - u0), np.uint8)
        hull = cv2.convexHull(uv.astype(np.float32))
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
        return mask > 0

    def iou(self, P):
        sil = self.render(P)
        inter = np.count_nonzero(sil & self.obs & self.known)
        union = np.count_nonzero((sil | self.obs) & self.known)
        return inter / union if union else 0.0

    def fit_standing(self, x, y, h, r):
        def cost(p):
            x, y, h, r = p
            if not (0.10 < h < 0.30 and 0.022 < r < 0.045):
                return 1.0
            return -self.iou(_cyl_points(np.array([x, y, TABLE_Z]), np.array([x, y, TABLE_Z + h]), r))

        s0 = np.array([x, y, h, r])
        simplex = np.array([s0, s0 + [0.015, 0, 0, 0], s0 + [0, 0.015, 0, 0], s0 + [0, 0, 0.04, 0],
                            s0 + [0, 0, 0, 0.006]])
        res = minimize(cost, s0, method="Nelder-Mead",
                       options={"xatol": 5e-4, "fatol": 1e-3, "maxiter": 160, "initial_simplex": simplex})
        x, y, h, r = res.x
        return {"kind": "standing", "xy": np.array([x, y]), "h": float(h), "r": float(r), "iou": float(-res.fun)}

    def fit_lying(self, x, y, yaw, L, r):
        def cost(p):
            x, y, yaw, L, r = p
            if not (0.10 < L < 0.30 and 0.022 < r < 0.045):
                return 1.0
            u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            c = np.array([x, y, TABLE_Z + r])
            return -self.iou(_cyl_points(c - u * L / 2, c + u * L / 2, r))

        s0 = np.array([x, y, yaw, L, r])
        simplex = np.array([s0, s0 + [0.012, 0, 0, 0, 0], s0 + [0, 0.012, 0, 0, 0], s0 + [0, 0, 0.12, 0, 0],
                            s0 + [0, 0, 0, 0.03, 0], s0 + [0, 0, 0, 0, 0.005]])
        res = minimize(cost, s0, method="Nelder-Mead",
                       options={"xatol": 5e-4, "fatol": 1e-3, "maxiter": 220, "initial_simplex": simplex})
        x, y, yaw, L, r = res.x
        return {"kind": "lying", "xy": np.array([x, y]), "u": np.array([np.cos(yaw), np.sin(yaw)]),
                "len": float(L), "r": float(r), "iou": float(-res.fun)}

_UV = None

def _pixel_rays(shape):
    global _UV
    if _UV is None or _UV.shape[:2] != shape:
        vs, us = np.mgrid[0:shape[0], 0:shape[1]]
        _UV = np.stack([(us - WRIST.CX) / WRIST.FX, -(vs - WRIST.CY) / WRIST.FX, -np.ones(shape)], axis=-1)
    return _UV

def _fill_boxes(mask, T_cam, box_corners):
    for corners in box_corners:
        c = (corners - T_cam[:3, 3]) @ T_cam[:3, :3]
        front = -c[:, 2] > 0.02
        if front.sum() < 3:
            continue
        if not front.all():
            c = c.copy()
            c[~front, 2] = -0.02
        uv = np.stack([WRIST.CX + WRIST.FX * c[:, 0] / -c[:, 2], WRIST.CY - WRIST.FX * c[:, 1] / -c[:, 2]], axis=1)
        uv = np.clip(uv, -3000, 3000)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)

STAND_BOX = np.array([[x, y, z] for x in (-0.075, 0.075) for y in (-0.56, -0.37) for z in (TABLE_Z, 1.33)])

def wrist_unknown(shape, T_cam, gripper_open=True, other_links=()):
    unk = finger_mask(shape).copy() if gripper_open else np.zeros(shape, bool)
    d = _pixel_rays(shape) @ T_cam[:3, :3].T
    o = T_cam[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z + 0.02 - o[2]) / d[..., 2]
    x, y = o[0] + t * d[..., 0], o[1] + t * d[..., 1]

    unk |= ~((t > 0) & (x > REGION[0]) & (x < TABLE[1]) & (y > -0.42) & (y < REGION[3] + 0.08))
    m = np.zeros(shape, np.uint8)
    _fill_boxes(m, T_cam, [STAND_BOX, *other_links])
    unk |= m > 0
    return unk

def wrist_measure(rgb, T_cam, hint_xy, max_dist=0.14, h0=0.2, other_links=()):
    unk = wrist_unknown(rgb.shape[:2], T_cam, other_links=other_links)
    fg = ~wood(rgb) & ~unk & (V.hsv(rgb)[..., 2] >= 0)
    fg = ndimage.binary_opening(fg, iterations=2)
    lab, n = ndimage.label(fg)
    best = None
    others = []
    huv = WRIST.project(T_cam, np.array([hint_xy[0], hint_xy[1], TABLE_Z + 0.05]))
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(vs) < 400:
            continue
        o, d = T_cam[:3, 3], (np.stack([(us - WRIST.CX) / WRIST.FX, -(vs - WRIST.CY) / WRIST.FX,
                                        -np.ones(len(us))], axis=1) @ T_cam[:3, :3].T)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (TABLE_Z + R_DEF - o[2]) / d[:, 2]
        ok = t > 0
        if ok.sum() < 200:
            continue
        foot = (o + t[ok, None] * d[ok])[:, :2]
        dist = float(np.linalg.norm(foot.mean(0) - hint_xy))
        pix_d = 1e9 if huv is None else float(np.min(np.hypot(us - huv[0], vs - huv[1])))
        rec = (dist, i, foot, us, vs, pix_d)
        others.append(rec)
        if pix_d < 150 or dist < max_dist:
            key = (pix_d, dist)
            if best is None or key < best[0]:
                best = (key, rec)
    if best is None:
        return None
    dist, i, foot, us, vs, pix_d = best[1]
    blob = lab == i
    c, u, (lo, hi) = V.line_fit(foot)
    nrm = np.array([-u[1], u[0]])
    w = (foot - c) @ nrm
    length, width = float(hi - lo), float(np.percentile(w, 97) - np.percentile(w, 3))
    aspect = length / max(width, 1e-3)
    edge = bool(np.any(blob & ndimage.binary_dilation(unk, iterations=3)) or us.min() <= 2 or us.max() >= 637
                or vs.min() <= 2 or vs.max() >= 477)
    f = WristFit(T_cam, blob, unk)
    r0 = float(np.clip(width / 2, 0.024, 0.042))
    yaw0 = float(np.arctan2(u[1], u[0]))
    mid = c + u * (lo + hi) / 2
    fl = f.fit_lying(mid[0], mid[1], yaw0, float(np.clip(length, 0.12, 0.28)), r0)
    fs = f.fit_standing(foot[:, 0].mean(), foot[:, 1].mean(), h0, r0)
    b0 = standing_base(T_cam, us, vs)
    if b0 is not None:
        fs2 = f.fit_standing(b0[0], b0[1], h0, r0)
        if fs2["iou"] > fs["iou"]:
            fs = fs2
    kind = "lying" if fl["iou"] >= fs["iou"] else "standing"
    if aspect > 2.4 and length > 0.11:
        kind = "lying"
    elif aspect < 1.45:
        kind = "standing"
    fl["wide"], fl["taper"] = taper(T_cam, blob, fl)
    res = dict(fl if kind == "lying" else fs)
    res.update(aspect=round(aspect, 2), width=round(width, 3), length=round(length, 3), n=int(blob.sum()),
               edge=edge, iou_l=round(fl["iou"], 3), iou_s=round(fs["iou"], 3), dist=round(dist, 3),
               alt={"lying": fl, "standing": fs})
    nb = []
    for d2, j, foot2, us2, vs2, _ in others:
        if j == i or d2 > 0.30:
            continue
        nb.append(_quick_fit(T_cam, lab == j, unk, foot2, h0))
    res["neighbours"] = [x for x in nb if x is not None]
    return res

def taper(T_cam, blob, fl):
    vs, us = np.nonzero(blob)
    o = T_cam[:3, 3]
    d = np.stack([(us - WRIST.CX) / WRIST.FX, -(vs - WRIST.CY) / WRIST.FX, -np.ones(len(us))], 1) @ T_cam[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z + fl["r"] - o[2]) / d[:, 2]
    ok = t > 0
    if ok.sum() < 100:
        return 0, 0.0
    P = (o + t[ok, None] * d[ok])[:, :2]
    u = np.asarray(fl["u"])
    nrm = np.array([-u[1], u[0]])
    a = (P - fl["xy"]) @ u
    w = (P - fl["xy"]) @ nrm
    L = fl["len"]
    widths = []
    for lo, hi in ((-L / 2, -L / 4), (L / 4, L / 2)):
        sel = (a >= lo) & (a < hi)
        widths.append(float(np.percentile(w[sel], 95) - np.percentile(w[sel], 5)) if sel.sum() > 30 else 0.0)
    if min(widths) <= 0.0:
        return 0, 0.0
    return (1 if widths[1] > widths[0] else -1), abs(widths[1] - widths[0])

def standing_base(T_cam, us, vs):
    nadir = T_cam[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    if nadir[2] >= -1e-3:
        return None
    vp = np.array([WRIST.CX + WRIST.FX * nadir[0] / -nadir[2], WRIST.CY - WRIST.FX * nadir[1] / -nadir[2]])
    pix = np.stack([us, vs], axis=1).astype(float)
    cen = pix.mean(0)
    dvec = vp - cen
    if np.linalg.norm(dvec) < 1e-6:
        return None
    dvec /= np.linalg.norm(dvec)
    proj = (pix - cen) @ dvec
    near = pix[proj >= np.percentile(proj, 97)]
    uv = near.mean(0)
    p = WRIST.to_plane(T_cam, uv[0], uv[1], TABLE_Z)
    away = p[:2] - T_cam[:2, 3]
    if np.linalg.norm(away) > 1e-6:
        away /= np.linalg.norm(away)
    return p[:2] + away * R_DEF

def _quick_fit(T_cam, blob, unk, foot, h0):
    c, u, (lo, hi) = V.line_fit(foot)
    nrm = np.array([-u[1], u[0]])
    w = (foot - c) @ nrm
    length, width = float(hi - lo), float(np.percentile(w, 97) - np.percentile(w, 3))
    if width < 0.015:
        return None
    f = WristFit(T_cam, blob, unk)
    r0 = float(np.clip(width / 2, 0.024, 0.042))
    mid = c + u * (lo + hi) / 2
    fl = f.fit_lying(mid[0], mid[1], float(np.arctan2(u[1], u[0])), float(np.clip(length, 0.12, 0.28)), r0)
    fs = f.fit_standing(foot[:, 0].mean(), foot[:, 1].mean(), h0, r0)
    best = fl if fl["iou"] >= fs["iou"] else fs
    best["truncated"] = bool(np.any(blob & ndimage.binary_dilation(unk, iterations=3)))
    return best

def side_measure(rgb, T_cam, hint_xy, h0, r0=R_DEF, other_links=()):
    unk = wrist_unknown(rgb.shape[:2], T_cam, other_links=other_links)
    fg = ~wood(rgb) & ~unk & (V.hsv(rgb)[..., 2] >= 0)
    fg = ndimage.binary_opening(fg, iterations=2)
    lab, n = ndimage.label(fg)
    huv = WRIST.project(T_cam, np.array([hint_xy[0], hint_xy[1], TABLE_Z + 0.06]))
    if huv is None:
        return None
    best = None
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(vs) < 600:
            continue
        pix_d = float(np.min(np.hypot(us - huv[0], vs - huv[1])))
        if pix_d < 120 and (best is None or pix_d < best[0]):
            best = (pix_d, i)
    if best is None:
        return None
    blob = lab == best[1]
    f = WristFit(T_cam, blob, unk)
    fs = f.fit_standing(hint_xy[0], hint_xy[1], h0, r0)
    for dx, dy in ((0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)):
        if fs["iou"] > 0.85:
            break
        f2 = f.fit_standing(hint_xy[0] + dx, hint_xy[1] + dy, h0, r0)
        if f2["iou"] > fs["iou"]:
            fs = f2

    vs, us = np.nonzero(blob)
    o, d = T_cam[:3, 3], (np.stack([(us - WRIST.CX) / WRIST.FX, -(vs - WRIST.CY) / WRIST.FX,
                                    -np.ones(len(us))], axis=1) @ T_cam[:3, :3].T)
    fl = {"iou": 0.0}
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z + R_DEF - o[2]) / d[:, 2]
    ok = t > 0
    if ok.sum() > 100:
        foot = (o + t[ok, None] * d[ok])[:, :2]
        c, u, (lo, hi) = V.line_fit(foot)
        mid = c + u * (lo + hi) / 2
        for yaw0 in (float(np.arctan2(u[1], u[0])), float(np.arctan2(u[1], u[0])) + np.pi / 2):
            f3 = f.fit_lying(mid[0], mid[1], yaw0, float(np.clip(hi - lo, 0.14, 0.26)), r0)
            if f3["iou"] > fl["iou"]:
                fl = f3
    fs.update(n=int(blob.sum()), pix_d=round(best[0], 1), iou_s=round(fs["iou"], 3), iou_l=round(fl["iou"], 3),
              edge=bool(np.any(blob & ndimage.binary_dilation(unk, iterations=3))))
    return fs
