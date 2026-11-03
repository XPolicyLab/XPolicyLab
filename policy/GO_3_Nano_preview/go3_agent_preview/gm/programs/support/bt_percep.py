
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from ...lib.kinematics import chain as kin
from ...lib.geometry.transforms import rot_x, rot_z
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import HeadCamera as C

def _box(sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    return np.array([[a * hx, b * hy, c * hz] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])

PILLAR_DIMS = (0.0381, 0.0619, 0.0378)
LONG_BOARD_DIMS = (0.3748, 0.0655, 0.020)
SHORT_BOARD_DIMS = (0.2437, 0.0655, 0.020)
PLATE_DIMS = (0.0966, 0.0504, 0.014)
ROOF_BASE, ROOF_H, ROOF_LEN = 0.0606, 0.0375, 0.046
_ROOF_TRI = ((-0.0303, -0.0125), (0.0303, -0.0125), (0.0, 0.025))

SHAPES = {
    "pillar": (_box(*PILLAR_DIMS), np.eye(3), PILLAR_DIMS[2] / 2),
    "long_board": (_box(*LONG_BOARD_DIMS), np.eye(3), LONG_BOARD_DIMS[2] / 2),
    "short_board": (_box(*SHORT_BOARD_DIMS), np.eye(3), SHORT_BOARD_DIMS[2] / 2),
    "plate": (_box(*PLATE_DIMS), np.eye(3), PLATE_DIMS[2] / 2),

    "roof": (np.array([[x, y, z] for x, y in _ROOF_TRI for z in (-ROOF_LEN / 2, ROOF_LEN / 2)]),
             rot_x(np.pi / 2), 0.0125),
}
HEIGHT = {"pillar": PILLAR_DIMS[2], "long_board": 0.02, "short_board": 0.02, "plate": PLATE_DIMS[2],
          "roof": ROOF_H}

STATIC_DC = ((205, 405, 435, 480), (0, 140, 60, 222))

def world_vertices(kind, x, y, yaw, z_support=TABLE_Z):
    V, R0, h0 = SHAPES[kind]
    R = rot_z(yaw) @ R0
    return V @ R.T + np.array([x, y, z_support + h0])

PIX = 0.5

def project(P):
    c = (P - C.POS) @ C.R
    z = -c[:, 2]
    return np.stack([C.CX + C.FX * c[:, 0] / z, C.CY - C.FX * c[:, 1] / z], axis=1) - PIX

def class_masks(rgb):
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = h[..., 0].astype(np.int16), h[..., 1].astype(np.int16), h[..., 2].astype(np.int16)
    creamy = (H >= 15) & (H <= 26)
    board = creamy & (S >= 12) & (S < 60) & (V > 200)
    plate = (H >= 14) & (H <= 26) & (S >= 60) & (V > 150)
    roof = (H >= 30) & (H <= 50) & (S >= 100) & (V > 60)
    white = (S < 40) & (V > 90) & ~(creamy & (S >= 14))
    wood = ((H <= 25) | (H >= 168)) & (S >= 40) & (S <= 220) & (V >= 45) & (V <= 205) & ~plate
    return {"pillar": white, "board": board, "plate": plate, "roof": roof, "wood": wood}

def arm_mask(arm, q, g=1.0, shape=(480, 640), pad=0.0):
    m = np.zeros(shape, np.uint8)
    fr = kin.fk_chain(arm, q)
    P = [kin.base_T(arm)[:3, 3]] + [T[:3, 3] for T in fr]
    segs = ((0, 1, 0.05), (1, 2, 0.05), (2, 3, 0.04), (3, 4, 0.04), (4, 5, 0.045), (5, 6, 0.045))
    for i, j, r in segs:
        for t in np.linspace(0, 1, 12):
            p = P[i] + t * (P[j] - P[i])
            c = C.R.T @ (p - C.POS)
            if c[2] >= -0.05:
                continue
            u, v = C.CX + C.FX * c[0] / -c[2], C.CY - C.FX * c[1] / -c[2]
            if -200 < u < 840 and -200 < v < 680:
                cv2.circle(m, (int(round(u)), int(round(v))), int(np.ceil(C.FX * (r + pad) / -c[2])), 255, -1)
    T6 = fr[-1]
    gap = kin.unit_to_width(g)
    parts = [np.array([[x, y, z] for x in (0.0, 0.11) for y in (-0.062, 0.062) for z in (-0.042, 0.042)])]
    for s in (-1, 1):
        yc = s * (gap / 2 + 0.006)
        parts.append(np.array([[x, yc + d, z] for x in (0.08, kin.FINGER_TIP_X + 0.004)
                               for d in (-0.008, 0.008) for z in (-0.013, 0.013)]))
    for V in parts:
        W = V @ T6[:3, :3].T + T6[:3, 3]
        c = (W - C.POS) @ C.R
        if np.any(c[:, 2] >= -0.02):
            continue
        uv = project(W)
        if np.all(np.abs(uv) < 4000):
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    return m > 0

def occluders(qs, gs=None, dilate=4):
    m = np.zeros((480, 640), bool)
    for arm, q in qs.items():
        m |= arm_mask(arm, q, 1.0 if gs is None else gs[arm])
    for u0, v0, u1, v1 in STATIC_DC:
        m[v0:v1, u0:u1] = True
    if dilate:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    return m

_SHIFT = 4
_SCALE = 1 << _SHIFT

def render(polys_uv, roi, out=None):
    u0, v0, w, h = roi
    img = np.zeros((h, w), np.uint8) if out is None else out
    for uv in polys_uv:
        hull = cv2.convexHull(((uv - (u0, v0)) * _SCALE).astype(np.int32))
        cv2.fillConvexPoly(img, hull, 255, lineType=cv2.LINE_AA, shift=_SHIFT)
    return img.astype(np.float32) / 255.0

TOP_ONLY = ("plate",)

def top_vertices(kind, x, y, yaw, z_support=TABLE_Z):
    V = world_vertices(kind, x, y, yaw, z_support)
    return V[V[:, 2] > V[:, 2].max() - 1e-6]

class Fit:

    def __init__(self, kind, target, weight, roi, z_support=TABLE_Z, fixed=()):
        self.kind, self.roi, self.z = kind, roi, z_support
        u0, v0, w, h = roi
        t = target[v0:v0 + h, u0:u0 + w].astype(np.float32)
        self.T = cv2.GaussianBlur(t, (0, 0), 0.8)
        self.W = weight[v0:v0 + h, u0:u0 + w].astype(np.float32)
        self.fixed = list(fixed)
        self.top = kind in TOP_ONLY

    def polys(self, params, top=False):
        k = len(params) // 3
        f = top_vertices if top else world_vertices
        return [project(f(self.kind, *params[3 * i:3 * i + 3], self.z)) for i in range(k)]

    def loss(self, params):
        if self.top:
            Pt = render(self.polys(params, True) + self.fixed, self.roi)
            Ph = render(self.polys(params), self.roi)
            side = np.clip(Ph - Pt, 0.0, 1.0)
            return float(np.sum(self.W * (1.0 - side) * (Pt - self.T) ** 2))
        P = render(self.polys(params) + self.fixed, self.roi)
        return float(np.sum(self.W * (P - self.T) ** 2))

    def run(self, x0, yaw_period=np.pi, yaw_window=None):
        x0 = np.asarray(x0, float)
        scale = np.tile([0.004, 0.004, np.deg2rad(4.0)], len(x0) // 3)
        k = len(x0) // 3

        def f(z):
            x = x0 + z * scale
            pen = 0.0
            if yaw_window is not None:
                c, hw = yaw_window
                for i in range(k):
                    d = abs((x[3 * i + 2] - c + yaw_period / 2) % yaw_period - yaw_period / 2)
                    pen += 1e4 * max(0.0, d - hw) ** 2
            return self.loss(x) + pen
        res = minimize(f, np.zeros_like(x0), method="Powell",
                       options={"xtol": 0.05, "ftol": 1e-4, "maxfev": 600})
        x = x0 + res.x * scale
        for i in range(len(x) // 3):
            x[3 * i + 2] = (x[3 * i + 2] + yaw_period / 2) % yaw_period - yaw_period / 2
        return x, float(res.fun)

def components(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out.append({"area": int(st[i, cv2.CC_STAT_AREA]), "bbox": tuple(int(v) for v in st[i, :4]),
                        "mask": lab == i, "uv": cen[i]})
    out.sort(key=lambda c: -c["area"])
    return out

def backproject(uv, z):
    uv = np.asarray(uv, float) + PIX
    d = np.stack([(uv[:, 0] - C.CX) / C.FX, -(uv[:, 1] - C.CY) / C.FX, -np.ones(len(uv))], axis=1) @ C.R.T
    t = (z - C.POS[2]) / d[:, 2]
    return C.POS + t[:, None] * d

def init_guess(comp, kind, z_support=TABLE_Z):
    vs, us = np.nonzero(comp["mask"])
    pts = backproject(np.stack([us, vs], axis=1).astype(float), z_support + HEIGHT[kind])[:, :2]
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ang = np.arctan2(vt[0][1], vt[0][0])

    return c, ang

def expected_area(kind, x, y, yaw=0.0, z_support=TABLE_Z):
    uv = project(world_vertices(kind, x, y, yaw, z_support))
    return float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32))))

def roi_of(bbox, pad=14):
    u, v, w, h = bbox
    u0, v0 = max(0, u - pad), max(0, v - pad)
    u1, v1 = min(640, u + w + pad), min(480, v + h + pad)
    return (u0, v0, u1 - u0, v1 - v0)

def fit_component(kind, comp, masks, dc, k=1, yaws=None, z_support=TABLE_Z, period=np.pi, init=None,
                  yaw_window=None):
    target = comp["mask"]
    weight = ~dc & (masks["wood"] | target | masks["board"] if kind != "long_board" else ~dc)

    roi = roi_of(comp["bbox"])
    fit = Fit(kind, target, weight, roi, z_support)
    if init is None:
        c, ang = init_guess(comp, kind, z_support)
        if k == 1:
            starts = [[c[0], c[1], y] for y in (yaws if yaws is not None else (ang,))]
        else:
            vs, us = np.nonzero(target)
            pts = backproject(np.stack([us, vs], axis=1).astype(float), z_support + HEIGHT[kind])[:, :2]
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
            _, lab, cen = cv2.kmeans(pts.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            starts = []
            for y in (yaws if yaws is not None else (0.0,)):
                s = []
                for i in range(k):
                    s += [float(cen[i][0]), float(cen[i][1]), y]
                starts.append(s)
    else:
        starts = [init]
    best = None
    for s in starts:
        x, f = fit.run(s, period, yaw_window)
        if best is None or f < best[1]:
            best = (x, f)
    x, f = best
    norm = f / max(1.0, float(target.sum()))
    return [tuple(x[3 * i:3 * i + 3]) for i in range(k)], norm

def measure_scene(rgb, qs, gs=None):
    masks = class_masks(rgb)
    dc = occluders(qs, gs)
    out = {"pillars": [], "plate": None, "roof": None, "long_board": None, "short_board": None,
           "notes": []}

    for kind, key in (("plate", "plate"), ("roof", "roof")):
        comps = [c for c in components(masks[key] & ~dc, 40) if _in_region(c, kind)]
        dcx = dc | _others(masks, key)
        yaws = [np.pi / 2 + d for d in np.deg2rad((-10, 0, 10))] if kind == "plate" else list(np.deg2rad((-10, 0, 10)))
        best = None
        for comp in comps[:3]:
            for c in comps:
                if c is not comp and np.linalg.norm(c["uv"] - comp["uv"]) < 45:
                    comp = _merge(comp, c)
            c0, _ = init_guess(comp, kind)
            exp = expected_area(kind, c0[0], c0[1], yaws[1])
            if comp["area"] < 0.3 * exp:
                continue
            window = (np.pi / 2, np.deg2rad(16)) if kind == "plate" else (0.0, np.deg2rad(16))
            (pose,), res = fit_component(kind, comp, masks, dcx, 1, yaws, yaw_window=window)
            score = res - 0.2 * min(1.0, comp["area"] / exp)
            if best is None or score < best[0]:
                best = (score, {"pose": pose, "res": res, "area": comp["area"]})
        if best is None:
            out["notes"].append(f"no {kind}")
        else:
            out[kind] = best[1]
    dcp = dc | _others(masks, "pillar")
    comps = components(masks["pillar"] & ~dc, 60)
    for comp in comps:
        if not _in_region(comp, "pillar"):
            continue
        c, _ = init_guess(comp, "pillar")
        exp = expected_area("pillar", c[0], c[1])
        k = int(np.clip(np.round(comp["area"] / exp), 1, 3))
        touches = (cv2.dilate(comp["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & dc
        side = -1.0 if c[0] < 0 else 1.0
        window = (side * np.deg2rad(12.5), np.deg2rad(22))
        starts = list(side * np.deg2rad((4, 12.5, 21)))
        poses, res = fit_component("pillar", comp, masks, dcp, k, starts, yaw_window=window)
        for p in poses:
            out["pillars"].append({"pose": p, "res": res, "area": comp["area"], "k": k,
                                   "occluded": bool(touches.any())})
    for kind in ("long_board", "short_board"):
        pass
    boards = components(masks["board"] & ~dc, 400)
    for comp in boards[:2]:
        c, _ = init_guess(comp, "long_board")
        kind = "long_board" if c[1] > -0.25 else "short_board"
        (pose,), res = fit_component(kind, comp, masks, dc | _others(masks, "board"), 1, [0.0])
        out[kind] = {"pose": pose, "res": res, "area": comp["area"]}
    return out

REGIONS = {"pillar": ((0.12, 0.47), (-0.25, 0.05)), "plate": ((0.24, 0.52), (-0.26, 0.06)),
           "roof": ((0.24, 0.52), (-0.26, 0.06))}

def _in_region(comp, kind):
    _, _, w, h = comp["bbox"]
    if h < 6 or w > 90:
        return False
    xy = backproject(np.array([comp["uv"]]), TABLE_Z + HEIGHT[kind] / 2)[0, :2]
    (ax0, ax1), (y0, y1) = REGIONS[kind]
    return ax0 <= abs(xy[0]) <= ax1 and y0 <= xy[1] <= y1

def _merge(a, b):
    m = a["mask"] | b["mask"]
    vs, us = np.nonzero(m)
    bbox = (int(us.min()), int(vs.min()), int(us.max() - us.min() + 1), int(vs.max() - vs.min() + 1))
    return {"area": int(m.sum()), "bbox": bbox, "mask": m, "uv": np.array([us.mean(), vs.mean()])}

def _others(masks, key):
    m = np.zeros_like(masks["wood"])
    for k in ("pillar", "plate", "roof", "board"):
        if k != key:
            m |= masks[k]
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

CLASS_OF = {"pillar": "pillar", "plate": "plate", "roof": "roof", "long_board": "board", "short_board": "board"}

def front_face(kind, x, y, yaw, z_support):
    V = world_vertices(kind, x, y, yaw, z_support)
    cam = np.array([C.POS[0], C.POS[1]])
    c = V[:, :2].mean(axis=0)

    d = np.linalg.norm(V[:, :2] - cam, axis=1)
    idx = np.argsort(d)[:4]
    return project(V[idx])

def measure_at(rgb, kind, xy, yaw, z_support, qs, gs=None, search=0.03, yaws=None, extra_dc=None, known=(),
               yaw_window=None):
    masks = class_masks(rgb)
    key = CLASS_OF[kind]
    dc = occluders(qs, gs)
    if extra_dc is not None:
        dc = dc | extra_dc
    lx = {"pillar": PILLAR_DIMS[0], "plate": PLATE_DIMS[0], "roof": ROOF_BASE, "long_board": LONG_BOARD_DIMS[0],
          "short_board": SHORT_BOARD_DIMS[0]}[kind]
    ly = {"pillar": PILLAR_DIMS[1], "plate": PLATE_DIMS[1], "roof": ROOF_LEN, "long_board": LONG_BOARD_DIMS[1],
          "short_board": SHORT_BOARD_DIMS[1]}[kind]
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_], [s_, c]])
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = np.asarray(xy) + R @ np.array([sx * (lx / 2 + search), sy * (ly / 2 + search)])
            for z in (z_support, z_support + HEIGHT[kind]):
                corners.append([p[0], p[1], z])
    uv = project(np.array(corners))
    u0, v0 = np.floor(uv.min(axis=0)).astype(int)
    u1, v1 = np.ceil(uv.max(axis=0)).astype(int)
    u0, v0, u1, v1 = max(0, u0), max(0, v0), min(640, u1), min(480, v1)
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    roi = (u0, v0, u1 - u0, v1 - v0)
    region = np.zeros_like(masks[key])
    region[v0:v1, u0:u1] = True
    fixed = []
    dc_known = np.zeros_like(dc)
    for k2, x2, y2, yaw2, z2 in known:
        if CLASS_OF[k2] == key:
            fixed.append(project(world_vertices(k2, x2, y2, yaw2, z2)))
        if key == "plate" and k2 in ("long_board", "short_board"):
            ff = front_face(k2, x2, y2, yaw2, z2)
            mk = np.zeros((480, 640), np.uint8)
            cv2.fillConvexPoly(mk, cv2.convexHull(ff.astype(np.float32)).astype(np.int32), 1)
            dc_known |= cv2.dilate(mk, np.ones((5, 5), np.uint8)) > 0
    dc = dc | dc_known
    target = masks[key] & region & ~dc
    vis = int(target.sum())
    exp = expected_area(kind, xy[0], xy[1], yaw, z_support)
    if vis < 0.15 * exp:
        return {"pose": None, "visible": vis / max(1.0, exp)}
    others = np.zeros_like(masks["wood"])
    for k in ("pillar", "plate", "roof", "board"):
        if k != key and not (k == "board" and kind in ("pillar", "plate", "roof")):
            others |= masks[k]
    others = cv2.dilate(others.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    weight = ~(dc | others) & region
    fx = []
    for poly in fixed:
        fx.append(poly - (0, 0))
    fit = Fit(kind, target, weight, roi, z_support, fixed=fx)
    best = None
    for y0 in (yaws if yaws is not None else (yaw,)):
        x, f = fit.run([xy[0], xy[1], y0], yaw_window=yaw_window)
        if best is None or f < best[1]:
            best = (x, f)
    x, f = best
    return {"pose": (float(x[0]), float(x[1]), float(x[2])), "res": f / max(1.0, vis), "visible": vis / max(1.0, exp)}
