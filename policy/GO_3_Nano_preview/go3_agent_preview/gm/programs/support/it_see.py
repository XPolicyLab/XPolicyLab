from __future__ import annotations

import cv2
import numpy as np

from ...lib.perception import colour as V
from ...lib.geometry.transforms import axis_angle
from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import WRIST, HeadCamera

TUBE_R_BODY = 0.01398
TUBE_R_CAP = 0.01747
TUBE_LEN = 0.117
TUBE_CAP_START = 0.10233
TUBE_TILT = np.deg2rad(1.95)
TUBE_TIP_Z = 0.778278

RACK_Z = 0.795
RACK_TOP = RACK_Z + 0.030
WRIST_FX = WRIST.FX
BORE_LX = (-0.07198, -0.03599, 0.0, 0.03599, 0.07198)
BORE_LY = (-0.01797, 0.01797)

WHITE_S_MAX = 10
WHITE_V_MIN = 210
# With num_envs=10 some env slots render the same tubes grey (body V 180-205
# where a single env renders 225-235), and the survey misses them.  The survey
# relaxes the body threshold one step at a time until it sees three capped
# tubes.  The cap threshold never moves: the table shares the cap hue, and only
# its brightness keeps it out of the cap mask.
WHITE_RELAX = ((10, 210), (11, 205), (12, 200), (13, 195), (14, 190), (15, 186), (16, 182))

CAP_H = (8, 24)
CAP_S_MIN = 50
CAP_V_MIN = 200
BLUE_H = (90, 125)
BLUE_S_MIN = 60
BLUE_V_MIN = 120
TABLE_XY = (-0.70, 0.70, -0.60, 0.50)

TIP_FROM_CAP = -0.1037
TIP_FROM_CENTROID = -0.06384
CAP_WEIGHT = 0.8

RACK_BIAS = np.array([0.00109, -0.00321])
RACK_YAW_BIAS = 0.0

def table_region(shape=(480, 640), z=TABLE_Z):
    x0, x1, y0, y1 = TABLE_XY
    uv = [HeadCamera.project([x, y, z]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    m = np.zeros(shape, np.uint8)
    uv = [p for p in uv if p is not None]
    if len(uv) >= 3:
        cv2.fillPoly(m, [np.round(np.array(uv)).astype(np.int32)], 255)
    return m

def arm_mask(q, shape=(480, 640), pad=0.010, grip=None, n=4):
    from ...lib.kinematics import chain as kin

    gr = kin.GRIP_MAX if grip is None else grip
    m = np.zeros(shape, np.uint8)
    for arm in ("left", "right"):
        frames = kin.link_frames(arm, np.asarray(q[arm], float), gr)
        for name, T in frames.items():
            lo, hi = kin.LINK_BOXES[name]
            lo = np.asarray(lo, float) - pad
            hi = np.asarray(hi, float) + pad
            g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n),
                                     np.linspace(lo[2], hi[2], n)), -1).reshape(-1, 3)
            W = g @ T[:3, :3].T + T[:3, 3]
            for p in W:
                uv = HeadCamera.project(p)
                if uv is None:
                    continue
                u, v = int(uv[0]), int(uv[1])
                if 0 <= u < shape[1] and 0 <= v < shape[0]:
                    m[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = 255
    return m

def masks(rgb, q=None, white_sv=(WHITE_S_MAX, WHITE_V_MIN)):
    h = V.hsv(rgb)
    tm = table_region(h.shape[:2])
    white = V.mask_hsv(h, (0, 180), (0, white_sv[0]), (white_sv[1], 255))
    cap = V.mask_hsv(h, CAP_H, (CAP_S_MIN, 255), (CAP_V_MIN, 255))
    if q is not None:
        keep = cv2.bitwise_not(arm_mask(q, shape=h.shape[:2]))
        white = cv2.bitwise_and(white, keep)
        cap = cv2.bitwise_and(cap, keep)
    else:
        white = V.blank_zones(white)
        cap = V.blank_zones(cap)
    white = cv2.bitwise_and(V.clean(white, 2, 3), tm)
    cap = cv2.bitwise_and(V.clean(cap, 2, 2), tm)
    both = V.clean(cv2.bitwise_and(cv2.bitwise_or(white, cap), tm), 2, 5)
    return h, tm, white, cap, both

def plane_xy(uv, z):
    return HeadCamera.to_plane(float(uv[0]), float(uv[1]), z)[:2]

def _fuse(frags, gap=16.0, ang_tol=12.0):
    used = [False] * len(frags)
    out = []
    for i in range(len(frags)):
        if used[i]:
            continue
        a = frags[i]
        used[i] = True
        for j in range(i + 1, len(frags)):
            if used[j]:
                continue
            b = frags[j]
            ang = np.degrees(np.arccos(np.clip(abs(float(a["u"] @ b["u"])), 0.0, 1.0)))
            if ang > ang_tol:
                continue
            d = min(float(np.linalg.norm(ea - eb)) for ea in a["ends"] for eb in b["ends"])
            if d > gap:
                continue
            pix = np.vstack([a["pix"], b["pix"]])
            c = pix.mean(0)
            _, _, vt = np.linalg.svd(pix - c, full_matrices=False)
            s = (pix - c) @ vt[0]
            a = {"uv": c, "u": vt[0], "pix": pix, "area": a["area"] + b["area"],
                 "ends": (c + vt[0] * float(s.min()), c + vt[0] * float(s.max())), "len": float(s.max() - s.min())}
            used[j] = True
        out.append(a)
    return out

def tubes(rgb, q=None, min_area=150, white_sv=(WHITE_S_MAX, WHITE_V_MIN)):
    h, tm, white, cap, both = masks(rgb, q, white_sv)
    zz = TABLE_Z + TUBE_R_BODY
    frags = []
    for b in V.blobs(both, min_area):
        pix = b["pix"].astype(float)
        c = pix.mean(0)
        _, _, vt = np.linalg.svd(pix - c, full_matrices=False)
        s = (pix - c) @ vt[0]
        L = float(s.max() - s.min())
        if L < 20:
            continue
        frags.append({"uv": c, "u": vt[0], "pix": pix, "area": b["area"], "len": L,
                      "ends": (c + vt[0] * float(s.min()), c + vt[0] * float(s.max()))})
    out = []
    for t in _fuse(frags):
        p0, p1 = plane_xy(t["ends"][0], zz), plane_xy(t["ends"][1], zz)
        L_m = float(np.linalg.norm(p1 - p0))
        w_m = float(t["area"]) / max(t["len"], 1.0) * L_m / max(t["len"], 1.0)
        if not (0.06 <= L_m <= 0.16 and 0.014 <= w_m <= 0.048):
            continue
        pix = t["pix"]
        on_cap = cap[pix[:, 1].astype(int), pix[:, 0].astype(int)] > 0
        P = V.pixels_to_plane(pix[::2], TABLE_Z + 1.5 * TUBE_R_BODY)[:, :2]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        d = vt[0]
        n_cap = int(on_cap.sum())
        if n_cap >= 8:
            Pc = V.pixels_to_plane(pix[on_cap], TABLE_Z + 2 * TUBE_R_CAP)[:, :2]
            cc = Pc.mean(0)
            if (cc - c) @ d < 0:
                d = -d
            along_cap = (cc - c) @ d

            s_tip = CAP_WEIGHT * (along_cap + TIP_FROM_CAP) + (1 - CAP_WEIGHT) * TIP_FROM_CENTROID
        else:
            s_tip = TIP_FROM_CENTROID
        tip = c + s_tip * d
        out.append({"tip": tip, "axis": d, "cap_px": n_cap, "uv": t["uv"], "len_m": L_m, "width_m": w_m,
                    "area": t["area"], "centroid": c})
    out.sort(key=lambda t: -t["area"])
    return out

def rack(rgb, min_area=250):
    from scipy.optimize import least_squares

    h = V.hsv(rgb)
    blue = V.mask_hsv(h, BLUE_H, (BLUE_S_MIN, 255), (BLUE_V_MIN, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    if int(st[i, 4]) < min_area:
        return None
    face = (lab == i).astype(np.uint8)
    ys, xs = np.nonzero(face)
    P = V.pixels_to_plane(np.stack([xs, ys], 1)[::3].astype(float), RACK_TOP)[:, :2]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    yaw = float(np.arctan2(vt[0][1], vt[0][0]))

    hull = cv2.convexHull(cv2.findNonZero(face))
    hm = np.zeros_like(blue)
    cv2.fillPoly(hm, [hull], 255)
    hm = cv2.erode(hm, np.ones((3, 3), np.uint8))
    holes = cv2.morphologyEx(cv2.bitwise_and(cv2.bitwise_not(blue), hm), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n2, _, st2, ce2 = cv2.connectedComponentsWithStats(holes, 8)
    blobs = np.array([ce2[k] for k in range(1, n2) if 45 <= st2[k, 4] <= 400])
    if len(blobs) < 4:
        return None
    local = np.array([(lx, ly) for lx in BORE_LX for ly in BORE_LY])

    def bores_uv(par):
        cx, cy, th = par
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        W = np.array([cx, cy]) + local @ R.T
        return [HeadCamera.project([p[0], p[1], RACK_TOP]) for p in W]

    def resid(par):
        r = []
        for uv in bores_uv(par):
            if uv is None:
                continue
            d = np.linalg.norm(blobs - uv, axis=1)
            j = int(np.argmin(d))
            if d[j] < 14:
                r.append(blobs[j] - uv)
        return np.array(r).ravel() if r else np.zeros(2)

    par = np.array([c[0], c[1], yaw])
    for _ in range(3):
        try:
            par = least_squares(resid, par, method="lm").x
        except Exception:
            break
    matched = len(resid(par)) // 2
    if matched < 6:
        return None
    yaw = float((par[2] + np.pi / 2) % np.pi - np.pi / 2) - RACK_YAW_BIAS
    xy = np.array(par[:2]) - RACK_BIAS
    return {"xy": xy, "yaw": yaw, "matched": matched}

def held_cap(rgb):
    h = V.hsv(rgb)
    cap = V.mask_hsv(h, CAP_H, (CAP_S_MIN, 255), (CAP_V_MIN, 255))
    n, _, st, _ = cv2.connectedComponentsWithStats(cap, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, hh, a = (int(v) for v in st[i])
        if a < 500 or abs(x + w / 2 - rgb.shape[1] / 2) > 120:
            continue
        if best is None or a > best["area"]:
            best = {"area": a, "w": w, "h": hh, "top": y, "bottom": y + hh, "cx": x + w / 2}
    return best

def cap_slide(ref, now):
    r0 = WRIST_FX * 2 * TUBE_R_CAP / max(ref["w"], 1)
    return float(r0 * (1.0 - np.sqrt(ref["area"] / max(now["area"], 1))))

HELD_AXIS = np.array([0.0165, 0.0, 0.0014])
HELD_GRIP_FROM_TIP = 0.085
_RING = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)

def _held_box(T_cam, T_tcp, tx, ty, slide):
    R = T_tcp[:3, :3]
    g = T_tcp[:3, 3] + R @ HELD_AXIS
    a = R @ (axis_angle([1, 0, 0], tx) @ axis_angle([0, 1, 0], ty) @ (R.T @ np.array([0.0, 0.0, 1.0])))
    tip = g - HELD_GRIP_FROM_TIP * a + slide * a
    e1 = np.cross(a, R[:, 1])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ring = TUBE_R_CAP * (np.outer(np.cos(_RING), e1) + np.outer(np.sin(_RING), e2))
    P = np.vstack([tip + TUBE_CAP_START * a + ring, tip + TUBE_LEN * a + ring])
    c = (P - T_cam[:3, 3]) @ T_cam[:3, :3]
    if np.any(c[:, 2] >= -1e-6):
        return None
    u = WRIST.CX + WRIST.FX * c[:, 0] / -c[:, 2]
    v = WRIST.CY - WRIST.FX * c[:, 1] / -c[:, 2]
    return np.array([u.min(), u.max(), v.min(), v.max()])

def held_lean(blob, T_cam, T_tcp):
    if blob is None:
        return None
    obs = np.array([blob["cx"] - blob["w"] / 2.0, blob["cx"] + blob["w"] / 2.0, blob["top"], blob["bottom"]], float)
    x = np.zeros(3)
    for _ in range(30):
        b = _held_box(T_cam, T_tcp, *x)
        if b is None:
            return None
        r = b - obs
        J = np.zeros((4, 3))
        for j, h in enumerate((1e-4, 1e-4, 1e-5)):
            d = np.zeros(3)
            d[j] = h
            b2 = _held_box(T_cam, T_tcp, *(x + d))
            if b2 is None:
                return None
            J[:, j] = (b2 - b) / h
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + np.clip(step, -0.05, 0.05)
        if np.linalg.norm(step) < 1e-7:
            break
    b = _held_box(T_cam, T_tcp, *x)
    if b is None:
        return None
    return {"tx": float(x[0]), "ty": float(x[1]), "slide": float(x[2]), "res": float(np.abs(b - obs).max())}

def bore_xy(rk, lx, ly):
    c, s = np.cos(rk["yaw"]), np.sin(rk["yaw"])
    return rk["xy"] + np.array([c * lx - s * ly, s * lx + c * ly])

def seated_score(rgb, xy, q=None):
    h = V.hsv(rgb)
    keep = None if q is None else arm_mask(q, shape=h.shape[:2])
    hits = 0
    tot = 0
    for z in np.linspace(RACK_TOP + 0.006, RACK_TOP + 0.060, 7):
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = [xy[0] + 0.012 * np.cos(th), xy[1] + 0.012 * np.sin(th), z]
            uv = HeadCamera.project(p)
            if uv is None:
                continue
            u, v = int(uv[0]), int(uv[1])
            if not (0 <= u < h.shape[1] and 0 <= v < h.shape[0]):
                continue
            if keep is not None and keep[v, u]:
                continue
            px = h[v, u]
            tot += 1
            white = px[1] <= WHITE_S_MAX + 8 and px[2] >= WHITE_V_MIN - 20
            tan = CAP_H[0] <= px[0] <= CAP_H[1] and px[1] >= CAP_S_MIN and px[2] >= CAP_V_MIN - 20
            hits += int(white or tan)
    return None if tot < 20 else hits / tot
