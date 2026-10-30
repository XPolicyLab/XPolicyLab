
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from ...lib.geometry.camera import HeadCamera as C
from ...lib.geometry.transforms import TABLE_Z

_ANG = np.linspace(0, 2 * np.pi, 24, endpoint=False)
_T = np.linspace(0, 1, 7)

def cylinder_points(p0, p1, r):
    a = p1 - p0
    L = np.linalg.norm(a)
    a = a / max(L, 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    ring = r * (np.cos(_ANG)[:, None] * e1 + np.sin(_ANG)[:, None] * e2)
    return (p0 + _T[:, None, None] * (p1 - p0)[None, None, :] + ring[None]).reshape(-1, 3)

def project_points(P):
    c = (P - C.POS) @ C.R
    z = -c[:, 2]
    return np.stack([C.CX + C.FX * c[:, 0] / z, C.CY - C.FX * c[:, 1] / z], axis=1)

def render(P, box):
    u0, v0, u1, v1 = box
    uv = project_points(P) - np.array([u0, v0])
    mask = np.zeros((v1 - v0, u1 - u0), np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask > 0

def standing_axis(x, y, h):
    return np.array([x, y, TABLE_Z]), np.array([x, y, TABLE_Z + h])

def lying_axis(x, y, yaw, L, r):
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    c = np.array([x, y, TABLE_Z + r])
    return c - u * L / 2, c + u * L / 2

class Fitter:

    def __init__(self, blob_pix, unknown, pad=40):
        us, vs = blob_pix[:, 0].astype(int), blob_pix[:, 1].astype(int)
        H, W = unknown.shape
        self.box = (max(0, us.min() - pad), max(0, vs.min() - pad), min(W, us.max() + pad), min(H, vs.max() + pad))
        u0, v0, u1, v1 = self.box
        self.obs = np.zeros((v1 - v0, u1 - u0), bool)
        self.obs[vs - v0, us - u0] = True
        self.known = ~unknown[v0:v1, u0:u1]

    def iou(self, P):
        sil = render(P, self.box)
        inter = np.count_nonzero(sil & self.obs & self.known)
        union = np.count_nonzero((sil | self.obs) & self.known)
        return inter / union if union else 0.0

    def fit_standing(self, x0, y0, h0=0.2, r=0.031):
        def cost(p):
            x, y, h = p
            if not 0.08 < h < 0.32:
                return 1.0
            return -self.iou(cylinder_points(*standing_axis(x, y, h), r))

        best = None
        for dh in (0.0, -0.05, 0.05):
            res = minimize(cost, [x0, y0, h0 + dh], method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 120, "initial_simplex":
                                    np.array([[x0, y0, h0 + dh], [x0 + 0.02, y0, h0 + dh], [x0, y0 + 0.02, h0 + dh],
                                              [x0, y0, h0 + dh + 0.03]])})
            if best is None or res.fun < best.fun:
                best = res
        x, y, h = best.x
        return {"kind": "standing", "xy": np.array([x, y]), "h": float(h), "iou": float(-best.fun)}

    def fit_lying(self, x0, y0, yaw0, L0=0.2, r=0.031):
        def cost(p):
            x, y, yaw, L = p
            if not 0.10 < L < 0.30:
                return 1.0
            return -self.iou(cylinder_points(*lying_axis(x, y, yaw, L, r), r))

        best = None
        for dyaw in (0.0, 0.4, -0.4):
            s0 = [x0, y0, yaw0 + dyaw, L0]
            res = minimize(cost, s0, method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 160, "initial_simplex":
                                    np.array([s0, [s0[0] + 0.02, *s0[1:]], [s0[0], s0[1] + 0.02, *s0[2:]],
                                              [*s0[:2], s0[2] + 0.3, s0[3]], [*s0[:3], s0[3] + 0.04]])})
            if best is None or res.fun < best.fun:
                best = res
        x, y, yaw, L = best.x
        return {"kind": "lying", "xy": np.array([x, y]), "yaw": float(yaw), "len": float(L), "iou": float(-best.fun)}
