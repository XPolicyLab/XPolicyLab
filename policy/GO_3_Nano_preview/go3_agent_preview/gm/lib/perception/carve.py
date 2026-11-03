
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ...lib.geometry.transforms import TABLE_Z
from ...lib.geometry.camera import WRIST, HeadCamera

def wood_mask(rgb):
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = h[..., 0], h[..., 1], h[..., 2]
    return ((hue <= 25) | (hue >= 168)) & (sat >= 40) & (sat <= 220) & (val >= 45) & (val <= 200)

def project_head(P):
    C = HeadCamera
    c = (P - C.POS) @ C.R
    z = -c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = C.CX + C.FX * c[:, 0] / z
        v = C.CY - C.FX * c[:, 1] / z
    return u, v, z > 0

def project_cam(T, P, cam=WRIST):
    c = (P - T[:3, 3]) @ T[:3, :3]
    z = -c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.CX + cam.FX * c[:, 0] / z
        v = cam.CY - cam.FX * c[:, 1] / z
    return u, v, z > 0

class Carver:
    def __init__(self, xlim, ylim, zlim, res=0.008):
        self.res = res
        self.xs = np.arange(xlim[0], xlim[1], res) + res / 2
        self.ys = np.arange(ylim[0], ylim[1], res) + res / 2
        self.zs = np.arange(zlim[0], zlim[1], res) + res / 2
        X, Y, Z = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.shape = X.shape
        self.P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def carve(self, views):
        keep = np.ones(len(self.P), bool)
        for project, fg in views:
            u, v, front = project(self.P)
            H, W = fg.shape
            inside = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            ui = np.clip(u, 0, W - 1).astype(int)
            vi = np.clip(v, 0, H - 1).astype(int)
            bg = inside & ~fg[vi, ui]
            keep &= ~bg
        return keep.reshape(self.shape)

    def clusters(self, occ, min_vox=12):
        lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        out = []
        for i in range(1, n + 1):
            idx = np.argwhere(lab == i)
            if len(idx) < min_vox:
                continue
            pts = np.stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]], self.zs[idx[:, 2]]], axis=1)
            out.append(pts)
        out.sort(key=lambda p: -len(p))
        return out

def describe(pts):
    xy = pts[:, :2]
    c = xy.mean(axis=0)
    _, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    u = vt[0]
    if u[0] < 0:
        u = -u
    n = np.array([-u[1], u[0]])
    a = (xy - c) @ u
    b = (xy - c) @ n
    return {
        "c": c,
        "u": u,
        "len": float(a.max() - a.min()),
        "wid": float(b.max() - b.min()),
        "mid": c + u * (a.max() + a.min()) / 2 + n * (b.max() + b.min()) / 2,
        "ztop": float(pts[:, 2].max()),
        "height": float(pts[:, 2].max() - TABLE_Z),
        "n": len(pts),
    }
