
from __future__ import annotations

import cv2
import numpy as np

from ...lib.geometry.camera import HeadCamera

ARM_ZONES = ((0, 230, 175, 480), (465, 230, 640, 480), (210, 400, 430, 480))

def hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

def mask_hsv(img_hsv, h=(0, 180), s=(0, 255), v=(0, 255)):
    h0, h1 = h
    if h0 <= h1:
        m = cv2.inRange(img_hsv, (h0, s[0], v[0]), (h1, s[1], v[1]))
    else:
        m = cv2.inRange(img_hsv, (h0, s[0], v[0]), (180, s[1], v[1])) | cv2.inRange(
            img_hsv, (0, s[0], v[0]), (h1, s[1], v[1])
        )
    return m

def blank_zones(mask, zones=ARM_ZONES):
    out = mask.copy()
    for u0, v0, u1, v1 in zones:
        out[v0:v1, u0:u1] = 0
    return out

def clean(mask, open_px=2, close_px=2):
    if open_px:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    if close_px:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return mask

def blobs(mask, min_area=20):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        vs, us = np.nonzero(labels == i)
        out.append(
            {
                "area": area,
                "uv": np.array(cents[i]),
                "bbox": tuple(int(x) for x in stats[i, :4]),
                "pix": np.stack([us, vs], axis=1),
            }
        )
    out.sort(key=lambda b: -b["area"])
    return out

def to_plane(uv, z):
    return HeadCamera.to_plane(float(uv[0]), float(uv[1]), z)

def pixels_to_plane(pix, z):
    C = HeadCamera
    d = np.stack([(pix[:, 0] - C.CX) / C.FX, -(pix[:, 1] - C.CY) / C.FX, -np.ones(len(pix))], axis=1) @ C.R.T
    t = (z - C.POS[2]) / d[:, 2]
    return C.POS + t[:, None] * d

def line_fit(points_xy):
    c = points_xy.mean(axis=0)
    _, _, vt = np.linalg.svd(points_xy - c, full_matrices=False)
    u = vt[0] / np.linalg.norm(vt[0])
    if u[0] < 0:
        u = -u
    s = (points_xy - c) @ u
    return c, u, (float(s.min()), float(s.max()))

def min_area_yaw(points_xy):
    rect = cv2.minAreaRect((points_xy * 1000).astype(np.float32))
    return np.deg2rad(rect[2]) % (np.pi / 2)


def log_fail(ctx, stage, **kw):
    ctx.log("fail", stage=stage, **kw)


def _inside(xy, box):
    return box[0] <= xy[0] <= box[1] and box[2] <= xy[1] <= box[3]
