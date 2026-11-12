from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...planning.kinematics import TABLE_Z
from ...perception.cameras import H, W, Camera

PLANE_FRAC = 0.5
MAX_POINTS = 4000
BORDER_PX = 3


@dataclass
class WristFootprint:
    xy: np.ndarray
    yaw: float
    L: float
    W: float
    shift: float
    score: float
    area: int
    cut: float

    def as_dict(self) -> dict:
        return {
            "xy": [round(float(v), 4) for v in self.xy],
            "yaw": round(math.degrees(self.yaw), 1),
            "L": round(self.L, 4),
            "W": round(self.W, 4),
            "shift": round(self.shift, 4),
            "score": round(self.score, 3),
            "area": self.area,
            "cut": round(self.cut, 3),
        }


def project_mask(cam: Camera, mask: np.ndarray, z: float) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if len(rows) > MAX_POINTS:
        k = np.linspace(0, len(rows) - 1, MAX_POINTS).astype(int)
        rows, cols = rows[k], cols[k]
    uv = np.stack([cols, rows], axis=1).astype(float)
    p = np.asarray(cam.on_plane_z(uv, z), float)[:, :2]
    return p[np.all(np.isfinite(p), axis=1)]


def footprint(pts: np.ndarray):
    import cv2

    (cx, cy), (a, b), ang = cv2.minAreaRect((pts * 1000.0).astype(np.float32))
    yaw = math.radians(ang) if a >= b else math.radians(ang + 90.0)
    yaw = math.remainder(yaw, math.pi)
    return np.array([cx, cy]) / 1000.0, yaw, max(a, b) / 1000.0, min(a, b) / 1000.0


def border_share(mask: np.ndarray) -> float:
    import cv2

    m = mask.astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    n = int(np.count_nonzero(edge))
    if not n:
        return 0.0
    b = np.zeros_like(m)
    b[:BORDER_PX, :] = b[-BORDER_PX:, :] = 1
    b[:, :BORDER_PX] = b[:, -BORDER_PX:] = 1
    return float(np.count_nonzero(edge & b)) / n


def measure(
    insts, cam: Camera, cat: str, xy_expected, h: float, max_shift: float = 0.04, plane_frac: float = PLANE_FRAC
) -> WristFootprint | None:
    z = TABLE_Z + plane_frac * float(h)
    xy_expected = np.asarray(xy_expected, float)[:2]
    best = None
    for inst in insts:
        if inst.category != cat or inst.mask.shape != (H, W):
            continue
        pts = project_mask(cam, inst.mask, z)
        if len(pts) < 20:
            continue
        c, yaw, L, Wd = footprint(pts)
        d = float(np.linalg.norm(c - xy_expected))
        if d <= max_shift and (best is None or d < best.shift):
            best = WristFootprint(
                xy=c,
                yaw=yaw,
                L=L,
                W=Wd,
                shift=d,
                score=float(inst.score),
                area=int(inst.area),
                cut=border_share(inst.mask),
            )
    return best
