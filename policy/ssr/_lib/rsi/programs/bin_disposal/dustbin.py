from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RIM_Z = 0.725
OPEN_HALF = 0.187
DARK_L = 40
MIN_AREA = 2000


@dataclass
class Bin:
    centre: np.ndarray
    x_right: float
    y_front: float
    y_back: float
    area: int


def detect(img: np.ndarray, cam, table_mask: np.ndarray | None = None) -> Bin | None:
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB)
    dark = (lab[:, :, 0] < DARK_L).astype(np.uint8)
    if table_mask is not None:
        dark[table_mask > 0] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        return None
    vs, us = np.nonzero(lab_cc == i)
    P = cam.on_plane_z(np.stack([us, vs], axis=1).astype(float), RIM_Z)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < MIN_AREA // 2:
        return None
    x_right = float(np.percentile(P[:, 0], 99.5))
    strip = P[P[:, 0] > x_right - 0.12]
    y_front = float(np.percentile(strip[:, 1], 0.5))
    y_back = float(np.percentile(strip[:, 1], 99.5))
    centre = np.array([x_right - OPEN_HALF, 0.5 * (y_front + y_back)])
    return Bin(centre, x_right, y_front, y_back, int(stats[i, cv2.CC_STAT_AREA]))
