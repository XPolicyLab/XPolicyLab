# Copyright © 2026 Martingale. All Rights Reserved.
import numpy as np
from martingale.hosts import robodojo_sight as sight, robodojo_kinematics as kin
ANGLES = np.arange(48) * (2 * np.pi / 48)
DIRECTIONS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)

def rotation(angles: np.ndarray) -> np.ndarray:
    c, s = (np.cos(angles), np.sin(angles))
    rz = np.array([[c[0], -s[0], 0], [s[0], c[0], 0], [0, 0, 1]])
    ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    rx = np.array([[1, 0, 0], [0, c[2], -s[2]], [0, s[2], c[2]]])
    return rz @ ry @ rx

def _support(observation: sight.Sight):
    rows, cols = np.nonzero(observation.mask)
    projected = np.stack([cols, rows], axis=1) @ DIRECTIONS.T
    target = projected.max(axis=0)
    unknown = observation.unknown
    touches_unknown = np.zeros(len(rows), dtype=bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = (rows + dr, cols + dc)
        inside = (rr >= 0) & (rr < unknown.shape[0]) & (cc >= 0) & (cc < unknown.shape[1])
        touches_unknown |= ~inside
        touches_unknown[inside] |= unknown[rr[inside], cc[inside]]
    clipped = np.any((projected >= target - 0.5) & touches_unknown[:, None], axis=0)
    return (target, clipped)

def _initial_center(observations: list[sight.Sight]) -> np.ndarray:
    a, b = (np.zeros((3, 3)), np.zeros(3))
    for ob in observations:
        rows, cols = np.nonzero(ob.mask)
        ray = ob.view.rays(np.array([(rows.min() + rows.max()) / 2]), np.array([(cols.min() + cols.max()) / 2]))[0]
        ray /= np.linalg.norm(ray)
        projection = np.eye(3) - np.outer(ray, ray)
        a += projection
        b += projection @ ob.view.pos
    return np.linalg.lstsq(a, b, rcond=None)[0]
