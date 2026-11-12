from __future__ import annotations

from .. import ASSETS

import json
from functools import lru_cache

import numpy as np

from ..planning import kinematics as K
from .cameras import Camera, H, W

_DATA = ASSETS / "x5_hulls.json"
_LINK_ORDER = "link1", "link2", "link3", "link4", "link5", "link6"
FINGER_OFFSETS = {"link7": (0.08657, 0.024896, -0.0002436), "link8": (0.08657, -0.0249, -0.00024366)}


def _xyz_rpy(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = K.rot_z(rpy[2]) @ K.rot_y(rpy[1]) @ K.rot_x(rpy[0])
    t[:3, 3] = xyz
    return t


T6_CAMBASE = _xyz_rpy((0.057, 0.0, 0.0), (0.0, 0.0, -3.1416))
TCAMBASE_CAM = _xyz_rpy((-0.0275, 0.0, 0.05), (0.0, 0.3491, 3.141592653589793))


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(_DATA.read_text())


@lru_cache(maxsize=1)
def _pieces() -> dict[str, list[np.ndarray]]:
    return {k: [np.asarray(p, dtype=float) for p in v] for k, v in _data()["hulls"].items()}


@lru_cache(maxsize=1)
def spheres() -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float).reshape(-1, 4) for k, v in _data()["spheres"].items()}


def finger_q(grip: float) -> float:
    return float(np.clip(grip, 0.0, 1.0)) * 0.054 - 0.01


def link_poses(arm: str, q, grip: float = 1.0) -> dict[str, np.ndarray]:
    fr = K.frames(arm, q)
    out = {"base_link": K.base_mat(arm)}
    for name, t in zip(_LINK_ORDER, fr):
        out[name] = t
    t6 = fr[-1]
    qf = max(0.0, finger_q(grip))
    for name, off in FINGER_OFFSETS.items():
        o = np.eye(4)
        sign = 1.0 if name == "link7" else -1.0
        o[:3, 3] = off[0], off[1] + sign * qf, off[2]
        out[name] = t6 @ o
    out["camera_base"] = t6 @ T6_CAMBASE
    out["camera"] = out["camera_base"] @ TCAMBASE_CAM
    return out


def arm_points(arm: str, q, grip: float = 1.0) -> list[np.ndarray]:
    pcs = _pieces()
    out = []
    for name, t in link_poses(arm, q, grip).items():
        for p in pcs.get(name, []):
            out.append(p @ t[:3, :3].T + t[:3, 3])
    return out


def mask(
    cam: Camera, q_left=None, q_right=None, g_left: float = 1.0, g_right: float = 1.0, dilate: int = 2, shape=(H, W)
) -> np.ndarray:
    import cv2

    m = np.zeros(shape, dtype=np.uint8)
    for arm, q, g in (("left", q_left, g_left), ("right", q_right, g_right)):
        if q is None:
            continue
        for pts in arm_points(arm, q, g):
            uv, d = cam.project(pts)
            ok = d > 0.01
            if ok.sum() < 3:
                continue
            uv = uv[ok]
            if np.abs(uv).max() > 100000.0:
                continue
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            cv2.fillConvexPoly(m, hull, 1)
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        m = cv2.dilate(m, k)
    return m.astype(bool)
