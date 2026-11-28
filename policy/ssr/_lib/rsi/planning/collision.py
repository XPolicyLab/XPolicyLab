from __future__ import annotations

import numpy as np

from . import kinematics as K

_FINGER_X = 0.09, 0.105, 0.12, 0.135, 0.15
_FINGER_R = 0.009


def _link_spheres():
    from ..perception import robot_mask as RM

    return RM.spheres()


def arm_spheres(arm: str, q, grip: float = 1.0) -> np.ndarray:
    from ..perception import robot_mask as RM

    sph = _link_spheres()
    poses = RM.link_poses(arm, q, grip)
    out = []
    for name, s in sph.items():
        t = poses.get(name)
        if t is None:
            continue
        c = s[:, :3] @ t[:3, :3].T + t[:3, 3]
        out.append(np.hstack([c, s[:, 3:4]]))
    t6 = poses["link6"]
    qf = max(0.0, RM.finger_q(grip))
    for sign in (1.0, -1.0):
        for x in _FINGER_X:
            p = t6[:3, 3] + t6[:3, 0] * x + t6[:3, 1] * sign * (0.0004 + qf + 0.006)
            out.append(np.array([[*p, _FINGER_R]]))
    return np.vstack(out)


def arm_distance(q_left, q_right, g_left: float = 1.0, g_right: float = 1.0) -> float:
    a = arm_spheres("left", q_left, g_left)
    b = arm_spheres("right", q_right, g_right)
    d = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=2) - a[:, None, 3] - b[None, :, 3]
    return float(d.min())


def table_clearance(arm: str, q, grip: float = 1.0) -> float:
    s = arm_spheres(arm, q, grip)
    s = s[3:]
    return float((s[:, 2] - s[:, 3]).min() - K.TABLE_Z)


def box_clearance(arm: str, q, boxes, grip: float = 1.0) -> float:
    if not boxes:
        return 9.0
    s = arm_spheres(arm, q, grip)
    b = np.asarray(boxes, dtype=float).reshape(-1, 5)
    lo = np.stack([b[:, 0] - b[:, 2], b[:, 1] - b[:, 2], b[:, 3]], axis=1)
    hi = np.stack([b[:, 0] + b[:, 2], b[:, 1] + b[:, 2], b[:, 4]], axis=1)
    c = s[:, None, :3]
    d = np.linalg.norm(np.maximum(np.maximum(lo[None] - c, c - hi[None]), 0.0), axis=2) - s[:, None, 3]
    return float(d.min())


RIG_BOXES = ((-0.015, -0.485, 0.717), (0.015, -0.455, 1.303)), ((-0.041, -0.49, 1.248), (0.089, -0.405, 1.303))


def rig_clearance(arm: str, q, grip: float = 1.0) -> float:
    s = arm_spheres(arm, q, grip)
    best = 9.0
    for lo, hi in RIG_BOXES:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        dd = np.maximum(0.0, np.maximum(lo - s[:, :3], s[:, :3] - hi))
        best = min(best, float((np.linalg.norm(dd, axis=1) - s[:, 3]).min()))
    return best
