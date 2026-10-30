from __future__ import annotations

import numpy as np

from ..kinematics import chain as kin
from .transforms import make_T


def side_pose(p, phi):
    x = np.array([np.cos(phi), np.sin(phi), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return make_T(np.stack([x, np.cross(z, x), z], 1), np.asarray(p, float))


def arm_points(arm, q):
    fr = kin.fk_chain(arm, q)
    joints = [T[:3, 3] for T in fr[1:]]
    pts = []
    for p0, p1 in zip(joints[:-1], joints[1:]):
        for t in (0.0, 0.5):
            pts.append(p0 + t * (p1 - p0))
    T6 = fr[-1]
    for x, y in ((0.06, 0.08), (0.06, -0.08), (0.10, 0.0),
                 (kin.FINGER_TIP_X, 0.03), (kin.FINGER_TIP_X, -0.03)):
        pts.append((T6 @ np.array([x, y, 0.0, 1.0]))[:3])
    return np.array(pts)
