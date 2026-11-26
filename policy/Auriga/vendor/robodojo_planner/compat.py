"""NumPy 2 compatibility for the vendored CuroboPlanner (not part of RoboDojo).

RoboDojo runs CuroboPlanner under NumPy 1.26. ``_trans_from_world_to_base`` feeds float32 quaternions
to ``transforms3d.quaternions.quat2mat``. Under NumPy 1, ``2.0 / Nq`` with ``Nq`` an ``np.float32``
scalar is promoted to float64 (value-based casting), so the rotation matrix and the goal quaternion
are computed in float64. Under NumPy 2 (NEP 50, the Auriga policy environment pins numpy 2.5.3) the
Python float is "weak" and the whole matrix stays float32. The goal quaternion then differs from the
in-simulation one by up to one float32 ulp (408 of 660 real plan_path requests), and cuRobo's
optimiser turns that into different trajectories (up to 1.4e-2 rad on long paths).

``CuroboPlanner`` below overrides only that method and reproduces the NumPy 1 arithmetic exactly:
the quaternion norm is summed in the input dtype, as before, and everything after ``2.0 / Nq`` is
float64. Under NumPy 1 it returns the same values as the original method.
"""

import numpy as np
import transforms3d as t3d

from .curobo_planner import CuroboPlanner as _RoboDojoCuroboPlanner

_FLOAT_EPS = np.finfo(np.float64).eps


def quat2mat_numpy1(q):
    """``transforms3d.quaternions.quat2mat`` (0.4.2) with NumPy 1 scalar promotion."""
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z  # same dtype rules in NumPy 1 and 2 (array scalars only)
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / np.float64(Nq)  # NumPy 1: np.float32 scalar with a Python float -> float64
    w, x, y, z = (np.float64(v) for v in (w, x, y, z))
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X; wY = w * Y; wZ = w * Z
    xX = x * X; xY = x * Y; xZ = x * Z
    yY = y * Y; yZ = y * Z; zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


class CuroboPlanner(_RoboDojoCuroboPlanner):
    def _trans_from_world_to_base(self, base_pose, target_pose):
        """Same statements as RoboDojo's method, with ``quat2mat_numpy1`` for ``t3d.quaternions.quat2mat``."""
        base_p, base_q = base_pose[0:3], base_pose[3:]
        target_p, target_q = target_pose[0:3], target_pose[3:]
        rel_p = target_p - base_p
        wRb = quat2mat_numpy1(base_q)
        wRt = quat2mat_numpy1(target_q)
        result_p = wRb.T @ rel_p
        result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
        return result_p, result_q
