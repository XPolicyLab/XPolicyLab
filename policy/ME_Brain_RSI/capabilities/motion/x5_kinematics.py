"""URDF forward kinematics and bounded, locally continuous X5 inverse kinematics.

This module reads robot geometry and solves joint targets. It never writes
simulation state and performs no collision checking. All poses are expressed in
the environment/world frame, with pose vectors ordered xyz + quaternion wxyz.
"""
from __future__ import annotations

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _transform(value):
    value = np.asarray(value, dtype=float)
    if value.shape == (4, 4):
        return value.copy()
    if value.shape != (7,):
        raise ValueError("Expected a 4x4 transform or xyz+wxyz pose")
    transform = np.eye(4)
    transform[:3, 3] = value[:3]
    transform[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    return transform


class X5Kinematics:
    def __init__(self, urdf_path, base_pose, limits=None):
        self.urdf_path = Path(urdf_path)
        self.base = _transform(base_pose)
        tree = ET.parse(self.urdf_path)
        self.origins = []
        self.axes = []
        self.joint_names = [f"joint{i}" for i in range(1, 7)]
        urdf_limits = []
        for name in self.joint_names:
            joint = tree.find(f"joint[@name='{name}']")
            if joint is None or joint.attrib.get("type") not in {"revolute", "continuous"}:
                raise ValueError(f"Expected rotational X5 joint {name}")
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            axis = joint.find("axis")
            axis_vector = np.fromstring(axis.attrib.get("xyz", "1 0 0"), sep=" ")
            axis_vector /= np.linalg.norm(axis_vector)
            self.origins.append(matrix)
            self.axes.append(axis_vector)
            limit = joint.find("limit")
            urdf_limits.append([float(limit.attrib.get("lower", -math.inf)),
                                float(limit.attrib.get("upper", math.inf))])
        urdf_limits = np.asarray(urdf_limits)
        if limits is None:
            self.limits = urdf_limits
        else:
            if hasattr(limits, "detach"):
                limits = limits.detach().cpu().numpy()
            self.limits = np.asarray(limits, dtype=float).copy()
            if self.limits.shape != (6, 2):
                raise ValueError("limits must have shape (6,2) in joint1..joint6 order")
            # Real USD joint limits are authoritative. Invalid values indicate
            # the caller used an articulation array in the wrong joint order.
            if not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
                raise ValueError("Expected finite, nonempty physical joint limits")
        self.lower = self.limits[:, 0]
        self.upper = self.limits[:, 1]
        self.last_info = {}

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("q must be a finite six-joint vector")
        result = self.base.copy()
        for value, origin, axis in zip(q, self.origins, self.axes):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(value * axis).as_matrix()
            result = result @ origin @ motion
        return result

    def normalize(self, q, near):
        """Return the legal 2pi-equivalent representation nearest ``near``.

        This only removes needless full revolutions. It cannot make an elbow
        branch change or crossing a physical wrist limit safe; callers must
        still enforce local step bounds or call solve_continuous.
        """
        q = np.asarray(q, dtype=float)
        near = np.asarray(near, dtype=float)
        if q.shape != (6,) or near.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(near).all():
            return None
        result = np.empty(6)
        period = 2 * math.pi
        for i, angle in enumerate(q):
            k_min = math.ceil((self.lower[i] - angle - 1e-10) / period)
            k_max = math.floor((self.upper[i] - angle + 1e-10) / period)
            if k_min > k_max:
                return None
            k = min(max(round((near[i] - angle) / period), k_min), k_max)
            result[i] = angle + period * k
        return np.clip(result, self.lower, self.upper)

    def solve_continuous(self, goal, seed, max_delta=.5):
        """Solve within both physical limits and ``seed +/- max_delta``.

        Returns None when the target cannot be achieved locally within 0.5mm
        and 0.2deg. It deliberately never falls back to a distant random seed.
        ``last_info`` exposes residuals and bound contact for diagnostics.
        """
        goal = _transform(goal)
        seed = np.asarray(seed, dtype=float)
        max_delta = np.broadcast_to(np.asarray(max_delta, dtype=float), (6,))
        if seed.shape != (6,) or not np.isfinite(seed).all() or np.any(max_delta <= 0):
            self.last_info = {"status": "invalid_seed_or_step"}
            return None
        lower = np.maximum(self.lower, seed - max_delta)
        upper = np.minimum(self.upper, seed + max_delta)
        if np.any(lower >= upper) or np.any(seed < self.lower - .002) or np.any(seed > self.upper + .002):
            self.last_info = {"status": "seed_outside_limits"}
            return None
        initial = np.clip(seed, lower + 1e-10, upper - 1e-10)

        def residual(q):
            current = self.fk(q)
            return np.r_[(current[:3, 3] - goal[:3, 3]) * 10.,
                         Rotation.from_matrix(goal[:3, :3].T @ current[:3, :3]).as_rotvec()]

        solution = least_squares(residual, initial, bounds=(lower, upper),
                                 max_nfev=90, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        current = self.fk(solution.x)
        position_error = float(np.linalg.norm(current[:3, 3] - goal[:3, 3]))
        angle_error = float(Rotation.from_matrix(goal[:3, :3].T @ current[:3, :3]).magnitude())
        success = position_error < .0005 and angle_error < math.radians(.2)
        self.last_info = dict(status="success" if success else "local_target_unreachable",
                              position_error=position_error, angle_error_rad=angle_error,
                              max_joint_delta=float(np.max(np.abs(solution.x - seed))),
                              nfev=int(solution.nfev), active_bounds=solution.active_mask.tolist())
        return solution.x if success else None
