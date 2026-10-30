"""Rank public X5 IK branches by motion distance and continuous fold posture."""
from collections import OrderedDict
import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from coap.kinematics import JOINT_LOWER, JOINT_UPPER, nearest_periodic, pose_matrix
from coap.motion import TCP_OFFSET


class PeriodicPostureKinematics:
    """Use soft forearm/wrist shape costs without rejecting issued motions."""

    def __init__(self, kinematics):
        self.kinematics = kinematics
        raw = kinematics
        while hasattr(raw, 'kinematics'):
            raw = raw.kinematics
        self.raw = raw
        self.cache = OrderedDict()

    def __getattr__(self, name):
        return getattr(self.kinematics, name)

    def fold_cost(self, arm, joints):
        """Measure smooth compactness between nonadjacent forearm/wrist solids."""
        centers, radii, links = self.raw.collision_spheres(arm, joints)
        forearm = links == 3
        wrist = (links == 5) | (links == 6)
        distances = np.linalg.norm(centers[forearm, None, :]-centers[None, wrist, :], axis=2)
        gaps = distances-radii[forearm, None]-radii[None, wrist]
        gap = float(np.min(gaps))
        cost = float(np.logaddexp(0., (.005-gap)/.003)**2)
        return cost, gap

    def path_posture(self, arm, reference, joints):
        values = [self.fold_cost(arm, reference+fraction*(joints-reference))
                  for fraction in np.linspace(1./16., 1., 16)]
        costs = np.array([item[0] for item in values])
        score = float(4.*costs[-1]+3.*costs.mean()+2.*costs.max())
        return score, dict(endpoint_fold_cost=float(costs[-1]),
                           maximum_fold_cost=float(costs.max()),
                           minimum_forearm_wrist_gap_m=min(item[1] for item in values))

    def solve(self, arm, target_pose, current_joints=None, tcp=False,
              position_tol=.002, rotation_tol=.020, max_starts=12, max_nfev=160,
              max_joint_step=None, joint_margin=0.):
        """Select an ordinary IK solution using a smooth physical posture prior."""
        started = time.monotonic()
        target = pose_matrix(target_pose)
        if tcp:
            target[:3, 3] -= target[:3, :3]@TCP_OFFSET
        reference = np.asarray(self.raw.initial_joints[arm] if current_joints is None
                               else current_joints, float).reshape(6)
        margin = np.broadcast_to(np.asarray(joint_margin, float), (6,))
        lower, upper = JOINT_LOWER+margin, JOINT_UPPER-margin
        if max_joint_step is not None:
            step = np.broadcast_to(np.asarray(max_joint_step, float), (6,))
            lower = np.maximum(lower, reference-step)
            upper = np.minimum(upper, reference+step)
        reference = np.clip(reference, lower+1e-7, upper-1e-7)
        cache_key = (arm, target.tobytes(), np.round(reference, 5).tobytes(),
                     float(position_tol), float(rotation_tol), margin.tobytes(),
                     None if max_joint_step is None else np.asarray(max_joint_step).tobytes())
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            result = dict(self.cache[cache_key])
            result['elapsed_seconds'] = time.monotonic()-started
            result['posture_cache_hit'] = True
            return result
        seeds = self.raw.seeds(arm, target, reference, max(10, int(max_starts)))
        records = []
        def residual(joints):
            frame = self.raw.forward_matrix(arm, joints)
            displacement = frame[:3, 3]-target[:3, 3]
            angular = Rotation.from_matrix(target[:3, :3].T@frame[:3, :3]).as_rotvec()
            return np.r_[4.*displacement, angular, 1e-5*(joints-reference)]
        for seed in seeds:
            candidate = least_squares(residual, np.clip(seed, lower+1e-7, upper-1e-7),
                bounds=(lower, upper), max_nfev=int(max_nfev),
                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            nearest = nearest_periodic(candidate.x, reference)
            periodic = [nearest]
            for turns in (-1, 1):
                variant = nearest.copy()
                variant[3] += turns*2.*math.pi
                if lower[3] <= variant[3] <= upper[3]:
                    periodic.append(variant)
            for joints in periodic:
                frame = self.raw.forward_matrix(arm, joints)
                position_error = float(np.linalg.norm(frame[:3, 3]-target[:3, 3]))
                rotation_error = float(np.linalg.norm(Rotation.from_matrix(
                    target[:3, :3].T@frame[:3, :3]).as_rotvec()))
                travel = float(np.linalg.norm(joints-reference))
                posture, geometry = self.path_posture(arm, reference, joints)
                accurate = position_error <= position_tol and rotation_error <= rotation_tol
                records.append(dict(joint_value=joints.tolist(), position_error=position_error,
                    rotation_error=rotation_error, joint_distance=travel, posture_cost=posture,
                    posture_geometry=geometry, accurate=accurate,
                    objective=travel+posture+1500.*position_error+12.*rotation_error))
        feasible = [record for record in records if record['accurate']]
        pool = feasible or records
        selected = min(pool, key=lambda record: record['objective'])
        result = dict(selected, status='Success' if feasible else 'Failure',
                      attempts=len(seeds), elapsed_seconds=time.monotonic()-started,
                      posture_candidates=[dict(joint4=float(record['joint_value'][3]),
                          objective=record['objective'], posture_cost=record['posture_cost'],
                          accurate=record['accurate']) for record in records],
                      posture_source='public_link3_link5_link6_path_with_joint4_periodic_alternatives',
                      runtime_rejection_added=False)
        self.cache[cache_key] = dict(result)
        while len(self.cache) > 512:
            self.cache.popitem(last=False)
        return result


def enable(ctx):
    """Attach branch ranking only to the current classification context."""
    if ctx.numeric_ik and not isinstance(ctx.kinematics, PeriodicPostureKinematics):
        ctx.kinematics = PeriodicPostureKinematics(ctx.kinematics)
