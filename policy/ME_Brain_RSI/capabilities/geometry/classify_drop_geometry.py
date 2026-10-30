"""Soft-rank unchanged IK targets with public full-chain robot geometry."""
import itertools
import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from coap.kinematics import (COLLISION_IGNORE, JOINT_LOWER, JOINT_UPPER,
                             nearest_periodic, pose_matrix)
from coap.motion import ARMS, tcp_to_link
from coap.tasks.classify_r28_drop_geometry_assets import TABLE_VERTICES


FRACTIONS = np.linspace(0., 1., 65)
TABLE_MESHES = tuple((name, link, np.asarray(vertices, float))
                    for name, link, vertices in TABLE_VERTICES)


def _penalty(gap, margin):
    """Give each finite clearance a smooth cost without accepting or rejecting it."""
    return np.logaddexp(0., (margin - np.asarray(gap)) / .005) ** 2


def _aggregate(cost):
    return float(4. * cost[-1] + 3. * np.mean(cost) + 2. * np.max(cost))


class FullGeometryRanking:
    """Enumerate ordinary endpoint solutions and rank synchronous joint chords."""

    def __init__(self, kinematics):
        raw = kinematics
        while hasattr(raw, 'kinematics'):
            raw = raw.kinematics
        self.raw = raw
        _, radii, links = raw.collision_spheres('left', raw.initial_joints['left'])
        self.radii = radii
        self.links = links
        low = np.minimum(links[:, None], links[None, :])
        high = np.maximum(links[:, None], links[None, :])
        eligible = np.triu(np.ones(low.shape, bool), 1) & (low != high)
        eligible &= ~np.isin(low * 7 + high,
                            [first * 7 + second for first, second in COLLISION_IGNORE])
        self.self_rows, self.self_cols = np.where(eligible)
        self.fold_rows, self.fold_cols = np.where((low == 3) & np.isin(high, [5, 6]))

    def candidates(self, arm, target_pose, reference, source_stage):
        """Keep the original Cartesian accuracy pool and all issued-motion semantics."""
        target = pose_matrix(target_pose)
        reference = np.asarray(reference, float)
        clipped = np.clip(reference, JOINT_LOWER + 1e-7, JOINT_UPPER - 1e-7)
        seeds = self.raw.seeds(arm, target, clipped, 10 if source_stage else 8)
        records = []

        def residual(joints):
            frame = self.raw.forward_matrix(arm, joints)
            displacement = frame[:3, 3] - target[:3, 3]
            angular = Rotation.from_matrix(target[:3, :3].T @ frame[:3, :3]).as_rotvec()
            return np.r_[4. * displacement, angular, 1e-5 * (joints - clipped)]

        for seed in seeds:
            solution = least_squares(residual,
                np.clip(seed, JOINT_LOWER + 1e-7, JOINT_UPPER - 1e-7),
                bounds=(JOINT_LOWER, JOINT_UPPER), max_nfev=140,
                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            nearest = nearest_periodic(solution.x, clipped)
            variants = [nearest]
            for turn in (-1, 1):
                value = nearest.copy()
                value[3] += turn * 2. * math.pi
                if JOINT_LOWER[3] <= value[3] <= JOINT_UPPER[3]:
                    variants.append(value)
            for joints in variants:
                if any(np.linalg.norm(joints - row['joint_value']) < 1e-5 for row in records):
                    continue
                frame = self.raw.forward_matrix(arm, joints)
                position_error = float(np.linalg.norm(frame[:3, 3] - target[:3, 3]))
                rotation_error = float(np.linalg.norm(Rotation.from_matrix(
                    target[:3, :3].T @ frame[:3, :3]).as_rotvec()))
                travel = float(np.linalg.norm(joints - clipped))
                records.append(dict(joint_value=joints, position_error=position_error,
                    rotation_error=rotation_error, joint_distance=travel,
                    accurate=position_error <= .002 and rotation_error <= .020,
                    endpoint_cost=travel + 1500. * position_error + 12. * rotation_error))
        # This is the unchanged accuracy preference from the existing IK solver.
        accurate = [row for row in records if row['accurate']]
        return accurate or records

    def path_geometry(self, arm, start, end, opening, table_z, source_stage):
        """Measure table hulls and official nonignored self-sphere pairs."""
        centers = []
        table_gaps = []
        self_gaps = []
        fold_gaps = []
        for fraction in FRACTIONS:
            joints = start + fraction * (end - start)
            frames = self.raw.forward_links(arm, joints)
            points, _, _ = self.raw.collision_spheres(arm, joints)
            centers.append(points)
            table = float('inf')
            for name, link, vertices in TABLE_MESHES:
                frame = frames[link]
                height = np.einsum('vi,i->v', vertices, frame[2, :3]) + frame[2, 3]
                if name in ('link7', 'link8'):
                    height += frame[2, 1] * (.044 if name == 'link7' else -.044) * opening
                table = min(table, float(height.min() - table_z))
            table_gaps.append(table)
            distances = (np.linalg.norm(points[self.self_rows] - points[self.self_cols], axis=1)
                         - self.radii[self.self_rows] - self.radii[self.self_cols])
            # The public self-collision model subtracts 1 cm from each link radius.
            self_gaps.append(float(distances.min() + .020))
            fold = (np.linalg.norm(points[self.fold_rows] - points[self.fold_cols], axis=1)
                    - self.radii[self.fold_rows] - self.radii[self.fold_cols])
            fold_gaps.append(float(fold.min()))
        table_gaps = np.asarray(table_gaps)
        self_gaps = np.asarray(self_gaps)
        fold_gaps = np.asarray(fold_gaps)
        table_cost = _aggregate(_penalty(table_gaps, .001))
        self_cost = _aggregate(_penalty(self_gaps, .001))
        # Preserve the original sixteen-sample forearm/wrist posture prior.
        fold_cost = _aggregate(np.logaddexp(0., (.005 - fold_gaps[4::4]) / .003) ** 2)
        return dict(centers=np.asarray(centers), table_cost=table_cost,
                    self_cost=self_cost, fold_cost=fold_cost if source_stage else 0.,
                    minimum_table_gap_m=float(table_gaps.min()),
                    minimum_self_gap_m=float(self_gaps.min()),
                    minimum_forearm_wrist_gap_m=float(fold_gaps.min()))

    def cross_geometry(self, first, second):
        """Use the opposite arm's proposed synchronous chord, including simultaneous moves."""
        gaps = (np.linalg.norm(first[:, :, None, :] - second[:, None, :, :], axis=3)
                - self.radii[None, :, None] - self.radii[None, None, :])
        minima = gaps.min(axis=(1, 2))
        return _aggregate(_penalty(minima, .005)), float(minima.min())

    def rank(self, targets, starts, openings, table_z, source_stage, tcp=True):
        """Always return one endpoint for every requested arm without a motion gate."""
        started = time.monotonic()
        pools = {}
        for arm in ARMS:
            if arm in targets:
                goal = np.asarray(targets[arm], float)
                pose = tcp_to_link(goal[:3], goal[3:]) if tcp else goal
                pool = self.candidates(arm, pose, starts[arm], source_stage)
            else:
                pool = [dict(joint_value=starts[arm], endpoint_cost=0.,
                             position_error=0., rotation_error=0., joint_distance=0., accurate=True)]
            for row in pool:
                row['geometry'] = self.path_geometry(arm, starts[arm], row['joint_value'],
                    openings[arm], table_z, source_stage and arm in targets)
                geometry = row['geometry']
                row['independent_cost'] = (row['endpoint_cost'] + geometry['fold_cost']
                    + geometry['table_cost'] + geometry['self_cost']) if arm in targets else 0.
            pools[arm] = pool
        pairs = []
        for left, right in itertools.product(pools['left'], pools['right']):
            cross_cost, cross_gap = self.cross_geometry(left['geometry']['centers'],
                                                       right['geometry']['centers'])
            pairs.append((left['independent_cost'] + right['independent_cost'] + cross_cost,
                          left, right, cross_cost, cross_gap))
        selected = min(pairs, key=lambda item: item[0])
        records = dict(zip(ARMS, selected[1:3]))
        diagnostics = []
        for arm in targets:
            row = records[arm]
            geometry = row['geometry']
            diagnostics.append(dict(arm=arm, position_error_m=row['position_error'],
                rotation_error_rad=row['rotation_error'], joint_distance_rad=row['joint_distance'],
                minimum_table_gap_m=geometry['minimum_table_gap_m'],
                minimum_self_gap_m=geometry['minimum_self_gap_m'],
                minimum_other_arm_gap_m=selected[4],
                table_soft_cost=geometry['table_cost'], self_soft_cost=geometry['self_cost'],
                other_arm_soft_cost=selected[3], original_fold_cost=geometry['fold_cost'],
                candidates=len(pools[arm]), joint_pair_candidates=len(pairs),
                elapsed_seconds=time.monotonic() - started))
        endpoints = {arm: records[arm]['joint_value'].copy() for arm in ARMS}
        return endpoints, diagnostics, pools


def move_targets(ctx, targets, duration, tcp=True, motion_profile=None):
    """Retain the original points, two-waypoint chords and original timing functions."""
    starts = {arm: np.asarray(ctx.state[arm + '_arm_joint_state'], float).copy() for arm in ARMS}
    ctx.joint_targets = {arm: joints.copy() for arm, joints in starts.items()}
    if not hasattr(ctx, '_classify_full_geometry_ranking'):
        ctx._classify_full_geometry_ranking = FullGeometryRanking(ctx.kinematics)
    source_stage = str(getattr(ctx, 'stage', '')).startswith((
        'classify_b_approach_', 'classify_b_pick_'))
    # Cover the observed loaded aperture and any commanded opening transition.
    openings = {arm: max(float(np.asarray(ctx.state[arm + '_ee_joint_state']).reshape(-1)[0]),
                         float(ctx.gripper_targets[arm])) for arm in ARMS}
    endpoints, diagnostics, _ = ctx._classify_full_geometry_ranking.rank(
        targets, starts, openings, ctx.table_z, source_stage, tcp=tcp)
    path = np.stack([np.concatenate([starts[arm] for arm in ARMS]),
                     np.concatenate([endpoints[arm] for arm in ARMS])])
    carrying = any(ctx.gripper_targets[arm] < .8 for arm in targets)
    velocity = 1.7 if getattr(ctx, 'sorting_heavy_grasp', False) else (2.0 if carrying else 2.4)
    yield from ctx._emit_path(path, tuple(targets), duration, velocity,
                              {arm: np.asarray(goal).tolist() for arm, goal in targets.items()})
    ctx.motion_plan.update(endpoint_ik_diagnostics=diagnostics,
                           strategy='classify_public_full_geometry_soft_ik_ranking')
