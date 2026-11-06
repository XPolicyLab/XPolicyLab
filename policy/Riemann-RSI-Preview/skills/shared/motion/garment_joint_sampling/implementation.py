from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from skills.shared.geometry.measured_cloth_arc.implementation import chunk_angles, fold_arc_wrist
from skills.shared.motion.declared_pickup_kinematics.implementation import run as own_model
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix, matrix_pose


CHUNK = 16
OWN_PAD_FORWARD = .155


@lru_cache(maxsize=1)
def _declared_rigid_geometry():
    """Original public link2..6 geometry; no fingers or scene measurements."""
    root = Path(__file__).with_name('resources')
    expected = {
        'fixed_x5_arm_bounds.json': '4f8b966967759a74b52fd0349c1092d35c07623f2e85435a05901295283c7f9e',
        'fixed_x5_rigid_hulls.npz': 'a0e7d2d86c3622d70618046a8eacf6bfb4a74bd1d0bb4de23d2e4a416af84e97',
    }
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Declared garment rigid geometry changed')
    bounds = json.loads((root / 'fixed_x5_arm_bounds.json').read_text())
    boxes, hulls, normals = {}, {}, {}
    with np.load(root / 'fixed_x5_rigid_hulls.npz', allow_pickle=False) as arrays:
        for item in bounds:
            name = item['link']
            low, high = np.asarray(item['bounding_box_m'], dtype=float)
            boxes[name] = np.asarray(list(itertools.product(*zip(low, high))))
            hulls[name] = arrays[name].copy()
            normals[name] = arrays[name + '_normals'].copy()
    if set(boxes) != {'link2', 'link3', 'link4', 'link5', 'link6'}:
        raise RuntimeError('The garment rigid model contains unexpected links')
    return boxes, hulls, normals


def _interarm_rigid_gap(model, joints, roots):
    """Original separating-axis test on declared own arm/housing hulls."""
    boxes, source_hulls, source_normals = _declared_rigid_geometry()
    parts, hulls, normals = {}, {}, {}
    for arm in ('left', 'right'):
        values = np.asarray(joints[arm], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Rigid clearance needs six finite own joints')
        frames, transform = {}, np.eye(4)
        for index, (origin, axis, angle) in enumerate(zip(model.model.origins, model.model.axes, values), 1):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ motion
            frames['link' + str(index)] = transform.copy()
        parts[arm], hulls[arm], normals[arm] = {}, {}, {}
        for link, vertices in boxes.items():
            world = roots[arm] @ frames[link]
            parts[arm][link] = (vertices @ world[:3, :3].T + world[:3, 3], world[:3, :3].T)
            hulls[arm][link] = source_hulls[link] @ world[:3, :3].T + world[:3, 3]
            normals[arm][link] = source_normals[link] @ world[:3, :3].T
    minimum = float('inf')
    for left, (lp, la) in parts['left'].items():
        for right, (rp, ra) in parts['right'].items():
            axes = np.r_[la, ra, np.cross(la[:, None, :], ra[None, :, :]).reshape(-1, 3)]
            lengths = np.linalg.norm(axes, axis=1)
            axes = axes[lengths > 1e-7] / lengths[lengths > 1e-7, None]
            l, r = lp @ axes.T, rp @ axes.T
            gap = float(np.maximum(r.min(0) - l.max(0), l.min(0) - r.max(0)).max())
            if gap < .008:
                axes = np.r_[axes, normals['left'][left], normals['right'][right]]
                l, r = hulls['left'][left] @ axes.T, hulls['right'][right] @ axes.T
                gap = float(np.maximum(r.min(0) - l.max(0), l.min(0) - r.max(0)).max())
            minimum = min(minimum, gap)
    return minimum


class GarmentOwnModel:
    def __init__(self):
        self.model = own_model()
        self.queries = 0

    def eef(self, joints):
        return self.model.eef(joints)

    def pad(self, joints, root):
        # v45 uses the front pad 0.155 m along link6 X.  The general pickup
        # model deliberately keeps its separate 0.145 m TCP convention.
        offset = np.eye(4)
        offset[0, 3] = OWN_PAD_FORWARD
        return np.asarray(root) @ self.eef(joints) @ offset

    def roots(self, api):
        return {side: self.model.root(api.own_joint_states()[side], api.targets()[side + '_ee_pose'])
                for side in ('left', 'right')}

    def solve(self, target, seed, root, position_tolerance=.004, rotation_tolerance=.05):
        target, seed, root = np.asarray(target), np.asarray(seed), np.asarray(root)
        if (target.shape != (4, 4) or root.shape != (4, 4) or seed.shape != (6,)
                or not np.isfinite(target).all() or not np.isfinite(root).all() or not np.isfinite(seed).all()
                or not .002 <= position_tolerance <= .006 or rotation_tolerance != .05):
            raise ValueError('A garment inverse solve takes only bounded own wrist geometry')
        # No query budget here.  The packaged rule called cuRobo without one,
        # and its own contact_frame() preflights a 48-step arc for each of ten
        # frames per arm: a torso contact costs 318-418 solves on its own, so a
        # 512 cap let the left hem plan and then starved the right hem into a
        # silent None.  The stage is bounded by its sixteen native actions, not
        # by an invented solver count.
        self.queries += 1
        local = np.linalg.inv(root) @ target
        seed = np.where(np.abs(seed) < 1e-8, 0., seed)
        initial = np.clip(seed, self.model.lower + 1e-8, self.model.upper - 1e-8)

        def residual(joints, weight):
            actual = self.eef(joints)
            orientation = Rotation.from_matrix(local[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[actual[:3, 3] - local[:3, 3], weight * orientation]

        for weight in (.17, .08, .05, .02, .35):
            solved = least_squares(lambda joints: residual(joints, weight), initial,
                bounds=(self.model.lower, self.model.upper), max_nfev=100 if weight == .17 else 150,
                ftol=1e-10, xtol=1e-10, gtol=1e-9)
            joints = solved.x.copy()
            initial = np.clip(joints, self.model.lower + 1e-8, self.model.upper - 1e-8)
            for index in range(6):
                alternatives = [joints[index] + winding * 2 * math.pi for winding in range(-3, 4)
                    if self.model.lower[index] <= joints[index] + winding * 2 * math.pi <= self.model.upper[index]]
                if not alternatives:
                    raise ValueError('The original garment own winding is outside its declared URDF')
                joints[index] = min(alternatives, key=lambda value: abs(value - seed[index]))
            actual = self.eef(joints)
            distance = np.linalg.norm(actual[:3, 3] - local[:3, 3])
            angle = np.linalg.norm(Rotation.from_matrix(local[:3, :3] @ actual[:3, :3].T).as_rotvec())
            if distance <= position_tolerance and angle <= rotation_tolerance:
                return joints
        raise ValueError('Original garment inverse kinematics rejected its own endpoint')


def downward_wrist(contact, closing_axis, tilt_degrees):
    contact, closing = np.asarray(contact, dtype=float), np.r_[np.asarray(closing_axis, dtype=float)[:2], 0.]
    if (contact.shape != (3,) or not np.isfinite(contact).all() or not np.isfinite(closing).all()
            or np.linalg.norm(closing) < 1e-8 or tilt_degrees not in (0., 20., -20., 40., -40.)):
        raise ValueError('An original own-pad contact frame needs current finite contact geometry')
    closing /= np.linalg.norm(closing)
    lateral = np.cross(closing, [0., 0., 1.])
    angle = math.radians(tilt_degrees)
    approach = np.asarray([0., 0., -math.cos(angle)]) + lateral * math.sin(angle)
    wrist = np.eye(4)
    wrist[:3, :3] = np.column_stack([approach, closing, np.cross(approach, closing)])
    wrist[:3, 3] = contact - approach * OWN_PAD_FORWARD
    return wrist


def current_snapshot(api):
    result = {side + '_arm_joint_state': api.own_joint_states()[side].tolist() for side in ('left', 'right')}
    result.update({side + '_ee_joint_state': api.targets()[side + '_ee_joint_state'].tolist()
                   for side in ('left', 'right')})
    return result


def sampled_wrist_path(api, model, goals, opening_goals=None, contact=False, starts=None):
    if not isinstance(goals, dict) or not goals or not set(goals).issubset({'left', 'right'}):
        raise ValueError('A garment path has one or two declared own arms')
    roots = model.roots(api)
    # Interpolate from the acknowledged wrist pose, not from the commanded
    # target.  The packaged rule measured both its start pose and its IK seed
    # from the same observation; mixing a commanded start with an acknowledged
    # seed left the first sample several centimetres off, so a full sixteen
    # command chunk could not cover the segment and the carry ended 18 cm short.
    initial = starts or {side: roots[side] @ model.eef(api.own_joint_states()[side]) for side in goals}
    seeds = {side: api.own_joint_states()[side].copy() for side in goals}
    rotations = {side: Slerp([0., 1.], Rotation.from_matrix([initial[side][:3, :3], goals[side][:3, :3]]))
                 for side in goals}
    snapshot = current_snapshot(api)
    controls = []
    for index in range(CHUNK):
        fraction = (index + 1) / CHUNK
        smooth = fraction * fraction * (3 - 2 * fraction)
        commands = deepcopy(snapshot)
        positions = []
        for side in goals:
            wrist = np.eye(4)
            wrist[:3, :3] = rotations[side]([smooth]).as_matrix()[0]
            wrist[:3, 3] = (1 - smooth) * initial[side][:3, 3] + smooth * goals[side][:3, 3]
            if wrist[2, 3] < roots[side][2, 3] + .07:
                raise ValueError('Original garment own-pad mounting clearance is not met')
            solved = model.solve(wrist, seeds[side], roots[side])
            if np.max(np.abs(solved - seeds[side])) > .30:
                raise ValueError('Original garment sample jumps its own joint branch')
            seeds[side] = solved
            commands[side + '_arm_joint_state'] = solved.tolist()
            if opening_goals is not None and side in opening_goals:
                start_opening = snapshot[side + '_ee_joint_state'][0]
                commands[side + '_ee_joint_state'] = [start_opening + fraction * (opening_goals[side] - start_opening)]
            positions.append(wrist[:3, 3])
        if len(positions) == 2 and np.linalg.norm(positions[0] - positions[1]) < .105:
            raise ValueError('Original paired garment wrist clearance is not met')
        controls.append(commands)
    return controls


def sampled_joint_path(api, goals):
    if not isinstance(goals, dict) or not goals or not set(goals).issubset({'left', 'right'}):
        raise ValueError('The original joint chunk has declared own arms')
    snapshot = current_snapshot(api)
    controls = []
    for index in range(CHUNK):
        fraction = (index + 1) / CHUNK
        smooth = fraction * fraction * (3 - 2 * fraction)
        commands = deepcopy(snapshot)
        for side, goal in goals.items():
            start = np.asarray(snapshot[side + '_arm_joint_state'])
            target = np.asarray(goal)
            if target.shape != (6,) or not np.isfinite(target).all():
                raise ValueError('Own joint goals must be six finite public values')
            commands[side + '_arm_joint_state'] = ((1 - smooth) * start + smooth * target).tolist()
        controls.append(commands)
    return controls


def sampled_gripper_clock(api, model, arms, release=False, cuff=False):
    if (not isinstance(arms, tuple) or not 1 <= len(arms) <= 2 or len(set(arms)) != len(arms)
            or not set(arms).issubset({'left', 'right'}) or type(release) is not bool or type(cuff) is not bool):
        raise ValueError('A garment jaw clock has declared own arms and close/release mode')
    roots = model.roots(api)
    starts = {side: pose_matrix(api.targets()[side + '_ee_pose']) for side in arms}
    seeds = {side: api.own_joint_states()[side].copy() for side in arms}
    snapshot = current_snapshot(api)
    controls = []
    stationary = 8 if cuff else 4
    lift_steps, lift_height = (8, .065) if cuff else (6, .025)
    for index in range(CHUNK):
        command = deepcopy(snapshot)
        for side in arms:
            opening = float(snapshot[side + '_ee_joint_state'][0])
            if release:
                if cuff:
                    value = opening + (1 - opening) * min(1., (index + 1) / stationary)
                else:
                    partial = opening + .30 * (1 - opening)
                    if index < stationary:
                        value = opening + (partial - opening) * (index + 1) / stationary
                    elif index < 10:
                        value = partial
                    else:
                        value = partial + (1 - partial) * (index + 1 - 10) / 6
                step, lift = index + 1 - stationary, lift_height
                total = lift_steps
            else:
                value = opening * max(0., 1 - (index + 1) / (CHUNK * .75))
                step, lift, total = index + 1 - 12, .012, 4
            command[side + '_ee_joint_state'] = [float(value)]
            if step > 0:
                fraction = min(1., step / total)
                target = starts[side].copy()
                target[2, 3] += lift * fraction * fraction * (3 - 2 * fraction)
                solved = model.solve(target, seeds[side], roots[side])
                if np.max(np.abs(solved - seeds[side])) > .30:
                    raise ValueError('Original garment jaw/lift clock jumps its own branch')
                seeds[side] = solved
                command[side + '_arm_joint_state'] = solved.tolist()
        controls.append(command)
    return controls


def sampled_body_arc(api, model, starts, displacements, begin, end, chunk_index, chunk_count):
    if set(starts) != {'left', 'right'} or set(displacements) != set(starts):
        raise ValueError('The original hem arc requires both own arms')
    roots = model.roots(api)
    seeds = api.own_joint_states()
    snapshot = current_snapshot(api)
    controls = []
    for angle in chunk_angles(begin, end, chunk_index, chunk_count):
        command, wrists = deepcopy(snapshot), {}
        for side in starts:
            wrist = fold_arc_wrist(starts[side], displacements[side], angle)
            if wrist[2, 3] < roots[side][2, 3] + .07:
                raise ValueError('The garment arc intersects its own mounting clearance')
            solved = model.solve(wrist, seeds[side], roots[side], position_tolerance=.006 if begin else .004)
            if np.max(np.abs(solved - seeds[side])) > .30:
                raise ValueError('Original paired hem arc has a discontinuous joint branch')
            seeds[side], wrists[side] = solved, wrist
            command[side + '_arm_joint_state'] = solved.tolist()
            command[side + '_ee_joint_state'] = [0.]
        if np.linalg.norm(wrists['left'][:3, 3] - wrists['right'][:3, 3]) < .105:
            raise ValueError('Original paired wrist clearance failed along the hem arc')
        controls.append(command)
    return controls


def cuff_carry_chunks(api, model, goals):
    """Original v45 measured-pose preflight: one chunk only below .24 rad."""
    if not isinstance(goals, dict) or not goals or not set(goals).issubset({'left', 'right'}):
        raise ValueError('A cuff carry needs its declared own arm endpoint')
    starts = {side: api.own_joint_states()[side].copy() for side in goals}
    try:
        controls = sampled_wrist_path(api, model, goals)
    except ValueError:
        return 2
    maximum = 0.
    for side, start in starts.items():
        sequence = np.vstack([start, [command[side + '_arm_joint_state'] for command in controls]])
        maximum = max(maximum, float(np.max(np.abs(np.diff(sequence, axis=0)))))
    return 1 if maximum < .24 else 2


def body_fold_chunks(api, model, starts, displacements):
    """Reserve the original five tail chunks before compacting the hem arc.

    This does not add the original settle/recheck stages to a workflow. The
    reservation and complete compact-path preflight are v45's unchanged
    selection rule; the external graph still owns its actual continuation.
    """
    epoch = api.issued_action_count()
    if type(epoch) is not int or not 0 <= epoch <= 500:
        raise ValueError('A garment budget uses acknowledged own action count')
    if epoch + 7 * CHUNK <= 500:
        return 2
    roots = model.roots(api)
    try:
        controls = sampled_body_arc(api, model, starts, displacements,
                                    math.pi / 2., math.pi, 0, 1)
        for command in controls:
            joints = {side: np.asarray(command[side + '_arm_joint_state']) for side in ('left', 'right')}
            if _interarm_rigid_gap(model, joints, roots) < .008:
                raise ValueError('Original compact hem rigid clearance is not met')
    except ValueError:
        # Preserve the full route when the original compact path was unsafe,
        # even if its nominal finish exceeds the remaining official budget.
        return 2
    return 1
