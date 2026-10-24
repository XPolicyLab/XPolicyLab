"""Explicit xyzw poses. All transformations are robot geometry, not perception."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def vector(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f'Expected {n} finite numbers')
    return a


def rotation(q):
    q = vector(q, 4)
    if np.linalg.norm(q) < 1e-8: raise ValueError('Zero quaternion')
    return Rotation.from_quat(q)


def tcp_from_link(link_wxyz, offset):
    p = vector(link_wxyz, 7)
    r = rotation(np.roll(p[3:], -1))
    return {'position_m': (p[:3] + r.apply(offset)).tolist(), 'quaternion_xyzw': r.as_quat().tolist()}


def link_from_tcp(pose, offset):
    r = rotation(pose['quaternion_xyzw'])
    return np.r_[vector(pose['position_m'], 3) - r.apply(offset), np.roll(r.as_quat(), 1)]


def interpolate(start, end, fraction):
    p = np.asarray(start['position_m']) + fraction * (np.asarray(end['position_m']) - start['position_m'])
    rs = Rotation.from_quat([start['quaternion_xyzw'], end['quaternion_xyzw']])
    return {'position_m': p.tolist(), 'quaternion_xyzw': Slerp([0, 1], rs)(fraction).as_quat().tolist()}


def errors(a, b):
    return (float(np.linalg.norm(np.asarray(a['position_m']) - b['position_m'])),
            float((rotation(a['quaternion_xyzw']).inv() * rotation(b['quaternion_xyzw'])).magnitude()))


def return_steps_estimate(state, initial, hz, joint_speed=1.5):
    """Nominal budget for existing sequential joint-bookmark returns, not a guarantee."""
    steps = {arm: int(np.ceil(max(.4, np.max(np.abs(
        np.asarray(values['joint_positions_rad']) - initial[arm]['joint_positions_rad'])) / joint_speed) * hz))
        for arm, values in state.items()}
    return {'per_arm': steps, 'sequential_total': sum(steps.values()),
            'together': max(steps.values()) if steps else 0,  # robot.move_to_mark('initial'): both arms at once
            'basis': f'Current joints to initial bookmarks at nominal {joint_speed:g} rad/s; includes minimum duration. '
                     'Estimate only, excluding gripper opening, inspection and any recovery.'}


def make_target(start, params, base_quat_xyzw):
    r = rotation(start['quaternion_xyzw'])
    frame = params.get('frame', 'world')
    frames = {'world': Rotation.identity(), 'tcp': r, 'base': rotation(base_quat_xyzw)}
    if frame not in frames: raise ValueError('frame must be world, base, or tcp (frozen at motion start)')
    rf = frames[frame]
    p = vector(start['position_m'], 3)
    if 'position_m' in params:
        if frame != 'world': raise ValueError('Absolute position currently requires world frame')
        p = vector(params['position_m'], 3)
    if 'delta_m' in params: p = p + rf.apply(vector(params['delta_m'], 3))
    if params.get('quaternion_xyzw') is not None: r = rotation(params['quaternion_xyzw'])
    if 'rotation_axis' in params:
        axis = params['rotation_axis']
        if isinstance(axis, str): axis = np.eye(3)[['x', 'y', 'z'].index(axis)]
        axis = vector(axis, 3)
        if np.linalg.norm(axis) < 1e-8: raise ValueError('Zero rotation axis')
        angle = float(params['angle_deg'])
        if not np.isfinite(angle): raise ValueError('Nonfinite angle')
        r = Rotation.from_rotvec(rf.apply(axis / np.linalg.norm(axis)) * np.deg2rad(angle)) * r
    return {'position_m': p.tolist(), 'quaternion_xyzw': r.as_quat().tolist()}
