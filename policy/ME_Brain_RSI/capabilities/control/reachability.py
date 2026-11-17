"""Nominal sampled approach/contact/lift IK, not scene collision checking."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import tcp_to_ee, quat_matrix
from .kinematics import SerialArm, pose_matrix


def lift_points(contact, side, inward=0.):
    contact = np.asarray(contact, float)
    direction = np.array([-.3 if side == 'left' else .3, -.45])-contact[:2]
    direction /= max(np.linalg.norm(direction), 1e-8)
    return [contact+np.array([0, 0, .035]),
            contact+np.r_[direction*inward*.5, .09],
            contact+np.r_[direction*inward, .16]]


def check_path(arm, q, points, rotation, calibration):
    errors, previous = [], None
    for point in points:
        samples = [point] if previous is None else [previous+(point-previous)*t for t in (.33, .67, 1.)]
        for sample in samples:
            pose = tcp_to_ee(sample, rotation, calibration['tool_offset_in_ee'])
            fit = arm.inverse(pose_matrix(pose), q, restarts=0)
            next_q = np.asarray(fit['joints'])
            errors.append([fit['position_error_m'], fit['orientation_error_rad']])
            if (fit['position_error_m'] >= .004 or fit['orientation_error_rad'] >= .06
                    or np.pi-abs(next_q[-1]) <= .16
                    or (previous is not None and np.max(np.abs(next_q-q)) > .9)):
                return False, errors
            q = next_q
        previous = np.asarray(point)
    return True, errors


def select_trajectory(grasp, calibration, state, compact=False):
    side = 'left' if grasp.xy[0] < 0 else 'right'
    q = np.asarray((state or {}).get(side+'_arm_joint_state', []), float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('GP4 requires own six arm joints for lift reachability')
    arm = SerialArm.public_x5(side)
    top = Rotation.from_euler('z', grasp.yaw).as_matrix() @ Rotation.from_euler('y', np.pi/2).as_matrix()
    original_z = max(grasp.surface_z+.09, grasp.contact_z+.12)
    compact_z = max(grasp.surface_z+.035, grasp.contact_z+.060)
    radial = np.asarray(grasp.xy)-np.array([-.3 if side == 'left' else .3, -.45])
    radial /= max(np.linalg.norm(radial), 1e-8)
    axis = np.array([-radial[1], radial[0], 0.])
    choices = [] if compact else [(top, original_z, 0., False)]
    for tilt in (0., 15., 25., 35.):
        rot = Rotation.from_rotvec(-np.deg2rad(tilt)*axis).as_matrix() @ top
        choices += [(rot, compact_z, tilt, False),
                    (rot @ Rotation.from_euler('x', np.pi).as_matrix(), compact_z, tilt, True)]
    trials = []
    contact = np.r_[grasp.xy, grasp.contact_z]
    for rot, z, tilt, flipped in choices:
        for inward in (0., .06, .12):
            points = [np.r_[grasp.xy, z], contact, *lift_points(contact, side, inward)]
            ok, errors = check_path(arm, q, points, rot, calibration)
            row = dict(approach_z=z, tilt_degrees=tilt, finger_symmetric_flip=flipped,
                       inward_m=inward, errors=errors, accepted=ok)
            trials.append(row)
            if ok:
                poses = [tcp_to_ee(p, rot, calibration['tool_offset_in_ee']) for p in points]
                return side, poses, dict(method='nominal_sampled_full_grasp_path', selected=row,
                                         trials=trials, collision_checked=False)
    raise ValueError('No nominal sampled approach/contact/lift path')


def recheck_lift(current, side, state, calibration):
    """After bounded contact adaptation, revalidate from ACTUAL own pose."""
    rot = quat_matrix(current[3:])
    contact = current[:3]+rot @ calibration['tool_offset_in_ee']
    q = np.asarray(state[side+'_arm_joint_state'], float)
    arm = SerialArm.public_x5(side)
    for inward in (0., .06, .12):
        points = [contact, *lift_points(contact, side, inward)]
        ok, errors = check_path(arm, q, points, rot, calibration)
        if ok:
            return [tcp_to_ee(p, rot, calibration['tool_offset_in_ee']) for p in points[1:]], dict(
                method='measured_contact_nominal_lift_recheck', inward_m=inward,
                errors=errors, collision_checked=False)
    raise ValueError('Adapted contact has no nominal lift path')
