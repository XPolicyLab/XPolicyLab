import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from harness.runtime.observation import public_observation
from skills.shared.motion.mature_x5_fallback.implementation import pose_matrix


PROFILES = {
    'vessel_grasp': (.09, .65), 'vessel_grasp_relay': (.12, .65),
    'vessel_lift': (.16, .65), 'vessel_lift_relay': (.20, .65),
    'vessel_loaded_move': (.18, .65), 'vessel_observe': (.12, .65),
    'vessel_tilt': (.12, .65), 'vessel_tilt_relay': (.12, .9),
    'vessel_tilt_slow': (.12, .4), 'vessel_upright': (.12, .65),
    'vessel_upright_small': (.12, 1.2), 'vessel_upright_relay': (.12, 1.5),
    'vessel_return_high': (.24, .65), 'vessel_return_lower': (.18, .65),
    'vessel_return_relay': (.26, .65), 'vessel_retreat': (.18, .65),
    'vessel_retreat_relay': (.24, .65), 'vessel_precise': (.04, .65),
    'vessel_recovery': (.26, 1.2),
    'vessel_grasp_recovery': (.09, 1.2), 'vessel_grasp_relay_recovery': (.12, 1.2),
    'vessel_ball_contact': (.08, 1.2),
    'vessel_upright_relay_recovery': (.26, 1.5),
}


def native_count(observed, destination, profile):
    if profile not in PROFILES:
        raise ValueError('Timed own motion requires a declared original sampling profile')
    start, target = pose_matrix(observed), pose_matrix(destination)
    speed, angular_speed = PROFILES[profile]
    distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    angle = float(Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude())
    return max(4, int(math.ceil(max(distance / speed, angle / angular_speed) * 25)))


def solve_samples(kinematics, observed, current, destination, profile, remaining_actions):
    if type(remaining_actions) is not int or not 1 <= remaining_actions <= 400:
        raise ValueError('Timed motion has a finite declared official-action allowance')
    current = np.asarray(current, dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError('Timed motion starts from public own-joint feedback')
    start, target = pose_matrix(observed), pose_matrix(destination)
    count = native_count(observed, destination, profile)
    if count > remaining_actions:
        return None
    root = start @ np.linalg.inv(kinematics.eef(current))
    eef_from_tcp = np.linalg.inv(kinematics.eef(current)) @ kinematics.tcp(current)
    rotations = Slerp([0., 1.], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
    fractions = [index / count for index in range(1, count + 1)]
    previous_fraction, subdivisions = 0., 0
    seed, samples = current.copy(), []
    while fractions:
        fraction = fractions.pop(0)
        blend = fraction * fraction * (3. - 2. * fraction)
        waypoint = np.eye(4)
        waypoint[:3, :3] = rotations([blend]).as_matrix()[0]
        waypoint[:3, 3] = (1. - blend) * start[:3, 3] + blend * target[:3, 3]
        try:
            solved = kinematics.solve_tcp(np.linalg.inv(root) @ waypoint @ eef_from_tcp, seed,
                position_tolerance=.004, rotation_tolerance=.06, rotation_weight=.2,
                branch_jump_tolerance=.35, retry_relaxed=False)
        except RuntimeError as error:
            if 'branch jump' not in str(error) or subdivisions >= 12 or len(samples) + len(fractions) + 2 > remaining_actions:
                raise
            fractions[:0] = [(previous_fraction + fraction) / 2, fraction]
            subdivisions += 1
            continue
        seed = np.asarray(solved, dtype=np.float64)
        if seed.shape != (6,) or not np.isfinite(seed).all():
            raise ValueError('Own IK returned an invalid command')
        samples.append(seed.copy())
        previous_fraction = fraction
    return np.asarray(samples, dtype=np.float32)


class NativeTimedCartesian:
    def __init__(self, urdf):
        from skills.shared.motion.mature_x5_fallback.kinematics import X5MeasuredFallbackKinematics
        self.kinematics = X5MeasuredFallbackKinematics(urdf)

    def plan(self, payload, substeps):
        if type(substeps) is not int or substeps < 1 or payload['operation'] not in PROFILES or payload['time_scale'] != 1.:
            raise ValueError('Original 25 Hz samples are issued once, without another time-scale conversion')
        observation = public_observation(payload['observation'])
        side = payload['side']
        failure = {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
        try:
            samples = solve_samples(self.kinematics, observation['state'][side + '_ee_pose'],
                observation['state'][side + '_arm_joint_state'], payload['pose'], payload['operation'],
                payload['remaining_actions'])
        except RuntimeError:
            return failure
        if samples is None:
            return dict(failure, status='Budget')
        return {'status': 'Success', 'positions': samples, 'dense_waypoints': len(samples) * substeps,
                'official_waypoints': len(samples)}
