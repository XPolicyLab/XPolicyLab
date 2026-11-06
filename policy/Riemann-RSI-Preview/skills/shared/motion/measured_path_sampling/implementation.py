import math

import numpy as np


PROFILES = {
    'contact': {'maximum_joint_step': .08, 'speed': 1., 'settle_steps': 4},
    'carry': {'maximum_joint_step': .065, 'speed': 1.3, 'settle_steps': 2},
    'free': {'maximum_joint_step': .13, 'speed': 2.6, 'settle_steps': 2},
    'home': {'maximum_joint_step': .17, 'speed': 3.8, 'settle_steps': 2},
    'contact_correction': {'maximum_joint_step': .04, 'speed': .6, 'settle_steps': 6},
    'view': {'maximum_joint_step': .10, 'speed': 2., 'settle_steps': 2},
}


def resample(path, current, maximum_joint_step=.075, speed=1.8, minimum_controls=5):
    positions = np.asarray(path, dtype=float)
    current = np.asarray(current, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 6 or not 1 <= len(positions) <= 20000
            or current.shape != (6,) or not np.isfinite(positions).all() or not np.isfinite(current).all()
            or not math.isfinite(maximum_joint_step) or maximum_joint_step <= 0
            or not math.isfinite(speed) or speed <= 0
            or type(minimum_controls) is not int or not 1 <= minimum_controls <= 250):
        raise ValueError('Sample only a bounded planned own-arm joint path')
    positions = np.vstack([current, positions])
    distance = np.r_[0., np.cumsum(np.max(np.abs(np.diff(positions, axis=0)), axis=1))]
    count = max(minimum_controls, int(np.ceil(distance[-1] / maximum_joint_step)),
                int(np.ceil((len(positions) - 1) / (10 * speed))))
    if count > 250:
        raise ValueError('Planned own-arm route exceeds its original per-phase action budget')
    if distance[-1] < 1e-8:
        return np.repeat(positions[-1:], minimum_controls, axis=0)
    samples = np.linspace(0, distance[-1], count + 1)[1:]
    return np.stack([np.interp(samples, distance, positions[:, joint]) for joint in range(6)], axis=1)


def run(path, current, profile, remaining_actions):
    if profile not in PROFILES or type(remaining_actions) is not int or not 0 <= remaining_actions <= 3600:
        raise ValueError('Use a declared measured-motion profile and remaining action budget')
    settings = PROFILES[profile]
    positions = resample(path, current, settings['maximum_joint_step'], settings['speed'])
    if len(positions) + settings['settle_steps'] > remaining_actions:
        return {'status': 'budget', 'positions': [], 'settle_steps': 0}
    return {'status': 'Success', 'positions': positions, 'settle_steps': settings['settle_steps']}
