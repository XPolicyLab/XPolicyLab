import math

import numpy as np


def dense_commands(start, target, steps, unchanged=False):
    if (type(steps) is not int or not 1 <= steps <= 6400 or type(unchanged) is not bool
            or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in (start, target))
            or not 0 <= float(start) <= 1):
        raise ValueError('A gripper ramp uses bounded public normalized command response and own targets')
    target = float(np.clip(target, 0., 1.))
    interpolation_steps = int(steps * .9)
    if unchanged or interpolation_steps <= 1:
        return np.full(steps, target, dtype=float)
    fractions = np.minimum(np.arange(steps) / (interpolation_steps - 1), 1.)
    return float(start) + (target - float(start)) * fractions


def sampled_commands(start, target, steps, unchanged=False, interval=10,
                     time_scale=1., includes_arm_targets=False):
    if (type(interval) is not int or not 1 <= interval <= 40 or type(includes_arm_targets) is not bool
            or isinstance(time_scale, bool) or not math.isfinite(float(time_scale))
            or not .1 <= float(time_scale) <= 4):
        raise ValueError('A ramp retains the declared official observation interval and mature control stride')
    dense = dense_commands(start, target, steps, unchanged)
    stride = interval * (float(time_scale) if includes_arm_targets else 1.)
    count = int(math.ceil(len(dense) / stride))
    source = np.minimum((np.arange(count) + 1) * stride, len(dense)) - 1.
    lower = np.maximum(0, np.floor(source).astype(int))
    upper = np.minimum(lower + 1, len(dense) - 1)
    fraction = np.maximum(0., source - lower)
    return (1. - fraction) * dense[lower] + fraction * dense[upper]
