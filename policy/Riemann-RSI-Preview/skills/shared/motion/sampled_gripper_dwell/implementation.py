import math

import numpy as np


def commands(start, target, dense_steps, interval=10, time_scale=1.0, includes_arm_targets=False):
    if (type(dense_steps) is not int or not 1 <= dense_steps <= 400
            or type(interval) is not int or not 1 <= interval <= 40
            or type(includes_arm_targets) is not bool
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in (start, target, time_scale))
            or not 0 <= start <= 1 or not 0 <= target <= 1 or not .1 <= time_scale <= 4):
        raise ValueError('A gripper dwell samples bounded normalized own commands only')
    stride = interval * (time_scale if includes_arm_targets else 1.0)
    count = max(1, int(math.ceil(dense_steps / stride)))
    indices = np.minimum((np.arange(count) + 1) * stride, dense_steps) - 1
    fractions = indices / max(dense_steps - 1, 1)
    return start + (target - start) * fractions
