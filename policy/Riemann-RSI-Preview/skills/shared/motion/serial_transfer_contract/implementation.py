from copy import deepcopy
import math

import numpy as np


STAGES = ('pick_left', 'park_left', 'pick_right', 'place_right', 'clear_right',
          'place_left', 'clear_left', 'settle')
SIDE = {stage: 'right' if 'right' in stage else 'left' for stage in STAGES}
TARGETS = ('approach', 'grasp', 'lift', 'above_destination', 'place', 'retreat')
MINIMUM = {'pick_left': 20, 'park_left': 2, 'pick_right': 13, 'place_right': 13,
           'clear_right': 2, 'place_left': 9, 'clear_left': 2, 'settle': 7}


def validate_recipe(value):
    if not isinstance(value, dict) or set(value) != {'left', 'right', 'support_plane_m'}:
        raise ValueError('A serial transfer recipe contains only two own-arm target sets and its public support hypothesis')
    if type(value['support_plane_m']) not in (int, float) or not math.isfinite(value['support_plane_m']):
        raise ValueError('A command support hypothesis must be finite')
    for side in ('left', 'right'):
        if not isinstance(value[side], dict) or set(value[side]) != set(TARGETS):
            raise ValueError('A serial transfer recipe requires the declared target sequence')
        for target in value[side].values():
            vector = np.asarray(target, dtype=float)
            if (vector.shape != (7,) or not np.isfinite(vector).all()
                    or not .99 <= np.linalg.norm(vector[3:]) <= 1.01):
                raise ValueError('Command targets must be finite own robot poses')
    return deepcopy(value)
