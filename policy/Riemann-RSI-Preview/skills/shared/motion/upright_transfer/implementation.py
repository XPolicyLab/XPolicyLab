"""Public-pose lift, transfer and lower phases from mature upright carrying."""
from copy import deepcopy
import math

import numpy as np

from harness.execution import ExecutionError


def unit_vector(value, size, label):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    length = np.linalg.norm(vector)
    if vector.size != size or not np.isfinite(vector).all() or not np.isfinite(length) or length < 1e-8:
        raise ExecutionError('A public ' + label + ' must be finite and nonzero')
    return vector / length


def current_targets(observation):
    targets = {side + suffix: np.asarray(observation['state'][side + suffix],
        dtype=np.float64).reshape(-1).copy()
        for side in ('left', 'right') for suffix in ('_ee_pose', '_ee_joint_state')}
    for side in ('left', 'right'):
        unit_vector(targets[side + '_ee_pose'][3:], 4, 'robot orientation')
    return targets


class UprightTransfer:
    def __init__(self, observation, request):
        self.side = request['side']
        self.targets = current_targets(observation)
        field = self.side + '_ee_pose'
        source = self.targets[field].copy()
        destination = source.copy()
        destination[:3] += np.asarray(request['delta'], dtype=float)
        height = max(source[2], destination[2]) + request['clearance']
        source_high, destination_high = source.copy(), destination.copy()
        source_high[2] = height
        destination_high[2] = height
        self.phases = [source_high, destination_high, destination]
        self.phase = 0
        self.finished = False
        self.targets[self.side + '_ee_joint_state'][0] = 0.0

    def proposal(self, observation):
        field = self.side + '_ee_pose'
        current = np.asarray(observation['state'][field]).reshape(-1)
        while self.phase < len(self.phases):
            target = self.phases[self.phase]
            if math.hypot(*(current[:3] - target[:3])) > 0.004:
                break
            self.phase += 1
        self.finished = self.phase >= len(self.phases)
        self.targets[field] = self.phases[min(self.phase, len(self.phases) - 1)].copy()
        return deepcopy(self.targets)


def axis_rotation(quaternion, axis, angle):
    quaternion = unit_vector(quaternion, 4, 'robot orientation')
    axis = unit_vector(axis, 3, 'rotation axis')
    rotation = np.concatenate(([math.cos(angle * 0.5)], axis * math.sin(angle * 0.5)))
    scalar, vector = rotation[0], rotation[1:]
    other_scalar, other_vector = quaternion[0], quaternion[1:]
    result = np.concatenate(([scalar * other_scalar - np.dot(vector, other_vector)],
        scalar * other_vector + other_scalar * vector + np.cross(vector, other_vector)))
    return result / np.linalg.norm(result)


class RelativeRotation:
    def __init__(self, observation, request):
        self.side = request['side']
        self.targets = current_targets(observation)
        field = self.side + '_ee_pose'
        self.targets[field][3:] = axis_rotation(self.targets[field][3:], request['axis'], request['angle'])
        self.error = math.inf
        self.finished = False

    def proposal(self, observation):
        current = np.asarray(observation['state'][self.side + '_ee_pose']).reshape(-1)[3:]
        current = unit_vector(current, 4, 'robot orientation')
        target = self.targets[self.side + '_ee_pose'][3:]
        self.error = 2 * math.acos(float(np.clip(abs(np.dot(current, target)), 0, 1)))
        self.finished = self.error <= 0.02
        return deepcopy(self.targets)
