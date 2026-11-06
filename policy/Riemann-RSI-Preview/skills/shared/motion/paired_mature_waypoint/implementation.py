from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np


CALLBACKS = {'wrist_pose', 'plan_pose', 'gripper_controls', 'execute_joint', 'observed_pose'}
STEP_FIELDS = {'points_m', 'height_m', 'support_height_m', 'quaternion_wxyz', 'phase', 'hold_closed', 'check_endpoint'}


def vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('Invalid finite own-robot ' + label)
    return result.copy()


def merge(*sequences):
    if not sequences:
        return []
    result = []
    for index in range(max(map(len, sequences))):
        command = {}
        for sequence in sequences:
            if sequence:
                command.update(deepcopy(sequence[min(index, len(sequence) - 1)]))
        result.append(command)
    return result


def align(sequences):
    if len(sequences) != 2 or not all(sequences):
        raise ValueError('A paired waypoint requires two nonempty own-arm paths')
    duration = max(map(len, sequences))
    return [[deepcopy(sequence[int(round(position))])
             for position in np.linspace(0, len(sequence) - 1, duration)]
            for sequence in sequences]


class PairedWaypointAPI:
    __slots__ = ('__callbacks', '__lower', '__upper', '__horizon', '__issued',
                 '__support_height', '__time_scale', '__failed')

    def __init__(self, callbacks, *, lower_limits, upper_limits, support_height_m,
                 remaining_actions, time_scale=1.):
        if (set(callbacks) != CALLBACKS or any(not callable(value) for value in callbacks.values())
                or type(remaining_actions) is not int or not 1 <= remaining_actions <= 1000
                or type(time_scale) not in (int, float) or not np.isfinite(time_scale)
                or not .25 <= time_scale <= 3.
                or type(support_height_m) not in (int, float) or not np.isfinite(support_height_m)):
            raise ValueError('A paired tool accepts only its declared own-robot capabilities and budget')
        self.__lower = vector(lower_limits, 6, 'lower limits')
        self.__upper = vector(upper_limits, 6, 'upper limits')
        if np.any(self.__lower >= self.__upper):
            raise ValueError('Own joint bounds must be ordered')
        self.__callbacks = MappingProxyType(dict(callbacks))
        self.__horizon, self.__issued = remaining_actions, 0
        self.__failed = False
        self.__support_height, self.__time_scale = float(support_height_m), float(time_scale)

    @property
    def actions_issued(self):
        return self.__issued

    @property
    def support_height_m(self):
        return self.__support_height

    def poses(self, points, height, quaternion):
        result = [vector(self.__callbacks['wrist_pose'](point.copy(),
                    self.__support_height + height, quaternion.copy()), 7, 'requested wrist pose')
                  for point in points]
        if any(abs(np.linalg.norm(pose[3:]) - 1.) > .01 for pose in result):
            raise ValueError('Own wrist poses require unit orientations')
        return result

    def sequence(self, side, pose, hold_closed, gripper_target=0.):
        if (side not in ('left', 'right') or type(gripper_target) not in (int, float)
                or not np.isfinite(gripper_target) or not 0. <= gripper_target <= 1.
                or hold_closed and gripper_target != 0.):
            raise ValueError('A mature waypoint declares its own side and normalized jaw target')
        path = self.__callbacks['plan_pose'](side, pose.copy())
        self.validate_commands(path, side, arm=True)
        grips = self.gripper_sequence(side, gripper_target, len(path))
        self.validate_commands(grips, side, arm=False)
        if hold_closed:
            if any(float(np.asarray(value[side + '_ee_joint_state']['position'])[0]) != 0.
                   for value in grips[-1:]):
                raise ValueError('A closed preload must end at the normalized closed command')
            grips = [deepcopy(grips[-1]) for unused in grips]
        return merge(path, grips)

    def gripper_sequence(self, side, target, steps):
        if (side not in ('left', 'right') or type(target) not in (int, float)
                or not np.isfinite(target) or not 0. <= target <= 1.
                or type(steps) is not int or not 1 <= steps <= 20000):
            raise ValueError('A jaw skill declares one normalized target and bounded dense sample count')
        controls = self.__callbacks['gripper_controls'](side, float(target), steps)
        self.validate_commands(controls, side, arm=False)
        if len(controls) != steps or controls[-1][side + '_ee_joint_state']['position'] != [float(target)]:
            raise ValueError('A jaw path must preserve the declared sample count and final request')
        return controls

    def validate_commands(self, controls, side, arm):
        key = side + ('_arm_joint_state' if arm else '_ee_joint_state')
        size = 6 if arm else 1
        if not isinstance(controls, list) or not 1 <= len(controls) <= 20000:
            raise ValueError('A paired route requires a finite declared target sequence')
        for command in controls:
            if not isinstance(command, dict) or set(command) != {key}:
                raise ValueError('A route contains only the requested own joint target')
            target = command[key]
            if not isinstance(target, dict) or not {'position'} <= set(target) <= {'position', 'velocity'}:
                raise ValueError('A route cannot contain simulator state or extra payload fields')
            position = vector(target['position'], size, 'joint target')
            if arm and (np.any(position < self.__lower) or np.any(position > self.__upper)):
                raise ValueError('Paired own-arm target exceeds fixed public URDF limits')
            if not arm and not 0. <= position[0] <= 1.:
                raise ValueError('Jaw targets are normalized commands, never measured fingers')
            if 'velocity' in target:
                vector(target['velocity'], size, 'target velocity')

    def execute(self, controls, phase, gripper_only=False):
        if self.__failed:
            raise RuntimeError('Previous paired execution is unacknowledged; do not replay an uncertain action suffix')
        allowed = {side + suffix for side in ('left', 'right')
                   for suffix in ('_arm_joint_state', '_ee_joint_state')}
        if (not isinstance(controls, list) or not 1 <= len(controls) <= 20000
                or not isinstance(phase, str) or not phase or len(phase) > 96):
            raise ValueError('Own execution requires a finite dense target list and named phase')
        for command in controls:
            if not isinstance(command, dict) or not command or not set(command) <= allowed:
                raise ValueError('An execution packet contains only declared own arm/jaw targets')
            for key in command:
                self.validate_commands([{key: command[key]}], key.split('_', 1)[0],
                    arm=key.endswith('_arm_joint_state'))
        if (type(gripper_only) is not bool or gripper_only and any(
                any(not key.endswith('_ee_joint_state') for key in control) for control in controls)):
            raise ValueError('The gripper-only clock cannot contain arm targets')
        scale = 1. if gripper_only else self.__time_scale
        count = max(1, int(math.ceil(len(controls) / (10. * scale))))
        if self.__issued + count > self.__horizon:
            raise RuntimeError('Paired waypoint would exceed the remaining official action budget')
        self.__failed = True
        receipt = self.__callbacks['execute_joint'](deepcopy(controls), phase)
        if (not isinstance(receipt, dict) or set(receipt) != {'actions_executed'}
                or type(receipt['actions_executed']) is not int or receipt['actions_executed'] != count):
            raise RuntimeError('The official own-action acknowledgement differs from dense clock sampling')
        self.__issued += count
        self.__failed = False

    def observed_pose(self, side):
        return vector(self.__callbacks['observed_pose'](side), 7, 'current public wrist pose')


def run(api, step):
    if type(api) is not PairedWaypointAPI or not isinstance(step, dict) or set(step) != STEP_FIELDS:
        raise ValueError('Use the declared paired own-arm API and exact waypoint fields')
    points = np.asarray(step['points_m'], dtype=float)
    quaternion = vector(step['quaternion_wxyz'], 4, 'orientation')
    if (points.shape != (2, 3) or not np.isfinite(points).all()
            or abs(np.linalg.norm(quaternion) - 1.) > .01
            or type(step['height_m']) not in (int, float) or not np.isfinite(step['height_m'])
            or not 0. <= step['height_m'] <= .5
            or type(step['support_height_m']) not in (int, float)
            or not np.isfinite(step['support_height_m'])
            or abs(float(step['support_height_m']) - api.support_height_m) > 1e-6
            or type(step['hold_closed']) is not bool or type(step['check_endpoint']) is not bool
            or not isinstance(step['phase'], str) or not step['phase'] or len(step['phase']) > 96):
        raise ValueError('A paired waypoint requires finite public geometry and explicit finite feedback conditions')
    poses = api.poses(points, float(step['height_m']), quaternion)
    errors = []
    for attempt in range(2 if step['check_endpoint'] else 1):
        sequences = [api.sequence(side, pose, step['hold_closed'])
                     for side, pose in zip(('left', 'right'), poses)]
        controls = merge(*align(sequences))
        phase = step['phase'] if not attempt else step['phase'] + '_endpoint_retry'
        api.execute(controls, phase)
        if not step['check_endpoint']:
            return {'status': 'issued_without_endpoint_check', 'attempts': attempt + 1,
                    'contact_confirmed': False, 'official_success_claimed': False}
        actual = [api.observed_pose(side) for side in ('left', 'right')]
        errors = [float(np.linalg.norm(observed[:3] - pose[:3]))
                  for observed, pose in zip(actual, poses)]
        if max(errors) <= .025:
            return {'status': 'own_endpoint_qualified', 'attempts': attempt + 1,
                    'endpoint_errors_m': errors, 'contact_confirmed': False,
                    'official_success_claimed': False}
    raise RuntimeError('Keyboard paired waypoint endpoint error ' + str(errors))
