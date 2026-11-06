from copy import deepcopy

import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.public_phase_execution.implementation import PhasedProgram, PublicRobotAPI, pose_phase


class NormalizedCommandClock(PhasedProgram):
    preserve_commanded_joint_targets = True
    remember_nonrepeatable_commands = True

    def __init__(self, api, request, openings, segment_end_indices):
        if (type(api) is not PublicRobotAPI or not isinstance(request, dict) or set(request) - {'skill', 'side', 'steps', 'reason'}
                or not isinstance(request.get('skill'), str) or not 1 <= len(request['skill']) <= 64
                or request.get('side') not in ('left', 'right')
                or type(request.get('steps')) is not int or not 1 <= request['steps'] <= 640
                or not isinstance(openings, list) or not 1 <= len(openings) <= 40
                or any(type(value) not in (int, float) or not np.isfinite(value)
                       or not 0. <= value <= 1. for value in openings)
                or not isinstance(segment_end_indices, list) or not segment_end_indices
                or any(type(value) is not int for value in segment_end_indices)
                or segment_end_indices != sorted(set(segment_end_indices))
                or segment_end_indices[0] < 1 or segment_end_indices[-1] != len(openings)):
            raise ExecutionError('A normalized clock contains only bounded already-sampled own commands')
        self.original_commands = deepcopy(openings)
        self.segment_end_indices = deepcopy(segment_end_indices)
        target = api.targets()[request['side'] + '_ee_pose']
        phases = [pose_phase('normalized_command_' + str(index), target, opening, 1, False)
                  for index, opening in enumerate(openings)]
        super().__init__(api, request, phases)

    def feedback(self, observation, budget=False):
        result = super().feedback(observation, budget)
        result['image_measurements']['normalized_command_clock'] = {
            'prepared_native_commands': len(self.original_commands),
            'issued_native_commands': self.actions,
            'segment_end_indices': deepcopy(self.segment_end_indices),
            'command_source': 'explicit_already_sampled_own_normalized_commands',
            'sampling_reapplied': False,
            'arm_trajectory_replanned': False,
            'held_joints_source': 'last_own_command_or_initial_public_joints',
            'true_fingers_or_contact_read': False,
            'official_success_claimed': False}
        return result


def run(api, request, openings, segment_end_indices):
    return NormalizedCommandClock(api, request, openings, segment_end_indices)
