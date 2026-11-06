from copy import deepcopy
import math

import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.normalized_gripper_ramp.implementation import sampled_commands
from skills.shared.motion.public_phase_execution.implementation import PhasedProgram
from skills.shared.motion.rigid_tool_command_tracking.implementation import (
    checked_pose, predicted_tool_pose, tracking_correction,
)


def move_phase(name, pose, opening, precise=True, tool_goal=None, attachment=None,
               tolerance=.004, rotation_tolerance_degrees=3., time_scale=1.4, operation='pose'):
    if type(precise) is not bool or operation not in ('pose', 'joint'):
        raise ValueError('Mature movement declares own pose or own home joints and explicit feedback precision')
    if (not 0 < tolerance <= .04 or not 0 < rotation_tolerance_degrees <= 20
            or not .1 <= time_scale <= 4 or (tool_goal is None) != (attachment is None)):
        raise ValueError('Mature feedback settings and public attachment hypotheses must be bounded')
    phase = {'name': name, 'pose': checked_pose(pose).copy(), 'opening': float(opening),
             'minimum_steps': 1, 'require_pose': True}
    profile = {'precise': precise, 'target': phase['pose'].copy(),
        'tool_goal': checked_pose(tool_goal).copy() if tool_goal is not None else None,
        'attachment': checked_pose(attachment).copy() if attachment is not None else None,
        'tolerance': float(tolerance), 'rotation_tolerance_degrees': float(rotation_tolerance_degrees),
        'time_scale': float(time_scale), 'operation': operation, 'dense_steps': None}
    return phase, profile


def grip_phase(name, pose, opening, seconds=.28, lift_settle=False):
    if (type(lift_settle) is not bool or type(seconds) not in (int, float)
            or not math.isfinite(seconds) or not .04 <= seconds <= .64):
        raise ValueError('A mature gripper clock uses a finite declared duration')
    duration = min(seconds, .12) if lift_settle else seconds
    dense_steps = max(10, round(duration / .004))
    official_steps = int(math.ceil(dense_steps / 10))
    phase = {'name': name, 'pose': checked_pose(pose).copy(), 'opening': float(opening),
             'minimum_steps': official_steps, 'require_pose': False}
    profile = {'precise': False, 'target': phase['pose'].copy(), 'tool_goal': None, 'attachment': None,
        'tolerance': .004, 'rotation_tolerance_degrees': 3., 'time_scale': 1.,
        'operation': 'pose', 'dense_steps': dense_steps}
    return phase, profile


def hold_phase(name, pose, opening, actions=1):
    if type(actions) is not int or not 1 <= actions <= 16:
        raise ValueError('An own-command hold has an explicit official action count')
    phase, profile = grip_phase(name, pose, opening, seconds=max(.04, actions * .04))
    phase['minimum_steps'], profile['dense_steps'] = actions, actions * 10
    return phase, profile


def validate_receipt(value):
    fields = {'schema', 'official_action_acknowledgments', 'completed_nominal_phases',
              'maximum_corrections_used', 'dense_gripper_steps', 'official_gripper_samples',
              'trajectory_sampling_reapplied', 'gripper_feedback_is_contact_sensor'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'riemann-mature-public-feedback-replay-v1'
            or value['trajectory_sampling_reapplied'] is not False
            or value['gripper_feedback_is_contact_sensor'] is not False):
        raise ValueError('Mature execution receipts describe only own commands and public acknowledgments')
    limits = {'official_action_acknowledgments': 640, 'completed_nominal_phases': 16,
              'maximum_corrections_used': 2, 'dense_gripper_steps': 102400, 'official_gripper_samples': 640}
    for name, limit in limits.items():
        if type(value[name]) is not int or not 0 <= value[name] <= limit:
            raise ValueError('Mature execution counters cannot exceed their declared public action allowance')
    if value['official_gripper_samples'] < value['official_action_acknowledgments']:
        raise ValueError('Every issued own action has one explicitly sampled normalized gripper command')
    return deepcopy(value)


class MatureFeedbackProgram(PhasedProgram):
    preserve_commanded_joint_targets = True

    def __init__(self, api, request, schedule, precondition=True, measurements=None):
        if not isinstance(schedule, (list, tuple)) or not 1 <= len(schedule) <= 16:
            raise ValueError('Compose between one and sixteen finite mature command phases per tool')
        phases, profiles = zip(*schedule)
        super().__init__(api, request, list(phases), precondition, measurements)
        self.profiles = deepcopy(list(profiles))
        if len(self.profiles) + 2 * sum(profile['precise'] for profile in self.profiles) > 40:
            raise ValueError('The finite correction schedule must fit the declared public feedback report budget')
        self.phase_origins = [phase['name'] for phase in self.phases]
        self.correction_counts = [0] * len(self.phases)
        self.last_target = float(api.targets()[self.side + '_ee_joint_state'][0])
        for event in reversed(api.command_memory()):
            feedback = event.get('execution_feedback', {})
            if feedback.get('side') == self.side and feedback.get('actions_executed', 0) > 0:
                command = feedback.get('last_gripper_command')
                if type(command) in (float, int) and math.isfinite(command) and 0 <= command <= 1:
                    self.last_target = float(command)
                    break
        self.ramp_key = None
        self.ramp = None
        self.dense_gripper_steps = 0
        self.official_gripper_samples = 0
        self.completed_nominal_phases = 0

    def phase_time_scale(self, unused_phase):
        return self.profiles[self.phase_index]['time_scale']

    def trajectory_operation(self, unused_phase):
        return self.profiles[self.phase_index]['operation']

    def _prepare_ramp(self, dense_steps, official_steps, includes_arm):
        phase = self.phases[self.phase_index]
        opening = phase['opening']
        start = float(self.api.targets()[self.side + '_ee_joint_state'][0])
        ramp = sampled_commands(start, opening, dense_steps, unchanged=opening == self.last_target,
            interval=10, time_scale=self.profiles[self.phase_index]['time_scale'], includes_arm_targets=includes_arm)
        if len(ramp) != official_steps:
            raise ExecutionError('Joint trajectory and original dense gripper replay must use exactly the same sampler')
        self.ramp_key = (self.phase_index, self.correction_counts[self.phase_index])
        self.ramp = ramp
        self.last_target = opening
        self.dense_gripper_steps += dense_steps
        self.official_gripper_samples += official_steps

    def accept_trajectory(self, unused_phase, response, unused_current_joints):
        self._prepare_ramp(response['dense_waypoints'], response['official_waypoints'], True)

    def command_opening(self, unused_observation, unused_steps):
        key = (self.phase_index, self.correction_counts[self.phase_index])
        if self.ramp_key != key:
            profile, phase = self.profiles[self.phase_index], self.phases[self.phase_index]
            if profile['dense_steps'] is None:
                raise ExecutionError('A moving gripper ramp must be prepared from actual trajectory metadata')
            self._prepare_ramp(profile['dense_steps'], phase['minimum_steps'], False)
        if self.phase_actions >= len(self.ramp):
            raise ExecutionError('Mature replay cannot append hidden gripper holds after its sampled command sequence')
        return float(self.ramp[self.phase_actions])

    def _tracking(self):
        profile = self.profiles[self.phase_index]
        actual = self.api.targets()[self.side + '_ee_pose']
        if profile['tool_goal'] is not None:
            actual = predicted_tool_pose(actual, profile['attachment'])
            goal = profile['tool_goal']
        else:
            goal = profile['target']
        return tracking_correction(goal, actual, self.phases[self.phase_index]['pose'],
                                   profile['tolerance'], profile['rotation_tolerance_degrees'])

    def _errors(self, unused_observation):
        result = self._tracking()
        return result['translation_m'], math.radians(result['rotation_degrees'])

    def _end_phase(self, status, distance, angle):
        self._report(status, distance, angle)
        self.completed_nominal_phases += 1
        self.phase_index += 1
        self.phase_actions, self.errors, self.ramp_key, self.ramp = 0, [], None, None
        if self.phase_index == len(self.phases):
            self.status, self.finished = 'completed', True

    def observe(self, observation):
        self.api.observe(observation)
        if self.finished or self.phase_actions == 0 or self.trajectory_pending:
            return
        phase, profile = self.phases[self.phase_index], self.profiles[self.phase_index]
        if self.phase_actions < phase['minimum_steps']:
            return
        result = self._tracking()
        distance, angle = result['translation_m'], math.radians(result['rotation_degrees'])
        if not phase['require_pose'] or not profile['precise']:
            self._end_phase('command_dwell', distance, angle)
        elif result['status'] == 'within_tolerance':
            self._end_phase('pose_reached', distance, angle)
        elif result['status'] == 'diverged' or self.correction_counts[self.phase_index] >= 2:
            self._report('stalled', distance, angle)
            self.status, self.finished, self.failed = 'uncertain', True, True
        else:
            self._report('stalled', distance, angle)
            self.correction_counts[self.phase_index] += 1
            phase['pose'] = result['command_pose'].copy()
            phase['name'] = self.phase_origins[self.phase_index] + '_correct_' + str(self.correction_counts[self.phase_index])
            self.phase_actions, self.errors, self.ramp_key, self.ramp = 0, [], None, None

    def execution_receipt(self):
        return validate_receipt({'schema': 'riemann-mature-public-feedback-replay-v1',
            'official_action_acknowledgments': self.actions,
            'completed_nominal_phases': self.completed_nominal_phases,
            'maximum_corrections_used': max(self.correction_counts),
            'dense_gripper_steps': self.dense_gripper_steps, 'official_gripper_samples': self.official_gripper_samples,
            'trajectory_sampling_reapplied': False, 'gripper_feedback_is_contact_sensor': False})
