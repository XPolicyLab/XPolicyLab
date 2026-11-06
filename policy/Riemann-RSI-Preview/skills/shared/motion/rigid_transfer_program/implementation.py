from copy import deepcopy

import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix as matrix
from skills.shared.motion.normalized_command_clock.implementation import NormalizedCommandClock
from skills.shared.motion.normalized_gripper_ramp.implementation import sampled_commands
from skills.shared.motion.public_phase_execution.implementation import PhasedProgram, pose_phase
from skills.shared.perception.current_metric_instances.implementation import instances
from skills.shared.verification.observed_rigid_transfer.implementation import recorded, validate


STAGE_TO_PHASE = {'pregrasp': 'approach', 'contact': 'grasp_adjust', 'lift': 'lift', 'carry': 'transfer',
                  'align': 'approach', 'place': 'preplace', 'retract': 'retreat'}
FOLLOWING = {'pregrasp': 'contact', 'contact': 'close', 'lift': 'carry', 'carry': 'align',
             'align': 'place', 'place': 'release', 'retract': 'verify'}


def own_opening(api, side):
    for event in reversed(api.command_memory()[-32:]):
        value = event.get('execution_feedback', {})
        if value.get('side') == side and type(value.get('last_gripper_command')) in (int, float):
            opening = value['last_gripper_command']
            if np.isfinite(opening) and 0 <= opening <= 1:
                return float(opening)
    return float(api.targets()[side + '_ee_joint_state'][0])


def actual_request(request, side):
    if (side not in ('left', 'right') or not isinstance(request, dict) or set(request) - {'skill', 'side', 'steps', 'reason'}
            or request.get('side') not in ('auto', side)):
        raise ExecutionError('A rigid stage selects only the own arm allowed by its public contract')
    return dict(request, side=side)


def visual_review(api, current, following):
    measured = current['scene']
    support = measured['support_memory']
    candidates = instances(api, measured['source_anchor']['description'], support['height_estimate_m'],
                           support['height_uncertainty_m'], require_supported=False)
    source = np.asarray(measured['source_reference'], dtype=float)
    contact = np.asarray(measured['route']['contact'], dtype=float)
    expected = (matrix(api.targets()[measured['side'] + '_ee_pose']) @ np.linalg.inv(matrix(contact))
                @ np.r_[source[:3], 1.])[:3]
    if following == 'placement':
        expected = np.asarray(measured['target_reference'][:3], dtype=float)
    previous_colour = np.asarray(measured['source_anchor']['appearance_rgb'], dtype=float)
    ranked = sorted(candidates, key=lambda item: np.linalg.norm(np.asarray(item['anchor']['center_m']) - expected))
    evidence = {'status': 'uncertain', 'position_error_m': 4., 'colour_difference': 450., 'is_contact_measurement': False}
    if not ranked:
        return evidence
    anchor = ranked[0]['anchor']
    evidence.update(position_error_m=float(np.linalg.norm(np.asarray(anchor['center_m']) - expected)),
                    colour_difference=float(np.linalg.norm(np.asarray(anchor['appearance_rgb']) - previous_colour)))
    uncertainty = measured['source_anchor']['uncertainty_m'] + anchor['uncertainty_m']
    if evidence['position_error_m'] < .025 + min(uncertainty, .025) and evidence['colour_difference'] < 100.:
        evidence['status'] = 'current_image_placement' if following == 'placement' else 'current_surface_follows'
    return evidence


def feedback_state(program, result):
    result['image_measurements'].update(public_rigid_transfer_state=validate(program.current),
        rigid_visual_review=deepcopy(program.visual_review),
        rigid_command_phases_completed=bool(program.finished and program.actions and program.status in ('completed', 'uncertain')))
    return result


class RigidClock(NormalizedCommandClock):
    def __init__(self, api, request, current, openings, boundaries, following=None, reviewer=None, precondition=True):
        self.current = validate(current)
        self.following, self.reviewer, self.transition_recorded = following, reviewer, False
        self.visual_review = {'status': 'not_reviewed', 'position_error_m': 0., 'colour_difference': 0., 'is_contact_measurement': False}
        side = self.current['scene']['side'] if self.current['scene'] else request['side'] if request['side'] != 'auto' else 'left'
        super().__init__(api, actual_request(request, side), openings, boundaries)
        if not precondition:
            self.status, self.finished = 'precondition_unmet', True

    def observe(self, observation):
        super().observe(observation)
        if self.finished and self.actions and not self.transition_recorded:
            self.transition_recorded = True
            if self.status == 'completed':
                if self.following is not None:
                    self.current['phase'] = self.following
                if self.reviewer is not None:
                    self.reviewer(self)

    def feedback(self, observation, budget=False):
        return feedback_state(self, super().feedback(observation, budget))


class RigidMotion(PhasedProgram):
    preserve_commanded_joint_targets = True

    def __init__(self, api, request, current, target, phase, opening, maximum=160, following=None, reviewer=None):
        if type(maximum) is not int or not 1 <= maximum <= 320:
            raise ExecutionError('A rigid trajectory uses a bounded declared own-motion budget')
        self.current = validate(current)
        self.following, self.reviewer, self.transition_recorded = following, reviewer, False
        self.visual_review = {'status': 'not_reviewed', 'position_error_m': 0., 'colour_difference': 0., 'is_contact_measurement': False}
        self.maximum_actions = min(request['steps'], maximum)
        self.dispatch_request = deepcopy(request)
        self.motion_phase = phase
        super().__init__(api, dict(request, steps=self.maximum_actions),
            [pose_phase('rigid_' + phase, np.asarray(target, dtype=float), opening, 1, True)],
            precondition=api.trajectory_available())
        support = self.current['scene']['support_memory'] if self.current['scene'] else self.current['support_memory']
        self.geometry = {'support_plane_m': support['height_estimate_m']} if support else {}
        self.pose_tolerance_m = .006

    def trajectory_operation(self, phase):
        return 'continuous_pose'

    def observe(self, observation):
        if not self.actions:
            self.api.observe(observation)
            return
        super().observe(observation)
        if self.finished and not self.transition_recorded:
            self.transition_recorded = True
            if self.status == 'completed':
                if self.following is not None:
                    self.current['phase'] = self.following
                if self.reviewer is not None:
                    self.reviewer(self)
            else:
                self.current['phase'] = 'reset'

    def record_action(self):
        if not self.actions and self.current['phase'] == 'pregrasp':
            self.current['attempts'] = min(3, self.current['attempts'] + 1)
        super().record_action()

    def feedback(self, observation, budget=False):
        result = super().feedback(observation, budget)
        if self.finished and self.status in ('uncertain', 'budget'):
            self.current['phase'] = 'reset'
        result['image_measurements']['rigid_motion_execution'] = {
            'backend': 'declared_public_own_curobo_trajectory', 'phase': self.motion_phase,
            'native_actions_issued': self.actions, 'maximum_native_actions': self.maximum_actions,
            'trajectory_completed': bool(self.finished and self.actions and self.status == 'completed'),
            'micro_ik_fallback_used': False, 'contact_measurement_claimed': False,
            'true_finger_measurement_used': False, 'official_success_claimed': False}
        return feedback_state(self, result)


def unavailable(api, request, current):
    side = current['scene']['side'] if current['scene'] else ('left' if request.get('side') == 'auto' else request.get('side'))
    return RigidClock(api, actual_request(request, side), current, [own_opening(api, side)], [1], precondition=False)


def clear_view(api, request, current):
    side = 'left' if current['view_leg'] == 0 else 'right'
    target = api.home_targets()[side + '_ee_pose'].copy()
    target[:3] += [0., -.06, .18]

    def completed(program):
        program.current['view_leg'] += 1
        program.current['phase'] = 'view' if program.current['view_leg'] == 2 else 'clear'

    return RigidMotion(api, actual_request(request, side), current, target, 'approach', 1., reviewer=completed)


def motion(api, request, stage):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != stage or current['scene'] is None or stage not in STAGE_TO_PHASE:
        return unavailable(api, request, current)
    measured, goal = current['scene'], current['goal']
    maximum = 70 if stage == 'place' and measured['source_anchor']['description'] == 'pliers' else 45 if stage == 'place' else 80 if stage == 'retract' else 160

    def completed(program):
        if stage == 'lift':
            program.visual_review = visual_review(program.api, program.current, 'following')
            if program.visual_review['status'] != 'current_surface_follows':
                program.current['phase'], program.status = 'reset', 'uncertain'

    opening = 1. if stage in ('pregrasp', 'contact', 'retract') else 0.
    return RigidMotion(api, actual_request(request, measured['side']), current, measured['route'][stage],
        STAGE_TO_PHASE[stage], opening, maximum, FOLLOWING[stage], completed)


def clock(api, request, stage):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != stage or current['goal'] is None:
        return unavailable(api, request, current)
    side = current['scene']['side'] if current['scene'] else 'left'
    opening = own_opening(api, side)
    goal = current['goal']
    if stage in ('close', 'release'):
        if current['scene'] is None:
            return unavailable(api, request, current)
        count = goal['close_commands'] if stage == 'close' else goal['release_commands']
        target = 0. if stage == 'close' else 1.
        values = [target] * count if goal.get('gripper_schedule') == 'settled_constant' else sampled_commands(opening, target, count * 10).tolist()
        if stage == 'release' and goal['release_strategy'] == 'gradual':
            values, boundaries, previous = [], [], opening
            for target in np.linspace(.1, 1., 10):
                values.extend(sampled_commands(previous, float(target), 20).tolist())
                previous = values[-1]
                boundaries.append(len(values))
            values.extend(sampled_commands(previous, 1., 60).tolist())
            boundaries.append(len(values))
        else:
            boundaries = [count]
        if stage == 'release' and goal.get('pre_release_hold_commands', 0):
            before = goal['pre_release_hold_commands']
            values = [opening] * before + values
            boundaries = [before] + [index + before for index in boundaries]
        after = goal.get('close_settle_commands' if stage == 'close' else 'release_settle_commands', 0)
        if after:
            values += [values[-1]] * after
            boundaries.append(len(values))
        return RigidClock(api, request, current, values, boundaries, 'lift' if stage == 'close' else 'retract')
    if stage == 'reset':
        def completed(program):
            program.current.update(phase='view', scene=None)
        return RigidClock(api, actual_request(request, side), current, sampled_commands(opening, 1., 80).tolist(),
                          [8], reviewer=completed)
    if stage == 'verify':
        def completed(program):
            program.visual_review = visual_review(program.api, program.current, 'placement')
            if program.visual_review['status'] != 'current_image_placement':
                program.current.update(phase='view', scene=None)
                program.status = 'uncertain'
                return
            program.current['completed_descriptions'] = (program.current['completed_descriptions'] + [program.current['scene']['source_anchor']['description']])[-32:]
            program.current['visited_targets'] = (program.current['visited_targets'] + [program.current['scene']['target_reference']])[-32:]
            program.current.update(item=min(32, program.current['item'] + 1), phase='view', attempts=0, scene=None)
        return RigidClock(api, request, current, [1., 1.], [2], reviewer=completed)
    if stage == 'finish':
        return RigidClock(api, actual_request(request, side), current, [1.] * 8, [8])
    return unavailable(api, request, current)


class RigidHome(PhasedProgram):
    preserve_commanded_joint_targets = True
    motion_time_scale = 1.

    def __init__(self, api, request, current):
        self.current, self.transition_recorded = validate(current), False
        self.visual_review = {'status': 'not_reviewed', 'position_error_m': 0., 'colour_difference': 0., 'is_contact_measurement': False}
        side = 'left' if current['home_leg'] == 0 else 'right'
        super().__init__(api, actual_request(request, side),
                         [pose_phase('return_rigid_home', api.home_targets()[side + '_ee_pose'], 1., 2, True)])

    def trajectory_operation(self, phase):
        return 'joint'

    def trajectory_joint_target(self, phase):
        return self.api.home_joint_states()[self.side]

    def observe(self, observation):
        super().observe(observation)
        if self.finished and self.actions and not self.transition_recorded:
            self.transition_recorded = True
            if self.status == 'completed':
                self.current['home_leg'] += 1
                self.current['phase'] = 'finish' if self.current['home_leg'] == 2 else 'home'

    def feedback(self, observation, budget=False):
        return feedback_state(self, super().feedback(observation, budget))


def home(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != 'home' or current['home_leg'] >= 2:
        return unavailable(api, request, current)
    return RigidHome(api, request, current)
