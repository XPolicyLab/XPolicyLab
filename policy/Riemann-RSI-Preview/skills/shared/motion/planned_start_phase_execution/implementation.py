from copy import deepcopy
import math
import numpy as np
from harness.execution import ExecutionError
from harness.runtime.observation import public_observation
from skills.shared.motion.upright_transfer.implementation import current_targets, axis_rotation
from skills.shared.perception.colour_components.implementation import colour_components, rgb_image
REPORT_STATUSES = {'pose_reached', 'command_dwell', 'stalled', 'budget'}
PROGRAM_STATUSES = {'running', 'completed', 'uncertain', 'precondition_unmet', 'budget'}

class PublicRobotAPI:
    __slots__ = ('_observation', '_home_targets', '_geometry_memory', '_trajectory', '_home_joints', '_base_inference', '_command_memory', '_ik_calls', '_dense_calls', '_issued_action_count')

    def __init__(self, observation, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0):
        """Copy public RGB (H,W,3), poses (7,), arm joints (6,), jaws (1,) and bounded episode facts.

        Optional callbacks expose own-robot kinematics/planning only; return None. The issued count
        accepts the fixed system's 3600-action ceiling and supplies no task progress or success signal.
        """
        self.observe(observation)
        if type(issued_action_count) is not int or not 0 <= issued_action_count <= 3600:
            raise ExecutionError('Own action count is a bounded integer of issued commands, not evaluator progress')
        self._issued_action_count = issued_action_count
        memory = command_memory if command_memory is not None else []
        if not isinstance(memory, (list, tuple)) or len(memory) > 32 or any((not isinstance(event, dict) for event in memory)):
            raise ExecutionError('Own command memory must be a bounded same-episode execution list')
        self._command_memory = deepcopy(list(memory))
        if trajectory is not None and (not callable(trajectory)):
            raise ExecutionError('A trajectory capability must expose a bounded callable contract')
        self._trajectory = trajectory
        self._ik_calls = 0
        self._dense_calls = 0
        if robot_bases is not None and (not callable(robot_bases)):
            raise ExecutionError('Own robot FK must expose a restricted public-observation callable')
        self._base_inference = robot_bases
        self._home_joints = deepcopy(home_joints if home_joints is not None else {side: self._observation['state'][side + '_arm_joint_state'] for side in ('left', 'right')})
        if not isinstance(self._home_joints, dict) or set(self._home_joints) != {'left', 'right'}:
            raise ExecutionError('Episode home joints contain only the two observed own arms')
        for side, joints in self._home_joints.items():
            joints = np.asarray(joints, dtype=float).reshape(-1)
            if joints.size != 6 or not np.isfinite(joints).all():
                raise ExecutionError('Episode home joints require six finite own-arm observations')
            self._home_joints[side] = joints.copy()
        self._home_targets = deepcopy(home_targets if home_targets is not None else self.targets())
        expected = {side + suffix for side in ('left', 'right') for suffix in ('_ee_pose', '_ee_joint_state')}
        if not isinstance(self._home_targets, dict) or set(self._home_targets) != expected:
            raise ExecutionError('Episode home contains only observed public robot targets')
        for name, value in self._home_targets.items():
            dimension = 7 if name.endswith('_ee_pose') else 1
            vector = np.asarray(value, dtype=float).reshape(-1)
            if vector.size != dimension or not np.isfinite(vector).all():
                raise ExecutionError('Episode home must contain finite public robot targets')
            if dimension == 7 and (not 0.99 <= np.linalg.norm(vector[3:]) <= 1.01):
                raise ExecutionError('Episode home requires a measured unit robot orientation')
            if dimension == 1 and (not 0 <= vector[0] <= 1):
                raise ExecutionError('Episode home gripper values are public normalized commands')
            self._home_targets[name] = vector.copy()
        self._geometry_memory = validate_geometry_memory(geometry_memory or {})

    def observe(self, observation):
        self._observation = public_observation(observation)

    def targets(self):
        return current_targets(self._observation)

    def instruction(self):
        return self._observation['instruction'] or ''

    def image(self, camera='cam_head'):
        return rgb_image(self._observation, camera).copy()

    def regions(self, colour, camera='cam_head'):
        return colour_components(self.image(camera), colour)

    def home_targets(self):
        return deepcopy(self._home_targets)

    def geometry_memory(self):
        return deepcopy(self._geometry_memory)

    def issued_action_count(self):
        return self._issued_action_count

    def command_memory(self):
        return deepcopy(self._command_memory)

    def trajectory_available(self):
        return self._trajectory is not None

    def home_joint_states(self):
        return deepcopy(self._home_joints)

    def own_joint_states(self):
        return {side: np.asarray(self._observation['state'][side + '_arm_joint_state']).copy() for side in ('left', 'right')}

    def own_robot_bases_available(self):
        return self._base_inference is not None

    def own_robot_bases(self):
        from skills.shared.motion.planned_start_joint_planning.implementation import validate_robot_bases
        if self._base_inference is None:
            raise ExecutionError('The public robot API has no own-robot FK capability')
        return validate_robot_bases(self._base_inference(deepcopy(self._observation)))

    def plan_trajectory(self, side, pose, support_plane_m, remaining_actions, time_scale, operation='pose', joint_target=None):
        from skills.shared.motion.planned_start_joint_planning.implementation import READ_ONLY_DENSE_OPERATIONS, validate_request, validate_result
        if self._trajectory is None:
            raise ExecutionError('The public robot API has no trajectory capability')
        if operation in READ_ONLY_DENSE_OPERATIONS:
            raise ExecutionError('Read-only original dense results cannot use the sampled trajectory API')
        request = validate_request({'observation': self._observation, 'side': side, 'pose': pose, 'support_plane_m': support_plane_m, 'remaining_actions': remaining_actions, 'time_scale': time_scale, 'operation': operation, 'joint_target': joint_target})
        return validate_result(self._trajectory(request), remaining_actions)

    def own_inverse_kinematics(self, side, pose):
        result = {'status': 'Budget', 'joints': None, 'robot_actions': 0, 'is_motion_command': False, 'query_count': self._ik_calls}
        if self._ik_calls >= 1024:
            return result
        self._ik_calls += 1
        response = self.plan_trajectory(side, pose, None, 1, 1.0, operation='own_ik')
        if response['status'] == 'Success' and (response['dense_waypoints'] != 1 or response['official_waypoints'] != 1):
            raise ExecutionError('Read-only own IK returns exactly one own joint solution, not a motion sequence')
        return dict(result, status=response['status'], query_count=self._ik_calls, joints=response['positions'][0].copy() if response['status'] == 'Success' else None)

    def dense_own_trajectory(self, side, pose, operation='original_dense_pose', joint_target=None, start_state=None):
        """Plan read-only dense own joints; start_state optionally replaces this arm's joints (6,) and EE pose (7,)
        with a previously planned own endpoint for route preflight."""
        from skills.shared.motion.planned_start_joint_planning.implementation import READ_ONLY_DENSE_OPERATIONS, validate_request, validate_dense_result
        if self._trajectory is None or operation not in READ_ONLY_DENSE_OPERATIONS:
            raise ExecutionError('The public robot API requires an explicit original dense own-robot capability')
        observation = self._observation
        if start_state is not None:
            if not isinstance(start_state, dict) or set(start_state) != {'joints', 'ee_pose'}:
                raise ExecutionError('A planned own start contains only one arm joint vector and EE pose')
            observation = deepcopy(self._observation)
            observation['state'][side + '_arm_joint_state'] = np.asarray(start_state['joints'], dtype=float).reshape(6).copy()
            observation['state'][side + '_ee_pose'] = np.asarray(start_state['ee_pose'], dtype=float).reshape(7).copy()
        request = validate_request({'observation': observation, 'side': side, 'pose': pose, 'support_plane_m': None, 'remaining_actions': 1, 'time_scale': 1.0, 'operation': operation, 'joint_target': joint_target})
        if self._dense_calls >= 512:
            return {'status': 'Budget', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
        self._dense_calls += 1
        return validate_dense_result(self._trajectory(request))

    def preview_trajectory(self, side, poses, support_plane_m, remaining_actions, time_scale):
        from skills.shared.motion.planned_start_joint_planning.implementation import validate_request, validate_result
        if self._trajectory is None or side not in ('left', 'right') or (not 1 <= len(poses) <= 4):
            raise ExecutionError('A route preview requires a bounded own-arm planning capability')
        self.own_robot_bases()
        planned = deepcopy(self._observation)
        results = []
        for pose in poses:
            request = validate_request({'observation': planned, 'side': side, 'pose': pose, 'support_plane_m': support_plane_m, 'remaining_actions': remaining_actions, 'time_scale': time_scale, 'operation': 'pose', 'joint_target': None})
            result = validate_result(self._trajectory(request), remaining_actions)
            results.append(result)
            if result['status'] != 'Success':
                break
            planned['state'][side + '_arm_joint_state'] = result['positions'][-1].copy()
            planned['state'][side + '_ee_pose'] = np.asarray(pose).copy()
        return results

def validate_geometry_memory(value):
    if not isinstance(value, dict) or set(value) - {'support_plane_m', 'raised_colours'}:
        raise ExecutionError('Geometry memory contains only same-episode RGB-derived support hypotheses')
    numbers = []
    if 'support_plane_m' in value:
        numbers.append(value['support_plane_m'])
    raised = value.get('raised_colours', {})
    if not isinstance(raised, dict) or len(raised) > 6 or set(raised) - {'blue', 'green', 'red', 'yellow', 'orange', 'cyan'}:
        raise ExecutionError('Geometry memory exceeds its declared colour vocabulary')
    numbers.extend(raised.values())
    if any((type(number) not in (int, float) or not math.isfinite(number) for number in numbers)):
        raise ExecutionError('Geometry memory must contain finite public-image estimates')
    return deepcopy(value)

def pose_phase(name, pose, opening, minimum_steps=2, require_pose=True):
    return {'name': name, 'pose': np.asarray(pose, dtype=float).copy(), 'opening': float(opening), 'minimum_steps': minimum_steps, 'require_pose': require_pose}

def translated(pose, delta):
    result = np.asarray(pose, dtype=float).copy()
    result[:3] += np.asarray(delta, dtype=float)
    return result

def relative_rotation(pose, axis, angle):
    result = np.asarray(pose, dtype=float).copy()
    result[3:] = axis_rotation(result[3:], axis, angle)
    return result

def downward_tool_pose(pose):
    result = np.asarray(pose, dtype=float).copy()
    quaternion = result[3:] / np.linalg.norm(result[3:])
    scalar, vector = (quaternion[0], quaternion[1:])
    local = np.asarray([1.0, 0.0, 0.0])
    direction = local + 2 * np.cross(vector, np.cross(vector, local) + scalar * local)
    desired = np.asarray([0.0, 0.0, -1.0])
    cosine = float(np.clip(np.dot(direction, desired), -1, 1))
    axis = np.cross(direction, desired)
    if np.linalg.norm(axis) < 1e-08:
        if cosine > 0:
            return result
        axis = np.asarray([1.0, 0.0, 0.0])
    axis /= np.linalg.norm(axis)
    result[3:] = axis_rotation(quaternion, axis, math.acos(cosine))
    return result

def source_measurements(api, colour=None, support_colour=None):
    return {'source_visible_before': bool(api.regions(colour)) if colour else False, 'source_visible_after': False, 'support_visible_before': bool(api.regions(support_colour)) if support_colour else False, 'support_visible_after': False, 'axis_error_before_radians': None, 'axis_error_after_radians': None, 'measurement_ambiguous': True}

class PhasedProgram:

    def __init__(self, api, request, phases, precondition=True, measurements=None):
        if type(api) is not PublicRobotAPI:
            raise ExecutionError('A task tool receives only the restricted public robot API')
        self.api = api
        self.request = deepcopy(request)
        self.side = request['side']
        self.targets = api.targets()
        self.phases = deepcopy(phases)
        if not 1 <= len(self.phases) <= 40:
            raise ExecutionError('A task tool requires a bounded phase list')
        for phase in self.phases:
            if set(phase) != {'name', 'pose', 'opening', 'minimum_steps', 'require_pose'} or not isinstance(phase['name'], str) or (not 1 <= len(phase['name']) <= 64) or (phase['pose'].shape != (7,)) or (not np.isfinite(phase['pose']).all()) or (not 0.99 <= np.linalg.norm(phase['pose'][3:]) <= 1.01) or (not 0 <= phase['opening'] <= 1) or (type(phase['minimum_steps']) is not int) or (not 1 <= phase['minimum_steps'] <= 16) or (type(phase['require_pose']) is not bool):
                raise ExecutionError('An executable phase has an invalid public pose or budget')
        self.minimum_actions = sum((phase['minimum_steps'] for phase in self.phases))
        if self.minimum_actions > request['steps']:
            precondition = False
        self.phase_index = 0
        self.phase_actions = 0
        self.actions = 0
        self.errors = []
        self.reports = []
        self.status = 'running' if precondition else 'precondition_unmet'
        self.finished = not precondition
        self.issued_cycles = 0
        self.pose_qualified_cycles = 0
        self.measurements = deepcopy(measurements or source_measurements(api))
        self.visual_lift_feedback = {'status': 'unavailable', 'head_displacement_fraction': None, 'head_area_ratio': None, 'robot_translation_m': 0.0}
        self.failed = False
        self.trajectory_pending = 0
        self.preplanned = {}
        self.pose_tolerance_m = 0.004
        self.rotation_tolerance_radians = 0.035

    def record_action(self):
        if self.finished:
            raise ExecutionError('An ended task tool cannot issue another action')
        if self.phase_actions == 0 and self.phases[self.phase_index]['name'].startswith('press_down_'):
            self.issued_cycles += 1
        self.phase_actions += 1
        self.actions += 1
        self.trajectory_pending = max(0, self.trajectory_pending - 1)

    def _errors(self, observation):
        current = np.asarray(observation['state'][self.side + '_ee_pose']).reshape(-1)
        goal = self.phases[self.phase_index]['pose']
        distance = float(np.linalg.norm(goal[:3] - current[:3]))
        orientation = current[3:] / np.linalg.norm(current[3:])
        angle = 2 * math.acos(float(np.clip(abs(np.dot(orientation, goal[3:])), 0, 1)))
        return (distance, angle)

    def _report(self, status, distance, angle):
        self.reports.append({'phase': self.phases[self.phase_index]['name'], 'status': status, 'actions': self.phase_actions, 'position_error_m': distance, 'rotation_error_radians': angle})

    def _abort(self, observation, distance, angle):
        self._report('stalled', distance, angle)
        self.failed = True
        remaining = self.request['steps'] - self.actions
        if remaining < 3 or self.phases[self.phase_index]['name'] == 'uncertain_retreat':
            self.status, self.finished = ('uncertain', True)
            return
        pose = np.asarray(observation['state'][self.side + '_ee_pose']).reshape(-1)
        opening = self.phases[self.phase_index]['opening']
        self.phases = self.phases[:self.phase_index + 1] + [pose_phase('uncertain_retreat', translated(pose, [0, 0, 0.025]), opening)]
        self.phase_index += 1
        self.phase_actions, self.errors = (0, [])

    def observe(self, observation):
        self.api.observe(observation)
        if self.finished:
            return
        current = self.api._observation
        phase = self.phases[self.phase_index]
        distance, angle = self._errors(current)
        qualified = not phase['require_pose'] or (distance <= self.pose_tolerance_m and angle <= self.rotation_tolerance_radians)
        if self.phase_actions >= phase['minimum_steps'] and qualified and (not self.trajectory_pending):
            self._report('pose_reached' if phase['require_pose'] else 'command_dwell', distance, angle)
            if phase['name'].startswith('press_release_'):
                self.pose_qualified_cycles += 1
            self.phase_index += 1
            self.phase_actions, self.errors = (0, [])
            if self.phase_index == len(self.phases):
                self.status, self.finished = ('uncertain' if self.failed else 'completed', True)
            return
        if self.phase_actions:
            self.errors.append(distance + angle * 0.02)
            self.errors = self.errors[-12:]
            if phase['require_pose'] and (not qualified) and (not self.trajectory_pending) and (self.phase_actions >= 64 or (self.phase_actions >= 16 and len(self.errors) == 12 and (max(self.errors) - min(self.errors) < 0.00015))):
                self._abort(current, distance, angle)

    def proposal(self, observation):
        self.observe(observation)
        if not self.finished:
            phase = self.phases[self.phase_index]
            self.targets[self.side + '_ee_pose'] = phase['pose'].copy()
            self.targets[self.side + '_ee_joint_state'] = np.asarray([phase['opening']])
        return deepcopy(self.targets)

    def feedback(self, observation, budget=False):
        self.api.observe(observation)
        if budget and (not self.finished):
            distance, angle = self._errors(self.api._observation)
            self._report('budget', distance, angle)
            self.status, self.finished = ('budget', True)
        colour = self.request.get('colour')
        support = self.request.get('support_colour') or self.request.get('reference_colour')
        if colour:
            self.measurements['source_visible_after'] = bool(self.api.regions(colour))
        if support:
            self.measurements['support_visible_after'] = bool(self.api.regions(support))
        phase = self.phases[min(self.phase_index, len(self.phases) - 1)]
        return {'kind': self.request['skill'], 'side': self.side, 'status': self.status, 'phase': phase['name'], 'phase_reports': deepcopy(self.reports[-8:]), 'phase_count': len(self.reports), 'actions_executed': self.actions, 'required_minimum_actions': self.minimum_actions, 'budget_sufficient': self.minimum_actions <= self.request['steps'], 'issued_cycles': self.issued_cycles, 'pose_qualified_cycles': self.pose_qualified_cycles, 'public_gripper_value': float(self.api.targets()[self.side + '_ee_joint_state'][0]), 'last_gripper_command': float(self.targets[self.side + '_ee_joint_state'][0]), 'visual_review_required': True, 'object_grasp_confirmed': False, 'official_success_claimed': False, 'image_measurements': deepcopy(self.measurements), 'visual_lift_feedback': deepcopy(self.visual_lift_feedback)}

def validate_swap_placement_review(request, value):
    """Validate current-RGB occupancy against a validated own transfer request; return a copy.

    Both occupancy lists contain only finite support-role names. They describe public visual
    assignments and the own plan, never object contact or official task completion.
    """
    fields = {'scene_status', 'observed_occupied', 'expected_occupied'}
    roles = {'mat_a', 'mat_b', 'mat_c'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('A swap placement review has only declared public occupancy fields')
    if value['scene_status'] not in ('uncertain', 'image_observed', 'metric_observed'):
        raise ValueError('A swap placement review needs the actual public scene-reader status')
    for name in ('observed_occupied', 'expected_occupied'):
        occupied = value[name]
        if not isinstance(occupied, list) or any((type(role) is not str or role not in roles for role in occupied)) or occupied != sorted(set(occupied)):
            raise ValueError('A swap placement review contains only distinct sorted public support roles')
    source = request.get('source_mat')
    if type(source) is not str or source not in roles:
        raise ValueError('A swap review requires its validated public source support')
    expected = sorted(roles - {source})
    observed_count = 0 if value['scene_status'] == 'uncertain' else 2
    if value['expected_occupied'] != expected or len(value['observed_occupied']) != observed_count:
        raise ValueError('A swap review preserves the current scene evidence and the requested own transfer')
    return deepcopy(value)

def validate_phase_feedback(request, value, automatic_side=False):
    fields = {'kind', 'side', 'status', 'phase', 'phase_reports', 'issued_cycles', 'pose_qualified_cycles', 'public_gripper_value', 'last_gripper_command', 'visual_review_required', 'object_grasp_confirmed', 'official_success_claimed', 'image_measurements', 'phase_count', 'actions_executed', 'required_minimum_actions', 'budget_sufficient', 'visual_lift_feedback'}
    if type(automatic_side) is not bool:
        raise ExecutionError('Automatic own-arm selection requires a trusted declared tool contract')
    if not isinstance(value, dict) or set(value) != fields or value['kind'] != request['skill'] or (value['status'] not in PROGRAM_STATUSES) or (not isinstance(value['phase'], str)) or (not 1 <= len(value['phase']) <= 64):
        raise ExecutionError('Task-tool feedback does not match its declared phase contract')
    if value['side'] not in ('left', 'right') or (value['side'] != request['side'] and (not (automatic_side and request['side'] == 'auto'))):
        raise ExecutionError('Task-tool feedback must preserve the declared own-arm selection')
    if value['visual_review_required'] is not True or value['object_grasp_confirmed'] is not False or value['official_success_claimed'] is not False:
        raise ExecutionError('Task-tool feedback cannot claim object contact, grasp or task success')
    minimum = 0 if request['skill'] in ('liquid_scene_view', 'liquid_observe_spout') else 1
    if type(value['required_minimum_actions']) is not int or not minimum <= value['required_minimum_actions'] <= 640 or type(value['budget_sufficient']) is not bool or (value['budget_sufficient'] != (value['required_minimum_actions'] <= request['steps'])):
        raise ExecutionError('Phase feedback must distinguish insufficient action allowance from visual uncertainty')
    for name in ('public_gripper_value', 'last_gripper_command'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or (not 0 <= value[name] <= 1):
            raise ExecutionError('Gripper feedback is a normalized public value, not measured fingers')
    issued, qualified = (value['issued_cycles'], value['pose_qualified_cycles'])
    if type(issued) is not int or type(qualified) is not int or (not 0 <= qualified <= issued <= request.get('count', 0)):
        raise ExecutionError('Robot press cycles cannot be reported as official button counts')
    reports = value['phase_reports']
    if not isinstance(reports, list) or len(reports) > 8 or type(value['phase_count']) is not int or (not len(reports) <= value['phase_count'] <= 40) or (type(value['actions_executed']) is not int) or (not 0 <= value['actions_executed'] <= request['steps']):
        raise ExecutionError('Task-tool phase reports exceed their bound')
    actions = 0
    for row in reports:
        if not isinstance(row, dict) or set(row) != {'phase', 'status', 'actions', 'position_error_m', 'rotation_error_radians'} or (not isinstance(row['phase'], str)) or (not 1 <= len(row['phase']) <= 64) or (row['status'] not in REPORT_STATUSES) or (type(row['actions']) is not int) or (row['actions'] < 0):
            raise ExecutionError('Task-tool phase feedback has an undeclared field')
        for name in ('position_error_m', 'rotation_error_radians'):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]) or row[name] < 0:
                raise ExecutionError('Task-tool pose feedback must be finite and public')
        if row['rotation_error_radians'] > math.pi:
            raise ExecutionError('Rotation feedback exceeds its public quaternion range')
        actions += row['actions']
    if actions > value['actions_executed']:
        raise ExecutionError('Task-tool reports exceed the shared action budget')
    measurements = value['image_measurements']
    if isinstance(measurements, dict) and 'bowl_scene_diagnostics' in measurements and (request['skill'] != 'bowl_stack_scene_view'):
        raise ExecutionError('Bowl admission diagnostics belong only to the bowl scene view')
    flags = {'source_visible_before', 'source_visible_after', 'support_visible_before', 'support_visible_after', 'measurement_ambiguous'}
    angles = {'axis_error_before_radians', 'axis_error_after_radians'}
    extra = {'public_command_recipe'} if request['skill'] == 'dual_bar_stage' and request.get('stage') == 'prepare' and isinstance(measurements, dict) and ('public_command_recipe' in measurements) else set()
    if extra and 'open_dwell_reused' in measurements:
        extra.add('open_dwell_reused')
        if type(measurements['open_dwell_reused']) is not bool:
            raise ExecutionError('Dwell reuse must describe an explicit own-command receipt')
    if request['skill'] == 't_transfer_stage' and request.get('stage') == 'pick_left' and isinstance(measurements, dict) and ('public_t_command_recipe' in measurements):
        extra = {'public_t_command_recipe'}
    if request['skill'] in ('swap_mat_move', 'swap_confirm_move') and isinstance(measurements, dict) and ('public_swap_fixture' in measurements):
        extra = {'public_swap_fixture'}
    if request['skill'] == 'swap_mat_move' and isinstance(measurements, dict) and ('public_swap_placement_review' in measurements):
        extra.add('public_swap_placement_review')
    if request['skill'] == 'cover_cup_transfer' and isinstance(measurements, dict):
        extra = set(measurements) & {'public_cover_anchor', 'public_release_evidence', 'command_phases_completed'}
        if 'command_phases_completed' in extra:
            if type(measurements['command_phases_completed']) is not bool or (measurements['command_phases_completed'] and value['status'] not in ('completed', 'uncertain')):
                raise ExecutionError('Completed cup command phases require an explicit finished robot-command receipt')
    if request['skill'] in ('board_turn_stage', 'board_wait_opponent', 'board_return_idle_wrist') and isinstance(measurements, dict):
        extra = set(measurements) & {'public_board_anchor', 'public_board_progress'}
        if request['skill'] == 'board_wait_opponent' and 'public_wait_observation' in measurements:
            extra.add('public_wait_observation')
    if request['skill'] in ('tube_scene_view', 'grasp_tube', 'lift_tube', 'upright_tube', 'seat_tube', 'reseat_tube', 'finish_tube_sequence') and isinstance(measurements, dict):
        extra = set(measurements) & {'public_insertion_anchor', 'public_insertion_progress', 'command_phases_completed'}
    if request['skill'] in ('number_scene_view', 'number_home', 'grasp_number', 'lift_number', 'carry_number', 'place_number', 'verify_number_placement', 'recover_number_grasp', 'relay_number') and isinstance(measurements, dict):
        declared = {'public_number_anchor', 'public_number_progress', 'public_number_route', 'public_number_head_reference', 'public_number_surface_reference', 'public_number_inventory', 'public_number_target_reference', 'command_phases_completed', 'public_number_grasp_precondition', 'public_number_relay', 'current_source_identity_before_grasp', 'actual_current_view_metric_status', 'current_visual_placement_feedback', 'public_visual_grasp_feedback', 'own_home_joint_error_radians', 'gripper_signal_semantics', 'number_phase_notes', 'number_placement_read_before_observer_reset', 'number_placement_observation_actions', 'number_observer_return_side', 'public_wrist_view_preflight', 'actual_current_view_metric_reason', 'visible_head_target_count', 'visible_head_digits', 'possible_payload_support_from_public_command_response', 'possible_support_is_contact_sensor'}
        extra = set(measurements) & declared
        from skills.shared.planning.observed_number_contract_observed_start.implementation import validate_measurements
        try:
            validate_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Numeral feedback cannot inject a private pose, contact measurement or scorer state') from error
    from skills.shared.orchestration.planned_start_action_dispatch.implementation import PLANAR_T_ACTIONS, RIGID_TRANSFER_ACTIONS, RIGID_VIEW_ACTIONS, RECIPE_TOOLS
    if request['skill'] in (*RIGID_TRANSFER_ACTIONS, *RIGID_VIEW_ACTIONS, *RECIPE_TOOLS) and isinstance(measurements, dict):
        from skills.shared.verification.observed_rigid_transfer_planned_start.implementation import measurements as validate_rigid_measurements
        try:
            extra = validate_rigid_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Transfer feedback permits only closed public geometry and acknowledged own command receipts') from error
    if request['skill'] in ('pickup_scene_view', 'pickup_pregrasp', 'pickup_contact', 'pickup_close', 'pickup_warmup_lift', 'pickup_full_lift', 'pickup_verify_lift', 'pickup_reset_grasp') and isinstance(measurements, dict):
        from skills.shared.verification.observed_pickup_state_planned_start.implementation import validate_measurements
        try:
            extra = validate_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Pickup feedback cannot carry task assets, private layout, real fingers or scorer state') from error
    if request['skill'] in ('garment_scene_view', 'garment_approach_contact', 'garment_lower_contact', 'garment_close_grasp', 'garment_lift_fold', 'garment_carry_fold', 'garment_settle_fold', 'garment_release_fold', 'garment_clear_wrist', 'garment_recover_contact', 'garment_return_home') and isinstance(measurements, dict):
        from skills.shared.verification.observed_garment_state_planned_start.implementation import measurements as validate_garment_measurements
        try:
            extra = validate_garment_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Garment feedback must contain only current image evidence and own acknowledged commands') from error
    if request['skill'] in ('instrument_scene_view', 'grasp_instrument_tool', 'lift_instrument_tool', 'carry_instrument_tool', 'strike_instrument_key', 'recover_instrument_tool') and isinstance(measurements, dict):
        from skills.shared.planning.observed_instrument_contract_planned_start.implementation import validate_measurements
        try:
            extra = validate_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Instrument feedback must contain only independently qualified public geometry and own commands') from error
    if request['skill'] in ('vessel_scene_view', 'recover_pour_ball', 'grasp_pour_cup', 'lift_pour_cup', 'relay_pour_cup', 'carry_pour_cup', 'observe_pour_rim', 'tilt_pour_cup', 'upright_pour_cup', 'return_pour_cup', 'home_vessel_arms', 'reset_vessel_grasp', 'finish_pour_sequence') and isinstance(measurements, dict):
        from skills.shared.verification.observed_vessel_execution_planned_start.implementation import measurements as validate_vessel_measurements
        try:
            extra = validate_vessel_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Vessel feedback cannot inject private size, fingers, layout, scorer state or success') from error
    if request['skill'] in ('liquid_scene_view', 'liquid_support_cup', 'liquid_approach_bottle', 'liquid_close_bottle', 'liquid_reset_grasp', 'liquid_lift_bottle', 'liquid_observe_spout', 'liquid_carry_bottle', 'liquid_tilt_bottle', 'liquid_drain', 'liquid_upright', 'liquid_hold_upright') and isinstance(measurements, dict):
        from skills.shared.verification.observed_liquid_execution_planned_start.implementation import measurements as validate_liquid_measurements
        try:
            extra = validate_liquid_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Liquid feedback permits only measured public geometry and exact own command receipts') from error
    if request['skill'] in ('cube_scene_view', 'grasp_measured_cube', 'lift_measured_cube', 'carry_measured_cube', 'release_measured_cube', 'verify_measured_stack', 'reset_cube_grasp', 'finish_cube_stack') and isinstance(measurements, dict):
        from skills.shared.planning.observed_cube_transfer_contract.implementation import measurements as validate_cube_measurements
        try:
            extra = validate_cube_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Cube feedback cannot expose GT geometry, true fingers, layout or scorer state') from error
    if request['skill'] in PLANAR_T_ACTIONS and isinstance(measurements, dict):
        from skills.shared.planning.observed_planar_contact_contract_observed_start.implementation import anchor as validate_contact_anchor, progress as validate_contact_progress
        extra = set(measurements) & {'public_planar_contact_anchor', 'public_planar_contact_progress'}
        if extra and extra != {'public_planar_contact_anchor', 'public_planar_contact_progress'}:
            raise ExecutionError('Planar contact geometry and own command progress must be supplied together')
        if extra:
            try:
                validate_contact_anchor(measurements['public_planar_contact_anchor'])
                validate_contact_progress(measurements['public_planar_contact_progress'])
            except (ValueError, KeyError, TypeError) as error:
                raise ExecutionError('Planar contact evidence cannot contain GT, layout or scorer fields') from error
    if not isinstance(measurements, dict) or set(measurements) != flags | angles | extra:
        raise ExecutionError('Task-tool image evidence contains an undeclared measurement')
    if 'public_command_recipe' in extra:
        from skills.shared.motion.paired_joint_execution_planned_start.implementation import validate_public_pair_recipe
        try:
            validate_public_pair_recipe(measurements['public_command_recipe'])
        except (ValueError, TypeError) as error:
            raise ExecutionError('A public command recipe is malformed') from error
    elif extra == {'public_t_command_recipe'}:
        from skills.shared.motion.serial_transfer_contract.implementation import validate_recipe
        try:
            validate_recipe(measurements['public_t_command_recipe'])
        except (ValueError, TypeError) as error:
            raise ExecutionError('A public T command recipe is malformed') from error
    elif 'public_swap_fixture' in extra:
        from skills.shared.perception.public_plane_memory.implementation import validate
        try:
            validate(measurements['public_swap_fixture'])
        except (ValueError, TypeError) as error:
            raise ExecutionError('A retained public support hypothesis is malformed') from error
    if 'public_swap_placement_review' in extra:
        try:
            validate_swap_placement_review(request, measurements['public_swap_placement_review'])
        except (ValueError, TypeError, KeyError) as error:
            raise ExecutionError('A swap placement review contains only current public support assignments') from error
    if request['skill'] == 'cover_cup_transfer':
        from skills.shared.orchestration.container_return_contract.implementation import validate_anchor, validate_release
        try:
            if 'public_cover_anchor' in extra:
                validate_anchor(measurements['public_cover_anchor'])
            if 'public_release_evidence' in extra:
                validate_release(measurements['public_release_evidence'])
        except (ValueError, TypeError, KeyError) as error:
            raise ExecutionError('A covering receipt contains only declared public goals and release imagery') from error
    if request['skill'] in ('board_turn_stage', 'board_wait_opponent', 'board_return_idle_wrist') and extra:
        from skills.shared.orchestration.turn_based_board_contract.implementation import SEQUENCE, validate_anchor, validate_progress, validate_wait_observation
        try:
            if 'public_wait_observation' in extra:
                waiting = validate_wait_observation(measurements['public_wait_observation'])
                if waiting['action_count'] != value['actions_executed'] or waiting['expected_count'] != 2 * request['turn'] + 2:
                    raise ValueError('Wait diagnostics must match the actual requested own turn and issued actions')
                if value['status'] == 'completed' and (waiting['visible_cells'] is None or sum(waiting['visible_cells']) < waiting['expected_count'] or waiting['stable_frames'] < 12 or (waiting['changed_pixels'] is None) or (waiting['changed_pixels'] >= 90)):
                    raise ValueError('A completed opponent wait requires independent current visibility and stable delivered board frames')
            board_extra = extra - {'public_wait_observation'}
            if request['skill'] == 'board_wait_opponent' and value['status'] == 'completed' and (board_extra != {'public_board_anchor', 'public_board_progress'}):
                raise ValueError('A completed current wait must also retain its qualified board and own turn advancement')
            if board_extra and board_extra != {'public_board_anchor', 'public_board_progress'}:
                raise ValueError('A board receipt must pair public anchors with own command progress')
            if board_extra:
                validate_anchor(measurements['public_board_anchor'])
                progress = validate_progress(measurements['public_board_progress'])
            if board_extra and request['skill'] == 'board_return_idle_wrist':
                if progress['next_stage'] >= len(SEQUENCE) or SEQUENCE[progress['next_stage']] != (request['turn'], 'wait') or progress['own_commanded_cells'][-1] != request['cell'] or (progress['used_source_slots'][-1] != request['source']):
                    raise ValueError('An idle own reset cannot advance the opponent wait or change the placed source')
            elif board_extra and (progress['next_stage'] < 1 or SEQUENCE[progress['next_stage'] - 1] != (request['turn'], request.get('stage', 'wait'))):
                raise ValueError('The board receipt must describe the requested own stage')
        except (ValueError, TypeError, KeyError) as error:
            raise ExecutionError('A board receipt cannot contain hidden state or advance another command stage') from error
    if request['skill'] in ('tube_scene_view', 'grasp_tube', 'lift_tube', 'upright_tube', 'seat_tube', 'reseat_tube', 'finish_tube_sequence') and extra:
        from skills.shared.motion.serial_insertion_contract.implementation import validate_anchor, validate_progress
        try:
            if extra != {'public_insertion_anchor', 'public_insertion_progress', 'command_phases_completed'}:
                raise ValueError('An insertion receipt pairs initial public goals with actual completed own commands')
            if measurements['command_phases_completed'] is not True or value['actions_executed'] <= 0 or value['status'] not in ('completed', 'uncertain'):
                raise ValueError('A command receipt cannot advance an unexecuted or unfinished tool')
            anchor = validate_anchor(measurements['public_insertion_anchor'])
            progress = validate_progress(measurements['public_insertion_progress'])
            if len(anchor['sources']) != 3 or progress['order'] != [1, 0, 2]:
                raise ValueError('Insertion handoffs must retain the actually declared source order')
            stages = {'grasp_tube': 1, 'lift_tube': 2, 'upright_tube': 3, 'seat_tube': 0}
            if request['skill'] in stages:
                if progress['stage'] != stages[request['skill']]:
                    raise ValueError('A receipt cannot advance a different insertion stage')
                cursor = progress['cursor'] - int(request['skill'] == 'seat_tube')
                slot = progress['order'][cursor] if cursor < 3 else 3
                if request['slot'] != slot or not 0 <= cursor <= 3:
                    raise ValueError('A stage receipt must belong to the requested own source slot')
            if request['skill'] == 'tube_scene_view' and progress['cursor'] != request['cursor']:
                raise ValueError('A view cannot advance the insertion cursor')
            if request['skill'] == 'reseat_tube' and (not progress['inspection_done'] or progress['stage'] != 0):
                raise ValueError('A reseat receipt describes only its single own attempt')
            if request['skill'] == 'finish_tube_sequence':
                final_stages = ('home_left', 'home_right', 'settle')
                if not progress['final_stage'] or final_stages[progress['final_stage'] - 1] != request['stage']:
                    raise ValueError('A reset receipt must retain the original own-home order')
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise ExecutionError('An insertion receipt cannot leak private state or advance another own command stage') from error
    if any((type(measurements[name]) is not bool for name in flags)):
        raise ExecutionError('Image visibility measurements must be explicit booleans')
    for name in angles:
        angle = measurements[name]
        if angle is not None and (type(angle) not in (int, float) or not math.isfinite(angle) or (not 0 <= angle <= math.pi / 2)):
            raise ExecutionError('An image-axis error is not a calibrated world angle')
    lift = value['visual_lift_feedback']
    if not isinstance(lift, dict) or set(lift) != {'status', 'head_displacement_fraction', 'head_area_ratio', 'robot_translation_m'} or lift['status'] not in ('unavailable', 'ambiguous', 'no_visible_follow', 'visible_change'):
        raise ExecutionError('Visual lift feedback is an uncertain public-image hypothesis')
    for name in ('head_displacement_fraction', 'head_area_ratio', 'robot_translation_m'):
        number = lift[name]
        if number is None and name != 'robot_translation_m':
            continue
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ExecutionError('Visual lift measurements must be finite public image/robot differences')
    if lift['head_displacement_fraction'] is not None and lift['head_displacement_fraction'] > math.sqrt(2) or lift['robot_translation_m'] > 0.3:
        raise ExecutionError('Visual lift measurements exceed the declared relative movement')
    return deepcopy(value)
