from copy import deepcopy
import math
from pathlib import Path
import json

import numpy as np

from harness.execution import ExecutionError
from harness.runtime.observation import public_observation
from skills.shared.motion.native_timed_cartesian.implementation import PROFILES as TIMED_PROFILES


REQUEST_FIELDS = {'observation', 'side', 'pose', 'support_plane_m', 'remaining_actions', 'time_scale',
                  'operation', 'joint_target'}
RESULT_FIELDS = {'status', 'positions', 'dense_waypoints', 'official_waypoints'}
TRAJECTORY_TIME_SCALES = (0.25, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4, 1.8, 2.5, 3.0)


def vector(value, dimension, label):
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != dimension or not np.isfinite(result).all():
        raise ExecutionError(label + ' must contain finite public robot values')
    return result.copy()


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != REQUEST_FIELDS:
        raise ExecutionError('A trajectory tool accepts only its declared public arguments')
    result = deepcopy(payload)
    if result['side'] not in ('left', 'right'):
        raise ExecutionError('A trajectory requires an explicitly selected own arm')
    result['observation'] = public_observation(result['observation'])
    result['pose'] = vector(result['pose'], 7, 'Trajectory target')
    if not 0.99 <= np.linalg.norm(result['pose'][3:]) <= 1.01:
        raise ExecutionError('A trajectory target requires a unit public robot orientation')
    if result['operation'] not in ('pose', 'canonical_pose', 'joint', 'own_ik', 'continuous_pose', 'curobo_local_descent',
                                  'x5_pose', 'x5_place', 'x5_place_calibrated', 'x5_retreat',
                                  'sampled_contact', 'sampled_carry', 'sampled_free', 'sampled_home',
                                  'sampled_correction', 'sampled_retreat', 'sampled_view', *TIMED_PROFILES):
        raise ExecutionError('A mature trajectory supports only declared original cuRobo or own-X5 planning')
    if result['operation'] in ('joint', 'sampled_home'):
        result['joint_target'] = vector(result['joint_target'], 6, 'Same-episode own-joint target')
    elif result['joint_target'] is not None:
        raise ExecutionError('A pose plan cannot carry an alternate joint target')
    plane = result['support_plane_m']
    if plane is not None and (type(plane) not in (int, float) or not math.isfinite(plane)):
        raise ExecutionError('An optional planning plane must be a current RGB hypothesis')
    if (type(result['remaining_actions']) is not int or not 1 <= result['remaining_actions'] <= 400
            or type(result['time_scale']) not in (int, float)
            or result['time_scale'] not in TRAJECTORY_TIME_SCALES):
        raise ExecutionError('The mature trajectory has a finite declared replay budget')
    if result['operation'].startswith('sampled_') and result['time_scale'] != 1.:
        raise ExecutionError('The measured-joint profiles already contain their original time sampling')
    if result['operation'] in TIMED_PROFILES and result['time_scale'] != 1.:
        raise ExecutionError('The original timed Cartesian profile is already sampled at the official clock')
    if (result['operation'] == 'own_ik' and (result['remaining_actions'] != 1
            or result['time_scale'] != 1. or result['support_plane_m'] is not None)):
        raise ExecutionError('Read-only own IK carries one pose and no scene plane or motion allowance')
    return result


def replay_waypoints(positions, interval, time_scale):
    dense = np.asarray(positions, dtype=float)
    if (dense.ndim != 2 or dense.shape[1] != 6 or not 1 <= len(dense) <= 20000
            or not np.isfinite(dense).all() or type(interval) is not int or interval < 1
            or not math.isfinite(time_scale) or time_scale <= 0):
        raise ExecutionError('The mature bridge requires a finite six-joint path')
    stride = interval * time_scale
    count = max(1, int(math.ceil(len(dense) / stride)))
    result = []
    for index in range(count):
        source = min((index + 1) * stride, len(dense)) - 1.0
        lower = max(0, int(math.floor(source)))
        upper = min(lower + 1, len(dense) - 1)
        fraction = max(0.0, source - lower)
        result.append((1.0 - fraction) * dense[lower] + fraction * dense[upper])
    return np.asarray(result, dtype=np.float32)


def joint_action(observation, side, joints, opening):
    observation = public_observation(observation)
    result = {}
    for arm in ('left', 'right'):
        result[arm + '_arm_joint_state'] = vector(observation['state'][arm + '_arm_joint_state'], 6, 'Own arm')
        result[arm + '_ee_joint_state'] = vector(observation['state'][arm + '_ee_joint_state'], 1, 'Public gripper')
    result[side + '_arm_joint_state'] = vector(joints, 6, 'Planned own-arm waypoint')
    if type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= 1:
        raise ExecutionError('A planned gripper command is a normalized public command')
    result[side + '_ee_joint_state'] = np.asarray([opening], dtype=np.float32)
    return result


def validate_result(value, remaining_actions):
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS or value['status'] not in ('Success', 'Fail', 'Budget'):
        raise ExecutionError('A trajectory worker returned an undeclared result')
    if any(type(value[name]) is not int or value[name] < 0 for name in ('dense_waypoints', 'official_waypoints')):
        raise ExecutionError('A trajectory result requires explicit waypoint counts')
    if value['status'] != 'Success':
        if value['official_waypoints'] or len(value['positions']):
            raise ExecutionError('An unsuccessful plan cannot issue robot waypoints')
        return value
    positions = np.asarray(value['positions'], dtype=np.float32)
    if (positions.ndim != 2 or positions.shape[1] != 6 or not np.isfinite(positions).all()
            or not 1 <= len(positions) <= remaining_actions
            or len(positions) != value['official_waypoints']):
        raise ExecutionError('A planned path exceeds its public execution budget')
    return dict(value, positions=positions)


def canonical_joint_goal(goal, current, lower, upper):
    """Choose, per revolute joint, the in-limit 2*pi equivalent closest to the current joint.

    Mirrors the mature swap_T rule: cuRobo may return an endpoint that is a full turn away
    from the current configuration, which the arm cannot track within the action budget.
    Returns None when some joint has no in-limit equivalent.
    """
    goal, current = vector(goal, 6, 'Planned endpoint'), vector(current, 6, 'Observed own joints')
    lower, upper = vector(lower, 6, 'Lower joint limits'), vector(upper, 6, 'Upper joint limits')
    result = goal.copy()
    for index, value in enumerate(goal):
        equivalents = [value + turn * 2. * np.pi for turn in range(-3, 4)
                       if lower[index] - 1e-5 <= value + turn * 2. * np.pi <= upper[index] + 1e-5]
        if not equivalents:
            return None
        result[index] = min(equivalents, key=lambda candidate: abs(candidate - current[index]))
    return result


def own_arm_joint_limits(urdf):
    import xml.etree.ElementTree as ElementTree
    joints = {element.get('name'): element for element in ElementTree.parse(urdf).getroot().iter('joint')}
    limits = [joints['joint' + str(index)].find('limit') for index in range(1, 7)]
    return (np.asarray([float(limit.get('lower')) for limit in limits]),
            np.asarray([float(limit.get('upper')) for limit in limits]))


def validate_robot_bases(payload):
    if not isinstance(payload, dict) or set(payload) != {'left', 'right'}:
        raise ExecutionError('Own robot bases contain only the two public-observation-derived roots')
    result = {side: vector(payload[side], 7, 'Inferred own robot base') for side in ('left', 'right')}
    if any(not 0.99 <= np.linalg.norm(pose[3:]) <= 1.01 for pose in result.values()):
        raise ExecutionError('Own robot bases require unit orientations')
    if abs(float(result['left'][2] - result['right'][2])) > 0.005:
        raise ExecutionError('Independent public robot-state root estimates disagree')
    return result


class NativeJointExecutor:
    def __init__(self, planner, record):
        self.planner = planner
        self.record = record
        self.key = None
        self.positions = []
        self.index = 0
        self.commanded_targets = None

    def _joint_action(self, program, observation, joints, opening):
        preserve = getattr(program, 'preserve_commanded_joint_targets', False)
        if type(preserve) is not bool:
            raise ExecutionError('Command-target preservation requires an explicit boolean capability')
        side = program.side
        baseline = self.commanded_targets if preserve and self.commanded_targets is not None else observation['state']
        selected = baseline[side + '_arm_joint_state'] if joints is None else joints
        action = joint_action(observation, side, selected, opening)
        if preserve and self.commanded_targets is not None:
            for name, target in self.commanded_targets.items():
                if name not in (side + '_arm_joint_state', side + '_ee_joint_state'):
                    action[name] = target.copy()
        peer_commands = getattr(program, 'additional_joint_commands', None)
        if peer_commands is not None:
            from skills.shared.motion.own_sampled_phase_execution.implementation import OwnSampledPhaseProgram
            if not isinstance(program, OwnSampledPhaseProgram) or not callable(peer_commands):
                raise ExecutionError('Peer command preservation requires a declared own sampled program')
            other = 'left' if side == 'right' else 'right'
            extras = peer_commands(program.api._observation)
            if not isinstance(extras, dict) or set(extras) != {other + '_arm_joint_state'}:
                raise ExecutionError('A preserved peer command cannot replace the active arm or add private fields')
            target = np.asarray(extras[other + '_arm_joint_state'], dtype=float)
            if target.shape != (6,) or not np.isfinite(target).all() or np.max(np.abs(target)) > 10.:
                raise ExecutionError('A peer snapshot contains only six bounded own joint commands')
            action[other + '_arm_joint_state'] = target.copy()
        extra_commands = getattr(program, 'additional_gripper_commands', None)
        if extra_commands is not None:
            if not callable(extra_commands):
                raise ExecutionError('Paired opening requires a declared bounded public command capability')
            extras = extra_commands(program.api._observation)
            if not isinstance(extras, dict) or set(extras) - {'left_ee_joint_state', 'right_ee_joint_state'}:
                raise ExecutionError('Additional dwell commands contain only own normalized grippers')
            for name, target in extras.items():
                target = np.asarray(target, dtype=float)
                if target.shape != (1,) or not np.isfinite(target).all() or not 0 <= target[0] <= 1:
                    raise ExecutionError('A paired dwell cannot supply real fingers or an unbounded target')
                action[name] = target.copy()
        self.commanded_targets = deepcopy(action)
        if len(self.positions) and self.index == len(self.positions):
            self.record({'event': 'mature_joint_endpoint_command_response', 'side': side,
                         'task_tool_phase': program.phases[program.phase_index]['name'],
                         'phase_actions': program.phase_actions,
                         'commanded_joints': np.asarray(action[side + '_arm_joint_state']).tolist(),
                         'observed_public_joints': np.asarray(observation['state'][side + '_arm_joint_state']).tolist(),
                         'public_joint_command_error_radians': float(np.max(np.abs(
                             action[side + '_arm_joint_state'] - observation['state'][side + '_arm_joint_state']))),
                         'trajectory_changed': False, 'is_contact_or_true_finger_sensor': False})
        return action

    def action(self, program, observation, remaining_actions):
        if program.finished:
            return None
        from skills.shared.motion.own_sampled_phase_execution.implementation import OwnSampledPhaseProgram
        if isinstance(program, OwnSampledPhaseProgram):
            binding = getattr(program, 'bind_original_command_targets', None)
            if binding is not None:
                if not callable(binding):
                    raise ExecutionError('A fixed hold exposes an explicit own-command binding interface')
                baseline = self.commanded_targets if self.commanded_targets is not None else public_observation(observation)['state']
                binding({side + '_arm_joint_state': np.asarray(baseline[side + '_arm_joint_state']).copy()
                    for side in ('left', 'right')})
            program.prepare_sampled_phase(public_observation(observation))
            if program.finished:
                return None
        if isinstance(program, OwnSampledPhaseProgram) and program.phase_index in program.own_joint_samples:
            current = public_observation(observation)
            phase = program.phases[program.phase_index]
            target = program.sampled_target(current)
            if program.phase_actions == 0:
                self.record({'event': 'declared_own_joint_samples', 'task_tool_phase': phase['name'],
                    'official_samples': len(program.own_joint_samples[program.phase_index]),
                    'source': 'public_own_robot_urdf_numerical_command', 'resampled': False,
                    'scene_geometry_is_private': False, 'contact_measurement_claimed': False})
            return self._joint_action(program, current, target, phase['opening'])
        if getattr(program, 'paired_joint_execution', False):
            from skills.shared.motion.paired_joint_execution.implementation import PairedJointProgram
            if not isinstance(program, PairedJointProgram):
                raise ExecutionError('Paired replay requires the declared public paired program')
            if program.paired_key is None and self.commanded_targets is not None:
                program.commanded_joints = {side: self.commanded_targets[side + '_arm_joint_state'].copy()
                                            for side in ('left', 'right')}
            action = program.paired_action(observation, remaining_actions, self.record)
            if action is not None:
                self.commanded_targets = deepcopy(action)
            return action
        phase = program.phases[program.phase_index]
        side = program.side
        current = public_observation(observation)
        program.targets[side + '_ee_pose'] = phase['pose'].copy()
        program.targets[side + '_ee_joint_state'] = np.asarray([phase['opening']], dtype=np.float32)
        key = (program, program.phase_index, phase['name'])
        if key != self.key:
            self.key, self.positions, self.index = key, [], 0
        moving = phase['require_pose'] or phase['name'] == 'lower_to_visible_support'
        if not moving:
            opening = (program.command_opening(current, phase['minimum_steps'])
                       if hasattr(program, 'command_opening') else phase['opening'])
            program.targets[side + '_ee_joint_state'] = np.asarray([opening], dtype=np.float32)
            return self._joint_action(program, current, None, opening)
        if len(self.positions) and self.index == len(self.positions):
            opening = (program.command_opening(current, len(self.positions))
                       if hasattr(program, 'command_opening') else phase['opening'])
            program.targets[side + '_ee_joint_state'] = np.asarray([opening], dtype=np.float32)
            return self._joint_action(program, current, self.positions[-1], opening)
        if not len(self.positions):
            geometry = getattr(program, 'geometry', program.api.geometry_memory())
            home = 'home' in phase['name'] and phase['name'].startswith('return_')
            operation = (program.trajectory_operation(phase) if hasattr(program, 'trajectory_operation') else
                         'joint' if home else 'pose')
            if operation == 'own_ik':
                raise ExecutionError('A read-only IK solution cannot be executed as a robot trajectory')
            request = validate_request({'observation': current, 'side': side, 'pose': phase['pose'],
                'support_plane_m': geometry.get('support_plane_m'), 'remaining_actions': remaining_actions,
                'time_scale': (program.phase_time_scale(phase) if hasattr(program, 'phase_time_scale') else
                    getattr(program, 'motion_time_scale', 1.4 if 'transfer' in phase['name'] or 'carry' in phase['name'] else 1.8)),
                'operation': operation,
                'joint_target': (program.trajectory_joint_target(phase) if hasattr(program, 'trajectory_joint_target')
                    else program.api.home_joint_states()[side]) if operation in ('joint', 'sampled_home') else None})
            cached = program.preplanned.pop(program.phase_index, None)
            response = validate_result(cached if cached is not None else self.planner(request), remaining_actions)
            self.record({'event': 'mature_joint_trajectory', 'task_tool_phase': phase['name'],
                         'status': response['status'], 'dense_waypoints': response['dense_waypoints'],
                         'official_waypoints': response['official_waypoints'],
                         'planner': ('original_x5_measured_fallback' if request['operation'].startswith('x5_') else
                             'existing_own_joint_path_measured_view' if request['operation'] == 'sampled_view' else
                             'original_timed_cartesian_ik' if request['operation'] in TIMED_PROFILES else
                             'original_curobo_plan_joint' if request['operation'] == 'joint' else 'original_curobo_plan_path'),
                         'new_ik_algorithm': False,
                         'robot_state_source': 'public_joint_and_ee_observations',
                         'scene_geometry_source': 'original_empty_planning_scene',
                         'simulator_collision_settings_changed': False})
            if response['status'] != 'Success':
                if hasattr(program, 'recover_plan_failure') and program.recover_plan_failure(current):
                    self.key, self.positions, self.index = None, [], 0
                    return self.action(program, current, remaining_actions)
                distance, angle = program._errors(current)
                program._report('stalled', distance, angle)
                program.status, program.finished, program.failed = 'uncertain', True, True
                return None
            if hasattr(program, 'accept_trajectory'):
                program.accept_trajectory(phase, response, current['state'][side + '_arm_joint_state'])
            self.positions, self.index = response['positions'], 0
            program.trajectory_pending = len(self.positions)
        position = self.positions[self.index]
        self.index += 1
        opening = (program.command_opening(current, len(self.positions))
                   if hasattr(program, 'command_opening') else phase['opening'])
        program.targets[side + '_ee_joint_state'] = np.asarray([opening], dtype=np.float32)
        return self._joint_action(program, current, position, opening)


class MaturePlanner:
    def __init__(self, source):
        from .planner_code import FrozenSingleArmPlanner, owned_configuration
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg

        self.source = Path(source).resolve()
        self.settings = json.loads((self.source / 'configs/mature-trajectory.json').read_text())
        self.configuration = self.source / self.settings['robot_yaml']
        configuration = owned_configuration(self.configuration)
        converter = object.__new__(FrozenSingleArmPlanner)
        robot_configuration, _ = converter._build_robot_and_scene_cfg(configuration, 0.0)
        self.kinematics = Kinematics(KinematicsCfg.from_data_dict(deepcopy(robot_configuration), device_cfg=DeviceCfg()))
        self.active_joints = ['joint' + str(index) for index in range(1, 7)]
        self.planners = {}
        self.bases = {}

    def reset(self):
        self.bases.clear()
        self.planners.clear()
        if hasattr(self, 'sampled_motion'):
            self.sampled_motion.reset()

    def _infer_base(self, observation, side):
        from curobo.types import JointState
        from .planner_code import pose_to_matrix, matrix_to_pose

        joints = vector(observation['state'][side + '_arm_joint_state'], 6, 'Observed own joints')
        names = list(self.kinematics.joint_names)
        position = self.kinematics.default_joint_position.detach().clone().reshape(-1)
        for index, name in enumerate(self.active_joints):
            position[names.index(name)] = float(joints[index])
        state = JointState.from_position(position.reshape(1, -1), joint_names=names)
        tool = self.kinematics.compute_kinematics(state).tool_poses
        tool_pose = np.r_[tool.position.detach().cpu().numpy().reshape(-1, 3)[0],
                          tool.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]]
        observed = vector(observation['state'][side + '_ee_pose'], 7, 'Observed own EE pose')
        root = pose_to_matrix(observed) @ np.linalg.inv(pose_to_matrix(tool_pose))
        return matrix_to_pose(root).astype(np.float32)

    def robot_bases(self, observation):
        observation = public_observation(observation)
        if not self.bases:
            self.bases = validate_robot_bases({side: self._infer_base(observation, side)
                                              for side in ('left', 'right')})
        return validate_robot_bases(self.bases)

    def _canonical_pose_plan(self, side, joints, pose):
        """Original swap_T `_plan`: plan_path, then replan in joint space to a canonical endpoint.

        A plan_path whose endpoint is already canonical is returned unchanged. Otherwise, or when
        plan_path fails, the in-limit 2*pi equivalent nearest the current joints is reached with
        plan_joint (from the plan_path endpoint, or from a fresh own IK solution).
        """
        if not hasattr(self, 'joint_limits'):
            self.joint_limits = own_arm_joint_limits(self.source / self.settings['robot_urdf'])
        planner = self.planners[side]
        result = planner.plan_path(joints, pose.tolist(), self.bases[side])
        if result.get('status') == 'Success':
            endpoint = np.asarray(result['position'], dtype=float)[-1]
        else:
            solved = planner.solve_ik_to_joint(joints, pose.tolist(), self.bases[side])
            if solved.get('status') != 'Success':
                return {'status': 'Fail'}
            endpoint = np.asarray(solved['joint_value'], dtype=float).reshape(-1)
        goal = canonical_joint_goal(endpoint, joints, *self.joint_limits)
        if goal is None:
            return {'status': 'Fail'}
        if result.get('status') == 'Success' and np.max(np.abs(goal - endpoint)) <= 1e-4:
            return result
        print('[Riemann-RSI-Preview] canonical endpoint joint replan side=' + side + ' from='
              + str(np.round(endpoint, 4).tolist()) + ' to=' + str(np.round(goal, 4).tolist()), flush=True)
        return planner.plan_joint(joints, goal.tolist())

    def plan(self, payload):
        from .planner_code import FrozenSingleArmPlanner

        payload = validate_request(payload)
        observation, side = payload['observation'], payload['side']
        if payload['operation'] in TIMED_PROFILES:
            from skills.shared.motion.native_timed_cartesian.implementation import NativeTimedCartesian
            if not hasattr(self, 'timed_cartesian'):
                self.timed_cartesian = NativeTimedCartesian(self.source / self.settings['robot_urdf'])
            return self.timed_cartesian.plan(payload, self.settings['official_substeps'])
        if payload['operation'].startswith('x5_'):
            from skills.shared.motion.mature_x5_fallback.implementation import OriginalX5Fallback
            if not hasattr(self, 'original_fallback'):
                self.original_fallback = OriginalX5Fallback(self.source / self.settings['robot_urdf'])
            try:
                dense = self.original_fallback.path(payload, self.settings['official_substeps'])
            except RuntimeError as error:
                print('[Riemann-RSI-Preview] original own-X5 fallback rejected: ' + str(error), flush=True)
                return {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
            positions = replay_waypoints(dense, self.settings['official_substeps'], payload['time_scale'])
            if len(positions) > payload['remaining_actions']:
                return {'status': 'Budget', 'positions': [], 'dense_waypoints': len(dense), 'official_waypoints': 0}
            return {'status': 'Success', 'positions': positions, 'dense_waypoints': len(dense), 'official_waypoints': len(positions)}
        self.robot_bases(observation)
        if side not in self.planners:
            self.planners[side] = FrozenSingleArmPlanner([0, 0, 0, 1, 0, 0, 0],
                self.active_joints, self.active_joints, dt=self.settings['physics_dt'],
                yml_path=str(self.configuration), table_height=None)
        joints = vector(observation['state'][side + '_arm_joint_state'], 6, 'Observed own joints')
        if payload['operation'] == 'own_ik':
            result = self.planners[side].solve_ik_to_joint(joints, payload['pose'].tolist(), self.bases[side])
            if result.get('status') != 'Success':
                return {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
            solved = vector(result['joint_value'], 6, 'Original own-arm IK solution')
            return {'status': 'Success', 'positions': solved[None, :], 'dense_waypoints': 1, 'official_waypoints': 1}
        if payload['operation'].startswith('sampled_'):
            from skills.shared.motion.mature_sampled_motion.implementation import CheckedSampledMotion
            if not hasattr(self, 'sampled_motion'):
                self.sampled_motion = CheckedSampledMotion(self.source / self.settings['robot_urdf'])
            return self.sampled_motion.plan(self.planners[side], payload)
        if payload['operation'] == 'curobo_local_descent':
            from skills.shared.motion.continuous_joint_descent.implementation import plan_local_descent
            planner = self.planners[side]

            def forward_kinematics(values):
                tool = planner.motion_planner.compute_kinematics(planner._build_joint_state(values)).tool_poses
                return (tool.position.detach().cpu().numpy().reshape(-1, 3)[0],
                        tool.quaternion.detach().cpu().numpy().reshape(-1, 4)[0])

            try:
                dense = plan_local_descent(joints, payload['pose'], self.bases[side], planner.frame_bias,
                                          forward_kinematics, planner.plan_joint)
            except (RuntimeError, ValueError):
                return {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
            result = {'status': 'Success', 'position': dense}
        elif payload['operation'] == 'joint':
            result = self.planners[side].plan_joint(joints, payload['joint_target'].tolist())
        elif payload['operation'] == 'canonical_pose':
            result = self._canonical_pose_plan(side, joints, payload['pose'])
        else:
            result = self.planners[side].plan_path(joints, payload['pose'].tolist(), self.bases[side])
        if result.get('status') != 'Success':
            return {'status': 'Fail', 'positions': [], 'dense_waypoints': 0, 'official_waypoints': 0}
        dense = result['position']
        if payload['operation'] == 'continuous_pose':
            from skills.shared.motion.continuous_joint_descent.implementation import unwrap_path
            dense = unwrap_path(dense, joints)
        positions = replay_waypoints(dense, self.settings['official_substeps'], payload['time_scale'])
        if len(positions) > payload['remaining_actions']:
            return {'status': 'Budget', 'positions': [], 'dense_waypoints': len(dense), 'official_waypoints': 0}
        return {'status': 'Success', 'positions': positions, 'dense_waypoints': len(dense),
                'official_waypoints': len(positions)}
