from copy import deepcopy
import math
import numpy as np
from harness.execution import ExecutionError
from harness.runtime.observation import public_observation
from skills.shared.motion.observed_start_joint_planning.implementation import TRAJECTORY_TIME_SCALES
from skills.shared.motion.observed_start_phase_execution.implementation import PhasedProgram


from skills.shared.motion.paired_joint_execution.implementation import validate_public_pair_recipe

class PairedJointProgram(PhasedProgram):
    paired_joint_execution = True
    preserve_commanded_joint_targets = True

    def __init__(self, api, request, paired_phases, **options):
        if request['side'] != 'left':
            raise ExecutionError('A paired program uses left only as its feedback anchor')
        if any((set(phase) != {'left', 'right', 'operation', 'time_scale'} for phase in paired_phases)):
            raise ExecutionError('Paired motion requires two explicit public arm targets')
        for phase in paired_phases:
            if phase['operation'] not in ('pose', 'joint', 'rule_joint', 'hold') or type(phase['time_scale']) not in (int, float) or phase['time_scale'] not in TRAJECTORY_TIME_SCALES or any((phase['left'][name] != phase['right'][name] for name in ('name', 'minimum_steps', 'require_pose'))):
                raise ExecutionError('Both arms must share the phase clock and bounded planner contract')
        super().__init__(api, request, [phase['left'] for phase in paired_phases], **options)
        PhasedProgram(api, dict(request, side='right'), [phase['right'] for phase in paired_phases])
        self.paired_phases = deepcopy(paired_phases)
        self.paired_paths = None
        self.paired_index = 0
        self.paired_key = None
        self.commanded_joints = {side: api._observation['state'][side + '_arm_joint_state'].copy() for side in ('left', 'right')}

    def _errors(self, observation):
        phase = self.paired_phases[min(self.phase_index, len(self.paired_phases) - 1)]
        errors = []
        for side in ('left', 'right'):
            current = observation['state'][side + '_ee_pose']
            target = phase[side]['pose']
            errors.append((float(np.linalg.norm(current[:3] - target[:3])), 2 * math.acos(float(np.clip(abs(current[3:] @ target[3:]), 0, 1)))))
        return (max((value[0] for value in errors)), max((value[1] for value in errors)))

    def _abort(self, observation, distance, angle):
        self._report('stalled', distance, angle)
        self.status, self.finished, self.failed = ('uncertain', True, True)

    def paired_action(self, observation, remaining_actions, record):
        current = public_observation(observation)
        phase = self.paired_phases[self.phase_index]
        if self.paired_key != self.phase_index:
            self.paired_key, self.paired_paths, self.paired_index = (self.phase_index, None, 0)
        if phase['operation'] != 'hold' and self.paired_paths is None:
            paths = {}
            for side in ('left', 'right'):
                response = self.api.plan_trajectory(side, phase[side]['pose'], getattr(self, 'geometry', {}).get('support_plane_m'), remaining_actions, phase['time_scale'], operation=phase['operation'], joint_target=self.api.home_joint_states()[side] if phase['operation'] in ('joint', 'rule_joint') else None)
                record({'event': 'mature_joint_trajectory', 'task_tool_phase': phase[side]['name'], 'side': side, 'paired_clock': True, 'status': response['status'], 'dense_waypoints': response['dense_waypoints'], 'official_waypoints': response['official_waypoints'], 'planner': 'original_curobo_plan_joint' if phase['operation'] in ('joint', 'rule_joint') else 'original_curobo_plan_path', 'trajectory_operation': phase['operation'], 'time_scale': phase['time_scale'], 'requested_public_pose': phase[side]['pose'].tolist(), 'new_ik_algorithm': False, 'robot_state_source': 'public_joint_and_ee_observations', 'simulator_collision_settings_changed': False})
                if response['status'] != 'Success':
                    distance, angle = self._errors(current)
                    self._abort(current, distance, angle)
                    return None
                paths[side] = response['positions']
            if any((not len(path) for path in paths.values())):
                raise ExecutionError('A paired trajectory cannot silently omit either arm')
            self.paired_paths = paths
            self.trajectory_pending = max((len(path) for path in paths.values()))
        action = {}
        for side in ('left', 'right'):
            if self.paired_paths is not None:
                path = self.paired_paths[side]
                self.commanded_joints[side] = np.asarray(path[min(self.paired_index, len(path) - 1)], dtype=np.float32)
            action[side + '_arm_joint_state'] = self.commanded_joints[side].copy()
            opening = phase[side]['opening']
            action[side + '_ee_joint_state'] = np.asarray([opening], dtype=np.float32)
            self.targets[side + '_ee_joint_state'] = action[side + '_ee_joint_state'].copy()
            self.targets[side + '_ee_pose'] = phase[side]['pose'].copy()
        self.paired_index += 1
        return action
