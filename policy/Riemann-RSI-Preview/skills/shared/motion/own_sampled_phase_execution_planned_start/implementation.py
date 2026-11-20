from copy import deepcopy
import numpy as np
from harness.execution import ExecutionError
from skills.shared.motion.planned_start_phase_execution.implementation import PhasedProgram
from skills.shared.motion.own_robot_kinematic_model_observed_start.implementation import run as own_model

class OwnSampledPhaseProgram(PhasedProgram):
    preserve_commanded_joint_targets = True
    remember_nonrepeatable_commands = True

    def __init__(self, api, request, phases, samples=None, precondition=True, measurements=None):
        super().__init__(api, request, phases, precondition=precondition, measurements=measurements)
        self.own_joint_samples = {}
        model = own_model()
        for index, positions in (samples or {}).items():
            values = np.asarray(positions, dtype=float)
            if type(index) is not int or not 0 <= index < len(phases) or values.ndim != 2 or (values.shape != (phases[index]['minimum_steps'], 6)) or (len(values) > 16) or (not np.isfinite(values).all()) or np.any(values < model.lower_limits) or np.any(values > model.upper_limits) or (phases[index]['require_pose'] is not False):
                raise ExecutionError('An own sampled phase contains only its finite declared robot joint commands')
            self.own_joint_samples[index] = values.copy()
        self.motion_time_scale = 1.0
        self.joint_targets = {}
        self.geometry = {}
        self.stop_after_held_action = False

    def sampled_target(self, observation):
        values = self.own_joint_samples.get(self.phase_index)
        if values is None:
            return None
        current = np.asarray(observation['state'][self.side + '_arm_joint_state'], dtype=float)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise ExecutionError('Sample replay requires actual public own-joint feedback')
        if self.phase_actions == 0 and float(np.max(np.abs(values[-1] - current))) > 0.8:
            self.stop_after_held_action = True
            return current.copy()
        return values[min(self.phase_actions, len(values) - 1)].copy()

    def prepare_sampled_phase(self, unused_observation):
        return None

    def trajectory_operation(self, phase):
        return 'joint' if self.phase_index in self.joint_targets else 'pose'

    def trajectory_joint_target(self, unused_phase):
        return deepcopy(self.joint_targets[self.phase_index])

    def observe(self, observation):
        if self.stop_after_held_action and self.phase_actions:
            self.api.observe(observation)
            distance, angle = self._errors(self.api._observation)
            self._report('stalled', distance, angle)
            self.failed, self.finished, self.status = (True, True, 'uncertain')
            return
        super().observe(observation)
