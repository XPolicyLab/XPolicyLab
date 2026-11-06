from copy import deepcopy
import math
from skills.shared.motion.observed_start_phase_execution.implementation import PhasedProgram

class MaturePoseEndpoints(PhasedProgram):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_tolerance_m = 0.05
        self.rotation_tolerance_radians = math.pi

    def record_action(self):
        name = self.phases[self.phase_index]['name']
        repeated_down = self.phase_actions == 0 and name.startswith('press_down_') and name.endswith('_endpoint_retry')
        super().record_action()
        if repeated_down:
            self.issued_cycles -= 1

    def _abort(self, observation, distance, angle):
        self._report('stalled', distance, angle)
        phase = self.phases[self.phase_index]
        if phase['name'].endswith('_endpoint_retry'):
            if hasattr(self, 'recover_plan_failure') and self.recover_plan_failure(observation):
                return
            self.status, self.finished, self.failed = ('uncertain', True, True)
            return
        if len(self.phases) >= 40 or self.request['steps'] - self.actions < 2:
            self.status, self.finished, self.failed = ('uncertain', True, True)
            return
        retry = deepcopy(phase)
        retry['name'] += '_endpoint_retry'
        self.phases.insert(self.phase_index + 1, retry)
        self.phase_index += 1
        self.phase_actions, self.errors, self.trajectory_pending = (0, [], 0)

    def observe(self, observation):
        self.api.observe(observation)
        if not self.finished and self.phase_actions:
            phase = self.phases[self.phase_index]
            distance, angle = self._errors(self.api._observation)
            if phase['require_pose'] and (not self.trajectory_pending) and (self.phase_actions >= phase['minimum_steps']) and (distance > self.pose_tolerance_m):
                self._abort(self.api._observation, distance, angle)
                return
        super().observe(observation)
