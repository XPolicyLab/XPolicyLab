"""Public-state execution feedback. No scene queries, scoring or private state.

Limits below are controller diagnostics, not benchmark success thresholds.
Gripper state is a command target, so it is never treated as finger feedback.
"""
import numpy as np
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import JOINT_CHANNELS, ActionSpace

class ExecutionMonitor:
    VERSION = 'public-joint-feedback-v1'
    TRACKING_LIMIT_STEPS = 3.0
    OBSERVED_JUMP_LIMIT_STEPS = 6.0

    def __init__(self, spec):
        self.spec = spec
        self.offsets = ActionSpace(JOINT_CHANNELS).offsets()
        self.limits = np.asarray(spec.max_step, dtype=float)
        if self.limits.shape != spec.low.shape or not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
            raise InfrastructureFailure('execution_monitor_invalid_step_limits')
        self.previous = None

    @staticmethod
    def arm_state(observation):
        result = {}
        for arm in ('left', 'right'):
            key = arm+'_arm_joint_state'
            x = np.asarray(observation.state[key], dtype=float)
            if x.shape != (6,) or not np.isfinite(x).all():
                raise InfrastructureFailure('execution_monitor_invalid_public_state: '+key)
            result[key] = x.copy()
        return result

    def begin(self, observation):
        self.previous = self.arm_state(observation)

    def before(self, action, observation):
        """Reject malformed or discontinuous targets before sending any action."""
        current = self.arm_state(observation)
        if self.previous is None:
            self.previous = current
        if set(action.data) != {k for k, _ in JOINT_CHANNELS}:
            raise InfrastructureFailure('execution_monitor_action_channels_mismatch')
        for key, size in JOINT_CHANNELS:
            x = np.asarray(action.data[key], dtype=float)
            start, _ = self.offsets[key]
            sl = slice(start, start+size)
            if x.shape != (size,) or not np.isfinite(x).all():
                raise InfrastructureFailure('execution_monitor_invalid_action: '+key)
            if np.any(x < self.spec.low[sl]-1e-6) or np.any(x > self.spec.high[sl]+1e-6):
                raise InfrastructureFailure('execution_monitor_joint_limit: '+key)
            if key.endswith('_arm_joint_state'):
                delta = np.abs(x-self.previous[key])
                if np.any(delta > self.limits[sl]+1e-6):
                    raise InfrastructureFailure('execution_monitor_command_jump: '+key+' max_rad='+str(float(delta.max())))

    def after(self, action, before, after):
        old, actual = self.arm_state(before), self.arm_state(after)
        rows = {}; reasons = []
        for key, q in actual.items():
            start, size = self.offsets[key];limit = self.limits[start:start+size]
            target = np.asarray(action.data[key], dtype=float)
            error = np.abs(target-q);movement = np.abs(q-old[key])
            if np.any(error > limit*self.TRACKING_LIMIT_STEPS+1e-6):
                reasons.append('tracking_deviation:'+key)
            if np.any(movement > limit*self.OBSERVED_JUMP_LIMIT_STEPS+1e-6):
                reasons.append('observed_joint_jump:'+key)
            rows[key] = {'before':old[key].tolist(), 'command':target.tolist(),
                         'observed':q.tolist(), 'error_rad':error.tolist(),
                         'observed_delta_rad':movement.tolist()}
            self.previous[key] = target.copy()
        return {'monitor':self.VERSION, 'interrupted':bool(reasons), 'reasons':reasons,
                'joints':rows, 'gripper_feedback_used':False}

    def receipt(self):
        return {'version':self.VERSION, 'tracking_limit_steps':self.TRACKING_LIMIT_STEPS,
                'observed_jump_limit_steps':self.OBSERVED_JUMP_LIMIT_STEPS,
                'threshold_source':'multiples of fixed robot joint command step budgets',
                'state_source':'official public arm joint state', 'gripper_feedback_used':False}
