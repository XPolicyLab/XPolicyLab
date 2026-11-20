"""Blocking-policy counterpart of the mature dense action sampler."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np

from skills.shared.orchestration.public_rule_policy_interface.implementation import normalized_action


class PublicActionBudgetReached(RuntimeError):
    pass


def color_frame(frame):
    return {
        'state': {key: np.asarray(value).copy() for key, value in frame['state'].items()},
        'vision': {name: {'color': np.asarray(camera.get('color', camera.get('colors'))).copy()}
                   for name, camera in frame['vision'].items()},
    }


class PublicHistory:
    """Observation-only legacy history; no source object behind the view."""
    def __init__(self, initial, output_root):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.frame_interval = 10
        self._sim_step = 0
        self.images = {name: [] for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}
        self.phases = []
        self.public_environment = None
        for side in ('left', 'right'):
            for field in ('arm', 'ee', 'gripper'):
                setattr(self, 'state_' + side + '_' + field, [])
        self.update(initial, phase=None)

    def update(self, observation, *, phase):
        native = color_frame(observation)
        self.current_public_observation = native
        for name in self.images:
            self.images[name].append(native['vision'][name]['color'])
        for side in ('left', 'right'):
            for field, suffix in (('arm', 'arm_joint_state'), ('ee', 'ee_pose'), ('gripper', 'ee_joint_state')):
                getattr(self, 'state_' + side + '_' + field).append(native['state'][side + '_' + suffix])
        if phase is not None:
            self.phases.append(str(phase))
            self._sim_step += self.frame_interval
        if self.public_environment is not None:
            self.public_environment.robot_manager.update(native)
            self.public_environment.take_action_cnt[0] = len(self.phases)


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if not result.size or not np.isfinite(result).all():
        raise ValueError('Joint controls must be nonempty finite vectors')
    return result.copy()


class PublicCommandClock:
    def __init__(self, environment, history, exchange, *, time_scale):
        self._environment = environment
        self._history = history
        self._exchange = exchange
        self.time_scale = float(time_scale)
        if not np.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError('A finite positive mature action speed is required')
        self.interval, self.frequency = 10, 25.0
        self.normalized_actions, self.action_sim_steps, self.action_status = [], [], []
        self.after_action = None
        self._targets, self._grippers = {}, {}
        manager = environment.robot_manager
        for robot in manager.robot_list:
            if robot.type != 'target':
                continue
            arm = manager.process_name(robot.arm_name)
            grip = manager.process_name(robot.gripper_name)
            self._targets[arm] = array(history.current_public_observation['state'][arm])
            lower, upper = array(robot.gripper_scale)
            if upper <= lower:
                raise ValueError('Invalid declared gripper range')
            value = float(array(history.current_public_observation['state'][grip])[0])
            physical = lower + value * (upper - lower) if robot.gripper_move['sign'] == 1 else upper - value * (upper - lower)
            mimic = robot.gripper_move.get('mimic', [None, 1., 0.])
            self._targets[grip] = np.array([physical, physical * mimic[1] + mimic[2]])
            self._grippers[grip] = robot
        if len(self._targets) != 4:
            raise ValueError('Two declared X5 target arms are required')

    @property
    def action_count(self):
        return len(self.normalized_actions)

    def _raise_if_ended(self):
        if self.action_count >= self._environment.step_lim:
            raise PublicActionBudgetReached('Declared public action budget reached')

    def _normalized_action(self, targets):
        result = {}
        for key, target in targets.items():
            if key not in self._grippers:
                result[key] = target.tolist()
            else:
                robot = self._grippers[key]
                lower, upper = robot.gripper_scale
                value = float(np.clip((target[0] - lower) / (upper - lower), 0., 1.))
                if robot.gripper_move['sign'] != 1:
                    value = 1. - value
                result[key] = [value]
        return result

    def _issue(self, action, targets, phase):
        frame = self._exchange(deepcopy(action))
        self._targets = {key: value.copy() for key, value in targets.items()}
        self.normalized_actions.append(deepcopy(action))
        self.action_sim_steps.append(len(self.normalized_actions) * self.interval)
        self._history.update(frame, phase=phase)
        self.action_status.append({'phase': str(phase), 'own_action_count': self.action_count})
        self._raise_if_ended()
        if self.after_action is not None:
            self.after_action(phase)

    def execute_normalized(self, command, phase):
        self._raise_if_ended()
        action = normalized_action(command)
        if 'left_ee_pose' in action:
            raise ValueError('Dense clock accepts joint targets, not pose targets')
        targets = {}
        for key, value in action.items():
            value = array(value)
            if key in self._grippers:
                robot = self._grippers[key]
                fraction = value[0] if robot.gripper_move['sign'] == 1 else 1. - value[0]
                lower, upper = robot.gripper_scale
                physical = lower + fraction * (upper - lower)
                mimic = robot.gripper_move.get('mimic', [None, 1., 0.])
                targets[key] = np.array([physical, physical * mimic[1] + mimic[2]])
            else:
                targets[key] = value.copy()
        self._issue(action, targets, phase)

    def execute(self, controls, phase):
        self._raise_if_ended()
        if not controls:
            return
        dense = []
        current = {key: value.copy() for key, value in self._targets.items()}
        for control in controls:
            for key, value in control.items():
                if key not in current:
                    raise ValueError(f'Unknown joint control key {key!r}')
                value = value.get('position') if isinstance(value, dict) else value
                if value is None:
                    continue
                target = array(value)
                if target.shape != current[key].shape:
                    raise ValueError(f'Joint control shape changed for {key!r}')
                current[key] = target
            dense.append({key: value.copy() for key, value in current.items()})
        arm_keys = set(self._targets) - set(self._grippers)
        has_arm_motion = any(key in arm_keys for control in controls for key in control)
        stride = self.interval * (self.time_scale if has_arm_motion else 1.)
        count = max(1, int(math.ceil(len(dense) / stride)))
        for index in range(count):
            self._raise_if_ended()
            source = min((index + 1) * stride, len(dense)) - 1.
            lower = max(0, int(math.floor(source)))
            upper = min(lower + 1, len(dense) - 1)
            fraction = max(0., source - lower)
            targets = {key: (1. - fraction) * dense[lower][key] + fraction * dense[upper][key] for key in current}
            self._issue(self._normalized_action(targets), targets, phase)
