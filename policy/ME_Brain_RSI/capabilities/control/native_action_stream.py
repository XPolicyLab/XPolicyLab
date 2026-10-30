"""Track native policy targets without restarting a trajectory at every action."""
import math

import numpy as np

from .kinematics import JOINT_LOWER, JOINT_UPPER
from .motion import (ARMS, CONTROL_DT, FINAL_JOINT_TOLERANCE,
                     MAX_ACTUAL_TARGET_LEAD, MAX_JOINT_ACCELERATION,
                     MAX_JOINT_VELOCITY, MAX_TRACKING_ERROR, WRIST_MARGIN,
                     _check_native_command_path, _joint_action, _measured_state)
from .robot_motion_geometry import check_joint_path


_ACCELERATION_STEP = MAX_JOINT_ACCELERATION * CONTROL_DT
_LOWER = np.tile(JOINT_LOWER + WRIST_MARGIN, len(ARMS))
_UPPER = np.tile(JOINT_UPPER - WRIST_MARGIN, len(ARMS))
_TOLERANCE = 1e-10
_TERMINAL_SETTLE_FRAMES = 8


def stopping_distance(speed):
    """Include the proposed command and its fastest discrete deceleration."""
    speed = np.asarray(speed, float)
    if np.any(speed < 0) or not np.isfinite(speed).all():
        raise ValueError('A finite nonnegative command speed is required')
    count = np.ceil(speed / _ACCELERATION_STEP).astype(int)
    return CONTROL_DT * (count * speed - _ACCELERATION_STEP * count * (count - 1) / 2)


def _stopping_speed(distance):
    """Invert the discrete stopping distance without assuming continuous time."""
    distance = np.maximum(np.asarray(distance, float), 0.)
    lower = np.zeros_like(distance)
    upper = np.full_like(distance, MAX_JOINT_VELOCITY)
    for _ in range(45):
        middle = (lower + upper) / 2
        accepted = stopping_distance(middle) <= distance
        lower = np.where(accepted, middle, lower)
        upper = np.where(accepted, upper, middle)
    return lower


def bounded_following_point(previous, velocity, target):
    """Return one command with discrete speed, acceleration and limit bounds."""
    previous, velocity, target = (np.asarray(value, float) for value in
                                   (previous, velocity, target))
    if any(value.shape != (12,) or not np.isfinite(value).all()
           for value in (previous, velocity, target)):
        raise RuntimeError('Invalid native stream command state')
    _check_native_command_path(np.stack([previous, target]))
    if np.max(np.abs(velocity)) > MAX_JOINT_VELOCITY + _TOLERANCE:
        raise RuntimeError('Native stream incoming velocity exceeds its bound')
    # Reserve the distance required to stop before a native or wrist boundary.
    negative_cap = _stopping_speed(previous - _LOWER)
    positive_cap = _stopping_speed(_UPPER - previous)
    lower = np.maximum(velocity - _ACCELERATION_STEP, -negative_cap)
    upper = np.minimum(velocity + _ACCELERATION_STEP, positive_cap)
    if np.any(lower > upper + _TOLERANCE):
        raise RuntimeError('Native stream cannot brake before a joint boundary')
    delta = target - previous
    requested = np.sign(delta) * _stopping_speed(np.abs(delta))
    # A moving command can pass a suddenly reversed target while braking.
    # Removing that overshoot by snapping to the target would break acceleration.
    next_velocity = np.clip(requested, lower, upper)
    point = previous + CONTROL_DT * next_velocity
    _check_native_command_path(np.stack([previous, point]))
    if (np.max(np.abs(next_velocity)) > MAX_JOINT_VELOCITY + _TOLERANCE or
            np.max(np.abs(next_velocity - velocity)) > _ACCELERATION_STEP + _TOLERANCE):
        raise RuntimeError('Native stream discrete timing bounds are infeasible')
    return point, next_velocity


class NativeActionStream:
    """Emit at most one checked motion command per original policy action.

    The context owns the last emitted position and velocity. Original targets
    may remain unachieved when they update; measured tracking and that residual
    are recorded independently. No per-action arrival wait or rest assumption
    is made. A final explicit stop is available at the end of a policy stream.
    """
    def __init__(self, ctx):
        if not ctx.numeric_ik or ctx.kinematics is None:
            raise RuntimeError('Native streaming requires public numerical kinematics')
        self.ctx = ctx
        self.last_target = None
        self.terminal = False
        self.evidence = ctx.memory.setdefault('native_action_stream', {
            'schema': 'b88r42wq.native-action-stream/v1',
            'control_dt_s': CONTROL_DT,
            'maximum_velocity_rad_s': MAX_JOINT_VELOCITY,
            'maximum_acceleration_rad_s2': MAX_JOINT_ACCELERATION,
            'model_targets': 0, 'command_frames': 0, 'terminal_command_frames': 0,
            'maximum_tracking_error_rad': 0., 'maximum_target_lead_rad': 0.,
            'maximum_original_target_residual_rad': 0.,
            'history': [], 'original_timing_preserved': False,
            'model_action_rate_preserved': True,
            'per_action_rest_to_rest_retiming': False,
            'original_target_arrival_required_per_action': False,
            'policy_reads_simulator_truth': False,
        })

    def _command_state(self):
        commanded = np.concatenate([np.asarray(self.ctx.joint_targets[arm], float)
                                    for arm in ARMS])
        velocity = np.concatenate([np.asarray(self.ctx.joint_command_velocity[arm], float)
                                   for arm in ARMS])
        return commanded, velocity

    def _record(self, report):
        self.ctx.report_motion(report)
        for source, maximum in [('tracking_error_rad', 'maximum_tracking_error_rad'),
                                ('next_target_lead_rad', 'maximum_target_lead_rad'),
                                ('original_target_residual_rad', 'maximum_original_target_residual_rad')]:
            if source in report:
                self.evidence[maximum] = max(self.evidence[maximum], float(report[source]))
        self.evidence['history'].append({key: value for key, value in report.items()
                                        if key != 'robot_geometry'})
        self.evidence['history'] = self.evidence['history'][-64:]

    def _checked_action(self, point, original_target, phase):
        measured = _measured_state(lambda: self.ctx.state)
        actual = np.concatenate([np.asarray(measured[arm+'_arm_joint_state'], float)
                                 for arm in ARMS])
        previous, incoming = self._command_state()
        velocity = (point - previous) / CONTROL_DT
        tracking = float(np.max(np.abs(actual - previous)))
        lead = float(np.max(np.abs(point - actual)))
        report = {
            'mode': 'lxm6wuec_native_stream', 'phase': phase,
            'tracking_error_rad': tracking, 'next_target_lead_rad': lead,
            'maximum_command_velocity_rad_s': float(np.max(np.abs(velocity))),
            'maximum_command_acceleration_rad_s2':
                float(np.max(np.abs(velocity - incoming)) / CONTROL_DT),
            'original_joint_target': original_target.tolist(),
            'emitted_joint_target': point.tolist(),
            'original_target_residual_rad': float(np.max(np.abs(point-original_target))),
            'measured_original_target_error_rad': float(np.max(np.abs(actual-original_target))),
            'original_target_arrived':
                bool(np.max(np.abs(actual-original_target)) <= FINAL_JOINT_TOLERANCE),
            'geometry_path_kind': 'actual_previous_next_emitted_joint_chords',
            'native_timing_preserved': False,
            'model_action_rate_preserved': phase == 'native_stream_tracking',
            'stream_command_frame': self.evidence['command_frames'] + 1,
        }
        if tracking > MAX_TRACKING_ERROR + _TOLERANCE or lead > MAX_ACTUAL_TARGET_LEAD + _TOLERANCE:
            report['phase'] = 'native_stream_tracking_rejected'
            self._record(report)
            raise RuntimeError(f'Native stream tracking lost: error={tracking:.6f}, '
                               f'next_target_lead={lead:.6f}')
        _check_native_command_path(np.stack([actual, previous, point]))
        if (report['maximum_command_velocity_rad_s'] > MAX_JOINT_VELOCITY + _TOLERANCE or
                report['maximum_command_acceleration_rad_s2'] > MAX_JOINT_ACCELERATION + _TOLERANCE):
            raise RuntimeError('Native stream command violates discrete timing bounds')
        geometry = check_joint_path(self.ctx.kinematics, np.stack([actual, previous, point]),
                                    self.ctx.table_z)
        report['robot_geometry'] = geometry
        if not geometry['accepted']:
            report['phase'] = 'native_stream_geometry_rejected'
            self._record(report)
            raise RuntimeError('Native stream robot geometry rejected: '+str(geometry))
        self.evidence['command_frames'] += 1
        self._record(report)
        return _joint_action({arm: point[index*6:(index+1)*6]
                              for index, arm in enumerate(ARMS)}, self.ctx.gripper_targets)

    def emit_joint_target(self, targets, grips=None):
        """Track one native joint target using exactly one ordinary action frame."""
        def command():
            if self.terminal:
                raise RuntimeError('A stopped native stream cannot accept another policy target')
            if set(targets) != set(ARMS):
                raise ValueError('A native joint target must contain both arms')
            target = np.concatenate([np.asarray(targets[arm], float) for arm in ARMS])
            if any(np.asarray(targets[arm]).shape != (6,) for arm in ARMS):
                raise ValueError('Native joint targets must each have six values')
            if grips is not None:
                if set(grips) != set(ARMS) or any(not np.isfinite(grips[arm]) or
                        not 0 <= float(grips[arm]) <= 1 for arm in ARMS):
                    raise ValueError('Native gripper commands must each lie in 0..1')
                self.ctx.gripper_targets.update({arm: float(grips[arm]) for arm in ARMS})
            previous, velocity = self._command_state()
            point, _ = bounded_following_point(previous, velocity, target)
            self.last_target = target.copy()
            self.evidence['model_targets'] += 1
            yield self._checked_action(point, target, 'native_stream_tracking')
        try:
            yield from self.ctx.emit_numeric(command(), ARMS)
        except (RuntimeError, ValueError):
            self.terminal = True
            self.evidence['original_timing_preserved'] = False
            self.evidence['model_action_rate_preserved'] = False
            self.evidence['terminated_after_rejection'] = True
            raise

    def finish(self, settle_frames=_TERMINAL_SETTLE_FRAMES):
        """Brake to rest, then check measured tracking without claiming task arrival."""
        if not isinstance(settle_frames, int) or not 0 <= settle_frames <= _TERMINAL_SETTLE_FRAMES:
            raise ValueError('Invalid native stream terminal settling budget')
        if self.terminal:
            return
        self.terminal = True
        def stop():
            count = int(math.ceil(MAX_JOINT_VELOCITY / _ACCELERATION_STEP)) + 1
            for _ in range(count):
                previous, velocity = self._command_state()
                target = previous if self.last_target is None else self.last_target
                next_velocity = np.sign(velocity) * np.maximum(np.abs(velocity)-_ACCELERATION_STEP, 0.)
                next_velocity[np.abs(next_velocity) <= _TOLERANCE] = 0.
                point = previous + CONTROL_DT * next_velocity
                yield self._checked_action(point, target, 'native_stream_terminal_braking')
                self.evidence['terminal_command_frames'] += 1
                if np.max(np.abs(next_velocity)) <= _TOLERANCE:
                    break
            else:
                raise RuntimeError('Native stream could not stop within its braking budget')
            for index in range(settle_frames+1):
                measured = _measured_state(lambda: self.ctx.state)
                previous, velocity = self._command_state()
                actual = np.concatenate([np.asarray(measured[arm+'_arm_joint_state'], float)
                                         for arm in ARMS])
                target = previous if self.last_target is None else self.last_target
                error = float(np.max(np.abs(actual-previous)))
                if error <= FINAL_JOINT_TOLERANCE:
                    self._record({'phase': 'native_stream_stopped',
                        'tracking_error_rad': error,
                        'original_target_residual_rad': float(np.max(np.abs(actual-target))),
                        'original_target_arrived':
                            bool(np.max(np.abs(actual-target)) <= FINAL_JOINT_TOLERANCE),
                        'terminal_command_velocity_rad_s': float(np.max(np.abs(velocity))),
                        'terminal_settling_frames': index})
                    self.evidence['stopped'] = True
                    return
                if index == settle_frames:
                    raise RuntimeError(f'Native stream terminal tracking stalled: error={error:.6f}')
                yield self._checked_action(previous, target, 'native_stream_terminal_settling')
                self.evidence['terminal_command_frames'] += 1
        try:
            yield from self.ctx.emit_numeric(stop(), ARMS)
        except RuntimeError:
            self.evidence['terminated_after_rejection'] = True
            raise
        finally:
            if self.evidence['terminal_command_frames']:
                self.evidence['original_timing_preserved'] = False

    def fail(self, reason):
        """Route perception or IK rejection through the existing bounded stop."""
        self.terminal = True
        self.evidence['original_timing_preserved'] = False
        self.evidence['model_action_rate_preserved'] = False
        self.evidence['terminated_after_rejection'] = True
        self.ctx.report_motion({'phase': 'native_stream_external_rejection',
                                'native_stream_rejection_reason': str(reason)})
        def failed():
            raise RuntimeError(str(reason))
            yield
        yield from self.ctx.emit_numeric(failed(), ARMS)
