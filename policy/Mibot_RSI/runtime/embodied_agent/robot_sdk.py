"""Robot client usable from persistent Python, scripts, or another agent host."""
import json
import math
import os
import re
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path


def _numpy_json(value):
    """Accept NumPy motion vectors/scalars without importing NumPy into the SDK."""
    if type(value).__module__.startswith('numpy') and hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not a JSON robot parameter')


def _optional(**values):
    """The given optional parameters (those not None), so a request carries only what the caller set."""
    return {key: value for key, value in values.items() if value is not None}


class Robot:
    def __init__(self, connection=None):
        connection = connection or os.environ['ROBOT_CONNECTION']
        self.connection = json.loads(Path(connection).read_text()) if isinstance(connection, (str, Path)) else connection
        self.active_command = None
        self.active_request = None
        self.result_observer = None  # Optional harness hook; does not alter action semantics.

    def _http(self, path, body):
        req = urllib.request.Request(self.connection['url'] + path,
            data=json.dumps(body, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.connection['token']})
        try:
            with urllib.request.urlopen(req, timeout=15) as f:
                result = json.load(f)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                detail = e.read().decode(errors='replace')
                raise ValueError(f'Robot service rejected request ({e.code}): {detail}') from e
            raise
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def submit(self, operation, command_id=None, **params):
        cid = command_id or uuid.uuid4().hex
        try:
            # Normalize before marking a command active. A local JSON failure
            # sends no HTTP request and must not be reported as uncertain motion.
            body = json.loads(json.dumps({'id': cid, 'method': operation, 'params': params},
                                         allow_nan=False, default=_numpy_json))
        except (TypeError, ValueError) as exc:
            raise ValueError('Robot command has non-JSON or non-finite arguments; '
                             'no request was submitted') from exc
        # A lost reply must be resolved with status(cid), not a fresh motion ID.
        self.active_command = cid
        self.active_request = body
        try:
            return self._http('/submit', body)
        except ValueError:
            self.active_command = None
            self.active_request = None
            raise
        except Exception as e:
            raise RuntimeError(f'Request outcome uncertain; query robot.status({cid!r}) before retrying: {e}') from e

    def status(self, command_id):
        return self._http('/status', {'id': command_id})

    def cancel(self, command_id=None):
        cid = command_id or self.active_command
        return self._http('/cancel', {'id': cid}) if cid else {'status': 'no_active_command'}

    def wait(self, command_id, timeout=600):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            result = self.status(command_id)
            if result['status'] not in ('queued', 'running'):
                if self.active_command == command_id:
                    self.active_command = None
                    self.active_request = None
                if result['status'] == 'fault':
                    raise RuntimeError(result)
                return result.get('result', result)
            time.sleep(.1)
        raise TimeoutError(f'Wait timed out; command may still be active. Query or cancel {command_id}')

    def call(self, operation, **params):
        handle = self.submit(operation, **params)
        result = self.wait(handle['id'])
        if self.result_observer is not None:
            self.result_observer(operation, params, result)
        return result

    def describe(self): return self.call('describe')
    def observe(self): return self.call('observe')
    def replay(self, observation_id): return self.call('replay', observation_id=observation_id)
    def history(self, method=None, arm=None, status=None, label=None, before_sequence=None, limit=10):
        return self.call('history', method=method, arm=arm, status=status, label=label,
                         before_sequence=before_sequence, limit=limit)
    def inspect(self, command_id, phase='both', camera='cam_head', region=None):
        if phase not in ('before', 'after', 'both'): raise ValueError('Invalid phase')
        if isinstance(command_id, str) and re.fullmatch(r'frame_[0-9]+', command_id):
            if phase != 'both': raise ValueError('before/after applies to command IDs; a frame already identifies one observation')
            return self.call('inspect', selections=[{'observation_id': command_id, 'camera': camera, 'region': region}])
        phases = ['before', 'after'] if phase == 'both' else [phase]
        return self.call('inspect', selections=[{'command_id': command_id, 'phase': p,
                          'camera': camera, 'region': region} for p in phases])
    def compare(self, before_observation_id, after_observation_id, camera='cam_head', region=None):
        return self.call('inspect', selections=[{'observation_id': oid, 'camera': camera, 'region': region}
                         for oid in (before_observation_id, after_observation_id)])
    def annotate(self, command_id, label, expected='', observed=''):
        return self.call('annotate', target_command_id=command_id, label=label, expected=expected, observed=observed)
    def arm(self, name): return Arm(self, name)
    def mark_pose(self, name): return self.call('mark_pose', name=name)

    def move_to_mark(self, name='initial', timeout_s=None, speed_rad_s=None):
        # Both arms to the bookmark at once (joint space, one duration); paths are not checked against each other.
        extra = _optional(timeout_s=timeout_s, speed_rad_s=speed_rad_s)
        return self.call('move_to_mark', arm='both', name=name, **extra)
    def hold(self, duration_s=1.):
        # Both arms stay still while duration_s of simulated time passes (0.04..8 s); the simulation otherwise pauses
        # between commands, so this is how to watch whether something settles or falls before the next motion.
        return self.call('hold', duration_s=duration_s)
    def finish(self): return self.call('finish')
    def execute_trajectory(self, waypoints, dry_run=False):
        # dry_run=True: report what the waypoints' per-step checks would decide; nothing moves, no step is used.
        # An arm's waypoint entry may hold gripper_width_m: that gripper starts toward it with the waypoint.
        if dry_run: return self.call('plan', motion='trajectory', waypoints=waypoints)
        return self.call('trajectory', waypoints=waypoints)


class Arm:
    def __init__(self, robot, name):
        self.robot, self.name = robot, name
        self.gripper = Gripper(robot, name)

    def _move(self, dry_run, gripper_width_m=None, timeout_s=None, **params):
        # dry_run=True: report what the motion's per-step checks would decide; nothing moves, no step is used.
        # gripper_width_m: this arm's gripper starts toward that width with the motion's first step (the arm does
        # not wait for it); a dry run only checks the width.
        # timeout_s: optional; given, a motion that would last longer is rejected (without it, the service's own
        # duration limit applies).
        params.update(_optional(gripper_width_m=gripper_width_m, timeout_s=timeout_s))
        return self.robot.call('plan', motion='move', **params) if dry_run else self.robot.call('move', **params)

    def move_by(self, dx_m=0., dy_m=0., dz_m=0., frame='world', speed_m_s=.2, timeout_s=None, dry_run=False,
                gripper_width_m=None):
        return self._move(dry_run, gripper_width_m, timeout_s, arm=self.name, delta_m=[dx_m, dy_m, dz_m],
            frame=frame, speed_m_s=speed_m_s)

    def move_to(self, position_m, quaternion_xyzw=None, speed_m_s=.2, timeout_s=None, speed_rad_s=.5, dry_run=False,
                gripper_width_m=None):
        return self._move(dry_run, gripper_width_m, timeout_s, arm=self.name, position_m=position_m,
            quaternion_xyzw=quaternion_xyzw, frame='world', speed_m_s=speed_m_s, speed_rad_s=speed_rad_s)

    def rotate_by(self, axis, angle_deg, frame='tcp', speed_rad_s=.5, timeout_s=None, dry_run=False,
                  gripper_width_m=None):
        return self._move(dry_run, gripper_width_m, timeout_s, arm=self.name, rotation_axis=axis,
            angle_deg=angle_deg, frame=frame, speed_rad_s=speed_rad_s)

    def adjust(self, delta_mm=(0., 0., 0.), rotation_axis=None, angle_deg=0., frame='tcp',
               speed_mm_s=20., speed_deg_s=20., timeout_s=None):
        """Apply one actor-selected small Cartesian correction.

        Millimetres make visually estimated contact corrections easier to express.
        This is an ergonomic view of the normal move controller: it performs no
        perception, target selection, collision planning or automatic retry.
        """
        if not isinstance(delta_mm, (list, tuple)) or len(delta_mm) != 3:
            raise ValueError('delta_mm must contain three values')
        delta = [float(value) for value in delta_mm]
        angle = float(angle_deg)
        linear_speed = float(speed_mm_s)
        angular_speed = math.radians(float(speed_deg_s))
        if not all(math.isfinite(value) for value in delta + [angle, linear_speed, angular_speed]):
            raise ValueError('Adjustment values must be finite')
        if linear_speed <= 0 or angular_speed <= 0:
            raise ValueError('Adjustment speeds must be positive')
        if angle and rotation_axis is None:
            raise ValueError('A nonzero angle requires rotation_axis')
        if not angle and not any(delta):
            raise ValueError('Adjustment must translate or rotate')
        if angle and any(delta) and (math.dist(delta, [0., 0., 0.]) > 10. or abs(angle) > 10.):
            raise ValueError('A combined adjustment is limited to 10 mm translation and '
                             '10 degrees rotation; no request was submitted. Separate larger '
                             'translation and rotation, then inspect the held object and clearance.')
        params = {'arm': self.name, 'frame': frame,
                  'speed_m_s': linear_speed / 1000., 'speed_rad_s': angular_speed, **_optional(timeout_s=timeout_s)}
        if any(delta): params['delta_m'] = [value / 1000. for value in delta]
        if angle:
            params['rotation_axis'] = rotation_axis
            params['angle_deg'] = angle
        return self.robot.call('move', **params)

    def move_to_mark(self, name='initial', timeout_s=None, speed_rad_s=None):
        # speed_rad_s: joint speed of this joint-space move, up to the service's limit (default: the limit).
        extra = _optional(timeout_s=timeout_s, speed_rad_s=speed_rad_s)
        return self.robot.call('move_to_mark', arm=self.name, name=name, **extra)


class Gripper:
    def __init__(self, robot, arm): self.robot, self.arm = robot, arm
    def set_width(self, width_m, duration_s=.4):
        return self.robot.call('gripper', arm=self.arm, width_m=width_m, duration_s=duration_s)
    def _range(self):
        description = self.robot.describe()
        return (description['arms'][self.arm].get('gripper_width_range_m')
                or description['gripper_width_range_m'])
    def open(self): return self.set_width(self._range()[1])
    def close(self): return self.set_width(self._range()[0])
