"""Control-sampled robot-pose disturbances; no object or contact-state access."""
import copy
import math
from geometry import errors

THRESHOLDS = {'position_m': .005, 'rotation_rad': .05}
# Waypoint keys that move an arm. An entry that only sets a gripper width (no pose key) holds its arm, so that arm
# stays watched as during gripper.set_width; any other entry counts as moving, as before.
POSE_KEYS = ('position_m', 'delta_m', 'quaternion_xyzw', 'rotation_axis')


def moves_arm(entry):
    if isinstance(entry, dict) and entry.get('gripper_width_m') is not None:
        return any(entry.get(key) is not None for key in POSE_KEYS)
    return True


class HeldPoseMonitor:
    def __init__(self, method, params, observation, control_hz=None):
        params = params if isinstance(params, dict) else {}
        states = observation.get('robot_state', {})
        if method in ('move', 'move_to_mark'):
            held = set() if params.get('arm') == 'both' else set(states) - {params.get('arm')}
        elif method in ('gripper', 'hold'):
            held = set(states)  # Finger motion commands and a hold keep both arm TCP targets.
        elif method == 'trajectory':
            waypoints = params.get('waypoints', [])
            # Leave invalid input to the existing executor validation and fault
            # journal; an observer must not throw before that path runs.
            moving = (set(states) if not isinstance(waypoints, list) else
                      {arm for wp in waypoints if isinstance(wp, dict) for arm in states
                       if arm in wp and moves_arm(wp[arm])})
            held = set(states) - moving
        else:
            held = set()
        self.gaps, self.gap_count = [], 0
        self.reference = {}
        for arm in sorted(held):
            try:
                tcp = states[arm]['tcp']
                position, rotation = errors(tcp, tcp)
                if not math.isfinite(position) or not math.isfinite(rotation): raise ValueError('nonfinite initial robot pose')
                self.reference[arm] = copy.deepcopy(tcp)
            except (KeyError, TypeError, ValueError) as error:
                self.gap_count += 1
                if len(self.gaps) < 3:
                    self.gaps.append({'arm': arm, 'observation_id': observation.get('observation_id'), 'error': str(error)})
        self.control_hz = control_hz
        self.peaks = {arm: {'position_m': 0., 'rotation_rad': 0.,
                           'position_peak_observation_id': observation['observation_id'],
                           'rotation_peak_observation_id': observation['observation_id']} for arm in self.reference}

    def update(self, observation):
        for arm, reference in self.reference.items():
            try:
                position, rotation = errors(observation['robot_state'][arm]['tcp'], reference)
                if not math.isfinite(position) or not math.isfinite(rotation): raise ValueError('nonfinite robot pose')
                peak = self.peaks[arm]
                for key, value, frame_key in (
                        ('position_m', position, 'position_peak_observation_id'),
                        ('rotation_rad', rotation, 'rotation_peak_observation_id')):
                    if value > peak[key]: peak[key], peak[frame_key] = value, observation['observation_id']
            except (KeyError, TypeError, ValueError) as error:
                # An evidence observer must not change or abort physical execution.
                self.gap_count += 1
                if len(self.gaps) < 3:
                    self.gaps.append({'arm': arm, 'observation_id': observation.get('observation_id'), 'error': str(error)})

    def report(self):
        if not self.reference and not self.gap_count: return None
        attention = bool(self.gap_count) or any(p['position_m'] > THRESHOLDS['position_m']
                      or p['rotation_rad'] > THRESHOLDS['rotation_rad'] for p in self.peaks.values())
        result = {'status': 'incomplete' if self.gap_count else 'observed', 'attention': attention,
                  'control_hz': self.control_hz, 'thresholds': dict(THRESHOLDS),
                  'by_arm': copy.deepcopy(self.peaks)}
        if self.gap_count: result.update(gap_count=self.gap_count, gaps=copy.deepcopy(self.gaps))
        return result
