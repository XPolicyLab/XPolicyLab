"""Motion speed limits, and what set each motion's duration. Runs in the simulator process.

The limits default to the original adapter values; a harness may raise them for its own episodes
through ROBOT_SPEED_LIMITS, a JSON object with any of the DEFAULT_LIMITS keys. Each motion result
carries a `timing` entry saying what set its duration, so an agent can see where physical steps go.
"""
import json
import os

DEFAULT_LIMITS = {'speed_m_s': 0.2, 'speed_rad_s': 1.0, 'mark_joint_speed_rad_s': 1.5}
# The adapter stops any control step that would move a joint faster than this (joint_jump_or_speed_limit).
JOINT_STEP_LIMIT_RAD_S = 5.0
MIN_MOVE_S = 0.2
MIN_MARK_S = 0.4


def speed_limits(environ=None):
    """DEFAULT_LIMITS, overridden by the JSON object in ROBOT_SPEED_LIMITS."""
    environ = os.environ if environ is None else environ
    limits = dict(DEFAULT_LIMITS)
    text = environ.get('ROBOT_SPEED_LIMITS')
    if not text:
        return limits
    given = json.loads(text)
    if not isinstance(given, dict) or set(given) - set(limits):
        raise ValueError(f'ROBOT_SPEED_LIMITS takes a JSON object with keys among {sorted(limits)}')
    for key, value in given.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be a number')
        value = float(value)
        if not 0 < value <= 10:
            raise ValueError(f'{key} must be in (0, 10]')
        limits[key] = value
    if limits['mark_joint_speed_rad_s'] > JOINT_STEP_LIMIT_RAD_S:
        raise ValueError(f'mark_joint_speed_rad_s above {JOINT_STEP_LIMIT_RAD_S} would stop every joint-space move')
    return limits


def within(value, limit):
    """A requested move speed within its limit (exactly as the adapter always checked it)."""
    return 0 < value <= limit


# Motion profile of Cartesian moves and trajectory waypoints. By default
# ('constant', the original behaviour) the TCP goes along its straight path at one constant speed from the first
# control step to the last, so the commanded speed jumps from 0 to full speed within one step. 'smooth' keeps
# the same path and the same cruising speed but adds a ramp at each end, in which the speed rises (falls)
# smoothly, so the peak acceleration is about accel_m_s2 / accel_rad_s2. The ramps cost ramp_s of extra time
# per motion: ramp_s = 1.5 * max(speed / accel_m_s2, angular speed / accel_rad_s2) at the motion's cruising
# speeds. ROBOT_MOTION_PROFILE selects it: unset or 'constant', 'smooth', or a JSON object
# {"name": "smooth", "accel_m_s2": ..., "accel_rad_s2": ...}.
SMOOTH_DEFAULTS = {'accel_m_s2': 2.5, 'accel_rad_s2': 12.0}


def motion_profile(text):
    """The motion profile named by a ROBOT_MOTION_PROFILE value: None for the original constant speed, else
    {'name': 'smooth', 'accel_m_s2': ..., 'accel_rad_s2': ...}."""
    text = (text or '').strip()
    if not text or text == 'constant':
        return None
    given = {'name': 'smooth'} if text == 'smooth' else json.loads(text)
    if not isinstance(given, dict) or given.get('name') not in ('constant', 'smooth') or \
            set(given) - {'name', *SMOOTH_DEFAULTS}:
        raise ValueError("ROBOT_MOTION_PROFILE is 'constant', 'smooth' or a JSON object with name "
                         f"'constant' or 'smooth' and any of {sorted(SMOOTH_DEFAULTS)}")
    if given['name'] == 'constant':
        return None
    profile = {'name': 'smooth', **SMOOTH_DEFAULTS}
    for key in SMOOTH_DEFAULTS:
        if key in given:
            value = given[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 100:
                raise ValueError(f'{key} must be a number in (0, 100]')
            profile[key] = float(value)
    return profile


def ramp_time(profile, speed_m_s, speed_rad_s):
    """Extra seconds a motion cruising at these speeds takes under the profile (0 for the constant profile).

    A ramp of length t (one at each end, see progress) covers half the distance the cruise would cover in t,
    so two ramps cost t extra in all; the smoothstep ramp's peak acceleration is 1.5 * speed / t."""
    if not profile:
        return 0.0
    return 1.5 * max(speed_m_s / profile['accel_m_s2'], speed_rad_s / profile['accel_rad_s2'])


def progress(fraction, ramp_fraction=0.0):
    """Share of the straight path covered after `fraction` of the motion's duration.

    With ramp_fraction r = 0 this is `fraction` itself (constant speed). Otherwise the speed rises along a
    smoothstep curve over the first r of the duration, stays at the cruising speed, and falls symmetrically over
    the last r; the acceleration is continuous and zero at both ends. r is capped at 0.5 (no cruise phase)."""
    fraction = min(1.0, max(0.0, float(fraction)))
    r = min(0.5, max(0.0, float(ramp_fraction)))
    if r <= 0:
        return fraction

    def ramp(x):  # area under the smoothstep speed curve over the first x of a ramp lasting r
        p = x / r
        return r * (p ** 3 - p ** 4 / 2)
    if fraction <= r:
        covered = ramp(fraction)
    elif fraction >= 1 - r:
        covered = (1 - r) - ramp(1 - fraction)
    else:
        covered = r / 2 + (fraction - r)
    return covered / (1 - r)


def move_timing(distance_m, angle_rad, speed_m_s, speed_rad_s, limits, profile=None):
    """(duration, timing) of a move: the longest of the minimum duration and the translation and
    rotation at the requested speeds, plus the profile's ramps (none by default)."""
    terms = {'minimum_duration': MIN_MOVE_S, 'translation': distance_m / speed_m_s,
             'rotation': angle_rad / speed_rad_s}
    set_by = max(terms, key=terms.get)
    duration = terms[set_by]
    # Rounded once, as the waypoint ramps are: the same value sets the duration, ramp_s and the ramp fraction, and a
    # negligible ramp (e.g. the rounding-error rotation of a pure translation) is no ramp at all.
    ramp = round(ramp_time(profile, distance_m / duration, angle_rad / duration), 3)
    timing = {'set_by': set_by, 'planned_s': round(duration + ramp, 3),
              'distance_m': round(distance_m, 4), 'angle_rad': round(angle_rad, 4),
              'requested_m_s': speed_m_s, 'requested_rad_s': speed_rad_s,
              'limit_m_s': limits['speed_m_s'], 'limit_rad_s': limits['speed_rad_s']}
    if profile:
        timing['ramp_s'] = ramp
    return duration + ramp, timing


def waypoint_timing(motions, duration_s, limits, ramp_s=None):
    """Timing of a trajectory waypoint of the requested duration; `motions` maps each moving arm to
    its (distance_m, angle_rad). ramp_s: the motion profile's ramps (None without a profile), added to the
    requested duration."""
    translation = max((d for d, _ in motions.values()), default=0.0) / duration_s
    rotation = max((r for _, r in motions.values()), default=0.0) / duration_s
    timing = {'set_by': 'requested_duration', 'planned_s': round(duration_s + (ramp_s or 0.0), 3),
              'translation_m_s': round(translation, 4), 'rotation_rad_s': round(rotation, 4),
              'limit_m_s': limits['speed_m_s'], 'limit_rad_s': limits['speed_rad_s']}
    if ramp_s is not None:
        timing['ramp_s'] = round(ramp_s, 3)
    return timing


def mark_timing(joint_change_rad, limits, speed_rad_s=None):
    """(duration, timing) of a joint-space move to a marked pose, at the requested joint speed
    (speed_rad_s, checked by the caller against the limit) or else at the joint-speed limit."""
    if speed_rad_s is None:
        terms = {'minimum_duration': MIN_MARK_S, 'joint_speed': joint_change_rad / limits['mark_joint_speed_rad_s']}
    else:
        terms = {'minimum_duration': MIN_MARK_S, 'requested_joint_speed': joint_change_rad / speed_rad_s}
    set_by = max(terms, key=terms.get)
    timing = {'set_by': set_by, 'planned_s': round(terms[set_by], 3),
              'largest_joint_change_rad': round(joint_change_rad, 4),
              'limit_joint_rad_s': limits['mark_joint_speed_rad_s']}
    if speed_rad_s is not None:
        timing['requested_joint_rad_s'] = speed_rad_s
    return terms[set_by], timing


def gripper_width_floor(scale_min_m, width_zero_m, squeeze):
    """Lowest width target a gripper command may use: 0, or with squeeze the robot's own firmest
    closing command (both fingers at scale_min_m), which makes position-controlled fingers press harder."""
    return 2.0 * float(scale_min_m) + float(width_zero_m) if squeeze else 0.0


def gripper_timing(duration_s):
    return {'set_by': 'requested_duration', 'planned_s': round(duration_s, 3)}
