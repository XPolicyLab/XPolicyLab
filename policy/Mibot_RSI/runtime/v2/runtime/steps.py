"""Where an episode's physical steps went: the step_report helper of an actor session.

Built from the simulator's command history: each motion's steps (its start and end frames) and the
`timing` entry the adapter records (what set the motion's duration; embodied_agent/motion_timing.py).
"""
from __future__ import annotations

import math

from v2.runtime.view import frame_number

SETTING = {'translation': 'translation at the requested speed', 'rotation': 'rotation at the requested speed',
           'minimum_duration': 'minimum duration', 'requested_duration': 'requested duration',
           'joint_speed': 'joint-space speed limit', 'requested_joint_speed': 'requested joint speed',
           'not_recorded': 'timing not recorded'}


def command_steps(action):
    start = frame_number(action.get('start_observation_id'))
    end = frame_number(action.get('end_observation_id'))
    return max(0, end - start) if start is not None and end is not None else 0


def describe_timing(method, timing, requested=None):
    """One phrase on what set a command's duration; `requested` holds the command's parameters, to
    tell a speed the command set from the default speed of a parameter it left out."""
    requested = requested or {}
    if not timing:
        return 'timing not recorded'
    if isinstance(timing, list):  # the waypoints of a trajectory that ran
        ramp = sum(w.get('ramp_s') or 0 for w in timing)
        total = sum(w.get('planned_s', 0) for w in timing) - ramp
        fastest_m = max(w.get('translation_m_s', 0) for w in timing)
        fastest_rad = max(w.get('rotation_rad_s', 0) for w in timing)
        asked = len(requested.get('waypoints') or []) or len(timing)
        return (f"ran {len(timing)} of {asked} waypoint(s), {total:.2f} s requested"
                + (f" + {ramp:.2f} s of speed ramps" if ramp else '') + f"; fastest {fastest_m:.2f} m/s "
                f"(limit {timing[0].get('limit_m_s'):g}) and {fastest_rad:.2f} rad/s "
                f"(limit {timing[0].get('limit_rad_s'):g})")
    set_by = timing.get('set_by')
    # A smoothed motion profile adds speed ramps to a move's duration (none by default).
    ramps = f" + {timing['ramp_s']:.2f} s of speed ramps" if timing.get('ramp_s') else ''
    if set_by == 'translation':
        source = 'requested' if 'speed_m_s' in requested else 'default'
        return (f"translation {timing['distance_m'] * 100:.1f} cm at the {source} {timing['requested_m_s']:g} m/s "
                f"(limit {timing['limit_m_s']:g})" + ramps)
    if set_by == 'rotation':
        source = 'requested' if 'speed_rad_s' in requested else 'default'
        return (f"rotation {math.degrees(timing['angle_rad']):.0f} deg at the {source} "
                f"{timing['requested_rad_s']:g} rad/s (limit {timing['limit_rad_s']:g})" + ramps)
    if set_by == 'requested_joint_speed':
        return (f"largest joint change {timing['largest_joint_change_rad']:.2f} rad at the requested joint speed "
                f"{timing['requested_joint_rad_s']:g} rad/s (limit {timing['limit_joint_rad_s']:g})")
    if set_by == 'joint_speed':
        return (f"largest joint change {timing['largest_joint_change_rad']:.2f} rad at the joint-space limit "
                f"{timing['limit_joint_rad_s']:g} rad/s")
    if set_by == 'minimum_duration':
        return f"minimum duration {timing['planned_s'] - (timing.get('ramp_s') or 0):g} s" + ramps
    return f"requested duration {timing.get('planned_s', 0):g} s"


def step_report(actions, top=8):
    """(text, data) summarizing the steps of the given history actions."""
    rows = []
    for action in actions:
        steps = command_steps(action)
        if steps <= 0:
            continue
        timing = action.get('timing')
        if isinstance(timing, list):
            set_by = 'requested_duration' if timing else 'not_recorded'
        else:
            set_by = (timing or {}).get('set_by') or 'not_recorded'
        rows.append({'sequence': action.get('sequence'), 'command_id': action.get('command_id'),
                     'method': action.get('method'), 'steps': steps, 'status': action.get('status'),
                     'set_by': set_by, 'detail': describe_timing(action.get('method'), timing,
                                                                  action.get('requested'))})
    by_method, by_setting = {}, {}
    for row in rows:
        for table, key in ((by_method, row['method']),
                           (by_setting, f"{row['method']}: {SETTING.get(row['set_by'], row['set_by'])}")):
            entry = table.setdefault(key, {'steps': 0, 'commands': 0})
            entry['steps'] += row['steps']
            entry['commands'] += 1
    total = sum(row['steps'] for row in rows)
    costliest = sorted(rows, key=lambda row: -row['steps'])[:max(0, int(top))]

    def ordered(table):
        return sorted(table.items(), key=lambda item: -item[1]['steps'])
    lines = [f'Physical steps so far: {total} in {len(rows)} motion commands, including any replayed ones.',
             'By method: ' + ', '.join(f"{key} {v['steps']} ({v['commands']})" for key, v in ordered(by_method)),
             'By what set the duration:']
    lines += [f"  {key}: {v['steps']} steps ({v['commands']})" for key, v in ordered(by_setting)]
    if costliest:
        lines.append(f'Costliest {len(costliest)} commands:')
        lines += [f"  seq {row['sequence']} {row['command_id']} {row['method']} {row['steps']} steps, {row['status']}: "
                  f"{row['detail']}" for row in costliest]
    data = {'total_steps': total, 'commands': len(rows), 'by_method': dict(ordered(by_method)),
            'by_setting': dict(ordered(by_setting)), 'costliest': costliest}
    return '\n'.join(lines), data
