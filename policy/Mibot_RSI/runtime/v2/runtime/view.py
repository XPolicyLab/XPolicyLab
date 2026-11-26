"""Compact model view of robot reports and observations.

The recording keeps every original report; the model sees one line per action
and one state line: millimetres and degrees with one decimal, no quaternions,
joint angles, link poses or capture details.
"""
from __future__ import annotations

import math
import re

MAX_OUTPUT_CHARS = 40000  # about 10k tokens per tool output
CONTROL_HZ = 25.0
MOTION_METHODS = ('move', 'gripper', 'move_to_mark', 'trajectory', 'hold', 'finish')
TERMINAL_REASONS = ('benchmark_terminal', 'episode_already_ended')


def fixed(value, digits=1):
    """Fixed-point text without a negative zero ('-0.0' becomes '0.0')."""
    text = f'{float(value):.{digits}f}'
    return text[1:] if text.startswith('-') and float(text) == 0 else text


def mm(value):
    return fixed(float(value) * 1000)


def vec_mm(values):
    return '(' + ','.join(mm(v) for v in values) + ')'


def signed_vec_mm(values):
    def signed(value):
        text = f'{float(value) * 1000:+.1f}'
        return '+0.0' if float(text) == 0 else text
    return '(' + ','.join(signed(v) for v in values) + ')'


def rpy_deg(quaternion_xyzw):
    """Roll, pitch, yaw (degrees, ZYX convention) of an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def frame_number(observation_id):
    match = re.fullmatch(r'frame_(\d+)', str(observation_id or ''))
    return int(match.group(1)) if match else None


def arm_state(label, state):
    tcp = state.get('tcp') or {}
    parts = [label]
    if tcp.get('position_m') is not None:
        parts.append(vec_mm(tcp['position_m']) + 'mm')
    if tcp.get('quaternion_xyzw') is not None:
        parts.append('rpy(' + ','.join(fixed(a) for a in rpy_deg(tcp['quaternion_xyzw'])) + ')°')
    if state.get('gripper_width_m') is not None:
        parts.append('grip ' + mm(state['gripper_width_m']))
    return ' '.join(parts)


def state_line(robot_state, step=None, step_limit=None, return_estimate=None, frame_id=None, eval_budget=None):
    """`L (x,y,z)mm rpy(r,p,y)° grip w | R ... | step n[/limit (return≈k)] | frame_...`.

    An evaluation shows its step limit: `step n/limit (return≈k)`. A development episode has no limit and
    shows the evaluation budget for reference instead: `step n (eval budget N; return≈k)`.
    """
    parts = [arm_state(name[0].upper(), robot_state[name])
             for name in ('left', 'right') if isinstance((robot_state or {}).get(name), dict)]
    if step is not None:
        budget = f'step {step}'
        if step_limit:
            budget += f'/{step_limit}'
            if return_estimate is not None:
                budget += f' (return≈{return_estimate})'
        elif eval_budget:
            budget += f' (eval budget {eval_budget}' + (f'; return≈{return_estimate})' if return_estimate is not None
                                                         else ')')
        parts.append(budget)
    if frame_id:
        parts.append(str(frame_id))
    return ' | '.join(parts)


def observation_text(meta, step_limit=None, eval_budget=None):
    """One line for an observation's metadata (images are sent separately). eval_budget: a development episode's
    evaluation step budget, shown for reference."""
    estimate = meta.get('return_to_initial_steps_estimate') or {}
    estimate = estimate.get('together', estimate.get('sequential_total'))  # robot.move_to_mark('initial')
    line = state_line(meta.get('robot_state') or {}, meta.get('control_step'),
                      step_limit or meta.get('step_limit'), estimate, meta.get('observation_id'), eval_budget)
    if meta.get('episode_ended'):
        line += ' | EPISODE ENDED'
    return line


def _describe_request(method, params):
    arm = params.get('arm')
    label = 'L+R' if arm == 'both' else arm[0].upper() if isinstance(arm, str) and arm else ''
    if method == 'move':
        bits = ['move', label]
        if params.get('delta_m') is not None:
            bits.append('Δ' + signed_vec_mm(params['delta_m']) + 'mm')
        if params.get('position_m') is not None:
            bits.append('to ' + vec_mm(params['position_m']) + 'mm')
        if params.get('quaternion_xyzw') is not None:
            bits.append('rpy(' + ','.join(fixed(a) for a in rpy_deg(params['quaternion_xyzw'])) + ')°')
        if params.get('rotation_axis') and params.get('angle_deg'):
            bits.append(f"rot {params['rotation_axis']} {float(params['angle_deg']):+.1f}°")
        bits.append(params.get('frame', 'world'))
        if params.get('speed_m_s') is not None:
            bits.append(f"{float(params['speed_m_s']) * 1000:.0f}mm/s")
        if params.get('gripper_width_m') is not None:
            try:
                bits.append(f"grip→{mm(params['gripper_width_m'])}mm")
            except (TypeError, ValueError):
                bits.append('grip→?')
        return ' '.join(b for b in bits if b)
    if method == 'gripper':
        return f"gripper {label} → {mm(params.get('width_m', 0))}mm"
    if method == 'move_to_mark':
        return f"move_to_mark {label} {params.get('name', 'initial')}"
    if method == 'hold':
        try:
            return f"hold {float(params.get('duration_s', 1.0)):g}s"
        except (TypeError, ValueError):
            return 'hold ?s'
    if method == 'trajectory':
        waypoints = params.get('waypoints') or []
        count = len(waypoints)
        text = f'trajectory {count} waypoint' + ('s' if count != 1 else '')
        grips = waypoint_widths(waypoints)
        if grips:
            text += ' ' + ' '.join(f"{arm[0].upper()} grip→{mm(width)}mm" for arm, width in grips.items())
        return text
    return method


def waypoint_widths(waypoints):
    """{arm: the last gripper_width_m the waypoints give that arm}, left before right."""
    widths = {}
    for waypoint in waypoints if isinstance(waypoints, list) else []:
        for arm in ('left', 'right'):
            entry = waypoint.get(arm) if isinstance(waypoint, dict) else None
            if isinstance(entry, dict) and entry.get('gripper_width_m') is not None:
                try:
                    widths[arm] = float(entry['gripper_width_m'])
                except (TypeError, ValueError):
                    pass
    return {arm: widths[arm] for arm in ('left', 'right') if arm in widths}


def gripper_ends(reports, labels=True):
    """`L grip 30.4mm` per arm whose width target the motion (any of its waypoints) set, measured when the whole
    motion ended (the fingers may still be moving), or `L grip unchanged` when no width target took effect (the
    motion stopped before the first step that would have set it). labels=False leaves out the arm letter (a move
    names its arm already)."""
    reports = [report or {} for report in reports]
    applied = {}
    for report in reports:
        for arm, value in (report.get('gripper') or {}).items():
            applied[arm] = applied.get(arm, False) or bool((value or {}).get('applied'))
    final = (reports[-1].get('measured_final_state') or {}) if reports else {}
    parts = []
    for arm in sorted(applied):
        label = f'{arm[0].upper()} ' if labels else ''
        width = (final.get(arm) or {}).get('gripper_width_m')
        if width is None:  # no final state recorded: the width the last report that set it measured
            width = next((r['gripper'][arm].get('width_m') for r in reversed(reports)
                          if (r.get('gripper') or {}).get(arm, {}).get('applied')), None)
        if not applied[arm]:
            parts.append(f'{label}grip unchanged')
        elif width is not None:
            parts.append(f'{label}grip {mm(width)}mm')
    return parts


def lag_text(lag):
    """`R joint3 at -0.034 rad, commanded -0.084 rad`: per arm, the joint furthest behind its command at a tracking_error."""
    return '; '.join(f"{arm[0].upper()} {value.get('joint')} at {value.get('measured_rad', 0):+.3f} rad, "
                     f"commanded {value.get('commanded_rad', 0):+.3f} rad" for arm, value in sorted(lag.items()))


def speed_excess_text(excess, limits, waypoint=True):
    """`waypoint 1 R 2.60 rad/s > 2 rad/s, needs duration_s ≥1.3` for each waypoint arm that a trajectory would move
    faster than the speed limits (the adapter's speed_excess entries)."""
    limits = limits or {}
    parts = []
    for item in excess or []:
        duration = float(item.get('duration_s') or 0)
        over = []
        for key, unit, amount in (('speed_m_s', 'm/s', 'distance_m'), ('speed_rad_s', 'rad/s', 'angle_rad')):
            speed = item.get(key)
            if speed is None and duration > 0 and item.get(amount) is not None:
                speed = item[amount] / duration
            limit = limits.get(key)
            if speed is None or limit is None:
                continue
            if key in (item.get('over') or ()) or speed > limit + 1e-8:
                digits = 2  # more digits until the shown speed is above the limit
                while digits < 8 and round(speed, digits) <= limit:
                    digits += 1
                shown = f'{speed:.{digits}f}'
                over.append(f'{shown} {unit} > {limit:g} {unit}')
        label = f"{str(item.get('arm') or '?')[0].upper()} " + (' and '.join(over) + ', ' if over else '')
        parts.append((f"waypoint {item.get('waypoint')} " if waypoint else '') + label
                     + f"needs duration_s ≥{float(item.get('required_duration_s') or 0):g}")
    return '; '.join(parts)


def rejection_detail(result):
    """` (waypoint 1 R 2.60 rad/s > 2 rad/s, needs duration_s ≥1.3)` for a trajectory refused before it started."""
    parts = []
    if result.get('termination_reason') == 'duration_exceeds_timeout' and result.get('waypoint'):
        parts.append(f"waypoint {result['waypoint']} lasts {float(result.get('required_duration_s') or 0):g} s > 20 s")
    if result.get('speed_excess'):
        parts.append(speed_excess_text(result['speed_excess'], result.get('limits')))
    return f" ({'; '.join(parts)})" if parts else ''


def tolerance_text(report):
    """`L err 4.1mm 3.6°` for each arm that ended outside the 3 mm / 0.06 rad tolerance (all arms if none did)."""
    errors = report.get('tcp_error') or {}
    outside = {arm: e for arm, e in errors.items()
               if (e.get('position_m') or 0) > .003 or (e.get('orientation_rad') or 0) > .06} or errors
    return ', '.join(f"{arm[0].upper()} err {mm(e.get('position_m') or 0)}mm "
                     f"{math.degrees(e.get('orientation_rad') or 0):.1f}°" for arm, e in sorted(outside.items()))


def action_line(index, method, params, result):
    """One line per executed command, e.g.
    `#23 c1a2b3c4 move L Δ(+0.0,+15.0,+0.0)mm world 25mm/s → reached err 0.1mm 15steps 0.6s (5s wall)`,
    where c1a2b3c4 is the command ID (for robot.status, robot.inspect and references)."""
    params = params or {}
    result = result or {}
    status = result.get('status', 'unknown')
    reason = result.get('termination_reason')
    command = f" {result['command_id']}" if result.get('command_id') else ''
    text = f'#{index}{command} {_describe_request(method, params)} → {status}'
    if reason and status != 'reached' and reason != 'trajectory_completed':
        text += f' {reason}'
        fraction = result.get('path_fraction')
        if isinstance(fraction, (int, float)) and status == 'partial' and fraction < 0.999:
            text += f' at {fraction * 100:.0f}%'
        if reason == 'tracking_error' and result.get('tracking_lag'):
            text += f" ({lag_text(result['tracking_lag'])})"
        if status == 'rejected':
            text += rejection_detail(result)
    arm = params.get('arm')
    final = ((result.get('measured_final_state') or {}).get(arm) or {}) if arm else {}
    arm_error = ((result.get('tcp_error') or {}).get(arm) or {}) if arm else {}
    if method == 'move' and arm:
        error = arm_error.get('position_m')
        if status == 'reached' and error is not None:
            text += f' err {mm(error)}mm'
        elif (final.get('tcp') or {}).get('position_m') is not None:
            text += f", stopped {arm[0].upper()}{vec_mm(final['tcp']['position_m'])}mm"
    if (method in ('move', 'move_to_mark') and reason == 'final_tolerance_not_met'
            and arm_error.get('position_m') is not None):
        # The motion ran to its end but settled just outside the 3 mm / 0.06 rad tolerance: say how far.
        text += f" err {mm(arm_error['position_m'])}mm {math.degrees(arm_error.get('orientation_rad') or 0):.1f}°"
    if method == 'hold' and reason == 'final_tolerance_not_met' and result.get('tcp_error'):
        text += ' ' + tolerance_text(result)  # an arm was pushed off its held pose
    if method == 'gripper' and final.get('gripper_width_m') is not None:
        text += f" width {mm(final['gripper_width_m'])}mm"
    if method == 'move' and result.get('gripper'):
        text += ''.join(', ' + part for part in gripper_ends([result], labels=False))
    if method == 'trajectory' and result.get('waypoint_reports'):
        reports = result['waypoint_reports']
        done = sum(1 for report in reports if report.get('status') == 'reached')
        last = reports[-1]
        stop = ''
        why = last.get('termination_reason')
        if last.get('status') != 'reached' and why:
            # Where and why the trajectory stopped, as a single move's line says it.
            stop = f'; waypoint {len(reports)} stopped by {why}'
            fraction = last.get('path_fraction')
            if isinstance(fraction, (int, float)) and last.get('status') == 'partial' and fraction < 0.999:
                stop += f' at {fraction * 100:.0f}%'
            if why == 'tracking_error' and last.get('tracking_lag'):
                stop += f": {lag_text(last['tracking_lag'])}"
            elif why == 'final_tolerance_not_met' and last.get('tcp_error'):
                stop += f': {tolerance_text(last)}'
            elif why == 'exceeds_speed_limit' and last.get('speed_excess'):
                stop += f": {speed_excess_text(last['speed_excess'], last.get('limits'), waypoint=False)}"
        text += f' ({done}/{len(reports)} waypoints reached{stop}' + ''.join('; ' + part for part in gripper_ends(reports)) + ')'
    start, end = frame_number(result.get('start_observation_id')), frame_number(result.get('end_observation_id'))
    if start is not None and end is not None and end >= start and method != 'finish':
        text += f' {end - start}steps {(end - start) / CONTROL_HZ:.1f}s'
        if isinstance(result.get('wall_s'), (int, float)):
            text += f" ({result['wall_s']:.0f}s wall)"
    motion = result.get('uncommanded_tcp_motion') or {}
    if motion.get('attention'):
        peaks = []
        for other, value in (motion.get('by_arm') or {}).items():
            peaks.append(f"{other[0].upper()} {mm(value.get('position_m', 0))}mm "
                         f"{math.degrees(value.get('rotation_rad', 0)):.1f}° at "
                         f"{value.get('position_peak_observation_id')}")
        text += '\n  ! uncommanded motion of a held arm: ' + '; '.join(peaks)
    return text


def plan_line(params, result):
    """One line for a dry run (dry_run=True), e.g.
    `dry run move L to (-70.0,15.0,840.0)mm world 350mm/s → feasible 29steps 1.2s (nothing moved)` or
    `dry run move L … → would_stop joint_jump_or_speed_limit:left at 4% after 1steps (nothing moved)`."""
    params = dict(params or {})
    motion = params.pop('motion', 'move')
    result = result or {}
    status = result.get('status', 'unknown')
    text = f'dry run {_describe_request(motion, params)} → {status}'
    reports = result.get('waypoint_reports') or []
    if motion == 'trajectory' and reports:
        feasible = sum(1 for report in reports if report.get('status') == 'feasible')
        text += f" ({feasible}/{len(params.get('waypoints') or reports)} waypoints feasible"
        last = reports[-1]
        if last.get('status') == 'would_stop':
            text += (f"; waypoint {len(reports)} would stop: {last.get('termination_reason')} "
                     f"at {float(last.get('path_fraction') or 0) * 100:.0f}%")
        text += ')'
    elif status == 'would_stop':
        text += f" {result.get('termination_reason')} at {float(result.get('path_fraction') or 0) * 100:.0f}%"
    elif result.get('termination_reason') and status != 'feasible':
        text += f" {result.get('termination_reason')}" + rejection_detail(result)
    steps = result.get('planned_steps')
    if isinstance(steps, int) and status in ('feasible', 'would_stop'):
        text += (f' {steps}steps {steps / CONTROL_HZ:.1f}s' if status == 'feasible' else f' after {steps}steps')
    return text + ' (nothing moved)'


def ended_by(result):
    """True when a report shows that the environment has ended the episode."""
    result = result or {}
    if result.get('termination_reason') in TERMINAL_REASONS:
        return True
    return any(report.get('termination_reason') in TERMINAL_REASONS
               for report in result.get('waypoint_reports') or [])


def truncate(text, limit=MAX_OUTPUT_CHARS, saved_path=None):
    """Keep head and tail of an over-long output, pointing to the full copy."""
    text = str(text)
    if len(text) <= limit:
        return text
    keep = (limit - 200) // 2
    note = f'\n…[{len(text) - 2 * keep} characters omitted'
    note += f'; full output saved at {saved_path}]…\n' if saved_path else ']…\n'
    return text[:keep] + note + text[-keep:]
