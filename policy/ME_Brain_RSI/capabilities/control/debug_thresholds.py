"""Task-local execution without threshold-based interruption for simulator debugging."""
import math
import os

import numpy as np

TASKS = frozenset(('fill_egg_holder', 'fill_pen_holder'))


def enable(ctx, task_name=None):
    """Enable only the two explicitly requested ordinary tasks."""
    current = getattr(ctx, 'task_name', '')
    name = current or task_name
    if name not in TASKS or (task_name is not None and name != task_name):
        raise ValueError('Threshold bypass is restricted to the two ordinary tasks')
    if os.environ.get('EVAL_ENV_TYPE') != 'sim':
        raise RuntimeError('Threshold bypass requires the RoboDojo simulator environment')
    ctx.task_name = name
    ctx.debug_threshold_bypass = True
    ctx.memory['threshold_debug_execution'] = {
        'enabled': True, 'task': ctx.task_name, 'threshold_interruptions_enabled': False,
        'collision_preflight_enabled': False, 'simulator_only': True, 'physical_success_verified': False,
        'official_episode_limit_unchanged': True, 'bypassed_checks': {}}


def disabled(ctx):
    return (getattr(ctx, 'task_name', '') in TASKS and
            getattr(ctx, 'debug_threshold_bypass', False) is True and
            os.environ.get('EVAL_ENV_TYPE') == 'sim')


def audit_bypass(ctx, name):
    """Record skipped checks without claiming that geometry or arrival passed."""
    if not disabled(ctx):
        raise ValueError('Threshold bypass is not enabled for this context')
    counts = ctx.memory['threshold_debug_execution']['bypassed_checks']
    counts[name] = counts.get(name, 0) + 1
    return {'accepted': True, 'audit_performed': False, 'thresholds_bypassed': True,
            'physical_success_verified': False, 'frames': 0, 'includes_required_home': True,
            'minimum_clearances_m': {}}


def execute_path(ctx, path, arms, duration, profile, goals=None, ik_diagnostics=None):
    """Emit a finite timed path; measured errors are recorded without waiting or aborting."""
    from .motion import ARMS, _joint_action, _cartesian_errors, time_parameterize_joint_path
    if not disabled(ctx):
        raise ValueError('Threshold bypass is not enabled for this context')
    samples = time_parameterize_joint_path(path, duration,
                                           velocity_limit=profile.joint_velocity_limit)
    ctx.report_motion({'mode': 'threshold_debug_joint', 'phase': 'command_path_start',
        'frames': len(samples), 'threshold_interruptions_enabled': False,
        'collision_preflight_performed': False, 'motion_profile': profile.as_dict(),
        'nominal_frame_goals': {arm: np.asarray(goal).tolist() for arm, goal in (goals or {}).items()},
        'ik_diagnostics': ik_diagnostics or []})
    for point in samples:
        commands = {arm: point[index*6:(index+1)*6].copy() for index, arm in enumerate(ARMS)}
        yield _joint_action(commands, ctx.gripper_targets)
    measured = ctx.state
    endpoint = np.asarray(path[-1])
    joint_error = max((float(np.max(np.abs(np.asarray(measured[arm+'_arm_joint_state']) -
        endpoint[index*6:(index+1)*6]))) for index, arm in enumerate(ARMS) if arm in arms), default=0.)
    errors = _cartesian_errors(measured, goals, False) if goals else {}
    ctx.report_motion({'phase': 'command_path_complete', 'issued_frames': len(samples),
        'measured_pose_errors': errors, 'joint_tracking_error_rad': joint_error,
        'errors_are_diagnostic_only': True, 'physical_arrival_verified': False})


def numeric_move(ctx, targets, duration, profile):
    """Keep Cartesian waypoint interpolation and use finite approximate IK without a gate."""
    from .motion import ARMS, interpolate_pose
    starts = {arm: np.asarray(ctx.joint_targets[arm], float).copy() for arm in ARMS}
    references = {arm: value.copy() for arm, value in starts.items()}
    poses = {arm: ctx.kinematics.forward(arm, starts[arm]) for arm in targets}
    goals = {arm: np.asarray(target['pose'], float).copy() for arm, target in targets.items()}
    for arm, goal in goals.items():
        if goal.shape != (7,) or not np.isfinite(goal).all():
            raise ValueError('A finite Cartesian target is required for debugging execution')
    count = 1
    for arm, goal in goals.items():
        distance = float(np.linalg.norm(goal[:3]-poses[arm][:3]))
        angle = 2*math.acos(float(np.clip(abs(np.dot(goal[3:], poses[arm][3:])), 0., 1.)))
        count = max(count, min(12, max(2, math.ceil(distance/.04), math.ceil(angle/.3))))
    path = [np.concatenate([starts[arm] for arm in ARMS])]
    diagnostics = []
    for index in range(1, count+1):
        for arm, goal in goals.items():
            waypoint = interpolate_pose(poses[arm], goal, index/count)
            result = ctx.kinematics.solve(arm, waypoint, current_joints=references[arm],
                max_starts=4, max_nfev=80, joint_margin=0.)
            joints = np.asarray(result['joint_value'], float)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError('IK did not return a finite command vector')
            references[arm] = joints.copy()
            diagnostics.append({'arm': arm, 'waypoint': index, 'status': result['status'],
                'position_error_m': result['position_error'],
                'rotation_error_rad': result['rotation_error'],
                'residual_gate_applied': False})
        path.append(np.concatenate([references[arm] for arm in ARMS]))
    audit_bypass(ctx, 'motion_arrival_tracking_ik_and_collision')
    yield from execute_path(ctx, np.stack(path), tuple(targets), duration, profile,
                            goals=goals, ik_diagnostics=diagnostics)
