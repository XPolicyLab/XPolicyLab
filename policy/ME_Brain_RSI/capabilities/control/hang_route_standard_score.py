"""Soft-rank a finite endpoint-plan beam by its complete fixed Cartesian carry route."""
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from coap.motion import ARMS, interpolate_pose, link_to_tcp, quat_matrix
from coap.tasks.hang_candidate_motion import cartesian_joint_path, fixed_duration_samples


def _quaternion(rotation):
    """Preserve the existing matrix-to-wxyz convention without changing a plan."""
    return np.roll(Rotation.from_matrix(np.asarray(rotation, float)).as_quat(), 1)


def fixed_carry_route(source, plan, durations, include_approach=False):
    """Describe the existing AL/D/R01 route through seating with unchanged timing."""
    arm = plan['arm']
    grasp = np.asarray(source['grasp'], float)
    pick = _quaternion(plan['pick'])
    final_matrix = np.asarray(plan['final'], float)
    final = _quaternion(final_matrix)
    center = np.asarray(plan['center'], float)
    axis = np.asarray(plan['axis'], float)
    hole_tool = np.asarray(plan['hole_tool'], float)
    stage = np.array([-.28 if arm == 'left' else .28, -.24, 1.025])
    prehang = center+axis*.090-final_matrix @ hole_tool
    thread = center+axis*.020-final_matrix @ hole_tool
    transfer = prehang+axis*.040
    route = []
    if include_approach:
        route.append(dict(label='approach', position=grasp+[0., 0., .100],
                          quaternion=pick, grip=1., duration=durations['approach']))
    route.extend([
        dict(label='descend', position=grasp, quaternion=pick, grip=1., duration=durations['descend']),
        dict(label='close', position=None, quaternion=None, grip=0., duration=durations['close']),
        dict(label='lift', position=grasp+[0., 0., .120], quaternion=pick, grip=0., duration=durations['lift']),
        dict(label='stage', position=stage, quaternion=pick, grip=0., duration=durations['stage']),
        dict(label='rotate', position=stage, quaternion=final, grip=0., duration=durations['rotate']),
        dict(label='transfer', position=transfer, quaternion=final, grip=0., duration=durations['transfer']),
        dict(label='prehang', position=prehang, quaternion=final, grip=0., duration=durations['prehang']),
        dict(label='thread', position=thread, quaternion=final, grip=0., duration=durations['thread']),
        dict(label='seat', position=np.asarray(plan['target'], float), quaternion=final, grip=0., duration=durations['seat']),
    ])
    return route


def _end(path):
    """Split a shared-arm path endpoint without modifying measured state."""
    return {arm: path[-1, index*6:(index+1)*6].copy() for index, arm in enumerate(ARMS)}


def score_route(ctx, source, plan, durations, include_approach=False):
    """Score every returned IK result continuously without rejecting a construction."""
    started = time.monotonic()
    arm = plan['arm']
    offset = ARMS.index(arm)*6
    current = {side: np.asarray(ctx.state[side+'_arm_joint_state'], float).copy() for side in ARMS}
    commands = [current[arm].copy()]
    waypoints = [current[arm].copy()]
    diagnostics, shape_position, shape_rotation, segments = [], [], [], []
    for item in fixed_carry_route(source, plan, durations, include_approach):
        start_pose = ctx.kinematics.forward(arm, current[arm])
        start_tcp = np.r_[link_to_tcp(start_pose), start_pose[3:]]
        if item['position'] is None:
            row = np.concatenate([current[side] for side in ARMS])
            path = np.stack([row, row])
            detail = []
            goal = start_tcp
        else:
            goal = np.r_[item['position'], item['quaternion']]
            path, detail = cartesian_joint_path(ctx.kinematics, current, {arm: goal}, tcp=True)
        samples = fixed_duration_samples(path, item['duration'])
        active = samples[:, offset:offset+6]
        delta = np.diff(np.vstack([current[arm], active]), axis=0)
        local_position, local_rotation = [], []
        for index, joints in enumerate(active):
            fraction = (index+1)/len(active)
            smooth = fraction*fraction*(3.-2.*fraction)
            ideal = interpolate_pose(start_tcp, goal, smooth)
            actual = ctx.kinematics.forward(arm, joints)
            position_error = float(np.linalg.norm(link_to_tcp(actual)-ideal[:3]))
            rotation_error = float(np.linalg.norm(Rotation.from_matrix(
                quat_matrix(ideal[3:]).T @ quat_matrix(actual[3:])).as_rotvec()))
            local_position.append(position_error)
            local_rotation.append(rotation_error)
        commands.extend(active)
        waypoints.extend(path[1:, offset:offset+6])
        diagnostics.extend(detail)
        shape_position.extend(local_position)
        shape_rotation.extend(local_rotation)
        segments.append(dict(segment=item['label'], frames=len(active), waypoint_count=len(path),
            maximum_command_step_rad=float(np.abs(delta).max()),
            maximum_position_residual_m=max((float(row['position_error_m']) for row in detail), default=0.),
            maximum_rotation_residual_rad=max((float(row['rotation_error_rad']) for row in detail), default=0.),
            maximum_nominal_TCP_deviation_m=max(local_position),
            maximum_nominal_rotation_deviation_rad=max(local_rotation)))
        current = _end(path)
    commands = np.asarray(commands)
    waypoints = np.asarray(waypoints)
    maximum_step = float(np.abs(np.diff(commands, axis=0)).max())
    waypoint_step = float(np.abs(np.diff(waypoints, axis=0)).max())
    travel = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    position_error = max(float(row['position_error_m']) for row in diagnostics)
    rotation_error = max(float(row['rotation_error_rad']) for row in diagnostics)
    fit_cost = 6000.*position_error+10.*rotation_error
    jump_cost = 8.*maximum_step**2
    travel_cost = .03*travel
    shape_cost = 20.*max(shape_position)+.1*float(np.mean(shape_rotation))
    wrist_cost = .015*float(np.mean(1./(.10+np.abs(np.cos(waypoints[:, 4])))))
    cost = fit_cost+jump_cost+travel_cost+shape_cost+wrist_cost
    review = dict(cost=float(cost), fit_cost=float(fit_cost), jump_cost=float(jump_cost),
        travel_cost=float(travel_cost), shape_cost=float(shape_cost), wrist_cost=float(wrist_cost),
        maximum_command_step_rad=maximum_step, maximum_waypoint_step_rad=waypoint_step,
        maximum_position_residual_m=position_error, maximum_rotation_residual_rad=rotation_error,
        total_joint_travel_rad=travel, maximum_nominal_TCP_deviation_m=max(shape_position),
        mean_nominal_rotation_deviation_rad=float(np.mean(shape_rotation)),
        simulated_frames=len(commands)-1, segments=segments,
        planning_seconds=time.monotonic()-started, include_approach=bool(include_approach))
    return review


def rank_endpoint_plans(ctx, source, plans, durations, beam_size=2, include_approach=False, beam_mode='endpoint'):
    """Return a supplied plan by soft route cost, without changing any action or input."""
    started = time.monotonic()
    indexed = sorted(enumerate(plans), key=lambda pair: (float(pair[1]['cost']), pair[0]))
    size = min(4, max(1, int(beam_size)))
    beam = indexed[:size]
    if beam_mode == 'diverse':
        beam, used_pegs = [], set()
        for index, plan in indexed:
            if int(plan['peg']) not in used_pegs:
                beam.append((index, plan))
                used_pegs.add(int(plan['peg']))
        beam = beam[:size]
        used_indices = {index for index, _ in beam}
        beam.extend(pair for pair in indexed if pair[0] not in used_indices)
        beam = beam[:size]
    options = []
    for endpoint_rank, (index, plan) in enumerate(beam):
        review = score_route(ctx, source, plan, durations, include_approach)
        options.append(dict(plan_index=index, endpoint_rank=endpoint_rank,
            endpoint_cost=float(plan['cost']), arm=plan['arm'], peg=int(plan['peg']), **review))
    selected = min(options, key=lambda option: (option['cost'], option['endpoint_cost'], option['plan_index']))
    report = dict(selected_plan_index=selected['plan_index'], original_endpoint_best_index=indexed[0][0],
        supplied_plan_count=len(plans), evaluated_plan_count=len(options), maximum_beam_size=4,
        beam_size=size, beam_mode=beam_mode, options=options,
        planning_seconds=time.monotonic()-started, runtime_route_rejection=False,
        runtime_abort_enabled=False, action_durations_modified=False,
        phase='before_approach_nominal_cartesian_prefix' if include_approach else 'after_approach_before_descend',
        limitation='Public robot IK only. No self/scene collision, cup dynamics, grasp retention, aperture truth or actual tracking guarantee. AG hover branch and future wrist refresh must already be reflected by the caller state/plans when used after approach.')
    return plans[selected['plan_index']], report
