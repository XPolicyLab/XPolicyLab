"""Place a single-handle plier head first, then lower the handle about its support."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from coap.tasks.toolbox_support import measure, _lift
from coap.tasks.toolbox_e import BALANCED_SHIFT
from coap.tasks.toolbox_f import ToolboxCurveContext
FIXED_TILT_DEGREES = 65.0
SINGLE_HANDLE_LOCAL = np.array([0.035, 0.0, 0.024])
FINGER_BELOW_TCP_M = 0.013565
RELEASE_ABOVE_BOX_TOP_M = 0.01

@lru_cache(maxsize=1)
def public_geometry():
    """Load the retained public tool and box meshes, never episode object state."""
    folder = Path(__file__).parent
    pliers = np.load(folder / 'toolbox_public.npz')['pliers_hull'].astype(float)
    box = np.load(folder.parents[1] / 'vendor/physicalrsi_open/toolbox/box_public.npz')['points'].astype(float)
    tip = pliers[pliers[:, 0] < pliers[:, 0].min() + 0.004].mean(axis=0)
    return (pliers, box, tip)

def inclined_frame(ctx, item, arm):
    """Use the retained contacts except for the observed AM single-handle pinch."""
    source = np.asarray(item['source'])
    local = SINGLE_HANDLE_LOCAL if item['label'] == 'pliers' else [BALANCED_SHIFT[item['label']], 0.0, 0.0]
    grasp = source + np.asarray(item['source_rotation']) @ local
    grasp[2] = ctx.table_z + {'wrench': 0.012, 'hammer': 0.015, 'pliers': 0.015, 'tape_measure': 0.02}[item['label']]
    target = np.asarray(item['destination']) + (grasp - source) + [0.0, 0.0, 0.018]
    angle = math.radians(FIXED_TILT_DEGREES)
    forward, across = (np.array([0.0, math.cos(angle), -math.sin(angle)]), np.array([-1.0, 0.0, 0.0]))
    matrix = np.column_stack([forward, across, np.cross(forward, across)])
    quat = np.roll(Rotation.from_matrix(matrix).as_quat(), 1)
    return (grasp, target, quat)

def head_supported_route(ctx, item, grasp):
    """Construct a fixed head-contact arc from public geometry and the RGB box pose.

    The motion assumes gravity can swing the pinched handle about world X. The
    metal tip receives support before the hand lowers; no fitted attachment is
    used and no observation score interrupts execution.
    """
    pliers, box_points, tip = public_geometry()
    box = ctx.memory['toolbox_geometry']['box']
    box_rotation, box_translation = (np.asarray(box['box_rotation']), np.asarray(box['box_translation']))
    source_rotation = np.asarray(item['source_rotation'])
    box_yaw = math.atan2(box_rotation[1, 0], box_rotation[0, 0]) + math.pi / 2
    desired = Rotation.from_euler('z', box_yaw - float(item['source_yaw'])).as_matrix() @ source_rotation
    head = np.asarray(item['destination']) + desired @ tip
    box_local = box_rotation.T @ (head - box_translation)
    distances = np.linalg.norm(box_points[:, :2] - box_local[:2], axis=1)
    nearest = box_points[distances < 0.004]
    if not len(nearest):
        nearest = box_points[np.argsort(distances)[:24]]
    local_support_z = float(nearest[:, 2].max())
    head[2] = box_translation[2] + local_support_z + 0.001
    grasp_local = source_rotation.T @ (np.asarray(grasp) - np.asarray(item['source']))
    lever = desired @ (grasp_local - tip)
    radius = float(np.linalg.norm(lever[1:]))
    box_top = float(box_translation[2] + box_points[:, 2].max())
    release_z = box_top + FINGER_BELOW_TCP_M + RELEASE_ABOVE_BOX_TOP_M
    final_angle = math.acos(float(np.clip((release_z - head[2]) / radius, 0.05, 0.99)))
    carry = head + [lever[0], 0.0, radius + 0.075]
    support = head + [lever[0], 0.0, radius]
    angles = np.linspace(0.0, final_angle, 7)
    original_angle = math.atan2(lever[2], lever[1])
    rolling_rotations = [Rotation.from_euler('x', math.atan2(radius * math.cos(angle), -radius * math.sin(angle)) - original_angle).as_matrix() @ desired for angle in angles]
    metal_head = pliers[pliers[:, 0] < -0.035] - tip
    lowest = [float(np.einsum('ni,ji->nj', metal_head, rotation)[:, 2].min()) for rotation in rolling_rotations]
    rolling_lifts = [lowest[0] - value for value in lowest]
    pivots = [head + [0.0, 0.0, lift] for lift in rolling_lifts]
    arc = [pivot + [lever[0], -radius * math.sin(angle), radius * math.cos(angle)] for pivot, angle in zip(pivots[1:], angles[1:])]
    return dict(carry=carry, support=support, arc=arc, head_world=head, native_head_tip=tip, grasp_local=grasp_local, desired_rotation=desired, lever_world=lever, support_box_local_z=local_support_z, global_box_top_z=box_top, radius_yz=radius, final_arc_degrees=math.degrees(final_angle), support_pivots=pivots, rolling_rotations=rolling_rotations, native_head_rolling_lifts_m=rolling_lifts, contact_hypothesis='gravity swing followed by metal-tip support and handle lowering')

def run(ctx):
    """Retain the other three tool motions and compare a supported plier placement."""
    from coap.tasks.open_motion import enable
    ctx.stage = 'toolbox_measure'
    moves = measure(ctx)
    enable(ctx)
    ctx.__class__ = ToolboxCurveContext
    ctx.memory['toolbox_strategy'] = dict(candidate='AZ', planned_tools=len(moves), safety_checks_enabled=False, execution='fixed_duration_cartesian_numeric_ik', balanced_local_x_m=BALANCED_SHIFT, fixed_tilt_degrees=FIXED_TILT_DEGREES, pliers_single_handle_local=SINGLE_HANDLE_LOCAL.tolist(), pliers_approach_opening=0.5, pliers_placement='head_support_then_handle_arc', postcarry_pose_estimation=False)
    priority = {'tape_measure': 0, 'wrench': 1, 'pliers': 2, 'hammer': 3}
    moves.sort(key=lambda item: priority[item['label']])
    ctx.memory.setdefault('toolbox_strategy', {})['r02_parent_candidate'] = ctx.memory.get('toolbox_strategy', {}).get('candidate')
    ctx.memory['toolbox_strategy']['candidate'] = R02_CANDIDATE
    for item in moves:
        name = item['label']
        pliers = name == 'pliers'
        opening = 0.5 if pliers else 1.0
        arm = 'left' if item['source'][0] < 0 else 'right'
        grasp, target, quat = inclined_frame(ctx, item, arm)
        if name == 'pliers':
            quat = _r02_pick(ctx, item, arm, grasp, quat)
        ctx.stage = 'toolbox_approach_' + name
        yield from ctx.move(arm, _lift(grasp, 0.085), quat, opening, 18)
        yield from ctx.move(arm, grasp, quat, opening, 18)
        ctx.stage = 'toolbox_close_' + name
        yield from ctx.grip(arm, 0.0, duration=26)
        yield from ctx.hold(3)
        yield from ctx.move(arm, _lift(grasp, 0.185 if pliers else 0.1), quat, 0.0, 26)
        ctx.stage = 'toolbox_carry_' + name
        if pliers:
            route = _r02_route(ctx, head_supported_route(ctx, item, grasp))
            ctx.memory['pliers_head_supported_route'] = {key: value.tolist() if isinstance(value, np.ndarray) else [point.tolist() for point in value] if key in ('arc', 'support_pivots', 'rolling_rotations') else value for key, value in route.items()}
            yield from ctx.move(arm, route['carry'], quat, 0.0, 36)
            ctx.stage = 'toolbox_head_settle_pliers'
            yield from ctx.hold(6)
            ctx.stage = 'toolbox_head_support_pliers'
            yield from ctx.move(arm, route['support'], quat, 0.0, 20)
            yield from ctx.hold(3)
            ctx.stage = 'toolbox_head_lower_handle_pliers'
            for point in route['arc']:
                yield from ctx.move(arm, point, quat, 0.0, 6)
            target = route['arc'][-1]
        else:
            yield from ctx.move(arm, _lift(target, 0.02), quat, 0.0, 30)
            yield from ctx.move(arm, target, quat, 0.0, 20)
        ctx.stage = 'toolbox_release_' + name
        yield from ctx.grip(arm, 1.0, duration=12)
        yield from ctx.move(arm, _lift(target, 0.055), quat, 1.0, 18)
        yield from ctx.home(arm, duration=20)
    ctx.stage = 'done'
    yield from ctx.home(duration=28)
    yield from ctx.hold(10)

R02_CANDIDATE = 'candidate-20261022-r02-toolbox-b'
R02_PARENT_ENTRY = 'coap.tasks.toolbox:run'
R02_MODE = 'lower_transfer'
R02_TIMING = {'approach': 18, 'approach_lift': 0.085, 'descend': 18, 'lift': 26, 'source_lift': 0.185, 'retreat': 30, 'retreat_lift': 0.02, 'carry': 36, 'support': 20, 'arc': 6}

import time as _r02_time
from coap.motion import ARMS as _R02_ARMS, quat_matrix as _r02_matrix
from coap.tasks.toolbox_f import (cartesian_joint_path as _r02_path,
                                 fixed_duration_samples as _r02_samples)


def _r02_quaternion(matrix):
    """Return a wxyz quaternion for a public geometric frame."""
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def _r02_route(ctx, route):
    """Vary only the fixed high-transfer height; keep support and arc endpoints."""
    if R02_MODE != 'lower_transfer':
        return route
    route = dict(route)
    previous = np.asarray(route['carry']).copy()
    route['carry'] = np.asarray(route['support']) + [0., 0., .050]
    ctx.memory.setdefault('toolbox_r02', {})['transfer'] = dict(
        previous_carry=previous.tolist(), carry=route['carry'].tolist(),
        vertical_clearance_above_support_m=.050, support_and_arc_unchanged=True)
    return route


def _r02_preview(ctx, arm, grasp, quat, route):
    """Measure a numerical route's soft travel cost without accepting or rejecting it."""
    current = {side: np.asarray(ctx.state[side+'_arm_joint_state'], float).copy()
               for side in _R02_ARMS}
    offset = _R02_ARMS.index(arm)*6
    values = [current[arm].copy()]
    diagnostics = []
    legs = [('approach', np.asarray(grasp)+[0.,0.,R02_TIMING['approach_lift']], R02_TIMING['approach']),
            ('descend', grasp, R02_TIMING['descend']),
            ('lift', np.asarray(grasp)+[0.,0.,R02_TIMING['source_lift']], R02_TIMING['lift']),
            ('carry', route['carry'], R02_TIMING['carry']),
            ('support', route['support'], R02_TIMING['support'])]
    legs.extend(('arc_'+str(i), point, R02_TIMING['arc']) for i,point in enumerate(route['arc']))
    legs.append(('retreat', np.asarray(route['arc'][-1])+[0.,0.,R02_TIMING['retreat_lift']], R02_TIMING['retreat']))
    leg_steps = []
    for name, point, duration in legs:
        path, detail = _r02_path(ctx.kinematics, current, {arm: np.r_[point, quat]})
        samples = _r02_samples(path, duration)[:, offset:offset+6]
        delta = np.diff(np.vstack([current[arm], samples]), axis=0)
        leg_steps.append(dict(name=name, maximum_step_rad=float(np.abs(delta).max())))
        values.extend(samples)
        diagnostics.extend(detail)
        current = {side: path[-1, i*6:(i+1)*6].copy() for i,side in enumerate(_R02_ARMS)}
    values = np.asarray(values)
    delta = np.diff(values, axis=0)
    maximum_step = float(np.abs(delta).max())
    travel = float(np.linalg.norm(delta, axis=1).sum())
    position = max(float(row['position_error_m']) for row in diagnostics)
    rotation = max(float(row['rotation_error_rad']) for row in diagnostics)
    cost = 30.*maximum_step**2 + .08*travel + 6000.*position + 15.*rotation
    return dict(cost=float(cost), maximum_step_rad=maximum_step, joint_travel_rad=travel,
        maximum_position_error_m=position, maximum_rotation_error_rad=rotation,
        wrist_joint_min=values[:,3:].min(axis=0).tolist(),
        wrist_joint_max=values[:,3:].max(axis=0).tolist(), leg_steps=leg_steps,
        physics_verified=False, route_rejection=False)


def _r02_pick(ctx, item, arm, grasp, original):
    """Select a fixed pinch frame before contact, using only geometry and proprioception."""
    record = ctx.memory.setdefault('toolbox_r02', {})
    record.update(candidate=R02_CANDIDATE, mode=R02_MODE, parent_entry=R02_PARENT_ENTRY,
                  runtime_gates=False, motion_abort=False)
    original = np.asarray(original, float)
    if R02_MODE == 'lower_transfer':
        return original
    matrix = _r02_matrix(original)
    if R02_MODE == 'pickup_tilt55':
        tilt = math.atan2(-matrix[2,0], float(np.linalg.norm(matrix[:2,0])))
        changed = matrix @ Rotation.from_euler('y', -(tilt-math.radians(55.))).as_matrix()
        selected = _r02_quaternion(changed)
        record['frame'] = dict(original=original.tolist(), selected=selected.tolist(),
                               target_inclination_degrees=55., opening_axis_unchanged=True)
        return selected
    route = head_supported_route(ctx, item, grasp)
    candidates = [(1., original), (-1., _r02_quaternion(matrix @ np.diag([1.,-1.,-1.])))]
    started = _r02_time.monotonic()
    options = [(sign, quat, _r02_preview(ctx, arm, grasp, quat, route)) for sign,quat in candidates]
    sign, selected, review = min(options, key=lambda item: item[2]['cost'])
    record['frame'] = dict(original=original.tolist(), selected=selected.tolist(),
        selected_opening_sign=sign, selected_review=review,
        options=[dict(opening_sign=sign, **detail) for sign,_,detail in options],
        planning_seconds=_r02_time.monotonic()-started,
        selection='soft full-route numerical travel cost; every option remains executable')
    return selected
