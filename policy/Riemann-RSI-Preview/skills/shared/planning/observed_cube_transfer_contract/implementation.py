from copy import deepcopy
import math

import numpy as np


ACTIONS = ('cube_scene_view', 'grasp_measured_cube', 'lift_measured_cube', 'carry_measured_cube',
           'release_measured_cube', 'verify_measured_stack', 'reset_cube_grasp', 'finish_cube_stack')
EXTRA = {'public_cube_anchor', 'public_cube_progress', 'public_cube_route', 'public_cube_grasp_feedback',
         'command_phases_completed', 'observed_source_axis_rejection', 'public_cube_reason'}
REASONS = ('source_not_relocated', 'route_not_planned', 'placement_not_relocated', 'placement_displaced',
           'placement_contradicted')


def vector(value, size):
    checked = np.asarray(value, dtype=float)
    if checked.shape != (size,) or not np.isfinite(checked).all() or np.max(np.abs(checked)) > 3:
        raise ValueError('Cube evidence contains only finite bounded public vectors')
    if size == 7 and not .99 <= np.linalg.norm(checked[3:]) <= 1.01:
        raise ValueError('Own cube commands need a normalized public robot orientation')
    return checked


def anchor(value):
    """Validate public three-cube geometry with XYZ centers (3,); return a detached anchor or None."""
    if value is None:
        return None
    fields = {'schema', 'boxes', 'support_estimate_m', 'source', 'physical_asset_geometry_used',
              'object_ground_truth_used', 'stack_destination_m'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'riemann-current-public-cube-family-v1'
            or value['source'] != 'independent_current_multiview_free_cube_model'
            or value['physical_asset_geometry_used'] is not False or value['object_ground_truth_used'] is not False
            or not isinstance(value['boxes'], list) or len(value['boxes']) != 3):
        raise ValueError('A cube anchor has exactly three current independently measured free models')
    floor = value['support_estimate_m']
    if type(floor) not in (int, float) or not math.isfinite(floor) or abs(floor) > 3:
        raise ValueError('The current visual support estimate must be finite')
    slots = set()
    for box in value['boxes']:
        if set(box) != {'visual_slot', 'center', 'span_estimate_m', 'yaw_radians', 'appearance', 'pixel_rms', 'cameras'}:
            raise ValueError('Cube models cannot add GT identities or physical size tables')
        slot = box['visual_slot']
        if type(slot) is not int or slot not in (0, 1, 2) or slot in slots:
            raise ValueError('Visual cube slots are unique same-episode labels')
        slots.add(slot)
        center = vector(box['center'], 3)
        appearance = np.asarray(box['appearance'], dtype=float)
        if appearance.shape != (3,) or not np.isfinite(appearance).all() or np.any(appearance < 0) or np.any(appearance > 255):
            raise ValueError('Cube appearance comes only from current public RGB')
        span = box['span_estimate_m']
        if type(span) not in (int, float) or not math.isfinite(span) or not .003 <= span <= .2:
            raise ValueError('Cube span is a bounded freely fitted estimate')
        if abs(center[2] - span / 2 - floor) > .006:
            raise ValueError('The initial three cube estimates must agree on visible support')
        if (type(box['pixel_rms']) not in (int, float) or not math.isfinite(box['pixel_rms']) or not 0 <= box['pixel_rms'] <= 2
                or type(box['yaw_radians']) not in (int, float) or not math.isfinite(box['yaw_radians'])
                or not isinstance(box['cameras'], list) or not 2 <= len(box['cameras']) == len(set(box['cameras'])) <= 3
                or not set(box['cameras']) <= {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}):
            raise ValueError('Each cube needs independently qualified current camera fits')
    spans = [box['span_estimate_m'] for box in value['boxes']]
    if min(spans) < .6 * float(np.median(spans)):
        raise ValueError('A cube family cannot contain a fragmentary outline far smaller than its members')
    destination = vector(value['stack_destination_m'], 3)
    if abs(destination[2] - floor) > 1e-8:
        raise ValueError('The stack destination uses the current support estimate')
    positions = np.asarray([box['center'][:2] for box in value['boxes']])
    if np.any(destination[:2] < positions.min(axis=0)) or np.any(destination[:2] > positions.max(axis=0)):
        raise ValueError('Stack placement is selected from the current observed family, not a saved layout')
    return deepcopy(value)


def empty_progress():
    return {'cursor': 0, 'stage': 0, 'route_attempt': 0, 'axis_rejections': 0,
            'view_attempts': 0, 'final_stage': 0, 'placed_spans': []}


def progress(value):
    fields = {'cursor', 'stage', 'route_attempt', 'axis_rejections', 'view_attempts', 'final_stage', 'placed_spans'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Cube progress contains only bounded own-command state')
    bounds = {'cursor': 3, 'stage': 4, 'route_attempt': 6, 'axis_rejections': 3, 'view_attempts': 6, 'final_stage': 2}
    if any(type(value[name]) is not int or not 0 <= value[name] <= maximum for name, maximum in bounds.items()):
        raise ValueError('Cube commands cannot contain unbounded or boolean counters')
    spans = value['placed_spans']
    if (not isinstance(spans, list) or len(spans) != value['cursor']
            or any(type(span) not in (int, float) or not math.isfinite(span) or not .003 <= span <= .2 for span in spans)
            or value['cursor'] == 3 and value['stage'] != 0):
        raise ValueError('Only completed current visual placements advance the stack cursor')
    return deepcopy(value)


def route(value):
    if value is None:
        return None
    fields = {'side', 'visual_slot', 'orientation', 'tilted', 'span_estimate_m', 'source_tip', 'target_tip',
              'approach', 'grasp', 'carry', 'place', 'lift_poses'}
    if (not isinstance(value, dict) or set(value) != fields or value['side'] not in ('left', 'right')
            or type(value['visual_slot']) is not int or value['visual_slot'] not in (0, 1, 2)
            or type(value['orientation']) is not int or value['orientation'] not in (0, 1)
            or type(value['tilted']) is not bool):
        raise ValueError('A cube route is an explicit own-command plan, not a hidden teacher record')
    for name in ('source_tip', 'target_tip', 'approach', 'grasp', 'carry', 'place'):
        vector(value[name], 3 if name.endswith('_tip') else 7)
    if (type(value['span_estimate_m']) not in (int, float) or not .003 <= value['span_estimate_m'] <= .2
            or not isinstance(value['lift_poses'], list) or not 1 <= len(value['lift_poses']) <= 4):
        raise ValueError('A cube route has bounded measured span and finite original lift stages')
    for pose in value['lift_poses']:
        vector(pose, 7)
    return deepcopy(value)


def measurements(value, status, actions):
    """Validate public feedback dictionary and bounded command status/count; return declared fields."""
    selected = set(value) & EXTRA
    if not {'public_cube_anchor', 'public_cube_progress', 'command_phases_completed'} <= selected:
        raise ValueError('Cube tools require explicit current observation and own-command receipts')
    anchor(value['public_cube_anchor'])
    progress(value['public_cube_progress'])
    route(value.get('public_cube_route'))
    completed = value['command_phases_completed']
    if type(completed) is not bool or completed and (status != 'completed' or actions <= 0):
        raise ValueError('A cube phase receipt cannot pretend that an unexecuted command completed')
    if 'observed_source_axis_rejection' in value and type(value['observed_source_axis_rejection']) is not bool:
        raise ValueError('Source-axis rejection is measured own robot feedback')
    if 'public_cube_reason' in value and value['public_cube_reason'] not in REASONS:
        raise ValueError('A cube outcome reason is one of the declared bounded public causes')
    grasp = value.get('public_cube_grasp_feedback')
    if grasp is not None:
        fields = {'status', 'public_gripper_response', 'image_motion_observed', 'is_contact_sensor', 'object_grasp_confirmed'}
        if (not isinstance(grasp, dict) or set(grasp) != fields or grasp['status'] not in ('uncertain', 'visual_correspondence')
                or type(grasp['public_gripper_response']) not in (int, float) or not 0 <= grasp['public_gripper_response'] <= 1
                or type(grasp['image_motion_observed']) is not bool
                or grasp['is_contact_sensor'] is not False or grasp['object_grasp_confirmed'] is not False):
            raise ValueError('Public cube gripper response cannot become real finger width or contact')
    return selected
