from copy import deepcopy
import re

import numpy as np

from skills.shared.perception.public_support_geometry_context.implementation import validate as validate_support


ACTIONS = ('pickup_scene_view', 'pickup_pregrasp', 'pickup_contact', 'pickup_close',
           'pickup_warmup_lift', 'pickup_full_lift', 'pickup_verify_lift', 'pickup_reset_grasp')
STAGES = ('view', 'pregrasp', 'contact', 'close', 'warmup', 'lift', 'verify', 'reset', 'hold')
REASONS = (None, 'current_target_unqualified', 'support_unqualified', 'current_surface_unqualified',
           'original_route_unreachable', 'public_endpoint_unreached', 'current_target_changed',
           'visual_lift_uncertain', 'target_still_supported', 'own_branch_discontinuous', 'action_allowance_insufficient')


def vector(value, count, bound, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all() or np.max(np.abs(array)) > bound:
        raise ValueError('Invalid bounded public ' + label)
    return array


def pose(value):
    array = vector(value, 7, 3., 'robot target')
    if not .99 <= np.linalg.norm(array[3:]) <= 1.01:
        raise ValueError('An own tool target requires a unit public quaternion')
    return array


def validate_anchor(value):
    fields = {'source', 'prompt', 'center_m', 'spans_m', 'height_estimate_m', 'uncertainty_m',
              'cameras', 'coverage', 'head_outline', 'head_centroid', 'appearance_rgb', 'support_memory', 'resolution_semantics'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_public_rgb_visual_hull':
        raise ValueError('Pickup anchors contain only their current public visual reconstruction')
    if value['resolution_semantics'] != 'grid_and_support_resolution_not_hidden_object_error':
        raise ValueError('A visual hull resolution is not a certificate of hidden object geometry accuracy')
    if not isinstance(value['prompt'], str) or not 1 <= len(value['prompt']) <= 128 or re.search(r'[\x00-\x1f]', value['prompt']):
        raise ValueError('A target description is bounded text from the actual public instruction')
    vector(value['center_m'], 3, 3., 'target center estimate')
    spans = vector(value['spans_m'], 3, .5, 'observed object extent')
    if np.any(spans < 0) or max(spans[:2]) < .004:
        raise ValueError('An object extent must have been visually resolved')
    for name, lower, upper in [('height_estimate_m', .005, .35), ('uncertainty_m', 0., .018)]:
        if type(value[name]) not in (int, float) or not np.isfinite(value[name]) or not lower <= value[name] <= upper:
            raise ValueError('A visual reconstruction requires bounded measured height and uncertainty')
    cameras = value['cameras']
    if (not isinstance(cameras, list) or not 2 <= len(cameras) == len(set(cameras)) <= 3
            or cameras[0] != 'cam_head' or any(camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist') for camera in cameras)):
        raise ValueError('A current reconstruction needs independent native public views')
    coverage = np.asarray(value['coverage'], dtype=float)
    if coverage.shape != (len(cameras),) or not np.isfinite(coverage).all() or np.any(coverage < .55) or np.any(coverage > 1):
        raise ValueError('Every independent silhouette must qualify the current reconstruction')
    outline = np.asarray(value['head_outline'], dtype=float)
    if (outline.shape != (96, 2) or not np.isfinite(outline).all() or np.any(outline <= 1)
            or np.any(outline[:, 0] >= 638) or np.any(outline[:, 1] >= 478)):
        raise ValueError('The current source outline is a complete bounded head-image observation')
    centroid = vector(value['head_centroid'], 2, 640., 'current image centroid')
    if np.any(centroid < 0) or centroid[0] >= 640 or centroid[1] >= 480:
        raise ValueError('An image centroid stays in its current public frame')
    colour = vector(value['appearance_rgb'], 3, 255., 'current visible appearance')
    if np.any(colour < 0):
        raise ValueError('Visible colour is a public RGB estimate')
    validate_support(value['support_memory'])
    return deepcopy(value)


def validate_route(value):
    fields = {'side', 'pregrasp', 'contact', 'lift', 'pregrasp_joints', 'contact_joints', 'lift_joints',
              'width_estimate_m', 'cost', 'fraction', 'tilt_degrees'}
    if not isinstance(value, dict) or set(value) != fields or value['side'] not in ('left', 'right'):
        raise ValueError('A pickup route contains only computed own robot targets and visual section estimates')
    for name in ('pregrasp', 'contact', 'lift'):
        pose(value[name])
    for name in ('pregrasp_joints', 'contact_joints', 'lift_joints'):
        vector(value[name], 6, 10., 'own original joint branch')
    for name, lower, upper in [('width_estimate_m', .004, .082), ('cost', 0., 100.), ('fraction', 0., 1.)]:
        if type(value[name]) not in (int, float) or not np.isfinite(value[name]) or not lower <= value[name] <= upper:
            raise ValueError('The route uses bounded original section ranking estimates')
    if value['tilt_degrees'] not in (0, 15, 25, 35):
        raise ValueError('Use only the reviewed original own tool orientations')
    return deepcopy(value)


def initial():
    return {'stage': 'view', 'attempts': 0, 'view_attempts': 0, 'corrections': 0, 'verify_count': 0,
            'anchor': None, 'route': None, 'contact_actual': None, 'failure_reason': None}


def validate(value):
    if not isinstance(value, dict) or set(value) != set(initial()) or value['stage'] not in STAGES or value['failure_reason'] not in REASONS:
        raise ValueError('Pickup progress is a closed same-episode public command record, not evaluator progress')
    for name, upper in [('attempts', 3), ('view_attempts', 6), ('corrections', 1), ('verify_count', 16)]:
        if type(value[name]) is not int or not 0 <= value[name] <= upper:
            raise ValueError('Pickup retries and observation holds require finite counters')
    if value['anchor'] is not None:
        validate_anchor(value['anchor'])
    if value['route'] is not None:
        validate_route(value['route'])
    if value['contact_actual'] is not None:
        pose(value['contact_actual'])
    if value['stage'] not in ('view', 'pregrasp', 'reset') and (value['route'] is None or value['anchor'] is None):
        raise ValueError('A command stage requires its actual preceding visual anchor and own route')
    return deepcopy(value)


def recorded(memory):
    for event in reversed(memory[-32:]):
        if event.get('skill') not in ACTIONS:
            continue
        feedback = event.get('execution_feedback', {})
        if feedback.get('status') not in ('completed', 'uncertain', 'precondition_unmet', 'budget'):
            continue
        measured = feedback.get('image_measurements', {})
        value = measured.get('public_pickup_state')
        if value is not None:
            return validate(value)
    return initial()


def empty_visual():
    return {'status': 'unavailable', 'target_displacement_m': None, 'own_tcp_displacement_m': [0., 0., 0.],
            'is_contact_or_true_finger_sensor': False, 'official_success_claimed': False}


def validate_measurements(measured, status, actions):
    fields = {'public_pickup_state', 'pickup_visual_review', 'native_command_clock', 'command_phases_completed'}
    if not fields.issubset(measured):
        raise ValueError('A pickup tool must return its complete public command and uncertainty contract')
    validate(measured['public_pickup_state'])
    visual = measured['pickup_visual_review']
    if (not isinstance(visual, dict) or set(visual) != set(empty_visual())
            or visual['status'] not in ('unavailable', 'ambiguous', 'follows_hand', 'still_at_source')
            or visual['is_contact_or_true_finger_sensor'] is not False or visual['official_success_claimed'] is not False):
        raise ValueError('Visual lift feedback is not object contact, a real finger signal or official task success')
    vector(visual['own_tcp_displacement_m'], 3, .5, 'own measured tool displacement')
    if visual['target_displacement_m'] is not None:
        vector(visual['target_displacement_m'], 3, .5, 'current visual target displacement')
    if visual['status'] in ('follows_hand', 'still_at_source') and visual['target_displacement_m'] is None:
        raise ValueError('A positive visual conclusion requires an actual current target measurement')
    expected_clock = {'segment_width': 16, 'endpoint_padding_preserved': True, 'trajectory_resampled': False,
                      'opening_semantics': 'public_normalized_command_not_real_fingers'}
    if measured['native_command_clock'] != expected_clock:
        raise ValueError('Pickup feedback must retain the original native command clock and public gripper semantics')
    completed = measured['command_phases_completed']
    if type(completed) is not bool or completed and (status not in ('completed', 'uncertain') or actions < 16):
        raise ValueError('A completed command sequence requires actual counted native actions')
    return fields
