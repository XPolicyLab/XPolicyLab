from copy import deepcopy
import hashlib
import math
import numpy as np
from skills.shared.perception.public_support_geometry_context_observed_start.implementation import validate as validate_support
ACTIONS = ('garment_scene_view', 'garment_approach_contact', 'garment_lower_contact', 'garment_close_grasp', 'garment_lift_fold', 'garment_carry_fold', 'garment_settle_fold', 'garment_release_fold', 'garment_clear_wrist', 'garment_recover_contact', 'garment_return_home')
PHASES = ('view', 'approach', 'contact', 'close', 'lift', 'carry', 'lower', 'release', 'clear', 'recover', 'home', 'hold')
LANDMARKS = ('left_cuff', 'right_cuff', 'left_hem', 'right_hem', 'left_chest', 'right_chest')

def vector(value, length, bound=3.0):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all() or np.max(abs(array)) > bound:
        raise ValueError('A garment record contains only finite bounded public geometry')
    return array

def pose(value):
    array = vector(value, 7, 4.0)
    if not 0.99 <= np.linalg.norm(array[3:]) <= 1.01:
        raise ValueError('An own garment pose needs a unit public quaternion')
    return array

def anchor(value):
    fields = {'center', 'longitudinal_axis', 'lateral_axis', 'confidence', 'source', 'support_memory', 'height_m', 'support_uncertainty_m', 'head_outline', 'appearance_hsv', *LANDMARKS}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_public_rgb_garment_outline':
        raise ValueError('A garment anchor must contain only its current public outline evidence')
    for name in ('center', *LANDMARKS):
        vector(value[name], 2)
    for name in ('longitudinal_axis', 'lateral_axis'):
        if not 0.99 <= np.linalg.norm(vector(value[name], 2)) <= 1.01:
            raise ValueError('A measured garment axis must be normalized')
    if abs(np.dot(value['longitudinal_axis'], value['lateral_axis'])) > 0.01:
        raise ValueError('Observed garment axes must be orthogonal')
    for name, lower, upper in (('confidence', 0.0, 1.0), ('height_m', 0.1, 2.0), ('support_uncertainty_m', 0.0, 0.03)):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or (not lower <= value[name] <= upper):
            raise ValueError('A garment observation requires finite bounded confidence and support')
    outline = np.asarray(value['head_outline'], dtype=float)
    if outline.shape != (96, 2) or not np.isfinite(outline).all() or np.any(outline <= 1) or np.any(outline[:, 0] >= 638) or np.any(outline[:, 1] >= 478):
        raise ValueError('The retained garment silhouette belongs to a complete current native head image')
    appearance = vector(value['appearance_hsv'], 3, 255.0)
    if np.any(appearance < 0) or appearance[0] > 180:
        raise ValueError('Garment appearance comes only from current public HSV samples')
    validate_support(value['support_memory'])
    return deepcopy(value)

def route(value):
    fields = {'arms', 'contacts', 'destinations', 'contact_poses', 'approach_joints', 'carry_chunks', 'body_chunks'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('A garment route contains only current contact and computed own targets')
    arms = value['arms']
    if not isinstance(arms, list) or not 1 <= len(arms) <= 2 or len(set(arms)) != len(arms) or (not set(arms).issubset({'left', 'right'})):
        raise ValueError('A garment route has one or two distinct own arms')
    for name, length in (('contacts', 3), ('destinations', 3), ('contact_poses', 7), ('approach_joints', 6)):
        if not isinstance(value[name], dict) or set(value[name]) != set(arms):
            raise ValueError('Every own garment route field refers to the same declared arms')
        for item in value[name].values():
            pose(item) if length == 7 else vector(item, length, 10.0 if length == 6 else 3.0)
    for name in ('carry_chunks', 'body_chunks'):
        if type(value[name]) is not int or not 1 <= value[name] <= 2:
            raise ValueError('Original garment folding reserves a finite one/two-chunk schedule')
    return deepcopy(value)

def initial(instruction):
    return {'schema': 'riemann-public-garment-state-v1', 'instruction_fingerprint': hashlib.sha256(instruction.encode()).hexdigest(), 'phase': 'view', 'part': 0, 'chunk': 0, 'attempts': 0, 'anchor': None, 'route': None, 'start_wrists': None}

def validate(value):
    if not isinstance(value, dict) or set(value) != set(initial('')) or value['schema'] != 'riemann-public-garment-state-v1' or (not isinstance(value['instruction_fingerprint'], str)) or (len(value['instruction_fingerprint']) != 64) or any((character not in '0123456789abcdef' for character in value['instruction_fingerprint'])) or (value['phase'] not in PHASES):
        raise ValueError('Own garment progress must use its closed same-instruction state contract')
    for name, maximum in (('part', 3), ('chunk', 2), ('attempts', 3)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('Garment progress and retry counts must remain bounded')
    if value['anchor'] is not None:
        anchor(value['anchor'])
    if value['route'] is not None:
        route(value['route'])
    if value['start_wrists'] is not None:
        if value['route'] is None or set(value['start_wrists']) != set(value['route']['arms']):
            raise ValueError('Executed garment start poses must refer to the qualified own route')
        for item in value['start_wrists'].values():
            pose(item)
    return deepcopy(value)

def recorded(memory, instruction):
    current = initial(instruction)
    for event in reversed(memory[-32:]):
        if event.get('skill') not in ACTIONS:
            continue
        record = event.get('execution_feedback', {}).get('image_measurements', {}).get('public_garment_state')
        if record is not None:
            candidate = validate(record)
            return candidate if candidate['instruction_fingerprint'] == current['instruction_fingerprint'] else current
    return current

def measurements(value, status, action_count):
    extra = set(value) & {'public_garment_state', 'garment_visual_review', 'garment_clock', 'command_phases_completed'}
    if extra != {'public_garment_state', 'garment_visual_review', 'garment_clock', 'command_phases_completed'}:
        raise ValueError('A garment stage returns its complete closed public state and replay receipt')
    validate(value['public_garment_state'])
    review = value['garment_visual_review']
    if not isinstance(review, dict) or set(review) != {'status', 'head_colour_fraction', 'wrist_colour_fraction', 'gripper_measurement_is_contact'} or review['status'] not in ('uncertain', 'current_surface_follows_own_pad') or (review['gripper_measurement_is_contact'] is not False):
        raise ValueError('Garment visual evidence cannot claim true gripper contact')
    for name in ('head_colour_fraction', 'wrist_colour_fraction'):
        if type(review[name]) not in (int, float) or not math.isfinite(review[name]) or (not 0 <= review[name] <= 1):
            raise ValueError('Current garment colour fractions must be finite and normalized')
    if value['garment_clock'] != {'segment_width': 16, 'trajectory_resampled': False, 'true_finger_measurements_used': False}:
        raise ValueError('Garment execution must retain the declared sixteen-command public clock')
    if type(value['command_phases_completed']) is not bool or (value['command_phases_completed'] and (status not in ('completed', 'uncertain') or action_count < 16)):
        raise ValueError('A completed garment phase requires actual native command acknowledgments')
    return extra
