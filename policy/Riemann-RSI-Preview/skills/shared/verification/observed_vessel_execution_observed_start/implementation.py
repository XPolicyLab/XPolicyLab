from copy import deepcopy
import json
import math
import numpy as np
from skills.shared.motion.mature_feedback_execution_observed_start.implementation import validate_receipt
from skills.shared.perception.public_support_geometry_context_observed_start.implementation import validate as validate_view_support
from skills.shared.verification.observed_vessel_execution.implementation import vector

ACTIONS = ('vessel_scene_view', 'recover_pour_ball', 'grasp_pour_cup', 'lift_pour_cup', 'relay_pour_cup', 'carry_pour_cup', 'observe_pour_rim', 'tilt_pour_cup', 'upright_pour_cup', 'return_pour_cup', 'home_vessel_arms', 'reset_vessel_grasp', 'finish_pour_sequence')
STAGES = ('view', 'home_before_recovery', 'recover_initial', 'grasp', 'lift', 'relay', 'carry', 'observe', 'tilt', 'upright', 'return', 'home_after_return', 'recover_post', 'finish', 'done')
EXTRA = {'public_vessel_state', 'command_phases_completed', 'public_timed_execution', 'public_vessel_review'}

def number(value, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or (not lower <= value <= upper):
        raise ValueError('Vessel evidence contains finite bounded public measurements')
    return float(value)


def camera_names(value):
    if not isinstance(value, list) or not 2 <= len(value) == len(set(value)) <= 3 or (not set(value) <= {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}):
        raise ValueError('Metric vessels require independent current public cameras')

def measured_body(value):
    fields = {'center_xy_m', 'height_estimate_m', 'rim_radius_estimate_m', 'base_radius_estimate_m', 'support_estimate_m', 'support_uncertainty_m', 'rim_center_m', 'body_center_m', 'pixel_rms', 'cameras', 'source', 'physical_asset_geometry_used', 'object_ground_truth_used'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_independent_multiview_free_upright_frustum' or (value['physical_asset_geometry_used'] is not False) or (value['object_ground_truth_used'] is not False):
        raise ValueError('Cup state cannot contain asset dimensions or private identifiers')
    center = vector(value['center_xy_m'], 2)
    height = number(value['height_estimate_m'], 0.006, 0.35)
    floor = number(value['support_estimate_m'], -3.0, 3.0)
    number(value['support_uncertainty_m'], 0.0, 0.006)
    for name in ('rim_radius_estimate_m', 'base_radius_estimate_m'):
        number(value[name], 0.003, 0.18)
    number(value['pixel_rms'], 0.0, 2.5)
    camera_names(value['cameras'])
    if not np.allclose(vector(value['rim_center_m'], 3), np.r_[center, floor + height], atol=1e-06) or not np.allclose(vector(value['body_center_m'], 3), np.r_[center, floor + height / 2], atol=1e-06):
        raise ValueError('Free cup dimensions and support must agree internally')
    return deepcopy(value)

def measured_opening(value):
    fields = {'center_m', 'radius_estimate_m', 'pixel_rms', 'cameras', 'source', 'physical_asset_geometry_used', 'object_ground_truth_used'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_independent_multiview_free_upright_circle' or (value['physical_asset_geometry_used'] is not False) or (value['object_ground_truth_used'] is not False):
        raise ValueError('Vase state contains only a current free opening fit')
    vector(value['center_m'], 3)
    number(value['radius_estimate_m'], 0.003, 0.15)
    number(value['pixel_rms'], 0.0, 1.8)
    camera_names(value['cameras'])
    return deepcopy(value)

def scene(value):
    if value is None:
        return None
    fields = {'cup', 'vase_opening', 'support_estimate_m', 'support_uncertainty_m', 'source', 'official_success_claimed'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'independently_qualified_current_public_vessel_pair' or (value['official_success_claimed'] is not False):
        raise ValueError('A vessel scene is not a private task layout or native success')
    measured_body(value['cup'])
    measured_opening(value['vase_opening'])
    floor = number(value['support_estimate_m'], -3.0, 3.0)
    number(value['support_uncertainty_m'], 0.0, 0.006)
    if abs(floor - value['cup']['support_estimate_m']) > 0.006 or not 0.006 <= value['vase_opening']['center_m'][2] - floor <= 0.4:
        raise ValueError('The current support and free opening must agree')
    return deepcopy(value)

def compact_scene(value):
    current = deepcopy(value)
    for item in (current['cup'], current['vase_opening']):
        item.pop('head_centroid', None)
    return scene(current)

def rim(value):
    if value is None:
        return None
    fields = {'relative_rim_center_m', 'relative_rim_rotation', 'radius_estimate_m', 'pixel_rms', 'views', 'source', 'physical_asset_geometry_used', 'object_ground_truth_used', 'attachment_is_hypothesis'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_public_rim_arcs_and_observed_own_movement' or (value['physical_asset_geometry_used'] is not False) or (value['object_ground_truth_used'] is not False) or (value['attachment_is_hypothesis'] is not True) or (type(value['views']) is not int) or (value['views'] not in (2, 3)):
        raise ValueError('A held vessel is a public visual movement hypothesis, not measured attachment')
    vector(value['relative_rim_center_m'], 3)
    rotation = np.asarray(value['relative_rim_rotation'], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05)) or (not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-05)):
        raise ValueError('The held-rim own-relative orientation is invalid')
    number(value['radius_estimate_m'], 0.003, 0.18)
    number(value['pixel_rms'], 0.0, 1.2)
    return deepcopy(value)

def route(value):
    if value is None:
        return None
    fields = {'side', 'destination_side', 'approach', 'grasp', 'lift', 'source_body_m', 'own_tool_forward_m'}
    if not isinstance(value, dict) or set(value) != fields or value['side'] not in ('left', 'right') or (value['destination_side'] not in ('left', 'right')) or (value['own_tool_forward_m'] != 0.14):
        raise ValueError('A cup route uses its actual original own-tool point, not the generic Cube point')
    for name in ('approach', 'grasp', 'lift'):
        vector(value[name], 7)
    vector(value['source_body_m'], 3)
    return deepcopy(value)

def empty():
    return {'stage': 'view', 'scene': None, 'route': None, 'rim': None, 'initial_pour_pose': None, 'tilt_poses': [], 'return_point_m': None, 'relay_point_m': None, 'initial_cup_xy_m': None, 'initial_open_done': False, 'relayed': False, 'relay_completed': False, 'recovery_active': False, 'post_scan_done': False, 'ball_candidate': None, 'initial_ball_attempts': 0, 'post_ball_attempts': 0, 'view_attempts': 0, 'grasp_attempts': 0, 'tilt_index': 0, 'upright_index': 0, 'ball_radius_estimate_m': None, 'view_support': None}

def state(value):
    if not isinstance(value, dict) or set(value) != set(empty()) or value['stage'] not in STAGES:
        raise ValueError('Vessel state contains only bounded same-episode public evidence and own progress')
    scene(value['scene'])
    if value['view_support'] is not None:
        validate_view_support(value['view_support'])
    route(value['route'])
    rim(value['rim'])
    for name in ('initial_open_done', 'relayed', 'relay_completed', 'recovery_active', 'post_scan_done'):
        if type(value[name]) is not bool:
            raise ValueError('Vessel own-command flags cannot masquerade as contact or success')
    for name, maximum in {'initial_ball_attempts': 2, 'post_ball_attempts': 1, 'view_attempts': 6, 'grasp_attempts': 2, 'tilt_index': 9, 'upright_index': 8}.items():
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('Vessel command attempts are finite and shared-episode bounded')
    for name, size in (('initial_pour_pose', 7), ('return_point_m', 3), ('relay_point_m', 3), ('initial_cup_xy_m', 2)):
        if value[name] is not None:
            vector(value[name], size)
    if not isinstance(value['tilt_poses'], list) or len(value['tilt_poses']) not in (0, 9):
        raise ValueError('Pouring retains the nine original ordered angular stages')
    for pose in value['tilt_poses']:
        vector(pose, 7)
    if value['ball_radius_estimate_m'] is not None:
        number(value['ball_radius_estimate_m'], 0.001, 0.04)
    if value['ball_candidate'] is not None:
        candidate = value['ball_candidate']
        fields = {'center_m', 'radius_estimate_m', 'support_estimate_m', 'support_uncertainty_m', 'pixel_rms', 'cameras', 'source', 'object_ground_truth_used', 'physical_asset_geometry_used', 'contact_is_measured'}
        if not isinstance(candidate, dict) or set(candidate) != fields or candidate['source'] != 'current_independent_multiview_free_supported_sphere' or any((candidate[name] is not False for name in ('object_ground_truth_used', 'physical_asset_geometry_used', 'contact_is_measured'))):
            raise ValueError('A retained ball candidate is a public support hypothesis, never private state')
        vector(candidate['center_m'], 3)
        number(candidate['radius_estimate_m'], 0.001, 0.04)
        number(candidate['support_estimate_m'], -3.0, 3.0)
        number(candidate['support_uncertainty_m'], 0.0, 0.006)
        number(candidate['pixel_rms'], 0.0, 1.3)
        camera_names(candidate['cameras'])
    if len(json.dumps(value, allow_nan=False).encode()) > 6400:
        raise ValueError('Vessel feedback cannot hide an unbounded controller or scene dump')
    return deepcopy(value)

def recorded(memory):
    for event in reversed(memory):
        if event.get('skill') not in ACTIONS:
            continue
        evidence = event.get('execution_feedback', {}).get('image_measurements', {})
        if 'public_vessel_state' in evidence:
            return state(evidence['public_vessel_state'])
    return empty()

def measurements(value, status, actions):
    selected = set(value) & EXTRA
    if not {'public_vessel_state', 'command_phases_completed'} <= selected:
        raise ValueError('Vessel actions require explicit public state and acknowledged own commands')
    state(value['public_vessel_state'])
    completed = value['command_phases_completed']
    if type(completed) is not bool or (completed and (status != 'completed' or actions <= 0)):
        raise ValueError('Unexecuted vessel commands cannot advance the sequence')
    if 'public_timed_execution' in selected:
        receipt = validate_receipt(value['public_timed_execution'])
        if receipt['official_action_acknowledgments'] != actions:
            raise ValueError('Vessel timed receipts must account for every issued native action')
    if 'public_vessel_review' in selected:
        review = value['public_vessel_review']
        fields = {'status', 'source', 'is_contact_sensor', 'object_grasp_confirmed', 'official_success_claimed', 'scene_reason'}
        if not isinstance(review, dict) or set(review) != fields or review['status'] not in ('unavailable', 'uncertain', 'visual_correspondence') or (review['source'] != 'current_public_rgb_and_own_command_response') or (review['scene_reason'] not in (None, 'current_support_unqualified', 'current_cup_body_unqualified', 'current_vase_opening_unqualified', 'current_vessel_identities_overlap')) or any((review[name] is not False for name in ('is_contact_sensor', 'object_grasp_confirmed', 'official_success_claimed'))):
            raise ValueError('Vessel feedback cannot turn public gripper response into real fingers or contact')
    return selected
