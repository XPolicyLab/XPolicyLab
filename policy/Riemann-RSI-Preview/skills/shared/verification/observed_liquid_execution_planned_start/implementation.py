from copy import deepcopy
import numpy as np
from skills.shared.verification.observed_vessel_execution_planned_start.implementation import measured_opening, number, vector
from skills.shared.perception.public_support_geometry_context_planned_start.implementation import validate as validate_view_support
ACTIONS = ('liquid_scene_view', 'liquid_support_cup', 'liquid_approach_bottle', 'liquid_close_bottle', 'liquid_reset_grasp', 'liquid_lift_bottle', 'liquid_observe_spout', 'liquid_carry_bottle', 'liquid_tilt_bottle', 'liquid_drain', 'liquid_upright', 'liquid_hold_upright')
STAGES = ('view', 'support', 'approach', 'close', 'lift', 'observe', 'carry', 'tilt', 'drain', 'upright', 'hold', 'retry_view', 'done')
REASONS = (None, 'current_support_unqualified', 'current_bottle_unqualified', 'current_cup_opening_unqualified', 'current_spout_unqualified', 'current_lift_unqualified', 'current_path_unqualified', 'current_support_profile_unqualified')

def profile(value):
    fields = {'center_xy_m', 'height_estimate_m', 'radius_profile_m', 'height_fractions', 'support_estimate_m', 'support_uncertainty_m', 'pixel_rms', 'cameras', 'source', 'physical_asset_geometry_used', 'object_ground_truth_used'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_independent_multiview_free_axial_profile' or (value['physical_asset_geometry_used'] is not False) or (value['object_ground_truth_used'] is not False):
        raise ValueError('Liquid body state contains only a current free multiview radial hypothesis')
    vector(value['center_xy_m'], 2)
    number(value['height_estimate_m'], 0.006, 0.35)
    count = len(value['radius_profile_m']) if isinstance(value['radius_profile_m'], list) else 0
    if count not in (6, 8):
        raise ValueError('A current body retains six original radii or eight bounded observed radii')
    if not isinstance(value['height_fractions'], list) or any((type(fraction) not in (int, float) for fraction in value['height_fractions'])) or (count == 6 and value['height_fractions'] != [0.0, 0.12, 0.45, 0.78, 0.9, 1.0]):
        raise ValueError('Normalized radii retain the legacy profile or the bounded observed knot representation')
    radii = vector(value['radius_profile_m'], count)
    fractions = vector(value['height_fractions'], count)
    if np.any(radii < 0.001) or np.any(radii > 0.18) or fractions[0] != 0.0 or (fractions[-1] != 1.0) or np.any(np.diff(fractions) < 0.005):
        raise ValueError('Current body radii must retain the declared free normalized representation')
    number(value['support_estimate_m'], -3.0, 3.0)
    number(value['support_uncertainty_m'], 0.0, 0.006)
    number(value['pixel_rms'], 0.0, 2.5)
    cameras = value['cameras']
    if not isinstance(cameras, list) or not 2 <= len(cameras) == len(set(cameras)) <= 3 or any((name not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist') for name in cameras)):
        raise ValueError('A body fit needs independent delivered public cameras')
    return deepcopy(value)

def scene(value):
    if value is None:
        return None
    fields = {'bottle', 'cup_opening', 'cup_profile', 'support_estimate_m', 'support_uncertainty_m', 'source'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != 'current_independently_qualified_liquid_pair':
        raise ValueError('Liquid scene has no asset identifiers, saved layout or evaluator result')
    profile(value['bottle'])
    measured_opening(value['cup_opening'])
    if value['cup_profile'] is not None:
        profile(value['cup_profile'])
    height = number(value['support_estimate_m'], -3.0, 3.0)
    number(value['support_uncertainty_m'], 0.0, 0.006)
    if abs(height - value['bottle']['support_estimate_m']) > 0.006 or not 0.006 <= value['cup_opening']['center_m'][2] - height <= 0.4:
        raise ValueError('Current liquid pair and visual support must agree internally')
    return deepcopy(value)

def empty():
    return {'stage': 'view', 'scene': None, 'side': None, 'support_side': None, 'route': None, 'attachment': None, 'tilt_poses': [], 'view_attempts': 0, 'grasp_attempts': 0, 'spout_attempts': 0, 'retry_lowered': False, 'actions_seen': 0, 'reason': None, 'official_success_claimed': False, 'view_support': None}

def state(value):
    if not isinstance(value, dict) or set(value) != set(empty()) or value['stage'] not in STAGES:
        raise ValueError('Liquid progress is a finite public skill state, not a legacy controller snapshot')
    if value['official_success_claimed'] is not False or type(value['retry_lowered']) is not bool or value['reason'] not in REASONS:
        raise ValueError('Liquid feedback cannot assert physical contact, official success or an arbitrary message')
    scene(value['scene'])
    if value['view_support'] is not None:
        validate_view_support(value['view_support'])
    for name in ('side', 'support_side'):
        if value[name] not in (None, 'left', 'right'):
            raise ValueError('Liquid execution selects only its public own arm')
    for name, maximum in (('view_attempts', 4), ('grasp_attempts', 2), ('spout_attempts', 3), ('actions_seen', 1200)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('Liquid attempts and issued-command count are bounded within this episode')
    if value['route'] is not None:
        if not isinstance(value['route'], dict) or set(value['route']) != {'approach_pose', 'grasp_pose'}:
            raise ValueError('A liquid route contains only its two own command goals')
        for pose in value['route'].values():
            vector(pose, 7)
            if not 0.99 <= np.linalg.norm(pose[3:]) <= 1.01:
                raise ValueError('Liquid own command goals need a unit quaternion')
    if value['attachment'] is not None:
        attachment = value['attachment']
        fields = {'relative_body_pose', 'relative_spout_m', 'radius_estimate_m', 'source', 'attachment_is_hypothesis'}
        if not isinstance(attachment, dict) or set(attachment) != fields or attachment['source'] != 'current_spout_motion_and_own_wrist_feedback' or (attachment['attachment_is_hypothesis'] is not True):
            raise ValueError('Held geometry is a current visual movement hypothesis, not measured contact')
        vector(attachment['relative_body_pose'], 7)
        vector(attachment['relative_spout_m'], 3)
        number(attachment['radius_estimate_m'], 0.003, 0.15)
    if not isinstance(value['tilt_poses'], list) or len(value['tilt_poses']) not in (0, 49, 97):
        raise ValueError('Liquid pouring retains the original ordered tilt samples')
    for pose in value['tilt_poses']:
        vector(pose, 7)
        if not 0.99 <= np.linalg.norm(pose[3:]) <= 1.01:
            raise ValueError('A pouring sample needs an own unit orientation')
    return deepcopy(value)

def recorded(memory):
    for event in reversed(memory):
        if event.get('skill') not in ACTIONS:
            continue
        current = event.get('execution_feedback', {}).get('image_measurements', {}).get('public_liquid_state')
        if current is not None:
            return state(current)
    return empty()

def measurements(value, status, actions):
    extra = {'public_liquid_state', 'native_execution_receipt', 'public_liquid_review'}
    if not isinstance(value, dict) or not extra <= set(value):
        raise ValueError('Liquid tool feedback requires explicit public state, own clock and uncertainty')
    state(value['public_liquid_state'])
    receipt = value['native_execution_receipt']
    fields = {'schema', 'actions_executed', 'sampled_native_controls', 'whole_path_refinements', 'endpoint_settle_intervals', 'native_chunk_maximum', 'endpoint_settle_commands', 'gripper_commands', 'trajectory_sampling_reapplied', 'true_fingers_or_contact_read', 'official_success_claimed'}
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError('Liquid command receipt cannot carry an evaluator result or hidden retries')
    if receipt['schema'] != 'riemann-original-fixed-step-execution-v1' or receipt['native_chunk_maximum'] != 16 or receipt['gripper_commands'] != 'constant_original_normalized_opening' or (receipt['trajectory_sampling_reapplied'] is not False) or (receipt['true_fingers_or_contact_read'] is not False) or (receipt['official_success_claimed'] is not False):
        raise ValueError('Liquid command receipt preserves fixed native samples without task-success claims')
    for name, maximum in (('actions_executed', 400), ('sampled_native_controls', 400), ('whole_path_refinements', 10), ('endpoint_settle_intervals', 10), ('endpoint_settle_commands', 160)):
        if type(receipt[name]) is not int or not 0 <= receipt[name] <= maximum:
            raise ValueError('Liquid native receipt counters are bounded own commands')
    if receipt['actions_executed'] != actions or receipt['endpoint_settle_commands'] != 16 * receipt['endpoint_settle_intervals']:
        raise ValueError('Liquid receipt must match actually issued commands and original settling intervals')
    if status == 'completed' and receipt['sampled_native_controls'] + receipt['endpoint_settle_commands'] != actions:
        raise ValueError('A completed fixed-step tool executes every prepared native sample exactly once')
    review = value['public_liquid_review']
    fields = {'status', 'reason', 'source', 'gripper_is_contact_sensor', 'object_grasp_confirmed', 'official_success_claimed'}
    if not isinstance(review, dict) or set(review) != fields or review['status'] != status or (review['reason'] not in REASONS) or (review['source'] != 'current_public_rgb_and_own_command_response') or any((review[name] is not False for name in ('gripper_is_contact_sensor', 'object_grasp_confirmed', 'official_success_claimed'))):
        raise ValueError('Public liquid review is not a real finger, contact or task-success sensor')
    return extra
