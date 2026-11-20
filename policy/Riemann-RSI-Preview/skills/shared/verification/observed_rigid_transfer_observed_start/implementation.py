from copy import deepcopy
import hashlib
import math
import numpy as np
from skills.shared.perception.public_support_geometry_context_observed_start.implementation import validate as validate_support
PHASES = ('clear', 'view', 'pregrasp', 'contact', 'close', 'lift', 'carry', 'align', 'place', 'release', 'retract', 'verify', 'reset', 'home', 'finish', 'receiver_lift', 'actuate', 'turn', 'thread', 'recipe')
KINDS = ('tool_storage', 'nested_stack', 'ring_stack', 'holder_fill', 'colour_sort', 'box_pack', 'size_row', 'table_order', 'conveyor_match', 'image_match', 'axial_insert', 'handle_hang', 'toast_insert', 'screw_thread', 'imitated_sort')
GOAL_FIELDS = {'kind', 'sources', 'receiver', 'receiver_kind', 'ordering', 'matching', 'own_forward_m', 'close_commands', 'release_commands', 'release_strategy', 'after_release', 'axis_alignment', 'source_side_rule'}
GOAL_OPTIONAL = {'close_settle_commands': 0, 'release_settle_commands': 0, 'pre_release_hold_commands': 0, 'gripper_schedule': 'sampled_ramp'}

def vector(value, size, maximum=4.0):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or np.max(abs(result)) > maximum:
        raise ValueError('A transfer record contains only finite bounded public numeric evidence')
    return result

def pose(value):
    result = vector(value, 7)
    if not 0.99 <= np.linalg.norm(result[3:]) <= 1.01:
        raise ValueError('A public reference or own target needs a unit quaternion')
    return result

def goal(value):
    if not isinstance(value, dict) or not GOAL_FIELDS.issubset(value) or set(value) - (GOAL_FIELDS | set(GOAL_OPTIONAL)) or (value['kind'] not in KINDS) or (not isinstance(value['sources'], list)) or (not 1 <= len(value['sources']) <= 8) or any((not isinstance(item, str) or not 1 <= len(item) <= 100 or any((ord(character) < 32 for character in item)) for item in value['sources'])) or (not isinstance(value['receiver'], str)) or (not 1 <= len(value['receiver']) <= 100) or (value['receiver_kind'] not in ('compartments', 'container', 'row', 'stack', 'holes', 'peg', 'slot', 'handle_target')) or (value['ordering'] not in ('given', 'descending_width', 'ascending_width', 'source_x', 'demonstrated')) or (value['matching'] not in ('none', 'colour', 'current_image', 'current_shape')) or (value['own_forward_m'] not in (0.145, 0.155)) or any((type(value[name]) is not int or not 1 <= value[name] <= 40 for name in ('close_commands', 'release_commands'))) or (value['release_strategy'] not in ('dense', 'gradual', 'stationary')) or (value['after_release'] not in ('next', 'raise_receiver', 'press_actuator', 'turn_key', 'hold_inserted', 'thread')) or (value['axis_alignment'] not in ('none', 'major', 'plane_normal', 'hole_normal')) or (value['source_side_rule'] not in ('nearest', 'left', 'right', 'semantic_tools')):
        raise ValueError('A transfer goal declares only semantic roles and reviewed own-command settings')
    options = {name: value.get(name, default) for name, default in GOAL_OPTIONAL.items()}
    if options['gripper_schedule'] not in ('sampled_ramp', 'settled_constant') or any((type(options[name]) is not int or not 0 <= options[name] <= 16 for name in ('close_settle_commands', 'release_settle_commands', 'pre_release_hold_commands'))) or value['close_commands'] + options['close_settle_commands'] > 40 or ((26 if value['release_strategy'] == 'gradual' else value['release_commands']) + options['pre_release_hold_commands'] + options['release_settle_commands'] > 40):
        raise ValueError('Own close and release settle clocks are explicit, bounded and already sampled')
    return deepcopy(dict(value, **options))

def anchor(value):
    fields = {'source', 'description', 'center_m', 'spans_m', 'height_estimate_m', 'uncertainty_m', 'cameras', 'coverage', 'head_outline', 'head_centroid', 'appearance_rgb', 'reference_pose', 'orientation_is_chosen_geometry_gauge', 'surface_is_true_object_geometry', 'resolution_semantics'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] not in ('current_public_rgb_visual_hull', 'current_public_rgb_tracked_reference') or (value['surface_is_true_object_geometry'] is not False) or (value['orientation_is_chosen_geometry_gauge'] is not True) or (value['resolution_semantics'] != ('grid_and_support_resolution_not_hidden_object_error' if value['source'] == 'current_public_rgb_visual_hull' else 'previous_visual_hull_plus_public_optical_tracking_not_hidden_object_error')):
        raise ValueError('A retained instance must remain conservative current public geometry')
    for name in ('center_m', 'spans_m', 'appearance_rgb'):
        vector(value[name], 3, 255.0 if name == 'appearance_rgb' else 4.0)
    if np.any(np.asarray(value['spans_m']) <= 0):
        raise ValueError('Current surface spans are positive observed estimates')
    pose(value['reference_pose'])
    cameras = value['cameras']
    if not isinstance(cameras, list) or not 2 <= len(cameras) <= 3 or len(set(cameras)) != len(cameras) or any((camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist') for camera in cameras)) or (not isinstance(value['coverage'], list)) or (len(value['coverage']) != len(cameras)) or any((type(item) not in (int, float) or not 0 <= item <= 1 for item in value['coverage'])):
        raise ValueError('A current instance requires independent actual camera provenance')
    outline = np.asarray(value['head_outline'], dtype=float)
    if outline.shape != (96, 2) or not np.isfinite(outline).all() or np.any(outline <= 1) or np.any(outline[:, 0] >= 638) or np.any(outline[:, 1] >= 478):
        raise ValueError('A current head outline cannot contain clipping or a stored layout')
    vector(value['head_centroid'], 2, 640.0)
    for name in ('height_estimate_m', 'uncertainty_m'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or (not 0 < value[name] <= 0.5):
            raise ValueError('A current estimate has finite bounded, explicitly non-hidden uncertainty')
    if not isinstance(value['description'], str) or not 1 <= len(value['description']) <= 160:
        raise ValueError('A source description is bounded semantic text')
    return deepcopy(value)

def scene(value):
    fields = {'source_anchor', 'source_reference', 'target_reference', 'target_outline', 'support_memory', 'side', 'route', 'matching_confidence', 'observation_epoch'}
    if not isinstance(value, dict) or set(value) != fields or value['side'] not in ('left', 'right'):
        raise ValueError('A transfer scene contains only qualified public references and own targets')
    anchor(value['source_anchor'])
    for name in ('source_reference', 'target_reference'):
        pose(value[name])
    outline = np.asarray(value['target_outline'], dtype=float)
    if outline.shape != (96, 2) or not np.isfinite(outline).all() or np.max(abs(outline)) > 640:
        raise ValueError('A target outline is retained current image evidence')
    validate_support(value['support_memory'])
    expected = {'pregrasp', 'contact', 'lift', 'carry', 'align', 'place', 'retract'}
    if not isinstance(value['route'], dict) or set(value['route']) != expected:
        raise ValueError('A route contains the actual separately executable own stages')
    for target in value['route'].values():
        pose(target)
    if type(value['matching_confidence']) not in (int, float) or not 0 <= value['matching_confidence'] <= 1 or type(value['observation_epoch']) is not int or (value['observation_epoch'] < 0):
        raise ValueError('Current match confidence and own-command epoch are bounded')
    return deepcopy(value)

def initial(instruction):
    return {'schema': 'riemann-public-rigid-transfer-v3', 'instruction_fingerprint': hashlib.sha256(instruction.encode()).hexdigest(), 'goal': None, 'phase': 'clear', 'view_leg': 0, 'item': 0, 'attempts': 0, 'scene': None, 'completed_descriptions': [], 'visited_targets': [], 'last_source_epoch': None, 'home_leg': 0, 'planned_sources': [], 'planned_targets': [], 'observed_source_count': 0, 'recipe': None, 'recipe_index': 0, 'job': initial_job(), 'support_memory': None, 'view_attempts': 0, 'view_hint': None}

def initial_job():
    from skills.shared.verification.public_action_recipe_observed_start.implementation import initial_job as create
    return create()

def validate(value):
    if not isinstance(value, dict) or set(value) != set(initial('')) or value['schema'] != 'riemann-public-rigid-transfer-v3' or (not isinstance(value['instruction_fingerprint'], str)) or (len(value['instruction_fingerprint']) != 64) or any((character not in '0123456789abcdef' for character in value['instruction_fingerprint'])) or (value['phase'] not in PHASES):
        raise ValueError('Rigid transfer progress has a closed same-instruction public schema')
    for name, maximum in (('view_leg', 2), ('item', 32), ('attempts', 3), ('home_leg', 2), ('observed_source_count', 32), ('view_attempts', 4)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('Own stage and retry counters remain bounded')
    if value['goal'] is not None:
        goal(value['goal'])
    if value['scene'] is not None:
        scene(value['scene'])
    if value['support_memory'] is not None:
        validate_support(value['support_memory'])
    if value['view_hint'] is not None:
        hint = value['view_hint']
        if not isinstance(hint, dict) or set(hint) != {'source', 'description', 'head_centroid', 'unqualified_cameras'} or hint['source'] != 'current_public_head_semantic_hint' or (not isinstance(hint['description'], str)) or (not 1 <= len(hint['description']) <= 128) or any((ord(character) < 32 for character in hint['description'])):
            raise ValueError('View hints contain only a bounded currently observed semantic mask')
        pixels = vector(hint['head_centroid'], 2, 640.0)
        cameras = hint['unqualified_cameras']
        if not 0 <= pixels[0] < 640 or not 0 <= pixels[1] < 480 or (not isinstance(cameras, list)) or (not 1 <= len(cameras) <= 2) or (len(set(cameras)) != len(cameras)) or any((camera not in ('cam_left_wrist', 'cam_right_wrist') for camera in cameras)):
            raise ValueError('View hints use only current head pixels and declared wrist camera names')
    from skills.shared.verification.public_action_recipe_observed_start.implementation import job, recipe
    job(value['job'])
    recipe(value['recipe'])
    if type(value['recipe_index']) is not int or not 0 <= value['recipe_index'] <= 48 or (value['recipe'] is None and value['recipe_index'] != 0) or (value['recipe'] is not None and value['recipe_index'] > len(value['recipe']['steps'])):
        raise ValueError('Recipe progress is an acknowledged bounded own stage index')
    if not isinstance(value['planned_sources'], list) or len(value['planned_sources']) > 32 or (not isinstance(value['planned_targets'], list)) or (len(value['planned_targets']) > 32):
        raise ValueError('Planned arrangements retain only bounded previously observed public references')
    for item in value['planned_sources']:
        anchor(item)
    for item in value['planned_targets']:
        pose(item)
    if not isinstance(value['completed_descriptions'], list) or len(value['completed_descriptions']) > 32 or any((not isinstance(item, str) or not 1 <= len(item) <= 160 for item in value['completed_descriptions'])) or (not isinstance(value['visited_targets'], list)) or (len(value['visited_targets']) > 32):
        raise ValueError('Retain only bounded own placement history, never an external task ledger')
    for item in value['visited_targets']:
        pose(item)
    if value['last_source_epoch'] is not None:
        record = value['last_source_epoch']
        if not isinstance(record, dict) or set(record) != {'center_m', 'own_command_count'} or type(record['own_command_count']) is not int or (record['own_command_count'] < 0):
            raise ValueError('Source motion estimates use only a previous public center and own command count')
        vector(record['center_m'], 3)
    return deepcopy(value)

def recorded(memory, instruction):
    current = initial(instruction)
    for event in reversed(memory[-32:]):
        saved = event.get('execution_feedback', {}).get('image_measurements', {}).get('public_rigid_transfer_state')
        if saved is not None:
            checked = validate(saved)
            return checked if checked['instruction_fingerprint'] == current['instruction_fingerprint'] else current
    return current

def bowl_scene_diagnostics(records):
    """Validate at most two current public bowl admission records; return None.

    Input: scalar-only records emitted before/after one bowl view invocation.
    Output: no value; reject unknown stages/reasons, private payloads and invalid counts.
    These diagnostics do not contain geometry, command state or success evidence.
    """
    reasons = {'support': {'current_support_unqualified'}, 'collection': {'initial_three_bowls_unqualified'}, 'receiver': {'current_receiver_unqualified'}, 'opening': {'current_receiver_opening_unqualified'}, 'source': {'current_source_unqualified'}, 'waypoints': {'current_grasp_waypoints_unqualified', 'current_scene_qualified'}, 'complete': {'all_planned_sources_dispatched'}, 'view_motion': {'zero_action_view_plan_failed'}}
    fields = {'stage', 'reason', 'public_action_epoch', 'qualified_bowl_count', 'view_hint_available'}
    if not isinstance(records, list) or not 1 <= len(records) <= 2:
        raise ValueError('A bowl view has one or two current admission records')
    for record in records:
        if not isinstance(record, dict) or set(record) != fields or (not isinstance(record['stage'], str)) or (record['stage'] not in reasons) or (not isinstance(record['reason'], str)) or (record['reason'] not in reasons[record['stage']]) or (type(record['public_action_epoch']) is not int) or (record['public_action_epoch'] < 0) or (type(record['view_hint_available']) is not bool):
            raise ValueError('Bowl diagnostics use only fixed public stage/reason and scalar evidence')
        count = record['qualified_bowl_count']
        if record['stage'] == 'support' and count is not None or (record['stage'] != 'support' and (type(count) is not int or not 0 <= count <= 12)):
            raise ValueError('Bowl candidate counts are current bounded measurements or unknown before support')
    if any((right['public_action_epoch'] < left['public_action_epoch'] for left, right in zip(records, records[1:]))):
        raise ValueError('Bowl admission records follow actual public action order')

def measurements(value, status, action_count):
    required = {'public_rigid_transfer_state', 'rigid_visual_review', 'rigid_command_phases_completed'}
    if not required.issubset(value):
        raise ValueError('A rigid stage returns its complete closed public state and execution receipt')
    validate(value['public_rigid_transfer_state'])
    review = value['rigid_visual_review']
    if not isinstance(review, dict) or set(review) != {'status', 'position_error_m', 'colour_difference', 'is_contact_measurement'} or review['status'] not in ('not_reviewed', 'uncertain', 'current_surface_follows', 'current_image_placement') or (review['is_contact_measurement'] is not False) or any((type(review[name]) not in (int, float) or not math.isfinite(review[name]) or (not 0 <= review[name] <= maximum) for name, maximum in (('position_error_m', 4.0), ('colour_difference', 450.0)))):
        raise ValueError('Visual following and placement are uncertain public evidence, not true contact or success')
    if type(value['rigid_command_phases_completed']) is not bool or (value['rigid_command_phases_completed'] and (status not in ('completed', 'uncertain') or action_count < 1)):
        raise ValueError('A completed transfer phase requires actual native command acknowledgments')
    extras = set(value) & required
    if 'bowl_scene_diagnostics' in value:
        bowl_scene_diagnostics(value['bowl_scene_diagnostics'])
        extras.add('bowl_scene_diagnostics')
    for optional in ('own_feedback_motion', 'rigid_motion_execution', 'normalized_command_clock'):
        if optional in value:
            record = value[optional]
            if not isinstance(record, dict):
                raise ValueError('An own execution receipt is a closed scalar record')
            if optional == 'own_feedback_motion':
                fields = {'source', 'phase', 'native_actions_issued', 'maximum_native_actions', 'inhibited_observations', 'termination_reason', 'original_increment_resampled', 'contact_measurement_claimed', 'true_finger_measurement_used', 'official_success_claimed'}
                if set(record) != fields or record['source'] != 'current_public_own_pose_joint_feedback_and_declared_own_fk' or record['phase'] not in ('approach', 'contact', 'lift', 'transfer', 'preplace', 'grasp_adjust', 'retreat') or any((record[name] is not False for name in ('original_increment_resampled', 'contact_measurement_claimed', 'true_finger_measurement_used', 'official_success_claimed'))) or (record['native_actions_issued'] != action_count) or (type(record['maximum_native_actions']) is not int) or (not 1 <= record['maximum_native_actions'] <= 240) or (type(record['inhibited_observations']) is not int) or (not 0 <= record['inhibited_observations'] <= 240) or (record['termination_reason'] is not None and (not isinstance(record['termination_reason'], str) or len(record['termination_reason']) > 100)):
                    raise ValueError('Own feedback does not become a private contact sensor')
            elif optional == 'rigid_motion_execution':
                fields = {'backend', 'phase', 'native_actions_issued', 'maximum_native_actions', 'trajectory_completed', 'micro_ik_fallback_used', 'contact_measurement_claimed', 'true_finger_measurement_used', 'official_success_claimed'}
                if set(record) != fields or record['backend'] != 'declared_public_own_curobo_trajectory' or record['phase'] not in ('approach', 'contact', 'lift', 'transfer', 'preplace', 'grasp_adjust', 'retreat') or (record['native_actions_issued'] != action_count) or (type(record['maximum_native_actions']) is not int) or (not 1 <= record['maximum_native_actions'] <= 320) or (type(record['trajectory_completed']) is not bool) or (record['trajectory_completed'] != (status == 'completed' and action_count > 0)) or any((record[name] is not False for name in ('micro_ik_fallback_used', 'contact_measurement_claimed', 'true_finger_measurement_used', 'official_success_claimed'))):
                    raise ValueError('A declared trajectory receipt records own execution, not private contact or task success')
            else:
                fields = {'prepared_native_commands', 'issued_native_commands', 'segment_end_indices', 'command_source', 'sampling_reapplied', 'arm_trajectory_replanned', 'held_joints_source', 'true_fingers_or_contact_read', 'official_success_claimed'}
                if set(record) != fields or type(record['prepared_native_commands']) is not int or (not 1 <= record['prepared_native_commands'] <= 40) or (record['issued_native_commands'] != action_count) or (not isinstance(record['segment_end_indices'], list)) or (not record['segment_end_indices']) or any((type(index) is not int for index in record['segment_end_indices'])) or (record['segment_end_indices'] != sorted(set(record['segment_end_indices']))) or (record['segment_end_indices'][0] < 1) or (record['segment_end_indices'][-1] != record['prepared_native_commands']) or (record['command_source'] != 'explicit_already_sampled_own_normalized_commands') or (record['held_joints_source'] != 'last_own_command_or_initial_public_joints') or any((record[name] is not False for name in ('sampling_reapplied', 'arm_trajectory_replanned', 'true_fingers_or_contact_read', 'official_success_claimed'))):
                    raise ValueError('A normalized hold retains its exact declared command receipt')
            extras.add(optional)
    return extras
