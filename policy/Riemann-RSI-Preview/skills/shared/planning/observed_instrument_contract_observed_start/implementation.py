from copy import deepcopy
import math
import numpy as np
from skills.shared.motion.mature_feedback_execution_observed_start.implementation import validate_receipt
from skills.shared.motion.rigid_tool_command_tracking_observed_start.implementation import checked_pose
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

def vector(value, size, maximum=3.0):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or np.max(np.abs(result)) > maximum:
        raise ValueError('An instrument command retains only finite bounded public measurements')
    return result

def cameras(value):
    if not isinstance(value, list) or not 2 <= len(value) <= 3 or len(set(value)) != len(value) or any((camera not in CAMERAS for camera in value)):
        raise ValueError('Metric geometry must identify its independent current public cameras')
    return list(value)

def validate_frame(value):
    fields = {'schema', 'pose', 'contact_in_tool', 'radius_m', 'shaft_line_offset_in_tool', 'observed_shaft_intervals', 'cameras', 'cap_cameras', 'shaft_cameras', 'orientation_is_chosen_symmetric_tool_gauge', 'is_true_object_pose', 'is_contact_measurement', 'unseen_shaft_extent_filled'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != 'riemann-public-spherical-tool-frame-v1' or (value['orientation_is_chosen_symmetric_tool_gauge'] is not True) or any((value[name] is not False for name in ('is_true_object_pose', 'is_contact_measurement', 'unseen_shaft_extent_filled'))):
        raise ValueError('A tool frame cannot carry asset pose, contact or filled-in shaft geometry')
    checked_pose(value['pose'])
    radius = value['radius_m']
    if type(radius) not in (int, float) or not math.isfinite(radius) or (not 0.001 <= radius <= 0.15):
        raise ValueError('The cap radius must be independently measured from its public silhouette')
    contact = vector(value['contact_in_tool'], 3)
    if not np.allclose(contact, [0.0, 0.0, -radius], atol=1e-08, rtol=0):
        raise ValueError('A chosen symmetric contact reference must match the measured cap radius')
    offset = vector(value['shaft_line_offset_in_tool'], 3)
    if np.linalg.norm(offset) > 0.008 or abs(float(offset[1])) > 1e-06:
        raise ValueError('A grasp line preserves only the measured perpendicular shaft offset')
    views, cap_views, shaft_views = [cameras(value[name]) for name in ('cameras', 'cap_cameras', 'shaft_cameras')]
    intervals = value['observed_shaft_intervals']
    if not isinstance(intervals, dict) or set(intervals) != set(shaft_views) or set(views) != set(cap_views) | set(shaft_views):
        raise ValueError('Independent cap and shaft observations cannot be relabelled as the same camera evidence')
    for interval in intervals.values():
        bounds = vector(interval, 2)
        if not 0 < bounds[0] < bounds[1] <= 0.5:
            raise ValueError('Only actually visible continuous shaft intervals authorize a grasp bias')
    return deepcopy(value)

def validate_bars(value):
    fields = {'surface_xyz', 'major_axis_xyz', 'visible_marker_span_m', 'hue', 'current_head_anchor_image', 'cameras', 'maximum_ray_separation_m', 'is_true_object_pose'}
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError('The eight strike targets come from a complete current independent bar family')
    previous = -1.0
    for bar in value:
        if not isinstance(bar, dict) or set(bar) != fields or bar['is_true_object_pose'] is not False:
            raise ValueError('A strike target is current public surface evidence, not an object pose')
        vector(bar['surface_xyz'], 3)
        axis = vector(bar['major_axis_xyz'], 3)
        pixel = vector(bar['current_head_anchor_image'], 2, 640.0)
        if not 0.99 <= np.linalg.norm(axis) <= 1.01 or not 0 <= pixel[0] < 640 or (not 0 <= pixel[1] < 480) or (pixel[0] <= previous) or (type(bar['hue']) not in (int, float)) or (not 0 <= bar['hue'] < 180) or (type(bar['visible_marker_span_m']) not in (int, float)) or (not 0.008 <= bar['visible_marker_span_m'] <= 0.45) or (type(bar['maximum_ray_separation_m']) not in (int, float)) or (not 0 <= bar['maximum_ray_separation_m'] <= 0.004):
            raise ValueError('Targets retain independent marker geometry and actual head left-to-right ordering')
        cameras(bar['cameras'])
        previous = float(pixel[0])
    return deepcopy(value)

def validate_anchor(value):
    fields = {'schema', 'tool_frame', 'bars', 'initial_cap_image', 'is_ground_truth', 'official_success_claimed'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != 'riemann-public-instrument-anchor-v1' or (value['is_ground_truth'] is not False) or (value['official_success_claimed'] is not False):
        raise ValueError('An instrument anchor is only same-episode public image geometry')
    validate_frame(value['tool_frame'])
    validate_bars(value['bars'])
    pixel = vector(value['initial_cap_image'], 2, 640.0)
    if not 0 <= pixel[0] < 640 or not 0 <= pixel[1] < 480:
        raise ValueError('The initial cap reference is an actual current public head pixel')
    return deepcopy(value)

def empty_progress():
    return {'stage': 0, 'cursor': 0, 'attempt': 0}

def validate_progress(value):
    if not isinstance(value, dict) or set(value) != {'stage', 'cursor', 'attempt'} or any((type(value[name]) is not int for name in value)) or (not 0 <= value['stage'] <= 4) or (not 0 <= value['cursor'] <= 8) or (not 0 <= value['attempt'] <= 6) or ((value['stage'] == 4) != (value['cursor'] == 8)) or (value['stage'] < 3 and value['cursor'] != 0):
        raise ValueError('Instrument progress records bounded own commands, never native task success')
    return deepcopy(value)

def validate_route(value):
    fields = {'side', 'approach', 'grasp', 'working_orientation', 'planned_attachment', 'attachment', 'attachment_before_refine', 'motion_time_scale', 'handle_offset_m', 'clearance_m', 'fallback'}
    if not isinstance(value, dict) or set(value) != fields or value['side'] not in ('left', 'right') or (type(value['fallback']) is not bool):
        raise ValueError('A route is one finite mature command on public tool and own robot geometry')
    for name in ('approach', 'grasp', 'planned_attachment', 'attachment', 'attachment_before_refine'):
        checked_pose(value[name])
    orientation = vector(value['working_orientation'], 4)
    if not 0.99 <= np.linalg.norm(orientation) <= 1.01:
        raise ValueError('The working orientation is a normalized own command')
    if value['motion_time_scale'] not in (1.0, 1.4) or type(value['motion_time_scale']) not in (int, float) or type(value['handle_offset_m']) not in (int, float) or (not 0 < value['handle_offset_m'] <= 0.5) or (type(value['clearance_m']) not in (int, float)) or (value['clearance_m'] not in (0.075, 0.04, 0.06)):
        raise ValueError('The route preserves reviewed primary and secondary mature execution settings')
    return deepcopy(value)

def validate_attachment_evidence(value):
    fields = {'schema', 'status', 'initial_displacement_pixels', 'predicted_error_pixels', 'current_cap_image', 'translation_correction_m', 'shaft_contrast_score', 'attachment', 'public_gripper_response_interval', 'is_contact_sensor', 'is_true_object_pose', 'official_success_claimed'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != 'riemann-public-instrument-attachment-belief-v1' or (value['status'] not in ('consistent_public_attachment_belief', 'uncertain', 'unavailable')) or any((value[name] is not False for name in ('is_contact_sensor', 'is_true_object_pose', 'official_success_claimed'))):
        raise ValueError('Attachment feedback is a public prediction consistency check, not real grasp or contact')
    for name in ('initial_displacement_pixels', 'predicted_error_pixels', 'translation_correction_m'):
        maximum = 1.0 if name.endswith('_m') else 800.0
        if value[name] is not None and (type(value[name]) not in (int, float) or not math.isfinite(value[name]) or (not 0 <= value[name] <= maximum)):
            raise ValueError('Image feedback retains finite actual public differences')
    if value['current_cap_image'] is not None:
        pixel = vector(value['current_cap_image'], 2, 640.0)
        if not 0 <= pixel[0] < 640 or not 0 <= pixel[1] < 480:
            raise ValueError('The cap consistency reference is an actually visible current public pixel')
    if value['attachment'] is not None:
        checked_pose(value['attachment'])
    interval = vector(value['public_gripper_response_interval'], 2, 1.0)
    if not 0 <= interval[0] <= interval[1] <= 1 or type(value['shaft_contrast_score']) not in (int, float) or (not math.isfinite(value['shaft_contrast_score'])) or (not 0 <= value['shaft_contrast_score'] <= 150):
        raise ValueError('The normalized gripper response interval is not a true finger-joint measurement')
    return deepcopy(value)

def validate_measurements(measurements, status, actions):
    declared = {'public_instrument_anchor', 'public_instrument_progress', 'public_instrument_route', 'command_phases_completed', 'mature_execution_receipt', 'instrument_attachment_evidence', 'gripper_signal_semantics', 'public_instrument_partial_bars'}
    if 'gripper_signal_semantics' in measurements and measurements['gripper_signal_semantics'] != 'normalized_public_command_response_not_real_finger_width_or_contact':
        raise ValueError('A public gripper response is not finger width or contact feedback')
    if 'mature_execution_receipt' in measurements:
        receipt = validate_receipt(measurements['mature_execution_receipt'])
        if receipt['official_action_acknowledgments'] != actions:
            raise ValueError('Execution receipts must account for exactly the issued official own actions')
    if 'public_instrument_anchor' in measurements:
        validate_anchor(measurements['public_instrument_anchor'])
        validate_progress(measurements['public_instrument_progress'])
    if 'public_instrument_partial_bars' in measurements:
        validate_bars(measurements['public_instrument_partial_bars'])
    if 'public_instrument_route' in measurements:
        validate_route(measurements['public_instrument_route'])
    if 'command_phases_completed' in measurements:
        completed = measurements['command_phases_completed']
        if type(completed) is not bool or (completed and (actions <= 0 or status not in ('completed', 'uncertain'))):
            raise ValueError('Completed phases describe actual finished own commands only')
    if 'instrument_attachment_evidence' in measurements:
        validate_attachment_evidence(measurements['instrument_attachment_evidence'])
    return set(measurements) & declared
