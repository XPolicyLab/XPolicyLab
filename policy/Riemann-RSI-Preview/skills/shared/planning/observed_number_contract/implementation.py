from copy import deepcopy
import math

import numpy as np

from skills.shared.planning.descending_number_assignment.implementation import run as assign_numbers


STAGE_COUNT = 5


def validate_anchor(anchor):
    required = {'schema', 'digits', 'targets', 'support_reference_m', 'mounting_reference',
                'target_geometry_source', 'observed_target_count', 'current_layout_source'}
    if (not isinstance(anchor, dict) or set(anchor) != required
            or anchor['schema'] != 'riemann-current-number-scene-v1' or anchor['current_layout_source'] is not False
            or anchor['target_geometry_source'] not in ('independent_current_RGB_circular_surface_family',
                                                       'same_episode_public_RGB_circular_surface_family')
            or not 2 <= len(anchor['digits']) == len(anchor['targets']) <= 8
            or anchor['observed_target_count'] != len(anchor['targets'])
            or not math.isfinite(anchor['support_reference_m'])):
        raise ValueError('A number anchor contains only complete independent current public observations')
    sources = []
    reference = anchor['mounting_reference']
    reference_fields = {'status', 'height_m', 'source', 'semantics', 'object_support_requires_task_contract'}
    if (not isinstance(reference, dict) or set(reference) != reference_fields or reference['status'] != 'observed'
            or reference['source'] != 'current_own_robot_pose_and_declared_URDF'
            or reference['semantics'] != 'robot_mounting_reference_not_an_object_or_table_measurement'
            or reference['object_support_requires_task_contract'] is not True
            or abs(reference['height_m'] - anchor['support_reference_m']) > 1e-6):
        raise ValueError('A mounting reference cannot conceal an object or scene height query')
    allowed = {'digit', 'surface_xyz', 'estimated_grasp_span_m', 'yaw_radians', 'maximum_ray_separation_m',
               'cameras', 'height_semantics', 'current_relocalization_source', 'current_relocalization_cameras',
               'public_feature_height_interval_m'}
    required_source = allowed - {'current_relocalization_source', 'current_relocalization_cameras',
                                'public_feature_height_interval_m'}
    for source in anchor['digits']:
        if (not isinstance(source, dict) or not required_source.issubset(source) or not set(source).issubset(allowed)
                or type(source.get('estimated_grasp_span_m')) not in (int, float)
                or not 0. < source['estimated_grasp_span_m'] <= .08
                or not math.isfinite(source.get('yaw_radians', float('nan')))
                or not 0 <= source.get('maximum_ray_separation_m', float('inf')) <= .004
                or not 2 <= len(set(source.get('cameras', []))) <= 3
                or not set(source['cameras']).issubset({'cam_head', 'cam_left_wrist', 'cam_right_wrist'})
                or source['height_semantics'] not in ('current_visual_feature_above_public_mount_reference_not_asset_height',
                                                    'public_static_scene_multi_view_correspondence_not_asset_height')
                or source.get('current_relocalization_source', 'independent_current_head_label_and_remembered_public_surface')
                    not in ('independent_current_head_label_and_remembered_public_surface',
                            'independent_current_public_camera_label_and_remembered_public_surface')):
            raise ValueError('A numeral source cannot contain private pose, size, identity or layout fields')
        if 'current_relocalization_cameras' in source:
            cameras = source['current_relocalization_cameras']
            if (not isinstance(cameras, list) or not 1 <= len(cameras) == len(set(cameras)) <= 3
                    or not set(cameras).issubset({'cam_head', 'cam_left_wrist', 'cam_right_wrist'})):
                raise ValueError('Current relocalization names only actual delivered public cameras')
        point = np.asarray(source['surface_xyz'], dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 3
                or abs(point[2] - anchor['support_reference_m'] - source['estimated_grasp_span_m']) > 1e-6):
            raise ValueError('The source grasp span must remain tied to its public reference hypothesis')
        if 'public_feature_height_interval_m' in source:
            interval = source['public_feature_height_interval_m']
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
                    or not anchor['support_reference_m'] < interval[0] <= point[2] <= interval[1]
                        <= anchor['support_reference_m'] + .08):
                raise ValueError('A current visual feature interval must remain above its own public reference')
        elif source['estimated_grasp_span_m'] < .006:
            raise ValueError('A small visual feature needs independently propagated current pixel uncertainty')
        sources.append({'digit': source['digit'], 'surface_xyz': source['surface_xyz']})
    assign_numbers(sources, anchor['targets'])
    return deepcopy(anchor)


def empty_progress(count=0):
    return {'cursor': 0, 'stage': 0, 'count': count, 'final_stage': 0, 'retry': 0}


def validate_progress(progress, count):
    if (not isinstance(progress, dict) or set(progress) != {'cursor', 'stage', 'count', 'final_stage', 'retry'}
            or any(type(value) is not int for value in progress.values()) or progress['count'] != count
            or not 0 <= progress['cursor'] <= count or not 0 <= progress['stage'] < STAGE_COUNT
            or not 0 <= progress['final_stage'] <= 2 or not 0 <= progress['retry'] <= 1
            or progress['cursor'] == count and progress['stage'] != 0):
        raise ValueError('Number progress is bounded own-command state, never official success')
    return deepcopy(progress)


def advance(progress):
    result = deepcopy(progress)
    if result['stage'] == STAGE_COUNT - 1:
        result['cursor'] += 1
        result['stage'], result['retry'] = 0, 0
    else:
        result['stage'] += 1
    return validate_progress(result, result['count'])


def validate_route(route):
    fields = {'approach', 'grasp', 'lift', 'carry', 'place', 'source_tip', 'target_tip',
              'side', 'digit', 'closing_axis', 'tilt', 'branch_score', 'joint_limit_margin_rad'}
    if not isinstance(route, dict) or set(route) != fields or route['side'] not in ('left', 'right'):
        raise ValueError('An own number route has only bounded declared public fields')
    if type(route['digit']) is not int or not 0 <= route['digit'] <= 9:
        raise ValueError('A route cannot invent an identity outside the independently recognized vocabulary')
    for name in ('approach', 'grasp', 'lift', 'carry', 'place', 'source_tip', 'target_tip'):
        values = np.asarray(route[name], dtype=float)
        size = 3 if name.endswith('_tip') else 7
        if values.shape != (size,) or not np.isfinite(values).all() or np.max(np.abs(values[:3])) > 3:
            raise ValueError('An own route needs finite bounded public poses')
        if size == 7 and not .99 <= np.linalg.norm(values[3:]) <= 1.01:
            raise ValueError('A route orientation is not a public unit quaternion')
    for name in ('closing_axis', 'tilt', 'branch_score', 'joint_limit_margin_rad'):
        if not math.isfinite(route[name]):
            raise ValueError('An own branch estimate must be finite')
    if (route['closing_axis'] not in (0., math.pi, math.pi / 2, -math.pi / 2)
            or route['tilt'] not in (0., 5., 10., 15., 20., 25.) or route['joint_limit_margin_rad'] < .035):
        raise ValueError('A number route must retain declared mature candidate choices')
    return deepcopy(route)


def validate_measurements(measurements, status, actions):
    """Validate bounded public number feedback and own command receipts; return no value."""
    if 'public_number_grasp_precondition' in measurements:
        reason = measurements['public_number_grasp_precondition']
        if reason not in ('qualified', 'no_single_arm_route', 'current_source_unqualified', 'requested_side_unqualified'):
            raise ValueError('A grasp rejection reason names only its actual public precondition')
        if reason == 'no_single_arm_route' and 'current_source_identity_before_grasp' not in measurements:
            raise ValueError('A missing route cannot pretend that the source was freshly recognized')
    if 'public_number_relay' in measurements:
        relay = measurements['public_number_relay']
        fields = {'source_side', 'receiver_side', 'site', 'receiver_numeric_route_checked',
                  'relocalized_after_release', 'is_original_destination', 'cursor_advanced', 'official_success_claimed'}
        if (not isinstance(relay, dict) or not fields <= set(relay)
                or not set(relay) <= fields | {'current_visual_feedback'}
                or relay['source_side'] not in ('left', 'right') or relay['receiver_side'] not in ('left', 'right')
                or relay['source_side'] == relay['receiver_side'] or relay['receiver_numeric_route_checked'] is not True
                or any(relay[name] is not False for name in ('is_original_destination', 'cursor_advanced', 'official_success_claimed'))
                or type(relay['relocalized_after_release']) is not bool
                or status == 'completed' and (not relay['relocalized_after_release'] or actions <= 0)):
            raise ValueError('A relay contains only checked own commands and acknowledged current source relocalization')
        site = relay['site']
        site_fields = {'surface_xyz', 'visible_footprint_radius_m', 'occupied_clearance_m',
                       'current_RGB_appearance_agreement', 'current_RGB_material_fraction', 'physical_collision_free_claimed'}
        if (not isinstance(site, dict) or set(site) != site_fields or site['physical_collision_free_claimed'] is not False):
            raise ValueError('An intermediary site is a current RGB surface hypothesis, not a privileged collision result')
        point = np.asarray(site['surface_xyz'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 3:
            raise ValueError('Relay coordinates must be finite bounded current public measurements')
        bounds = {'visible_footprint_radius_m': (0., .1), 'occupied_clearance_m': (0., 3.),
                  'current_RGB_appearance_agreement': (.9, 1.), 'current_RGB_material_fraction': (0., .03)}
        for name, (lower, upper) in bounds.items():
            value = site[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('An intermediary site requires bounded current visible-footprint evidence')
        if relay['relocalized_after_release']:
            if 'current_visual_feedback' not in relay or relay['current_visual_feedback'].get('visual_placement_confirmed') is not True:
                raise ValueError('A predicted relay location is not an independently confirmed current source')
        if 'current_visual_feedback' in relay:
            validate_measurements({'current_visual_placement_feedback': relay['current_visual_feedback'],
                'gripper_signal_semantics': 'normalized_public_command_response_not_real_finger_width_or_contact'}, status, actions)
    timing = {'number_placement_read_before_observer_reset', 'number_placement_observation_actions',
              'number_observer_return_side'}
    if timing.intersection(measurements):
        if (not timing.issubset(measurements)
                or measurements['number_placement_read_before_observer_reset'] is not True
                or type(measurements['number_placement_observation_actions']) is not int
                or not 1 <= measurements['number_placement_observation_actions'] <= actions
                or measurements['number_observer_return_side'] not in ('left', 'right', 'none')
                or 'current_visual_placement_feedback' not in measurements
                or status == 'completed' and measurements['current_visual_placement_feedback'].get('visual_placement_confirmed') is not True):
            raise ValueError('Placement observation precedes only acknowledged own observer return, not private success')
    if 'public_wrist_view_preflight' in measurements:
        from skills.shared.geometry.public_wrist_view.implementation import validate_evidence
        validate_evidence(measurements['public_wrist_view_preflight'])
    if 'actual_current_view_metric_reason' in measurements:
        reasons = (None, 'instruction_not_number_sorting', 'own_robot_FK_capability_not_available',
                   'own_mount_reference_uncertain', 'current_destination_row_missing_or_ambiguous',
                   'destination_row_not_independently_metric', 'independent_numeral_surface_correspondence_uncertain',
                   'independent_visible_numeral_and_target_counts_disagree')
        if (measurements['actual_current_view_metric_reason'] not in reasons
                or measurements.get('actual_current_view_metric_status') == 'metric_observed'
                and measurements['actual_current_view_metric_reason'] is not None):
            raise ValueError('A view reason is a closed public geometry outcome, not a private result')
    if 'visible_head_target_count' in measurements:
        count = measurements['visible_head_target_count']
        if count is not None and (type(count) is not int or not 2 <= count <= 8):
            raise ValueError('Count only the actually visible bounded destination family')
    if 'visible_head_digits' in measurements:
        digits = measurements['visible_head_digits']
        if (not isinstance(digits, list) or len(digits) > 24
                or any(type(digit) is not int or not 0 <= digit <= 9 for digit in digits)):
            raise ValueError('Uncertain view evidence contains only its own current recognized labels')
    if 'public_number_surface_reference' in measurements:
        from skills.shared.perception.public_numeral_surface_reference.implementation import validate as validate_surfaces
        validate_surfaces(measurements['public_number_surface_reference'])
        if status not in ('completed', 'uncertain') or actions <= 0:
            raise ValueError('Partial numeral measurements follow actual open camera motion only')
    if 'public_number_target_reference' in measurements:
        from skills.shared.perception.public_circular_target_reference.implementation import validate as validate_targets
        validate_targets(measurements['public_number_target_reference'])
        if status not in ('completed', 'uncertain') or actions <= 0:
            raise ValueError('A measured target reference follows actual open camera motion only')
    if 'public_number_head_reference' in measurements:
        from skills.shared.perception.public_numeral_head_reference.implementation import validate as validate_head
        validate_head(measurements['public_number_head_reference'])
        if status not in ('completed', 'uncertain') or actions <= 0:
            raise ValueError('A public head reference is attached only after actual open camera motion')
    anchor = None
    if 'public_number_anchor' in measurements:
        anchor = validate_anchor(measurements['public_number_anchor'])
        validate_progress(measurements['public_number_progress'], len(anchor['digits']))
        if 'public_number_relay' in measurements and abs(measurements['public_number_relay']['site']['surface_xyz'][2]
                                                       - anchor['support_reference_m']) > 1e-6:
            raise ValueError('A relay support hypothesis must remain on its declared current public reference')
    elif 'public_number_progress' in measurements:
        raise ValueError('Own progress must refer to a complete public number anchor')
    if 'public_number_route' in measurements:
        route = validate_route(measurements['public_number_route'])
        if anchor is not None and route['digit'] not in {source['digit'] for source in anchor['digits']}:
            raise ValueError('A route identity must remain in the independently observed family')
    if 'command_phases_completed' in measurements:
        completed = measurements['command_phases_completed']
        if type(completed) is not bool or completed and (status not in ('completed', 'uncertain') or actions <= 0):
            raise ValueError('Own completion requires an actual finished command receipt')
    if measurements.get('gripper_signal_semantics') != 'normalized_public_command_response_not_real_finger_width_or_contact':
        raise ValueError('The public signal is not contact or real finger width')
    if measurements.get('possible_support_is_contact_sensor', False) is not False:
        raise ValueError('The shallow stop is a hypothesis, not a contact sensor')
    if 'possible_payload_support_from_public_command_response' in measurements:
        if type(measurements['possible_payload_support_from_public_command_response']) is not bool:
            raise ValueError('A support hypothesis is a bounded boolean, not private payload state')
    if 'current_source_identity_before_grasp' in measurements:
        digit = measurements['current_source_identity_before_grasp']
        if type(digit) is not int or not 0 <= digit <= 9:
            raise ValueError('A current recognized identity is a declared numeral label')
    if measurements.get('actual_current_view_metric_status', 'uncertain') not in ('metric_observed', 'uncertain'):
        raise ValueError('A current view cannot report native success or private state')
    if 'own_home_joint_error_radians' in measurements:
        error = measurements['own_home_joint_error_radians']
        if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 2 * math.pi:
            raise ValueError('Own reset error must be finite public robot feedback')
    if 'public_visual_grasp_feedback' in measurements:
        report = measurements['public_visual_grasp_feedback']
        required = {'status', 'grasp_is_true_contact_measurement', 'normalized_gripper_feedback_is_real_finger_width'}
        if (not isinstance(report, dict) or not required.issubset(report)
                or not set(report).issubset(required | {'reason', 'hypothesis_errors_pixels'})
                or report['status'] not in ('uncertain', 'visual_at_source', 'visual_held')
                or report['grasp_is_true_contact_measurement'] is not False
                or report['normalized_gripper_feedback_is_real_finger_width'] is not False
                or report.get('reason') not in (None, 'current_independent_outline_not_visible',
                    'current_view_cannot_separate_source_and_lift_hypotheses', 'visual_lift_correspondence_inconclusive')):
            raise ValueError('Visual lift evidence cannot inject a private contact or object record')
        if 'hypothesis_errors_pixels' in report:
            errors = np.asarray(report['hypothesis_errors_pixels'], dtype=float)
            if errors.shape != (2,) or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > 3000):
                raise ValueError('Current image errors must remain bounded public pixel distances')
    if 'current_visual_placement_feedback' in measurements:
        report = measurements['current_visual_placement_feedback']
        required = {'status', 'visual_placement_confirmed'}
        allowed = required | {'reason', 'horizontal_hypothesis_error_m', 'upright_visual_outline',
                              'point_semantics', 'official_success_claimed',
                              'current_public_camera_evidence', 'current_public_cameras_checked'}
        if (not isinstance(report, dict) or not required.issubset(report) or not set(report).issubset(allowed)
                or report['status'] not in ('observed', 'uncertain')
                or type(report['visual_placement_confirmed']) is not bool
                or report['visual_placement_confirmed'] != (report['status'] == 'observed')
                or report.get('official_success_claimed', False) is not False
                or report.get('reason') not in (None, 'current_independent_target_identity_not_visible',
                    'current_target_label_has_no_unique_local_public_camera_correspondence')
                or report.get('point_semantics', 'current_RGB_on_remembered_public_surface_hypothesis_not_object_ground_truth')
                    != 'current_RGB_on_remembered_public_surface_hypothesis_not_object_ground_truth'):
            raise ValueError('Current visual placement cannot conceal object poses or native scorer state')
        if 'horizontal_hypothesis_error_m' in report:
            error = report['horizontal_hypothesis_error_m']
            if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 3:
                raise ValueError('Current placement error must be a finite public hypothesis')
        if 'upright_visual_outline' in report and type(report['upright_visual_outline']) is not bool:
            raise ValueError('Current outline orientation is a public boolean, not a private rotation')
        cameras = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
        if 'current_public_cameras_checked' in report:
            checked = report['current_public_cameras_checked']
            if (not isinstance(checked, list) or tuple(checked) != cameras
                    or report['status'] != 'uncertain' or 'current_public_camera_evidence' in report):
                raise ValueError('A missing identity declares only the three delivered current public cameras')
        if 'current_public_camera_evidence' in report:
            evidence = report['current_public_camera_evidence']
            if not isinstance(evidence, list) or not 1 <= len(evidence) <= len(cameras):
                raise ValueError('Current camera correspondence has one bounded record per public camera')
            selected = set()
            for item in evidence:
                if (not isinstance(item, dict)
                        or set(item) != {'camera', 'horizontal_hypothesis_error_m', 'upright_visual_outline'}
                        or item['camera'] not in cameras or item['camera'] in selected
                        or type(item['upright_visual_outline']) is not bool):
                    raise ValueError('Camera correspondence cannot contain a private pose, contact or scorer record')
                error = item['horizontal_hypothesis_error_m']
                if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 3:
                    raise ValueError('Each camera error is a finite current visual hypothesis')
                selected.add(item['camera'])
            maximum = max(item['horizontal_hypothesis_error_m'] for item in evidence)
            upright = all(item['upright_visual_outline'] for item in evidence)
            if (not {'horizontal_hypothesis_error_m', 'upright_visual_outline'}.issubset(report)
                    or abs(report['horizontal_hypothesis_error_m'] - maximum) > 1e-9
                    or report['upright_visual_outline'] != upright
                    or report['visual_placement_confirmed'] != (maximum <= .055 and upright)):
                raise ValueError('Visual confirmation preserves the original spatial and orientation gates')
    if 'number_phase_notes' in measurements:
        notes = measurements['number_phase_notes']
        statuses = {'pose_reached', 'command_dwell', 'stalled', 'budget', 'contact_stabilized',
            'contact_correction_not_required', 'contact_stabilization_uncertain', 'contact_endpoint_requires_stabilization',
            'mature_endpoint_uncertain', 'bounded_public_placement_stop_requires_visual_verification'}
        if not isinstance(notes, list) or len(notes) > 8:
            raise ValueError('Own phase notes have a bounded command-only history')
        for note in notes:
            if (not isinstance(note, dict) or set(note) != {'phase', 'status', 'actions'}
                    or not isinstance(note['phase'], str) or not 1 <= len(note['phase']) <= 64
                    or note['status'] not in statuses or type(note['actions']) is not int or not 0 <= note['actions'] <= actions):
                raise ValueError('Own phase notes cannot contain private scene or scorer annotations')
