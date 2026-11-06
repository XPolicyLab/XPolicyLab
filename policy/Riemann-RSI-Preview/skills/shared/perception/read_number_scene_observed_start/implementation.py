from itertools import combinations
import math
import re
from skills.shared.orchestration.instruction_semantics.implementation import matches
from copy import deepcopy
import cv2
import numpy as np
from skills.shared.perception.circular_surface_geometry.implementation import run as circular_surface
from skills.shared.perception.elliptical_edge_regions.implementation import run as edge_regions
from skills.shared.perception.observed_row_geometry.implementation import run as observed_rows
from skills.shared.perception.public_camera_projection.implementation import points_on_plane
from skills.shared.perception.rgb_surface_geometry.implementation import triangulate, triangulated_height_interval
from skills.shared.perception.robot_mounting_reference.implementation import run as mounting_reference
from skills.shared.perception.surface_family_correspondence.implementation import run as surface_family
from skills.shared.perception.read_number_outline.implementation import run as read_outlines
from skills.shared.perception.public_numeral_head_reference.implementation import compatible as compatible_head, retained
from skills.shared.perception.public_circular_target_reference.implementation import capture as capture_targets, compatible as compatible_targets, family as target_family, retained as retained_targets
from skills.shared.perception.public_numeral_surface_reference.implementation import capture as capture_surfaces, combine as combine_surfaces, retained as retained_surfaces
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
INVENTORY_SEMANTICS = 'historical_public_RGB_cardinality_lower_bound_only_requires_current_reobservation'
NUMBER_TOOLS = ('number_scene_view', 'number_home', 'grasp_number', 'lift_number', 'carry_number', 'place_number', 'verify_number_placement', 'recover_number_grasp', 'relay_number')

def validate_inventory(value):
    """Validate historical public counts/labels and return a detached dictionary with no metric fields."""
    fields = {'schema', 'semantics', 'minimum_count', 'observed_labels'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != 'riemann-public-number-inventory-v1' or (value['semantics'] != INVENTORY_SEMANTICS) or (type(value['minimum_count']) is not int) or (not 0 <= value['minimum_count'] <= 10) or (not isinstance(value['observed_labels'], list)) or any((type(label) is not int or not 0 <= label <= 9 for label in value['observed_labels'])) or (value['observed_labels'] != sorted(set(value['observed_labels']))) or (len(value['observed_labels']) > value['minimum_count']):
        raise ValueError('A number inventory stores only bounded historical public cardinality and recognized labels')
    return deepcopy(value)

def public_inventory(memory=(), row=None, report=None, previous=None):
    """Combine public head reads/history into a count lower bound; never return pixels, poses or current qualification.

    Input row regions carry current centroids (2,), but only their count is retained. Each
    report label was independently recognized from current RGB; remembered labels only reject
    an incomplete new scene and never supply a source, target, identity or geometry measurement.
    """
    minimum_count, labels = (0, set())
    values = [] if previous is None else [validate_inventory(previous)]
    for event in memory:
        feedback = event.get('execution_feedback', {})
        if event.get('skill') not in NUMBER_TOOLS or feedback.get('actions_executed', 0) <= 0:
            continue
        measurements = feedback.get('image_measurements', {})
        if 'public_number_inventory' in measurements:
            values.append(validate_inventory(measurements['public_number_inventory']))
        count = measurements.get('visible_head_target_count')
        if type(count) is int and 2 <= count <= 8:
            minimum_count = max(minimum_count, count)
        observed = measurements.get('visible_head_digits', [])
        if isinstance(observed, list) and all((type(label) is int and 0 <= label <= 9 for label in observed)):
            labels.update(observed)
    for value in values:
        minimum_count = max(minimum_count, value['minimum_count'])
        labels.update(value['observed_labels'])
    if row is not None:
        if not 2 <= len(row) <= 8:
            raise ValueError('A public destination row has its original bounded observed cardinality')
        minimum_count = max(minimum_count, len(row))
    if report is not None:
        observed = [record['digit'] for record in report['digits']]
        if len(observed) > 24 or any((type(label) is not int or not 0 <= label <= 9 for label in observed)):
            raise ValueError('Inventory labels come only from bounded independently recognized current glyphs')
        labels.update(observed)
    return validate_inventory({'schema': 'riemann-public-number-inventory-v1', 'semantics': INVENTORY_SEMANTICS, 'minimum_count': max(minimum_count, len(labels)), 'observed_labels': sorted(labels)})

def applicable(instruction):
    return matches(instruction, actions=('arrange',), objects=('number',), any_terms=('largest', 'descending'))

def pad_regions(image):
    image = np.asarray(image)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError('Read destination surfaces only in current native public RGB')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pale = (hsv[..., 1] < 95) & (hsv[..., 2] > 110)
    found = [region for region in edge_regions(image) if np.mean(pale[region['mask']]) >= 0.55 and np.median(hsv[..., 1][region['mask']]) < 80]
    return sorted(found, key=lambda region: region['boundary_error'])[:16]

def target_row(regions):
    if not 2 <= len(regions) <= 16:
        return None
    centres = [region['centroid'] for region in regions]
    widths = [max(region['bbox'][2:]) for region in regions]
    for count in range(min(len(regions), 8), 1, -1):
        rows = observed_rows(centres, widths, [1.0, 0.0], count)
        rows = [row for row in rows if max((widths[index] for index in row['indices'])) <= 1.5 * min((widths[index] for index in row['indices']))]
        memberships = {tuple(sorted(row['indices'])) for row in rows}
        if not memberships:
            continue
        if len(memberships) != 1:
            return None
        selected = min(rows, key=lambda row: row['maximum_residual_pixels'])
        return [regions[index] for index in selected['indices']]
    return None

def numeral_surfaces(api, reports, reference_height, partial=False):
    """Read three public camera reports and height; return qualified source dictionaries or None.

    XYZ features have shape (3,). Partial mode keeps independently qualified digits while
    leaving unqualified digits absent; it never fills them from a single camera.
    """
    if set(reports) != set(CAMERAS) or not math.isfinite(reference_height):
        raise ValueError('A numeral surface needs current independent reads from the declared three cameras')
    maps = {}
    for camera in CAMERAS:
        records = reports[camera]['digits']
        labels = [record['digit'] for record in records]
        if len(records) > 24 or len(set(labels)) != len(labels) or any((type(label) is not int or not 0 <= label <= 9 for label in labels)) or any((record['camera'] != camera for record in records)):
            return None
        maps[camera] = {record['digit']: record for record in records}
    identities = sorted({digit for records in maps.values() for digit in records}, reverse=True)
    if not (1 if partial else 2) <= len(identities) <= 8:
        return None
    surfaces = []
    for digit in identities:
        evidence = []
        available = [camera for camera in CAMERAS if digit in maps[camera]]
        if len(available) < 2:
            if partial:
                continue
            return None
        first = maps[available[0]][digit]
        for first_camera, second_camera in combinations(available, 2):
            first_read, second_read = (maps[first_camera][digit], maps[second_camera][digit])
            estimate = triangulate(api, first_camera, first_read['anchor_image'], second_camera, second_read['anchor_image'])
            if estimate is None:
                continue
            interval = triangulated_height_interval(api, first_camera, first_read['anchor_image'], second_camera, second_read['anchor_image'], estimator=triangulate)
            if interval is not None and 0.0 < interval[0] - reference_height and (interval[1] - reference_height <= 0.08):
                evidence.append(dict(estimate, public_feature_height_interval_m=list(interval)))
        if not evidence:
            if partial:
                continue
            return None
        points = np.stack([estimate['point'] for estimate in evidence])
        if np.max(np.linalg.norm(points - np.median(points, axis=0), axis=1)) > 0.006:
            if partial:
                continue
            return None
        point = np.median(points, axis=0)
        surfaces.append({'digit': digit, 'surface_xyz': point.tolist(), 'estimated_grasp_span_m': float(point[2] - reference_height), 'yaw_radians': first['yaw_radians'], 'public_feature_height_interval_m': [min((estimate['public_feature_height_interval_m'][0] for estimate in evidence)), max((estimate['public_feature_height_interval_m'][1] for estimate in evidence))], 'maximum_ray_separation_m': max((estimate['ray_separation_m'] for estimate in evidence)), 'cameras': sorted({camera for estimate in evidence for camera in estimate['cameras']}), 'height_semantics': 'current_visual_feature_above_public_mount_reference_not_asset_height'})
    if not surfaces:
        return [] if partial else None
    if np.ptp([record['surface_xyz'][2] for record in surfaces]) > 0.012:
        return None
    return sorted(surfaces, key=lambda record: -record['digit'])

def measure(api, extended=False, head_reference=None, target_reference=None, inventory_reference=None):
    """Build a complete current public number anchor or uncertain report, preserving qualified partial views."""
    result = {'kind': 'number_sorting_scene', 'status': 'uncertain', 'anchor': None, 'metric_pose_is_ground_truth': False, 'missing_objects_filled': False, 'target_count_conditioned_on_digits': False, 'official_success_claimed': False}
    if not applicable(api.instruction()):
        return dict(result, reason='instruction_not_number_sorting')
    if not api.own_robot_bases_available():
        return dict(result, reason='own_robot_FK_capability_not_available')
    reference = mounting_reference(api.own_robot_bases())
    if reference['status'] != 'observed':
        return dict(result, reason='own_mount_reference_uncertain', mounting_reference=reference)
    reports = {camera: read_outlines(api, camera, extended=extended) for camera in CAMERAS}
    delivered_reports = deepcopy(reports)
    row = target_row(pad_regions(api.image('cam_head')))
    result['visible_head_digits'] = [record['digit'] for record in reports['cam_head']['digits']]
    result['visible_head_anchors'] = [record['anchor_image'] for record in reports['cam_head']['digits']]
    result['visible_head_target_count'] = len(row) if row is not None else None
    inventory = public_inventory(api.command_memory(), row, reports['cam_head'], inventory_reference)
    result['public_number_inventory'] = inventory
    if row is None:
        return dict(result, reason='current_destination_row_missing_or_ambiguous')
    if len(row) < inventory['minimum_count']:
        return dict(result, reason='historical_public_number_inventory_requires_reobservation')
    reference_head = head_reference if head_reference is not None else retained(api.command_memory())
    reused_head = len(reports['cam_head']['digits']) < len(row) and compatible_head(reference_head, reports['cam_head'], row)
    if reused_head:
        reports['cam_head'] = dict(reports['cam_head'], digits=deepcopy(reference_head['digits']))
    result['previous_public_head_reference_used'] = bool(reused_head)
    fresh = numeral_surfaces(api, reports, reference['height_m'], partial=True) or []
    if reused_head:
        for source in fresh:
            source['height_semantics'] = 'public_static_scene_multi_view_correspondence_not_asset_height'
    saved_surfaces = retained_surfaces(api, row, reference['height_m'])
    sources, remembered_sources = combine_surfaces(api, delivered_reports, fresh, saved_surfaces)
    retained_surface = capture_surfaces(api, remembered_sources, row, reference['height_m'])
    if retained_surface is not None:
        result['public_number_surface_reference'] = retained_surface
    family = surface_family(api, row, pad_regions, circular_surface, reference['height_m'], minimum_elevation_m=0.0, maximum_elevation_m=0.08, observed_planar_row=True)
    saved_targets = target_reference if target_reference is not None else retained_targets(api.command_memory(), 'number_scene_view', 'number_home', 'public_number_target_reference')
    reused_targets = family is None and compatible_targets(saved_targets, row, reference['height_m'])
    if reused_targets:
        family = target_family(saved_targets)
    result['previous_public_target_reference_used'] = bool(reused_targets)
    if family is None:
        return dict(result, reason='destination_row_not_independently_metric')
    if reused_targets:
        result['public_number_target_reference'] = saved_targets
    else:
        try:
            target_packet = capture_targets(family, row, api.image('cam_head'), reference['height_m'])
        except ValueError as error:
            return dict(result, reason='current_target_family_validation_rejected', target_family_rejection=str(error)[:256])
        if target_packet is None:
            return dict(result, reason='current_target_family_capture_incomplete')
        result['public_number_target_reference'] = target_packet
    if len(sources) < 2:
        return dict(result, reason='independent_numeral_surface_correspondence_uncertain')
    if len(sources) != len(row):
        return dict(result, reason='independent_visible_numeral_and_target_counts_disagree')
    if not set(inventory['observed_labels']) <= {source['digit'] for source in sources}:
        return dict(result, reason='historical_public_number_inventory_requires_reobservation')
    if reused_head:
        for source in sources:
            source['height_semantics'] = 'public_static_scene_multi_view_correspondence_not_asset_height'
    targets = sorted((surface['point'].tolist() for surface in family['surfaces']), key=lambda point: point[0])
    return dict(result, status='metric_observed', reason=None, anchor={'schema': 'riemann-current-number-scene-v1', 'digits': sources, 'targets': targets, 'support_reference_m': reference['height_m'], 'mounting_reference': reference, 'target_geometry_source': 'same_episode_public_RGB_circular_surface_family' if reused_targets else 'independent_current_RGB_circular_surface_family', 'observed_target_count': len(targets), 'current_layout_source': False})

def fresh_source(api, anchor, digit):
    from skills.shared.perception.visible_reference_correspondence.implementation import run as associate
    remembered = next((source for source in anchor['digits'] if source['digit'] == digit), None)
    if remembered is None:
        return None
    evidence = []
    for camera in CAMERAS:
        current = read_outlines(api, camera)
        matched = [record for record in current['digits'] if record['digit'] == digit]
        if not matched:
            continue
        points = points_on_plane(api, camera, [record['anchor_image'] for record in matched], remembered['surface_xyz'][2])
        nearby = np.linalg.norm(points - np.asarray(remembered['surface_xyz']), axis=1) <= 0.055
        if np.count_nonzero(nearby) > 1:
            return None
        selected = associate(points, [remembered['surface_xyz']], [0.055], 'metres')
        if selected is not None:
            evidence.append({'camera': camera, 'point': points[selected], 'record': matched[selected]})
    if not evidence:
        return None
    points = np.asarray([item['point'] for item in evidence])
    point = np.median(points, axis=0)
    if np.max(np.linalg.norm(points - point, axis=1)) > 0.008:
        return None
    strongest = max(evidence, key=lambda item: item['record'].get('similarity', 0.0))
    result = deepcopy(remembered)
    result['surface_xyz'], result['yaw_radians'] = (point.tolist(), strongest['record']['yaw_radians'])
    result['current_relocalization_source'] = 'independent_current_public_camera_label_and_remembered_public_surface'
    result['current_relocalization_cameras'] = [item['camera'] for item in evidence]
    return result

def placement(api, anchor, digit, target_index):
    from skills.shared.perception.visible_reference_correspondence.implementation import run as associate
    source = next((item for item in anchor['digits'] if item['digit'] == digit), None)
    if source is None or not 0 <= target_index < len(anchor['targets']):
        return {'status': 'uncertain', 'visual_placement_confirmed': False}
    target = np.asarray(anchor['targets'][target_index])
    height = float(target[2] + source['estimated_grasp_span_m'])
    evidence = []
    for camera in CAMERAS:
        report = read_outlines(api, camera, placed=True)
        matched = [record for record in report['digits'] if record['digit'] == digit]
        if not matched:
            continue
        points = points_on_plane(api, camera, [record['anchor_image'] for record in matched], height)
        selected = associate(points, [np.r_[target[:2], height]], [0.055], 'metres')
        if selected is None:
            continue
        distance = float(np.linalg.norm(points[selected][:2] - target[:2]))
        yaw = matched[selected]['yaw_radians']
        upright = abs(math.atan2(math.sin(yaw), math.cos(yaw))) <= math.pi / 4
        evidence.append({'camera': camera, 'horizontal_hypothesis_error_m': distance, 'upright_visual_outline': upright})
    if not evidence:
        return {'status': 'uncertain', 'visual_placement_confirmed': False, 'reason': 'current_target_label_has_no_unique_local_public_camera_correspondence', 'current_public_cameras_checked': list(CAMERAS)}
    distance = max((record['horizontal_hypothesis_error_m'] for record in evidence))
    upright = all((record['upright_visual_outline'] for record in evidence))
    confirmed = distance <= 0.055 and upright
    return {'status': 'observed' if confirmed else 'uncertain', 'visual_placement_confirmed': confirmed, 'horizontal_hypothesis_error_m': distance, 'upright_visual_outline': upright, 'current_public_camera_evidence': evidence, 'point_semantics': 'current_RGB_on_remembered_public_surface_hypothesis_not_object_ground_truth', 'official_success_claimed': False}
