from copy import deepcopy
import hashlib
import re

import numpy as np


SEMANTICS = 'previous_delivered_target_RGB_during_open_camera_only_commands_not_hidden_state'


def validate_row_correspondence(evidence, points, pixels, radii, reference, indices, cameras):
    fields = {'semantics', 'matched_height_interval_m', 'matched_radius_interval_m',
              'maximum_ray_separation_m', 'head_pixel_widths'}
    if (not isinstance(evidence, dict) or set(evidence) != fields
            or evidence['semantics'] != 'current_visible_planar_row_independent_head_wrist_pairs'
            or len(cameras) != 1 or not max(3, len(points) - 1) <= len(indices) < len(points)):
        raise ValueError('Partial target memory needs its measured current-row correspondence evidence')
    heights = np.asarray(evidence['matched_height_interval_m'], dtype=float)
    sizes = np.asarray(evidence['matched_radius_interval_m'], dtype=float)
    widths = np.asarray(evidence['head_pixel_widths'], dtype=float)
    separation = evidence['maximum_ray_separation_m']
    if (heights.shape != (2,) or sizes.shape != (2,) or widths.shape != (len(points),)
            or not all(np.isfinite(value).all() for value in (heights, sizes, widths))
            or not reference < heights[0] <= heights[1] <= reference + .08
            or heights[1] - heights[0] > .002 or np.any(points[:, 2] < heights[0])
            or np.any(points[:, 2] > heights[1]) or not .005 <= sizes[0] <= sizes[1] <= .12
            or sizes[1] / sizes[0] > 1.12 or radii.max() / radii.min() > 1.12
            or np.any(radii < .8 * sizes[0]) or np.any(radii > 1.2 * sizes[1])
            or np.any(widths <= 0) or widths.max() / widths.min() > 1.5
            or type(separation) not in (int, float) or not np.isfinite(separation)
            or not 0 <= separation <= .0015):
        raise ValueError('Partial target memory preserves tighter measured intervals, not an expected surface')
    from skills.shared.perception.observed_row_geometry.implementation import run as observed_rows
    _, _, directions = np.linalg.svd(pixels - pixels.mean(axis=0), full_matrices=False)
    if len(observed_rows(pixels, widths, directions[0], len(points))) != 1:
        raise ValueError('Partial target memory cannot infer targets outside its current visible row')


def validate(value):
    fields = {'schema', 'target_centres_pixels', 'targets_xyz', 'radii_m', 'observed_head_indices',
              'wrist_views', 'support_reference_m', 'image_sha256', 'semantics'}
    evidence = value.get('row_correspondence_evidence') if isinstance(value, dict) else None
    if evidence is not None:
        fields.add('row_correspondence_evidence')
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'riemann-public-static-circular-target-reference-v1'
            or value['semantics'] != SEMANTICS or not isinstance(value['image_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', value['image_sha256']) is None):
        raise ValueError('Retain only measured circular targets from a delivered public camera frame')
    points = np.asarray(value['targets_xyz'], dtype=float)
    pixels = np.asarray(value['target_centres_pixels'], dtype=float)
    radii = np.asarray(value['radii_m'], dtype=float)
    count = len(points)
    if (not 2 <= count <= 8 or points.shape != (count, 3) or pixels.shape != (count, 2)
            or radii.shape != (count,) or not np.isfinite(points).all() or np.max(np.abs(points)) > 3
            or not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels >= [640, 480])
            or not np.isfinite(radii).all() or np.any(radii < .005) or np.any(radii > .12)
            or radii.max() / radii.min() > 1.2 or np.ptp(points[:, 2]) > .012):
        raise ValueError('A retained family preserves all independently visible finite targets')
    reference = value['support_reference_m']
    indices, cameras = value['observed_head_indices'], value['wrist_views']
    if (type(reference) not in (int, float) or not np.isfinite(reference)
            or np.any(points[:, 2] - reference <= 0) or np.any(points[:, 2] - reference > .08)
            or not isinstance(indices, list) or not 2 <= len(set(indices)) == len(indices) <= count
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or not isinstance(cameras, list) or not 1 <= len(set(cameras)) == len(cameras) <= 2
            or not set(cameras).issubset({'cam_left_wrist', 'cam_right_wrist'})
            or len(cameras) == 1 and len(indices) < count and evidence is None):
        raise ValueError('Metric family memory requires the original independent cross-view coverage')
    if evidence is not None:
        validate_row_correspondence(evidence, points, pixels, radii, reference, indices, cameras)
    return deepcopy(value)


def capture(family, row, image, support_reference_m):
    if family is None or row is None or len(family['surfaces']) != len(row):
        return None
    value = {'schema': 'riemann-public-static-circular-target-reference-v1', 'semantics': SEMANTICS,
        'target_centres_pixels': [np.asarray(region['centroid']).tolist() for region in row],
        'targets_xyz': [np.asarray(surface['point']).tolist() for surface in family['surfaces']],
        'radii_m': [float(surface['radius_m']) for surface in family['surfaces']],
        'observed_head_indices': list(family['observed_head_indices']), 'wrist_views': list(family['wrist_views']),
        'support_reference_m': float(support_reference_m),
        'image_sha256': hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}
    if family.get('row_correspondence_evidence') is not None:
        value['row_correspondence_evidence'] = deepcopy(family['row_correspondence_evidence'])
    return validate(value)


def retained(memory, view_skill, home_skill, reference_field):
    for event in reversed(memory):
        feedback = event.get('execution_feedback', {})
        if feedback.get('actions_executed', 0) <= 0:
            continue
        if event.get('skill') not in (view_skill, home_skill):
            return None
        if event.get('skill') == home_skill and event.get('request', {}).get('purpose') != 'inspect':
            return None
        value = feedback.get('image_measurements', {}).get(reference_field)
        phases = feedback.get('phase_reports', [])
        settled = any(phase.get('phase') == 'settle_public_view_after_robot_motion'
                      and phase.get('actions', 0) > 0 for phase in phases)
        if value is not None and settled:
            return validate(value)
    return None


def compatible(value, row, support_reference_m):
    if value is None or row is None:
        return False
    value = validate(value)
    current = np.asarray([region['centroid'] for region in row])
    previous = np.asarray(value['target_centres_pixels'])
    return bool(current.shape == previous.shape and abs(value['support_reference_m'] - support_reference_m) <= .005
                and np.max(np.linalg.norm(current - previous, axis=1)) <= 4)


def family(value):
    value = validate(value)
    result = {'surfaces': [{'point': np.asarray(point), 'radius_m': radius}
                         for point, radius in zip(value['targets_xyz'], value['radii_m'])],
            'observed_head_indices': value['observed_head_indices'], 'wrist_views': value['wrist_views']}
    if value.get('row_correspondence_evidence') is not None:
        result['row_correspondence_evidence'] = deepcopy(value['row_correspondence_evidence'])
    return result
