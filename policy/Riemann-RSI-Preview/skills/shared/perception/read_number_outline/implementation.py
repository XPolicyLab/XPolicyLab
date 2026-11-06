from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from skills.shared.perception.connected_image_regions.implementation import run as components
from skills.shared.perception.material_outline_proposals.implementation import material_mask, run as material_proposals
from skills.shared.perception.normalized_outline_matcher.implementation import RASTER_SIZE, run as match_outline
from skills.shared.perception.public_camera_projection.implementation import points_on_plane
from skills.shared.perception.rgb_surface_geometry.implementation import pixel_rays
from skills.shared.perception.visible_outline_topology.implementation import run as visible_topology


MAXIMUM_CANDIDATES = 24
RECTIFIED_RASTER_SIZE = 320


@lru_cache(maxsize=1)
def dictionary():
    folder = Path(__file__).resolve().parent
    metadata = json.loads((folder / 'dictionary.json').read_text())
    declared = {'schema', 'sha256', 'labels', 'raster_size', 'fields', 'anchor_frame',
                'physical_object_dimensions', 'scene_coordinates', 'material_answer_conditioning'}
    if (set(metadata) != declared or metadata['schema'] != 'riemann-scale-free-visual-dictionary-v1'
            or metadata['labels'] != list(range(10)) or metadata['raster_size'] != RASTER_SIZE
            or metadata['fields'] != ['labels', 'outlines', 'anchors']
            or metadata['anchor_frame'] != 'dimensionless_intrinsic_outline_feature_not_current_image_or_layout'
            or any(metadata[key] is not False for key in
                   ('physical_object_dimensions', 'scene_coordinates', 'material_answer_conditioning'))):
        raise ValueError('Numeral recognition requires the complete scale-free public visual dictionary')
    source = folder / 'outline-dictionary.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('The declared visual dictionary changed')
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {'labels', 'outlines', 'anchors'}:
            raise ValueError('A visual dictionary cannot include hidden physical or layout fields')
        labels, outlines, anchors = (archive[key].copy() for key in ('labels', 'outlines', 'anchors'))
    if (not np.array_equal(labels, np.arange(10)) or outlines.shape != (10, RASTER_SIZE, RASTER_SIZE)
            or anchors.shape != (10, 2) or not np.isfinite(anchors).all()
            or np.any(anchors < 0) or np.any(anchors > 1)
            or any(np.count_nonzero(outline) < 6 for outline in outlines)):
        raise ValueError('The visual dictionary or intrinsic anchors are incomplete')
    for value in (labels, outlines, anchors):
        value.flags.writeable = False
    return labels, outlines, anchors


def identify(mask, placed=False, *, observed_mask=None, raster_to_image=None, clipped=False):
    if type(placed) is not bool:
        raise ValueError('Use an explicit bounded orientation range, not an expected identity')
    labels, outlines, anchors = dictionary()
    angles = tuple(range(-90, 91, 3) if placed else range(-60, 61, 3))
    result = match_outline(mask, tuple(int(label) for label in labels), outlines, angles)
    if result['ranking']:
        topology = visible_topology(mask if observed_mask is None else observed_mask,
            tuple(int(label) for label in labels), outlines, result['ranking'], raster_to_image, clipped)
        result['visible_topology'] = topology
        if result['status'] != 'confirmed' and topology['resolves_close_candidates']:
            result.update(status='confirmed', label=result['ranking'][0]['label'],
                          yaw_degrees=result['ranking'][0]['yaw_degrees'])
    result['anchor_pixel'] = None
    if result['status'] == 'confirmed':
        candidate = result['ranking'][0]
        anchor = np.append(anchors[result['label']] * RASTER_SIZE, 1.)
        observed = np.asarray(candidate['outline_to_observed']) @ anchor
        result['anchor_pixel'] = (observed[:2] / observed[2]).tolist()
    result['anchor_semantics'] = 'current_outline_feature_not_object_pose_or_center_of_mass'
    return result


def region_overlap(first, second):
    first_column, first_row, first_width, first_height = first['bbox']
    second_column, second_row, second_width, second_height = second['bbox']
    left, right = max(first_column, second_column), min(first_column + first_width, second_column + second_width)
    top, bottom = max(first_row, second_row), min(first_row + first_height, second_row + second_height)
    if left >= right or top >= bottom:
        return 0.
    intersection = np.count_nonzero(first['mask'][top:bottom, left:right] & second['mask'][top:bottom, left:right])
    return float(intersection / max(1, first['area'] + second['area'] - intersection))


def candidate_regions(image, extended=False, minimum_area=20):
    height, width = image.shape[:2]
    groups = []
    for mask in material_proposals(image, extended):
        for region in components(mask, minimum_area):
            if region['area'] > height * width * .04 or min(region['bbox'][2:]) < 5:
                continue
            column, row, span_x, span_y = region['bbox']
            if column <= 1 or row <= 1 or column + span_x >= width - 1 or row + span_y >= height - 1:
                continue
            group = next((prior for prior in groups if region_overlap(region, prior[0]) > .50), None)
            if group is None:
                if len(groups) < 3 * MAXIMUM_CANDIDATES:
                    groups.append([region])
            elif len(group) < 3 and not any(region_overlap(region, prior) > .94 for prior in group):
                group.append(region)
    proposals = [group[index] for index in range(3) for group in groups if index < len(group)]
    return proposals[:MAXIMUM_CANDIDATES]


def foreground_matches(recognized):
    enclosed_backgrounds = []
    for region, match in recognized:
        mask = np.asarray(region['mask'], dtype=np.uint8)
        contours, unused_hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        enclosed_backgrounds.append((region['area'], match, (filled != 0) & (mask == 0)))
    confirmed, excluded = [], []
    for region, match in recognized:
        nested = any(area > region['area'] and parent['similarity'] >= match['similarity']
            and parent['margin'] >= match['margin']
            and np.count_nonzero(region['mask'] & background) >= .98 * region['area']
            for area, parent, background in enclosed_backgrounds)
        if nested:
            excluded.append({'reason': 'enclosed_background_not_independent_foreground', 'bbox': region['bbox']})
        else:
            confirmed.append(match)
    return confirmed, excluded


def rectify_outline(api, camera, region):
    image = api.image(camera)
    height, width = image.shape[:2]
    column, row, span_x, span_y = region['bbox']
    padding = max(3, round(max(span_x, span_y) * .08))
    left, right = max(0, column - padding), min(width - 1, column + span_x + padding)
    top, bottom = max(0, row - padding), min(height - 1, row + span_y + padding)
    corners = np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
    origin, rays = pixel_rays(api, camera, corners)
    if np.any(rays[:, 2] >= -1e-4):
        return None
    virtual = points_on_plane(api, camera, corners, float(origin[2] - 1.))
    if not np.isfinite(virtual).all():
        return None
    coordinates = virtual[:, :2]
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    span = float(np.max(upper - lower))
    if not np.isfinite(span) or span <= 1e-8:
        return None
    scale = (RECTIFIED_RASTER_SIZE - 1) / span
    rectified = (coordinates - lower) * scale
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), rectified.astype(np.float32))
    if not np.isfinite(transform).all() or np.linalg.cond(transform) > 1e9:
        return None
    mask = cv2.warpPerspective(np.asarray(region['mask'], dtype=np.uint8), transform,
                              (RECTIFIED_RASTER_SIZE, RECTIFIED_RASTER_SIZE), flags=cv2.INTER_NEAREST)
    return {'mask': mask, 'rectified_to_current_image': np.linalg.inv(transform),
            'coordinate_semantics': 'scale_free_parallel_plane_not_measured_surface'}


def run(api, camera, extended=False, placed=False):
    image = api.image(camera)
    if image.shape != (480, 640, 3):
        raise ValueError('Use the declared native public RGB camera without a synthetic frame')
    height, width = image.shape[:2]
    proposals = candidate_regions(image, extended, 45 if camera == 'cam_head' else 70)
    recognized, uncertain = [], []
    for region in proposals:
        rectified = rectify_outline(api, camera, region)
        if rectified is None:
            uncertain.append({'reason': 'current_view_cannot_rectify_horizontal_outline', 'bbox': region['bbox']})
            continue
        match = identify(rectified['mask'], placed, observed_mask=region['mask'],
                         raster_to_image=rectified['rectified_to_current_image'])
        if match['status'] != 'confirmed':
            uncertain.append({'reason': 'independent_outline_match_ambiguous', 'bbox': region['bbox'],
                              'similarity': match['similarity'], 'margin': match['margin']})
            continue
        anchor = rectified['rectified_to_current_image'] @ np.append(match['anchor_pixel'], 1.)
        anchor = anchor[:2] / anchor[2]
        if (not np.isfinite(anchor).all() or not 0 <= anchor[0] < width or not 0 <= anchor[1] < height):
            uncertain.append({'reason': 'visual_anchor_outside_current_image', 'bbox': region['bbox']})
            continue
        record = {'digit': int(match['label']), 'camera': camera, 'anchor_image': anchor.tolist(),
            'anchor_image_normalized': (anchor / [width, height]).tolist(),
            'yaw_radians': float(np.radians(match['yaw_degrees'])), 'similarity': match['similarity'],
            'margin': match['margin'], 'bbox': region['bbox'], 'area': region['area'],
            'anchor_semantics': match['anchor_semantics'],
            'rectification_semantics': rectified['coordinate_semantics']}
        recognized.append((region, record))
    confirmed, excluded = foreground_matches(recognized)
    uncertain.extend(excluded)
    labels = [record['digit'] for record in confirmed]
    return {'kind': 'current_numeral_outlines', 'camera': camera,
        'status': 'observed' if confirmed and len(set(labels)) == len(labels) else 'uncertain',
        'digits': confirmed, 'uncertain_candidates': uncertain,
        'dictionary_labels': list(range(10)), 'expected_answer_conditioning': False,
        'physical_dimensions_used': False, 'metric_pose_estimated': False,
        'source': 'current_public_RGB_public_camera_robot_pose_and_scale_free_visual_dictionary'}
