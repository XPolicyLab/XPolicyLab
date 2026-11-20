from itertools import product
import hashlib
import json

import cv2
import numpy as np
from scipy.ndimage import binary_erosion

from skills.shared.motion.public_phase_execution.implementation import PublicRobotAPI
from skills.shared.perception.public_camera_projection.implementation import points_on_plane
from skills.shared.perception.rgb_surface_geometry.implementation import camera_model


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def diagnostic_inputs(api):
    observed = api.targets()
    observed.update({side + '_arm_joint_state': values for side, values in api.own_joint_states().items()})
    own_state = {name: np.asarray(values, dtype=float).reshape(-1).tolist() for name, values in observed.items()}
    images = {}
    for camera in CAMERAS:
        image = api.image(camera)
        images[camera] = {'sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'shape': list(image.shape), 'dtype': str(image.dtype), 'hash_encoding': 'current_rgb_c_order_bytes'}
    return {'source': 'current_public_rgb_and_own_robot_state', 'public_action_epoch': api.issued_action_count(),
        'instruction_sha256': hashlib.sha256(api.instruction().encode('utf-8')).hexdigest(),
        'rgb': images, 'own_state': own_state,
        'own_state_sha256': hashlib.sha256(json.dumps(own_state, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()}


def diagnostic_regions(regions):
    def values(value, maximum):
        try:
            observed = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return [float(number) if np.isfinite(number) else None for number in observed[:maximum]]
    outputs = {}
    for camera in CAMERAS:
        records = []
        for index, region in enumerate(regions[camera][:12]):
            try:
                outline = np.asarray(region.get('outline', []), dtype='<f8')
                outline_hash = hashlib.sha256(outline.tobytes()).hexdigest()
                outline_shape = list(outline.shape)
            except (TypeError, ValueError):
                outline_hash, outline_shape = None, None
            confidence = values(region.get('confidence'), 1)
            area = values(region.get('area'), 1)
            records.append({'index': index, 'clipped': region.get('clipped'),
                'confidence': confidence[0] if confidence else None, 'area_px': area[0] if area else None,
                'bbox': values(region.get('bbox'), 4), 'centroid_uv': values(region.get('centroid'), 2),
                'outline_sha256': outline_hash, 'outline_shape': outline_shape,
                'outline_hash_encoding': 'current_outline_little_endian_float64_c_order_bytes'})
        outputs[camera] = {'retained_count': len(regions[camera]),
            'clipped_count': sum(region.get('clipped') is True for region in regions[camera]),
            'records': records, 'records_truncated': len(regions[camera]) > 12}
    return outputs


def region_mask(region):
    outline = np.asarray(region['outline'], dtype=float)
    if (region.get('source') not in ('sam3_current_public_rgb_mask', 'current_public_colour_mask') or region.get('clipped') is not False
            or outline.shape != (96, 2) or not np.isfinite(outline).all()
            or np.any(outline <= 1.) or np.any(outline[:, 0] >= 638.) or np.any(outline[:, 1] >= 478.)):
        raise ValueError('The visual hull needs complete current semantic outlines, not stored object geometry')
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [np.rint(outline).astype(np.int32)], 1)
    return mask.astype(bool)


def projected(points, model):
    transform, focal = model
    coordinates = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = -coordinates[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    visible = depth > 1e-6
    pixels[visible] = np.column_stack((320 + focal * coordinates[visible, 0] / depth[visible],
                                     240 - focal * coordinates[visible, 1] / depth[visible]))
    return pixels, depth


def inside(points, mask, model):
    pixels, depth = projected(points, model)
    valid = (np.isfinite(pixels).all(axis=1) & (depth > 0) & (pixels[:, 0] >= 1.)
             & (pixels[:, 0] < 639.) & (pixels[:, 1] >= 1.) & (pixels[:, 1] < 479.))
    indices = np.rint(pixels[valid]).astype(int)
    selected = np.zeros(len(points), dtype=bool)
    selected[np.flatnonzero(valid)] = mask[indices[:, 1], indices[:, 0]]
    return selected


def projection_coverage(points, mask, model, spacing):
    pixels, depth = projected(points, model)
    valid = np.isfinite(pixels).all(axis=1) & (depth > 0)
    indices = np.rint(pixels[valid]).astype(int)
    valid_image = ((indices[:, 0] >= 0) & (indices[:, 0] < 640)
                   & (indices[:, 1] >= 0) & (indices[:, 1] < 480))
    indices = indices[valid_image]
    if not len(indices):
        return 0.
    prediction = np.zeros((480, 640), np.uint8)
    prediction[indices[:, 1], indices[:, 0]] = 1
    radius = min(6, max(1, int(np.ceil(model[1] * spacing / np.median(depth[valid])))))
    prediction = cv2.dilate(prediction, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)).astype(bool)
    union = np.count_nonzero(prediction | mask)
    return float(np.count_nonzero(prediction & mask) / union) if union else 0.


def one_hull(api, observations, support_height, support_uncertainty, require_supported, diagnostics=None):
    def rejected(reason, **facts):
        if diagnostics is not None:
            diagnostics.update(status='rejected', gate_reason=reason, facts=facts)
        return None
    if (not 2 <= len(observations) <= 3 or len({item['camera'] for item in observations}) != len(observations)
            or observations[0]['camera'] != 'cam_head'):
        return rejected('independent_camera_count_or_head_order', observation_count=len(observations))
    cameras = [item['camera'] for item in observations]
    if diagnostics is not None:
        diagnostics['active_gate'] = 'public_camera_model'
    models = [camera_model(api, camera) for camera in cameras]
    origins = [model[0][:3, 3] for model in models]
    if diagnostics is not None:
        diagnostics.update(camera_origins_m=[origin.tolist() for origin in origins],
            maximum_baseline_m=float(max(np.linalg.norm(first - second) for first in origins for second in origins)),
            grid_steps=[])
    if max(np.linalg.norm(first - second) for first in origins for second in origins) < .02:
        return rejected('camera_baseline', maximum_baseline_m=diagnostics['maximum_baseline_m']
            if diagnostics is not None else None)
    if diagnostics is not None:
        diagnostics['active_gate'] = 'current_semantic_outline'
    masks = [region_mask(item) for item in observations]
    outline = observations[0]['outline']
    if diagnostics is not None:
        diagnostics['active_gate'] = 'public_projected_search_bounds'
    bounds = np.vstack([points_on_plane(api, 'cam_head', outline, support_height + height)
                        for height in (.001, .35)])
    if not np.isfinite(bounds).all():
        return rejected('nonfinite_projected_bounds', bounds_finite=False)
    lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
    lower[2], upper[2] = support_height + .001, support_height + .35
    if np.max(np.abs([lower, upper])) > 3. or np.any(upper - lower < .001) or np.max(upper[:2] - lower[:2]) > 1.2:
        return rejected('bounded_search_volume', lower_m=lower.tolist(), upper_m=upper.tolist())
    original_lower, original_upper = lower.copy(), upper.copy()
    occupied = None
    for resolution in (32, 38):
        if diagnostics is not None:
            diagnostics['active_gate'] = 'voxel_intersection'
        axes = [np.linspace(first, last, resolution) for first, last in zip(lower, upper)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        spacing = (upper - lower) / (resolution - 1)
        accepted = np.ones(len(grid), dtype=bool)
        remaining = []
        for mask, model in zip(masks, models):
            accepted &= inside(grid, mask, model)
            if diagnostics is not None:
                remaining.append({'camera': cameras[len(remaining)], 'remaining_voxels': int(np.count_nonzero(accepted))})
        occupied = accepted.reshape((resolution,) * 3)
        points = grid[accepted]
        if diagnostics is not None:
            diagnostics['grid_steps'].append({'resolution': resolution, 'grid_voxels': len(grid),
                'spacing_m': spacing.tolist(), 'retained_points': len(points), 'per_camera': remaining})
        if len(points) < 16:
            return rejected('occupied_grid_points', resolution=resolution, retained_points=len(points))
        lower = np.maximum(original_lower, np.min(points, axis=0) - spacing)
        upper = np.minimum(original_upper, np.max(points, axis=0) + spacing)
    surface = occupied & ~binary_erosion(occupied)
    points = grid[surface.reshape(-1)]
    uncertainty = float(np.max(spacing) + support_uncertainty)
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    height, spans = float(upper[2] - support_height), upper - lower
    if (not .005 <= height <= .35 or not .004 <= max(spans[:2]) <= .5 or uncertainty > .018
            or require_supported and lower[2] - support_height > max(.012, uncertainty * 2)):
        failed = []
        if diagnostics is not None:
            if not .005 <= height <= .35:
                failed.append('height')
            if not .004 <= max(spans[:2]) <= .5:
                failed.append('planar_span')
            if uncertainty > .018:
                failed.append('uncertainty')
            if require_supported and lower[2] - support_height > max(.012, uncertainty * 2):
                failed.append('support_clearance')
        return rejected('surface_extent_uncertainty_or_support', failed_gates=failed, height_m=height,
            spans_m=spans.tolist(), uncertainty_m=uncertainty,
            support_clearance_m=float(lower[2] - support_height), require_supported=require_supported)
    if diagnostics is not None:
        diagnostics['active_gate'] = 'projection_coverage'
    coverage = [projection_coverage(points, mask, model, float(np.max(spacing)))
                for mask, model in zip(masks, models)]
    if min(coverage) < .55:
        return rejected('projection_coverage', coverage=coverage)
    if len(points) > 16000:
        points = points[np.linspace(0, len(points) - 1, 16000, dtype=int)]
    head_pixels, unused_depth = projected(points, models[0])
    indices = np.clip(np.rint(head_pixels).astype(int), [0, 0], [639, 479])
    colours = api.image('cam_head')[indices[:, 1], indices[:, 0]].copy()
    if diagnostics is not None:
        diagnostics.update(status='qualified', gate_reason='qualified', facts={'height_m': height,
            'spans_m': spans.tolist(), 'uncertainty_m': uncertainty, 'coverage': coverage,
            'surface_points': len(points), 'support_clearance_m': float(lower[2] - support_height)})
    return {'points_xyz': points, 'rgb_samples': colours, 'center_m': (lower + upper) / 2,
            'spans_m': spans, 'height_estimate_m': height, 'uncertainty_m': uncertainty,
            'cameras': cameras, 'coverage': coverage, 'head_region': observations[0],
            'source': 'current_public_rgb_visual_hull', 'surface_is_true_object_geometry': False,
            'resolution_semantics': 'grid_and_support_resolution_not_hidden_object_error'}


def run(api, regions, support_height, support_uncertainty, require_supported=True, diagnostics=None):
    if (type(api) is not PublicRobotAPI or not isinstance(regions, dict) or set(regions) != set(CAMERAS)
            or type(require_supported) is not bool or not np.isfinite(support_height)
            or not 0 <= support_uncertainty <= .006 or abs(support_height) > 3
            or any(not isinstance(values, list) or len(values) > 4 for values in regions.values())):
        raise ValueError('A current visual hull uses only bounded public masks and an independently observed support')
    if diagnostics is not None and not isinstance(diagnostics, list):
        raise ValueError('Visual-hull diagnostics use a bounded append-only list')
    for camera, observed_regions in regions.items():
        for region in observed_regions:
            if region.get('source') == 'current_public_colour_mask':
                digest = hashlib.sha256(api.image(camera).tobytes()).hexdigest()
                if (region.get('camera') != camera or region.get('current_rgb_sha256') != digest
                        or region.get('colour_family') not in ('red', 'yellow', 'purple', 'green', 'blue', 'white')):
                    raise ValueError('Colour outlines must belong to this current public camera image')
    receipt = {'source': 'current_public_visual_hull_gate_diagnostics',
        'support_height_m': float(support_height), 'support_uncertainty_m': float(support_uncertainty),
        'require_supported': require_supported, 'attempts': []} if diagnostics is not None else None
    candidates, attempts = [], 0
    for head in regions['cam_head'][:2]:
        available = [camera for camera in CAMERAS[1:] if regions[camera]]
        groups = ([available] if len(available) == 2 else []) + [[camera] for camera in available]
        for group in groups:
            for others in product(*(regions[camera][:2] for camera in group)):
                observations = [head, *others]
                attempt = None
                if receipt is not None:
                    attempt = {'attempt': attempts + 1, 'cameras': [item['camera'] for item in observations],
                        'semantic_indices': {item['camera']: next(index for index, region in enumerate(regions[item['camera']])
                            if region is item) for item in observations}}
                try:
                    if attempt is None:
                        candidate = one_hull(api, observations, support_height, support_uncertainty, require_supported)
                    else:
                        candidate = one_hull(api, observations, support_height, support_uncertainty,
                            require_supported, diagnostics=attempt)
                except ValueError as error:
                    candidate = None
                    if attempt is not None:
                        attempt.update(status='rejected', gate_reason=attempt.get('active_gate', 'invalid_outline_or_public_projection'),
                            error=str(error)[:240])
                if receipt is not None:
                    receipt['attempts'].append(attempt)
                attempts += 1
                if candidate is not None:
                    candidates.append(candidate)
                if attempts == 12:
                    break
            if attempts == 12:
                break
        if attempts == 12:
            break
    unique = []
    for candidate in sorted(candidates, key=lambda item: (len(item['cameras']), min(item['coverage'])), reverse=True):
        if not any(np.linalg.norm(candidate['center_m'] - item['center_m']) <=
                   max(.015, float(np.max(candidate['spans_m'])) * .45) for item in unique):
            unique.append(candidate)
    if receipt is not None:
        reason = ('qualified' if len(unique) == 1 else 'ambiguous_unique_hulls' if unique
            else 'no_qualified_hull' if attempts else 'no_retained_head_semantics' if not regions['cam_head']
            else 'no_retained_wrist_semantics')
        receipt.update(gate_reason=reason, attempt_count=attempts, candidate_count=len(candidates),
            unique_count=len(unique), maximum_attempts=12)
        diagnostics.append(receipt)
    return unique[0] if len(unique) == 1 else None
