from itertools import product
import math

import cv2
import numpy as np

from skills.shared.geometry.fit_current_planar_boundary.implementation import run as fit_boundary
from skills.shared.perception.current_closed_mask_boundaries.implementation import run as closed_boundaries
from skills.shared.perception.current_enclosed_rgb_edges.implementation import run as enclosed_edges
from skills.shared.perception.current_rgb_segmentation.implementation import prompted
from skills.shared.perception.current_semantic_descriptions.implementation import run as descriptions
from skills.shared.perception.current_visual_hull.implementation import diagnostic_inputs, diagnostic_regions, run as visual_hull
from skills.shared.perception.public_camera_projection.implementation import project_points
from skills.shared.perception.rgb_surface_geometry.implementation import pixel_rays


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


class InstanceAdmissionDiagnostics(list):
    pass


def appearance(image, region):
    mask = np.zeros((480, 640), np.uint8)
    contour = np.asarray(region['outline'], dtype=float)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 1)
    pixels = np.asarray(image)[mask > 0]
    return np.median(pixels, axis=0) if len(pixels) else np.zeros(3)


def snapshot(surface, description):
    points = np.asarray(surface['points_xyz'])
    center = np.asarray(surface['center_m'])
    unused_values, axes = np.linalg.eigh(np.cov(points[:, :2].T))
    long_axis = axes[:, 1]
    yaw = math.atan2(float(long_axis[1]), float(long_axis[0]))
    return {'source': 'current_public_rgb_visual_hull', 'description': description,
        'center_m': center.tolist(), 'spans_m': np.asarray(surface['spans_m']).tolist(),
        'height_estimate_m': float(surface['height_estimate_m']),
        'uncertainty_m': float(surface['uncertainty_m']), 'cameras': list(surface['cameras']),
        'coverage': list(surface['coverage']), 'head_outline': np.asarray(surface['head_region']['outline']).tolist(),
        'head_centroid': list(surface['head_region']['centroid']),
        'appearance_rgb': np.median(surface['rgb_samples'], axis=0).tolist(),
        'reference_pose': [*center, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
        'orientation_is_chosen_geometry_gauge': True, 'surface_is_true_object_geometry': False,
        'resolution_semantics': surface['resolution_semantics']}


def instances(api, description, support_height, support_uncertainty, require_supported=True, diagnostics=None):
    if (not isinstance(description, str) or not 1 <= len(description) <= 160
            or any(ord(character) < 32 for character in description)
            or type(require_supported) is not bool
            or diagnostics is not None and not isinstance(diagnostics, list)):
        raise ValueError('A current instance query uses only a bounded semantic description')
    detailed = isinstance(diagnostics, InstanceAdmissionDiagnostics)
    inputs = diagnostic_inputs(api) if detailed else None
    prompts = descriptions(description)
    regions = prompted(api, prompts)
    receipt = {'source': 'current_public_metric_instance_gate_diagnostics', 'inputs': inputs,
        'prompts': prompts, 'semantic': diagnostic_regions(regions),
        'counts_semantics': 'retained_current_prompted_regions_not_raw_SAM_proposals',
        'raw_semantic_counts_available': False, 'per_head': [], 'maximum_heads': 12} if detailed else None
    outputs = []
    for head_index, head in enumerate(regions['cam_head'][:12]):
        head_receipt = None
        if receipt is not None:
            head_receipt = {'semantic_index': head_index, 'association': {}, 'visual_hull': []}
            receipt['per_head'].append(head_receipt)
        if head['clipped']:
            if head_receipt is not None:
                head_receipt['gate_reason'] = 'clipped_head_semantics'
            continue
        colour = appearance(api.image('cam_head'), head)
        associated = {'cam_head': [head]}
        origin, rays = pixel_rays(api, 'cam_head', np.asarray([head['centroid']], dtype=float))
        if abs(rays[0, 2]) < 1e-8:
            if head_receipt is not None:
                head_receipt.update(gate_reason='head_ray_parallel_to_support',
                    ray_z=float(rays[0, 2]))
            continue
        distances = (np.linspace(support_height + .002, support_height + .30, 16) - origin[2]) / rays[0, 2]
        hypotheses = origin + distances[:, None] * rays[0]
        for camera in CAMERAS[1:]:
            predictions = project_points(api, camera, hypotheses)
            predictions = predictions[np.isfinite(predictions).all(axis=1)]
            costs = []
            def cost(row):
                geometric = float(np.min(np.linalg.norm(predictions - row['centroid'], axis=1))) if len(predictions) else 1000.
                colour_distance = np.linalg.norm(appearance(api.image(camera), row) - colour)
                total = geometric + .03 * colour_distance
                if head_receipt is not None:
                    costs.append({'semantic_index': next(index for index, region in enumerate(regions[camera])
                        if region is row), 'geometric_distance_px': geometric if np.isfinite(geometric) else None,
                        'colour_distance_rgb': float(colour_distance) if np.isfinite(colour_distance) else None,
                        'combined_cost': float(total) if np.isfinite(total) else None})
                return total
            associated[camera] = sorted([row for row in regions[camera] if not row['clipped']], key=cost)[:2]
            if head_receipt is not None:
                head_receipt['association'][camera] = {'finite_projected_hypotheses': len(predictions),
                    'excluded_clipped_indices': [index for index, row in enumerate(regions[camera]) if row['clipped']],
                    'evaluated_costs': costs[:12], 'cost_records_truncated': len(costs) > 12,
                    'selected_semantic_indices': [next(index for index, region in enumerate(regions[camera])
                        if region is row) for row in associated[camera]]}
        if head_receipt is None:
            surface = visual_hull(api, associated, support_height, support_uncertainty, require_supported)
        else:
            surface = visual_hull(api, associated, support_height, support_uncertainty, require_supported,
                diagnostics=head_receipt['visual_hull'])
        if surface is None:
            if head_receipt is not None:
                head_receipt['gate_reason'] = 'unqualified_visual_hull'
            continue
        if any(np.linalg.norm(surface['center_m'] - item['surface']['center_m']) < .015 for item in outputs):
            if head_receipt is not None:
                head_receipt['gate_reason'] = 'duplicate_current_metric_center'
            continue
        outputs.append({'anchor': snapshot(surface, description), 'surface': surface,
                        'regions': associated})
        if head_receipt is not None:
            head_receipt['gate_reason'] = 'qualified'
    if diagnostics is not None:
        missing_cameras = [camera for camera in CAMERAS[1:]
            if not regions[camera] or any(row['clipped'] for row in regions[camera])]
        summary = {'description': description,
            'heads': [{'head_centroid': list(row['centroid']), 'clipped': row['clipped'],
                'qualified': any(np.linalg.norm(np.asarray(row['centroid']) - item['anchor']['head_centroid']) < 1e-6
                    for item in outputs)} for row in regions['cam_head'][:12]],
            'unqualified_cameras': missing_cameras or list(CAMERAS[1:])}
        if receipt is not None:
            receipt.update(returned_instance_count=len(outputs), gate_reason='qualified_instances' if outputs
                else 'no_retained_head_semantics' if not regions['cam_head'] else 'no_qualified_instances')
            summary['instance_admission'] = receipt
        diagnostics.append(summary)
    return outputs


def fitted_openings(api, regions, extractor, upright_only, balance_independent_views=False):
    outlines = {camera: [dict(item, camera=camera) for region in regions[camera] if not region['clipped']
                for item in extractor(api.image(camera), region)][:8] for camera in CAMERAS}
    outputs = []
    for head in outlines['cam_head'][:8]:
        available = [camera for camera in CAMERAS[1:] if outlines[camera]]
        if not available:
            continue
        groups = ([available] if len(available) == 2 else []) + [[camera] for camera in available]
        attempts = 0
        for group in groups:
            group_attempts = 0
            group_budget = 4 if balance_independent_views and len(available) == 2 else 12
            for others in product(*(outlines[camera][:4] for camera in group)):
                frame = fit_boundary(api, [head, *others], upright_only=upright_only)
                attempts += 1
                group_attempts += 1
                if frame is not None:
                    if not any(np.linalg.norm(np.asarray(frame['pose'][:3]) - item['pose'][:3]) < .008 for item in outputs):
                        frame['head_outline'] = np.asarray(head['outline']).tolist()
                        if extractor is closed_boundaries:
                            frame['boundary_source'] = head['source']
                            frame['current_upper_edges_agree'] = all(item.get('observed_upper_body_edge') is True
                                for item in [head, *others])
                        outputs.append(frame)
                    break
                if attempts >= 12 or group_attempts >= group_budget:
                    break
            if frame is not None or attempts >= 12:
                break
    return outputs


def openings(api, regions, upright_only=False):
    original = fitted_openings(api, regions, enclosed_edges, upright_only)
    return original if original else fitted_openings(api, regions, closed_boundaries, upright_only,
        balance_independent_views=True)


def outside(region, boundary):
    polygon = np.asarray(boundary, dtype=np.float32)
    point = tuple(map(float, region['head_centroid']))
    return cv2.pointPolygonTest(polygon, point, False) < 0
