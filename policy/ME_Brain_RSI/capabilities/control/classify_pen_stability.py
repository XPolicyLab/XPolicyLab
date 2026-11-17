"""Select stable material-interior pen points after the retained observer merge."""
from copy import copy

import cv2
import numpy as np


CLEARANCE_WEIGHT = 4.0


def select_material_pixel(entity_mask, bbox):
    """Soft-rank every material pixel by centroid distance and thickness loss."""
    mask = np.asarray(entity_mask, np.uint8)
    rows, columns = np.nonzero(mask)
    pixels = np.column_stack((columns, rows)).astype(float)
    centroid = pixels.mean(axis=0)
    clearance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radii = clearance[rows, columns].astype(float)
    maximum = float(radii.max())
    center_cost = np.sum((pixels - centroid) ** 2, axis=1)
    thickness_cost = CLEARANCE_WEIGHT * (maximum - radii) ** 2
    costs = center_cost + thickness_cost
    selected_index = int(np.argmin(costs))
    local = pixels[selected_index]
    origin = np.asarray(bbox[:2], float)
    selected = local + origin
    centered = pixels - centroid
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0]
    axial = centered[:, 0] * axis[0] + centered[:, 1] * axis[1]
    axial_span = float(np.ptp(axial))
    evidence = dict(method='pen_continuous_centroid_thickness_soft_ranking',
        clearance_weight=CLEARANCE_WEIGHT, material_pixels=int(len(pixels)),
        material_centroid_pixel=(centroid + origin).tolist(),
        selected_pixel=selected.tolist(), selected_mask_value=int(mask[rows[selected_index], columns[selected_index]]),
        selected_clearance_pixels=float(radii[selected_index]), maximum_clearance_pixels=maximum,
        selected_axial_offset_pixels=float(np.sum((local - centroid) * axis)),
        axial_span_pixels=axial_span, selected_center_cost=float(center_cost[selected_index]),
        selected_thickness_cost=float(thickness_cost[selected_index]),
        selected_soft_cost=float(costs[selected_index]))
    return selected, evidence


def refine_pen_observation(ctx, item, camera='cam_head', image_shape=(480, 640)):
    """Refine only final pen XY using its existing material mask and calibration."""
    if item['category'] != 'pen':
        return item, None
    selected, evidence = select_material_pixel(item['entity_mask'], item['bbox'])
    point = np.asarray(ctx.perception.project(selected, float(item['surface_z']), camera,
                                            ctx.state, image_shape), float)
    previous = np.asarray(item['grasp_position'], float)
    point[2] = previous[2]
    result = copy(item)
    result['grasp_pixel'] = tuple(selected)
    result['grasp_position'] = point.copy()
    evidence.update(original_grasp_pixel=np.asarray(item['grasp_pixel'], float).tolist(),
        original_grasp_position=previous.tolist(), selected_grasp_position=point.tolist(),
        displacement_m=float(np.linalg.norm(point - previous)),
        category_unchanged=True, detection_and_merge_unchanged=True,
        height_unchanged=True, yaw_unchanged=True, source_anchor_unchanged=True,
        source_identity_algorithm_unchanged=True)
    return result, evidence
