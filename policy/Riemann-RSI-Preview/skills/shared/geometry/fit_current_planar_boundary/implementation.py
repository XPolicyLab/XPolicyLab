import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from skills.shared.perception.public_camera_projection.implementation import project_points
from skills.shared.perception.rgb_surface_geometry.implementation import pixel_rays, triangulate


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def tangent_basis(normal):
    normal = np.asarray(normal, dtype=float)
    candidate = np.asarray([1., 0., 0.]) if abs(normal[0]) < .8 else np.asarray([0., 1., 0.])
    first = np.cross(normal, candidate)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def intersections(origin, rays, normal, point):
    denominator = rays @ normal
    if np.any(np.abs(denominator) < .03):
        return None
    distance = np.dot(point - origin, normal) / denominator
    if not np.isfinite(distance).all() or np.any(distance <= .01) or np.any(distance >= 4.):
        return None
    return origin + rays * distance[:, None]


def run(api, observations, upright_only=False, *, reference_camera='cam_head'):
    if (not isinstance(observations, list) or not 2 <= len(observations) <= 3
            or observations[0].get('camera') != reference_camera
            or len({record.get('camera') for record in observations}) != len(observations)
            or any(record.get('camera') not in CAMERAS for record in observations)
            or type(upright_only) is not bool):
        return None
    outlines = [np.asarray(record.get('outline'), dtype=float) for record in observations]
    if any(value.shape != (96, 2) or not np.isfinite(value).all()
           or np.any(value <= 1) or np.any(value[:, 0] >= 638) or np.any(value[:, 1] >= 478) for value in outlines):
        return None
    centers = []
    for record, outline in zip(observations, outlines):
        center = np.asarray(record.get('projective_center', np.mean(outline, axis=0)), dtype=float)
        if (center.shape != (2,) or not np.isfinite(center).all()
                or 'projective_center' in record
                and cv2.pointPolygonTest(outline.astype(np.float32), tuple(center), False) <= 0):
            return None
        centers.append(center)
    seed = triangulate(api, reference_camera, centers[0], observations[1]['camera'], centers[1])
    if seed is None:
        return None
    origin, rays = pixel_rays(api, reference_camera, outlines[0])
    unused_origin, center_ray = pixel_rays(api, reference_camera, centers[0][None])
    ray = center_ray[0]
    depth = float(np.dot(np.asarray(seed['point']) - origin, ray))
    if not .03 <= depth <= 3.5:
        return None
    trees = [cKDTree(value) for value in outlines[1:]]
    seeds = ([np.asarray([0., 0., 1.])] if upright_only else
             [np.asarray(value, dtype=float) for value in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0])])
    accepted = []
    for normal_seed in seeds:
        first, second = tangent_basis(normal_seed)

        def model(parameters):
            normal = normal_seed + parameters[0] * first + parameters[1] * second
            normal /= np.linalg.norm(normal)
            point = origin + parameters[2] * ray
            return normal, intersections(origin, rays, normal, point)

        def residual(parameters):
            unused_normal, points = model(parameters)
            if points is None:
                return np.full(192 * len(trees), 100.)
            result = []
            for record, outline, tree in zip(observations[1:], outlines[1:], trees):
                predicted = project_points(api, record['camera'], points)
                if not np.isfinite(predicted).all():
                    return np.full(192 * len(trees), 100.)
                result.extend((tree.query(predicted)[0], cKDTree(predicted).query(outline)[0]))
            return np.concatenate(result)

        angular_bound = .08 if upright_only else 2.
        fit = least_squares(residual, [0., 0., depth],
            bounds=([-angular_bound, -angular_bound, max(.02, depth - .09)],
                    [angular_bound, angular_bound, min(3.8, depth + .09)]),
            max_nfev=65, diff_step=1e-4, loss='soft_l1', f_scale=.7)
        error = residual(fit.x)
        normal, points = model(fit.x)
        if points is None or np.sqrt(np.mean(error ** 2)) > 1.8 or np.mean(error < 2.5) < .8:
            continue
        center = np.mean(points, axis=0)
        if np.max(abs(points)) > 3. or not .003 <= np.max(np.ptp(points, axis=0)) <= 1.:
            continue
        if np.dot(normal, origin - center) < 0:
            normal = -normal
        accepted.append((float(np.mean(error ** 2)), center, normal, points, fit.jac))
    if not accepted:
        return None
    accepted.sort(key=lambda value: value[0])
    score, center, normal, points, jacobian = accepted[0]
    for other in accepted[1:]:
        disagreement = math.acos(float(np.clip(abs(np.dot(normal, other[2])), 0., 1.)))
        if abs(other[0] - score) < .12 and (np.linalg.norm(center - other[1]) > .008 or disagreement > .12):
            return None
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * max(.35, score)
    depth_error = max(.0005, float(np.sqrt(max(0., covariance[2, 2]))))
    axis_error = max(.003, float(np.sqrt(max(0., covariance[0, 0] + covariance[1, 1]))))
    if depth_error > .012 or axis_error > .12:
        return None
    first, second = tangent_basis(normal)
    local = (points - center) @ np.column_stack([first, second])
    unused_values, axes = np.linalg.eigh(np.cov(local.T))
    major = first * axes[0, 1] + second * axes[1, 1]
    minor = np.cross(normal, major)
    rotation = np.column_stack([major, minor, normal])
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return {'pose': np.r_[center, quaternion].tolist(), 'normal_xyz': normal.tolist(),
        'outline_xyz': points.tolist(), 'spans_m': np.ptp((points - center) @ rotation, axis=0).tolist(),
        'position_uncertainty_m': depth_error, 'axis_uncertainty_radians': axis_error,
        'pixel_rms': math.sqrt(score), 'cameras': [record['camera'] for record in observations],
        'source': 'current_independent_public_planar_boundary', 'orientation_is_chosen_geometry_gauge': True,
        'is_true_object_pose': False, 'is_contact_measurement': False, 'physical_asset_geometry_used': False}
