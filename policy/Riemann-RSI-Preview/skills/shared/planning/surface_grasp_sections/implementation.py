import math

import numpy as np


def candidate_sections(points, colors, plane, height):
    points, colors = np.asarray(points, dtype=float), np.asarray(colors)
    if (points.ndim != 2 or points.shape[1] != 3 or not 16 <= len(points) <= 20000
            or colors.shape != points.shape or colors.dtype != np.uint8
            or not np.isfinite(points).all() or np.max(np.abs(points)) > 3.
            or not math.isfinite(plane) or not math.isfinite(height) or not 0. < height < .5):
        raise ValueError('Grasp sections require bounded current visible surface points and RGB samples')
    center = np.mean(points[:, :2], axis=0)
    unused_left, unused_values, axes = np.linalg.svd(points[:, :2] - center, full_matrices=False)
    long_axis = axes[0]
    across = np.array([-long_axis[1], long_axis[0]])
    along_values = (points[:, :2] - center) @ long_axis
    fractions = (.5, .38, .62, .26, .74, .12, .88)
    along_bounds = np.percentile(along_values, [2, 98])
    sections = []
    for fraction in fractions:
        value = along_bounds[0] + fraction * np.ptp(along_bounds)
        selection = np.abs(along_values - value) < max(.005, .04 * np.ptp(along_bounds))
        levels = (None, .55, .78, .32) if height > .065 else (None,)
        for level in levels:
            selected = selection.copy()
            if level is not None:
                selected &= np.abs(points[:, 2] - (plane + level * height)) < .018
            band = points[selected]
            if len(band) < 16:
                continue
            contact_xy = .5 * (band[:, :2].min(0) + band[:, :2].max(0))
            width = float(np.ptp(band[:, :2] @ across))
            if width > .082 or width < .004:
                continue
            minimum_z, maximum_z = np.percentile(band[:, 2], [3, 97])
            color = colors[selected].mean(0)
            material_penalty = .9 if width < .015 and np.mean(color) < 125 else 0.
            quality = .7 * abs(fraction - .5) + .6 * max(0., .02 - width) / .02 + material_penalty
            if level is not None:
                quality += .15 * abs(level - .55)
            sections.append((quality, fraction, band, contact_xy, width, minimum_z, maximum_z, level))
    sections.sort(key=lambda section: section[0])
    return sections


def run(payload):
    fields = {'points_xyz', 'rgb_samples', 'support_plane_m', 'current_height_estimate_m', 'cameras',
              'current_public_RGB_only', 'task_asset_points_used'}
    if (not isinstance(payload, dict) or set(payload) != fields
            or payload['current_public_RGB_only'] is not True or payload['task_asset_points_used'] is not False
            or not isinstance(payload['cameras'], list) or len(set(payload['cameras'])) < 2
            or not set(payload['cameras']) <= {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}):
        raise ValueError('Measured grasp geometry cannot use a mesh catalog, layout or privileged point cloud')
    return {'sections': candidate_sections(payload['points_xyz'], payload['rgb_samples'], payload['support_plane_m'],
                                           payload['current_height_estimate_m']),
            'robot_actions_issued': 0, 'task_asset_points_used': False, 'full_pickup_completed': False}
