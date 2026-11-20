import math

import numpy as np

from skills.shared.perception.public_camera_projection.implementation import points_on_plane


def signature(horizontal):
    horizontal = np.asarray(horizontal, dtype=float)
    if horizontal.ndim != 2 or horizontal.shape[1] != 2 or len(horizontal) < 30 or not np.isfinite(horizontal).all():
        return None
    mean = horizontal.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(horizontal - mean, rowvar=False))
    axis = vectors[:, int(np.argmax(values))]
    projections = (horizontal - mean) @ axis
    normal = np.asarray([-axis[1], axis[0]])
    transverse = (horizontal - mean) @ normal
    low, high = projections < np.quantile(projections, .2), projections > np.quantile(projections, .8)
    if low.sum() < 2 or high.sum() < 2:
        return None
    low_width, high_width = float(np.ptp(transverse[low])), float(np.ptp(transverse[high]))
    if min(low_width, high_width) <= 0 or max(low_width, high_width) / min(low_width, high_width) < 1.4:
        return None
    if high_width > low_width:
        axis = -axis
    normal = np.asarray([-axis[1], axis[0]])
    longitudinal, transverse = (horizontal - mean) @ axis, (horizontal - mean) @ normal
    center = mean + axis * (longitudinal.min() + longitudinal.max()) * .5
    center += normal * (transverse.min() + transverse.max()) * .5
    return {'center': center, 'signed_axis': axis, 'yaw_radians': math.atan2(axis[1], axis[0]),
            'width_ratio': max(low_width, high_width) / min(low_width, high_width)}


def pose(api, pixels, top_height, support_height):
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 30 or not np.isfinite(pixels).all():
        return None
    indices = np.linspace(0, len(pixels) - 1, min(1024, len(pixels))).astype(int)
    world = points_on_plane(api, 'cam_head', pixels[indices], top_height)
    measured = signature(world[:, :2]) if np.isfinite(world).all() else None
    if measured is None:
        return None
    center, axis = measured['center'], measured['signed_axis']
    thickness = top_height - support_height
    if not .003 <= thickness <= .04:
        return None
    yaw = measured['yaw_radians']
    return {'pose': np.asarray([*center, (top_height + support_height) * .5,
                               math.cos(yaw * .5), 0., 0., math.sin(yaw * .5)]),
            'visible_thickness_m': float(thickness), 'signed_axis': axis,
            'source': 'current_public_silhouette_and_qualified_surface'}
