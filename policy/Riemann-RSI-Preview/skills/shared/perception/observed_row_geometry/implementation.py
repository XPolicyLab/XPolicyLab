from itertools import combinations
import math

import numpy as np


def run(centres, diameters, tangent, required_count):
    points = np.asarray(centres, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 16
            or diameters.shape != (len(points),) or tangent.shape != (2,)
            or not all(np.isfinite(value).all() for value in (points, diameters, tangent))
            or np.any(diameters <= 0) or np.linalg.norm(tangent) < 1e-8
            or type(required_count) is not int or not 2 <= required_count <= 8):
        raise ValueError('Group only a bounded current-image point family with observed feature widths')
    tangent = tangent / np.linalg.norm(tangent)
    rows = []
    for indices in combinations(range(len(points)), required_count):
        selected = points[list(indices)]
        center = selected.mean(axis=0)
        _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
        axis = axes[0]
        if abs(float(axis @ tangent)) < math.cos(math.radians(20)):
            continue
        if axis @ tangent < 0:
            axis = -axis
        normal = np.asarray([-axis[1], axis[0]])
        residual = float(np.max(np.abs((selected - center) @ normal)))
        width = float(np.median(diameters[list(indices)]))
        if residual > max(1., width * .25):
            continue
        along = selected @ axis
        ordering = np.argsort(along)
        gaps = np.diff(along[ordering])
        if np.min(gaps) <= width * .5 or np.max(gaps) > 1.7 * np.min(gaps):
            continue
        rows.append({'indices': [indices[int(index)] for index in ordering], 'centroid': center,
                     'axis': axis, 'normal': normal, 'maximum_residual_pixels': residual})
    return rows
