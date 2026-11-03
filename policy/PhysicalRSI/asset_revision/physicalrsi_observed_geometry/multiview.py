"""Triangulate visible features using current RGB camera calibration."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from . import Camera, GeometryUnavailable


@dataclass
class Feature:
    center: np.ndarray
    direction: np.ndarray
    score: float
    contour: np.ndarray | None = None


def projection(camera):
    return camera.k @ np.diag([1., -1., -1.]) @ np.linalg.inv(camera.transform)[:3]


def triangulate(a, b, uv, vv, *, max_error=2.):
    uv, vv = np.asarray(uv, float), np.asarray(vv, float)
    pa, pb = projection(a), projection(b)
    homogeneous = cv2.triangulatePoints(pa, pb, uv[:, None], vv[:, None])[:, 0]
    if abs(homogeneous[3]) < 1e-10:
        raise GeometryUnavailable('Feature rays do not intersect at finite depth')
    point = homogeneous[:3] / homogeneous[3]
    rays = [point - c.transform[:3, 3] for c in (a, b)]
    cosine = np.dot(*rays) / (np.linalg.norm(rays[0]) * np.linalg.norm(rays[1]))
    if np.arccos(np.clip(cosine, -1, 1)) < np.deg2rad(3):
        raise GeometryUnavailable('Insufficient triangulation baseline')
    ua, _ = a.project([point])
    ub, _ = b.project([point])
    error = max(np.linalg.norm(ua[0] - uv), np.linalg.norm(ub[0] - vv))
    if not np.isfinite(error) or error > max_error:
        raise GeometryUnavailable('Inconsistent feature correspondence')
    return point, float(error)


def feature_axis(a, b, fa, fb):
    normals = []
    for camera, feature in ((a, fa), (b, fb)):
        line = np.cross(np.r_[feature.center, 1.],
                        np.r_[feature.center + feature.direction, 1.])
        normal = (projection(camera).T @ line)[:3]
        normals.append(normal / np.linalg.norm(normal))
    axis = np.cross(*normals)
    if np.linalg.norm(axis) < 1e-5:
        raise GeometryUnavailable('Feature orientation is ambiguous across views')
    return axis / np.linalg.norm(axis)


def _matched_features(observation, detector):
    views = []
    for name in observation['vision']:
        camera = Camera.from_observation(observation, name)
        views.append((camera, detector(camera.image)))
    matches = []
    for (a, aa), (b, bb) in combinations(views, 2):
        for fa in aa:
            for fb in bb:
                try:
                    point, error = triangulate(a, b, fa.center, fb.center)
                    axis = feature_axis(a, b, fa, fb)
                except GeometryUnavailable:
                    continue
                quality = np.sqrt(fa.score * fb.score) / (1 + error**2)
                matches.append((float(quality), point, axis, a, b, fa, fb))
    return sorted(matches, key=lambda item: item[0], reverse=True)


def match_features(observation, detector):
    return [item[:3] for item in _matched_features(observation, detector)]


def _boundary(poly, samples=12):
    return np.concatenate([a + np.linspace(0, 1, samples)[:, None] * (b - a)
                           for a, b in zip(poly, np.roll(poly, -1, axis=0))])


def _boundary_distance(points, poly):
    edges = np.roll(poly, -1, axis=0) - poly
    delta = points[:, None] - poly
    t = np.clip((delta * edges).sum(-1) / np.maximum((edges * edges).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * edges, axis=-1).min(1)


def fit_rectangle(views, center, axis):
    """Fit a visible rectangular opening; all dimensions and pose are unknown.

    Gravity initializes the plane, but the optimization may freely tilt it.
    Only calibrated RGB contours constrain the final result.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    normal = np.array([0., 0., 1.]) - axis * axis[2]
    if np.linalg.norm(normal) < 1e-5:
        raise GeometryUnavailable('Rectangle plane initialization is ambiguous')
    normal /= np.linalg.norm(normal)
    rotation = np.column_stack((axis, np.cross(normal, axis), normal))
    camera, hull = max(views, key=lambda view: cv2.contourArea(view[1].astype('float32')))
    depth = camera.project([center])[1][0]
    dimensions = np.sort(cv2.minAreaRect(hull.astype('float32'))[1])[::-1] * depth / camera.k[0, 0]
    if np.min(dimensions) <= 0:
        raise GeometryUnavailable('Rectangle contour is degenerate')
    unit = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])

    def residual(values):
        points = ((unit * np.r_[np.exp(values[6:8]), 1]) @
                  Rotation.from_rotvec(values[3:6]).as_matrix().T + values[:3])
        result = []
        for camera, hull in views:
            poly = camera.project(points)[0]
            result.extend(_boundary_distance(_boundary(hull), poly))
            result.extend(_boundary_distance(_boundary(poly), hull))
        return np.asarray(result)

    start = np.r_[center, Rotation.from_matrix(rotation).as_rotvec(), np.log(dimensions)]
    try:
        fit = least_squares(residual, start, loss='soft_l1', f_scale=.5,
                            max_nfev=400, diff_step=1e-6)
    except (ValueError, FloatingPointError) as exc:
        raise GeometryUnavailable('Rectangle fit left observable geometry') from exc
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    if not fit.success or not np.isfinite(rms) or rms > 2.:
        raise GeometryUnavailable('Rectangle contours do not agree across views')
    rotation = Rotation.from_rotvec(fit.x[3:6]).as_matrix()
    dimensions = np.exp(fit.x[6:8])
    if dimensions[1] > dimensions[0]:
        rotation = rotation[:, [1, 0, 2]] * np.array([1, -1, 1])
        dimensions = dimensions[::-1]
    return fit.x[:3], rotation, dimensions, rms


def match_rectangles(observation, detector):
    results = []
    for score, center, axis, a, b, fa, fb in _matched_features(observation, detector)[:3]:
        if fa.contour is None or fb.contour is None:
            continue
        try:
            center, rotation, dimensions, rms = fit_rectangle(
                [(a, fa.contour), (b, fb.contour)], center, axis)
        except GeometryUnavailable:
            continue
        results.append((score / (1 + rms**2), center, rotation, dimensions))
    return sorted(results, key=lambda item: item[0], reverse=True)
