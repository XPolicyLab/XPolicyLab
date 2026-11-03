"""Fit an unknown circular opening to calibrated RGB silhouettes.

The supplied normal is an explicit shape hypothesis (for example parallel to an
observed support), not an asset measurement. Center and radius come from images.
"""
from itertools import combinations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from . import GeometryUnavailable
from .multiview import triangulate


def fit_circle(views, normal):
    if len(views) < 2:
        raise GeometryUnavailable('Two separated calibrated views required')
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise GeometryUnavailable('Finite nonzero circle normal required')
    normal = normal / np.linalg.norm(normal)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    x = np.cross(normal, reference); x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    unit = np.cos(angles)[:, None]*x + np.sin(angles)[:, None]*y
    contours = []
    for camera, mask in views:
        mask = np.asarray(mask, bool)
        if mask.shape != (camera.height, camera.width) or not mask.any():
            raise GeometryUnavailable('Nonempty image-sized opening mask required')
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise GeometryUnavailable('Opening must be unclipped')
        components = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(components) != 1 or cv2.contourArea(components[0]) < 12:
            raise GeometryUnavailable('One resolved opening required per view')
        contours.append(components[0].reshape(-1, 2).astype(float))
    starts = []
    for i, j in combinations(range(len(views)), 2):
        try:
            center, _ = triangulate(views[i][0], views[j][0], contours[i].mean(0), contours[j].mean(0), max_error=3.)
            starts.append(center)
        except GeometryUnavailable:
            pass
    if not starts:
        raise GeometryUnavailable('Opening correspondence or camera baseline is ambiguous')
    center = np.median(starts, axis=0)
    radius = max(np.sqrt(cv2.contourArea(v.astype('float32'))/np.pi)*c.project([center])[1][0]/c.k[0, 0]
                 for (c, _), v in zip(views, contours))
    trees = [cKDTree(v) for v in contours]
    def residual(parameters):
        points = parameters[:3] + np.exp(parameters[3])*unit
        parts = []
        for (camera, _), contour, tree in zip(views, contours, trees):
            uv = camera.project(points)[0]
            parts.extend([cKDTree(uv).query(contour)[0], tree.query(uv)[0]])
        return np.concatenate(parts)
    try:
        fit = least_squares(residual, np.r_[center, np.log(radius)], loss='soft_l1',
                            f_scale=.5, max_nfev=400, diff_step=1e-5)
    except (ValueError, FloatingPointError) as exc:
        raise GeometryUnavailable('Circle fit left observable geometry') from exc
    radius = float(np.exp(fit.x[3]))
    points = fit.x[:3] + radius*unit
    errors = []
    for (camera, _), contour, tree in zip(views, contours, trees):
        uv = camera.project(points)[0]
        errors.append(float(np.sqrt(np.mean(np.r_[cKDTree(uv).query(contour)[0], tree.query(uv)[0]]**2))))
    # Scale translation columns by fitted radius to compare dimensionless sensitivity.
    singular = np.linalg.svd(fit.jac * np.array([radius, radius, radius, 1]), compute_uv=False)
    if (not fit.success or not np.isfinite(fit.x).all() or max(errors) > 1.5
            or singular[-1] < singular[0]*1e-3):
        raise GeometryUnavailable('Circle contours are inconsistent or geometry is ambiguous')
    return {'center': fit.x[:3], 'radius': radius, 'normal': normal,
            'rms_px': float(np.sqrt(np.mean(fit.fun**2))), 'view_rms_px': errors,
            'provenance': 'current calibrated RGB silhouettes',
            'shape_hypothesis': 'circular opening with supplied normal'}


def select_circle(view_groups, normal):
    """Associate opening proposals geometrically, rejecting competing solutions.

    Callers must scope proposals to the target object; geometry cannot distinguish
    two physically valid openings on different objects by semantic intent.
    """
    candidates = []
    for (a, masks_a), (b, masks_b) in combinations(view_groups, 2):
        for ma in masks_a:
            for mb in masks_b:
                try:
                    candidates.append(fit_circle([(a, ma), (b, mb)], normal))
                except GeometryUnavailable:
                    continue
    if not candidates:
        raise GeometryUnavailable('No consistent multiview opening proposals')
    candidates.sort(key=lambda value: value['rms_px'])
    best = candidates[0]
    for other in candidates[1:]:
        if other['rms_px'] > best['rms_px'] + .5:
            continue
        # Relative geometric separation, independent of any known object size.
        scale = min(best['radius'], other['radius'])
        if (np.linalg.norm(best['center']-other['center']) > .2*scale
                or abs(best['radius']-other['radius']) > .2*scale):
            raise GeometryUnavailable('Multiple plausible opening correspondences')
    return best
