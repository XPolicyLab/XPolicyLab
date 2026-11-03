"""Measure a visible disk silhouette on its currently observed metric plane."""
import cv2
import numpy as np
from . import GeometryUnavailable


def observed_disk(camera, pixels, plane_point, plane_normal):
    pixels = np.asarray(pixels, float)
    point, normal = np.asarray(plane_point, float), np.asarray(plane_normal, float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 12
            or point.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(np.r_[pixels.ravel(), point, normal]).all()
            or np.linalg.norm(normal) < 1e-8):
        raise GeometryUnavailable('Disk pixels and an observed metric plane are required')
    if (np.any(pixels.min(0) <= 1)
            or np.any(pixels.max(0) >= [camera.width-2, camera.height-2])):
        raise GeometryUnavailable('A clipped disk silhouette does not determine its size')
    hull = cv2.convexHull(pixels.astype('float32')).reshape(-1, 2).astype(float)
    if len(hull) < 6 or min(np.ptp(hull, axis=0)) < 6:
        raise GeometryUnavailable('Disk silhouette is insufficiently resolved')
    normal = normal / np.linalg.norm(normal)
    origin = camera.transform[:3, 3]
    rays = camera.unproject(hull, np.ones(len(hull))) - origin
    denominator = rays @ normal
    if np.any(np.abs(denominator) < 1e-6):
        raise GeometryUnavailable('Disk plane is grazing its image rays')
    distances = ((point-origin) @ normal) / denominator
    if np.any(distances <= 0):
        raise GeometryUnavailable('Disk plane is behind the camera')
    world = origin + distances[:, None] * rays
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    axis -= normal * (axis @ normal)
    axis /= np.linalg.norm(axis)
    basis = np.column_stack([axis, np.cross(normal, axis)])
    xy = (world-point) @ basis
    design = np.c_[2*xy, np.ones(len(xy))]
    if np.linalg.matrix_rank(design) != 3:
        raise GeometryUnavailable('Disk boundary is degenerate')
    fit = np.linalg.lstsq(design, np.sum(xy*xy, axis=1), rcond=None)[0]
    radius_squared = fit[2] + fit[:2] @ fit[:2]
    if radius_squared <= 0:
        raise GeometryUnavailable('Disk radius is unresolved')
    radius = float(np.sqrt(radius_squared))
    residual = np.abs(np.linalg.norm(xy-fit[:2], axis=1)-radius)
    angles = np.sort(np.arctan2(*(xy-fit[:2])[:, ::-1].T))
    gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
    if np.quantile(residual, .9) > .12*radius or gap > np.pi/2:
        raise GeometryUnavailable('Observed boundary does not constrain a complete disk')
    return {'center': point+basis @ fit[:2], 'radius': radius,
            'boundary_residual': float(np.quantile(residual, .9)),
            'source': 'current_rgb_silhouette_and_observed_metric_plane'}
