"""Small transverse eye-in-hand corrections from observed slot and coin pixels."""
import numpy as np
from . import GeometryUnavailable


def stage_window(stages, name):
    """Half-open action interval, matching the incremental joint controller."""
    start = 0
    for stage in stages:
        end = start + int(stage['steps'])
        if stage['name'] == name:
            return start, end
        start = end
    return start, start


def transverse_correction(camera, plane_point, plane_normal, slot_center,
                          slot_axis, coin_pixels, *, maximum_step=.003):
    """Return an in-plane hand translation; this is not a release certificate.

    The wrist camera and held coin move together. Moving the camera changes the
    fixed slot's image position while the held coin's pixels remain stationary.
    Plane geometry must be measured, not supplied from asset metadata.
    """
    point, normal = np.asarray(plane_point, float), np.asarray(plane_normal, float)
    center, axis = np.asarray(slot_center, float), np.asarray(slot_axis, float)
    pixels = np.asarray(coin_pixels, float)
    if (point.shape != (3,) or normal.shape != (3,) or center.shape != (2,)
            or axis.shape != (2,) or pixels.ndim != 2 or pixels.shape[1] != 2
            or len(pixels) < 3 or not np.isfinite(np.r_[point,normal,center,axis,pixels.ravel()]).all()
            or np.linalg.norm(normal) < 1e-8 or np.linalg.norm(axis) < 1e-8
            or not np.isfinite(maximum_step) or maximum_step <= 0):
        raise GeometryUnavailable('Measured plane, slot line, and visible coin pixels required')
    normal = normal / np.linalg.norm(normal)
    axis = axis / np.linalg.norm(axis)
    transverse = np.array([-axis[1], axis[0]])
    # A median limits the influence of isolated specular pixels. Occlusion may
    # still bias the visible center, so alignment never authorizes release alone.
    coin = np.median(pixels, axis=0)
    error = float((coin-center) @ transverse)
    target = coin-error*transverse
    origin = camera.transform[:3, 3]
    rays = camera.unproject(np.array([coin,target]), np.ones(2))-origin
    denominator = rays @ normal
    if np.any(np.abs(denominator) < 1e-6):
        raise GeometryUnavailable('Slot plane is grazing the observed rays')
    distance = ((point-origin) @ normal)/denominator
    if np.any(distance <= 0):
        raise GeometryUnavailable('Slot plane is behind the camera')
    positions = origin+distance[:,None]*rays
    delta = positions[1]-positions[0]
    norm = float(np.linalg.norm(delta))
    delta *= min(1., maximum_step/max(norm,1e-12))
    return {'translation':delta, 'transverse_error_px':error,
            'coin_pixel':coin, 'target_pixel':target,
            'transverse_centered':abs(error)<=1., 'release_verified':False}
