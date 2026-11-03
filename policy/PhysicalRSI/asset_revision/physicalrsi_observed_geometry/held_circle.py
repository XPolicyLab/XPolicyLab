"""Unknown circular-object geometry from calibrated views during a rigid grasp."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from . import GeometryUnavailable
from .multiview import _boundary_distance


@dataclass
class HeldCircleFit:
    center: np.ndarray
    normal: np.ndarray
    radius: float
    pixel_rms: float
    view_errors: list


def fit_held_circle(views, initial_center, initial_radius):
    """Views are (Camera, observed world EE transform, visible boundary pixels).

    Fits the object's center/normal/radius in the EE frame. There is no object
    model, known radius, table height, or fixed grasp attachment transform.
    Initial size is an image-derived proposal, not a fixed asset dimension.
    """
    if len(views) < 4 or not np.isfinite(initial_radius) or initial_radius <= 0:
        raise GeometryUnavailable('Multiple calibrated views and visible size required')
    center = np.asarray(initial_center, float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise GeometryUnavailable('Finite initial relative position required')
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))

    def geometry(values):
        yaw, tilt = values[3:5]
        normal = np.array([np.cos(yaw) * np.cos(tilt),
                           np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
        x = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        rotation = np.column_stack((x, np.cross(normal, x), normal))
        return unit * np.exp(values[5]) @ rotation.T + values[:3], normal

    def residual(values):
        points, _ = geometry(values)
        errors = []
        for camera, ee, boundary in views:
            world = points @ ee[:3, :3].T + ee[:3, 3]
            projected = camera.project(world)[0]
            errors.extend(_boundary_distance(boundary, projected))
        return np.asarray(errors)

    solutions = []
    for yaw in (0., np.pi / 2):
        start = np.r_[center, yaw, 0., np.log(initial_radius)]
        lower = np.r_[center - 3 * initial_radius, yaw - np.pi / 2, -1.5,
                      np.log(initial_radius / 3)]
        upper = np.r_[center + 3 * initial_radius, yaw + np.pi / 2, 1.5,
                      np.log(initial_radius * 3)]
        try:
            fitted = least_squares(residual, start, bounds=(lower, upper),
                                   loss='soft_l1', f_scale=.75,
                                   max_nfev=350, diff_step=1e-6)
        except (GeometryUnavailable, ValueError, FloatingPointError):
            continue
        if fitted.success and np.isfinite(fitted.fun).all():
            solutions.append(fitted)
    if not solutions:
        raise GeometryUnavailable('No consistent held-circle fit')
    fitted = min(solutions, key=lambda solution: solution.cost)
    points, normal = geometry(fitted.x)
    view_errors = []
    for camera, ee, boundary in views:
        projected = camera.project(points @ ee[:3, :3].T + ee[:3, 3])[0]
        errors = _boundary_distance(boundary, projected)
        view_errors.append(float(np.sqrt(np.mean(errors**2))))
    return HeldCircleFit(fitted.x[:3], normal, float(np.exp(fitted.x[5])),
                         float(np.sqrt(np.mean(fitted.fun**2))), view_errors)
