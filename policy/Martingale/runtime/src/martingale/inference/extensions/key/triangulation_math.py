# Copyright 2026 PyRUA-Lean Contributors. Apache-2.0.
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.
from __future__ import annotations
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    import numpy as np
_MIN_BASELINE_M = 0.001
_MIN_RAY_ANGLE_DEG = 5.0
_MAX_REPROJECTION_PX = 3.0
_MAX_RAY_DISAGREEMENT_M = 0.005
_MAX_PIXEL_SENSITIVITY_M = 0.005
_MIN_DEPTH_M = 1e-09

@dataclass(frozen=True)
class _View:
    name: str
    pixel: np.ndarray
    intrinsic: np.ndarray
    rotation: np.ndarray
    origin: np.ndarray

    def ray(self, pixel: np.ndarray) -> np.ndarray:
        import numpy as np
        cv = np.linalg.solve(self.intrinsic, [pixel[1], pixel[0], 1.0])
        local = np.array([cv[0], -cv[1], -cv[2]])
        direction = self.rotation @ local
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 0:
            raise ValueError('calibration produces a nonfinite or zero ray')
        return direction / length

    def project(self, point: np.ndarray) -> tuple[np.ndarray | None, float]:
        import numpy as np
        local = self.rotation.T @ (point - self.origin)
        depth = float(-local[2])
        if not math.isfinite(depth) or depth <= _MIN_DEPTH_M:
            return (None, depth)
        projected = self.intrinsic @ np.array([local[0], -local[1], depth])
        rc = np.array([projected[1], projected[0]]) / projected[2]
        return (rc if np.isfinite(rc).all() else None, depth)

def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    import numpy as np
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must have shape {shape} and finite numbers') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must have shape {shape} and finite numbers')
    return result

def _view(value: object, index: int) -> _View:
    import numpy as np
    label = f'views[{index}]'
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a camera observation dictionary')
    name = value.get('camera')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'{label}.camera must be a nonempty camera name')
    pixel = _array(value.get('pixel_rc'), (2,), f'{label}.pixel_rc')
    shape = _array(value.get('shape'), (3,), f'{label}.shape')
    if (shape != np.floor(shape)).any() or (shape[:2] < 2).any() or shape[2] != 3:
        raise ValueError(f'{label}.shape must be integer [height, width, 3], height/width >= 2')
    if (pixel < 0).any() or (pixel > shape[:2] - 1).any():
        raise ValueError(f'{label}.pixel_rc must be within the original image pixel bounds')
    intrinsic = _array(value.get('intrinsic_matrix'), (3, 3), f'{label}.intrinsic_matrix')
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f'{label}.intrinsic_matrix must have positive focal lengths')
    if not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-09) or not np.isclose(intrinsic[1, 0], 0, rtol=0, atol=1e-09):
        raise ValueError(f'{label}.intrinsic_matrix must be an upper-triangular pinhole matrix')
    pose = _array(value.get('extrinsic_matrix'), (4, 4), f'{label}.extrinsic_matrix')
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-09):
        raise ValueError(f'{label}.extrinsic_matrix must be a homogeneous camera-to-world pose')
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-05) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-05):
        raise ValueError(f'{label}.extrinsic_matrix must contain a proper rotation')
    return _View(name, pixel, intrinsic, rotation, pose[:3, 3])

def _intersect(origins: np.ndarray, directions: np.ndarray) -> np.ndarray | None:
    """Least-squares intersection, recentered to avoid unnecessary cancellation."""
    import numpy as np
    reference = origins[0]
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    lhs = projectors.sum(axis=0)
    rhs = np.einsum('nij,nj->i', projectors, origins - reference)
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        return None
    try:
        eigenvalues = np.linalg.eigvalsh(lhs)
        if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            return None
        point = np.linalg.solve(lhs, rhs) + reference
    except np.linalg.LinAlgError:
        return None
    return point if np.isfinite(point).all() else None

def triangulate_checked(views: Sequence[Mapping[str, Any]], *, pixel_error_px: float=2.0) -> dict[str, Any]:
    """Triangulate one matched feature from two or three distinct RGB cameras.

    Each view supplies ``camera``, ``pixel_rc`` (row, column in the ORIGINAL
    image), ``shape`` ([height, width, 3]), ``intrinsic_matrix`` (3x3), and
    ``extrinsic_matrix`` (4x4 camera-to-world). World distances are metres.
    The caller must obtain all calibrations and pixels from one synchronized
    observation, with the feature visible and physically identical in every
    view. This pure calculation cannot verify image identity or freshness.

    Acceptance requires a >=1 mm camera baseline, >=5 degree useful ray
    angle, <=3 px reprojection error in every view, <=5 mm disagreement
    between the closest points on the rays, and <=5 mm pixel sensitivity.
    ``ray_angle_deg`` is the largest acute angle between any pair of ray
    LINES: near-antiparallel rays are as ill-conditioned as parallel ones.

    Pixel sensitivity is the largest displacement under all independent
    +/-``pixel_error_px`` row/column corner perturbations (16 or 64 solves).
    Perturbations are not clipped at image boundaries. This samples the
    error box corners; it is NOT a guaranteed global bound or calibrated
    confidence interval, and excludes calibration/semantic matching errors.
    Use the default 2 px or a larger estimate when feature selection is less
    precise. The supported range is 1 through 5 px.

    Malformed inputs raise ValueError. Numerically unreliable measurements
    return ``accepted=False``, ``xyz=None``, and diagnostic ``reasons``;
    rejected candidate coordinates are deliberately never exposed as targets.
    All result values are JSON-safe, including when geometry is degenerate.
    Inputs are never modified.
    """
    import numpy as np
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or len(views) not in (2, 3):
        raise ValueError('views must contain two or three distinct camera observations')
    if isinstance(pixel_error_px, (bool, np.bool_)):
        raise ValueError('pixel_error_px must be a finite number between 1 and 5')
    try:
        pixel_error = float(pixel_error_px)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('pixel_error_px must be a finite number between 1 and 5') from exc
    if not math.isfinite(pixel_error) or not 1.0 <= pixel_error <= 5.0:
        raise ValueError('pixel_error_px must be a finite number between 1 and 5')
    parsed = [_view(value, index) for index, value in enumerate(views)]
    cameras = [view.name for view in parsed]
    if len(set(cameras)) != len(cameras):
        raise ValueError('views must use distinct camera names')
    reasons: list[str] = []
    result: dict[str, Any] = {'accepted': False, 'xyz': None, 'reason': None, 'reasons': reasons, 'cameras': cameras, 'pixel_error_px': pixel_error, 'baseline_m': None, 'ray_angle_deg': None, 'reprojection_errors_px': None, 'ray_residuals_m': None, 'ray_disagreement_m': None, 'pixel_sensitivity_m': None, 'thresholds': {'min_baseline_m': _MIN_BASELINE_M, 'min_ray_angle_deg': _MIN_RAY_ANGLE_DEG, 'max_reprojection_px': _MAX_REPROJECTION_PX, 'max_ray_disagreement_m': _MAX_RAY_DISAGREEMENT_M, 'max_pixel_sensitivity_m': _MAX_PIXEL_SENSITIVITY_M}}

    def reject(reason: str) -> dict[str, Any]:
        if reason not in reasons:
            reasons.append(reason)
        result['reason'] = '; '.join(reasons)
        return result
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        try:
            directions = np.stack([view.ray(view.pixel) for view in parsed])
        except (ValueError, np.linalg.LinAlgError):
            return reject('invalid_camera_rays')
        origins = np.stack([view.origin for view in parsed])
        pairs = list(itertools.combinations(range(len(parsed)), 2))
        baselines = [float(np.linalg.norm(origins[a] - origins[b])) for a, b in pairs]
        if not all((math.isfinite(value) for value in baselines)):
            return reject('nonfinite_camera_baseline')
        baseline = max(baselines)
        angle = max((math.degrees(math.acos(float(np.clip(abs(directions[a] @ directions[b]), 0, 1)))) for a, b in pairs))
        result.update(baseline_m=baseline, ray_angle_deg=angle)
        if baseline < _MIN_BASELINE_M:
            reasons.append('insufficient_camera_baseline')
        if angle < _MIN_RAY_ANGLE_DEG:
            reasons.append('insufficient_ray_angle')
        point = _intersect(origins, directions)
        if point is None:
            return reject('degenerate_ray_geometry')
        projections = [view.project(point) for view in parsed]
        if any((pixel is None for pixel, _ in projections)):
            return reject('point_not_in_front_of_all_cameras')
        errors = [float(np.linalg.norm(pixel - view.pixel)) for view, (pixel, _) in zip(parsed, projections, strict=True)]
        along = np.einsum('ni,ni->n', point - origins, directions)
        closest = origins + along[:, None] * directions
        residuals = np.linalg.norm(closest - point, axis=1)
        disagreement = max((float(np.linalg.norm(closest[a] - closest[b])) for a, b in pairs))
        if not (all((math.isfinite(value) for value in errors)) and np.isfinite(residuals).all() and math.isfinite(disagreement)):
            return reject('nonfinite_triangulation_diagnostics')
        result.update(reprojection_errors_px=errors, ray_residuals_m=[float(value) for value in residuals], ray_disagreement_m=disagreement)
        if max(errors) > _MAX_REPROJECTION_PX:
            reasons.append('excessive_reprojection_error')
        if disagreement > _MAX_RAY_DISAGREEMENT_M:
            reasons.append('excessive_ray_disagreement')
        sensitivity = 0.0
        for signs in itertools.product((-1.0, 1.0), repeat=2 * len(parsed)):
            offsets = pixel_error * np.asarray(signs).reshape(len(parsed), 2)
            try:
                perturbed_rays = np.stack([view.ray(view.pixel + offset) for view, offset in zip(parsed, offsets, strict=True)])
            except (ValueError, np.linalg.LinAlgError):
                return reject('unstable_pixel_perturbation')
            perturbed = _intersect(origins, perturbed_rays)
            if perturbed is None or any((view.project(perturbed)[0] is None for view in parsed)):
                return reject('unstable_pixel_perturbation')
            displacement = float(np.linalg.norm(perturbed - point))
            if not math.isfinite(displacement):
                return reject('unstable_pixel_perturbation')
            sensitivity = max(sensitivity, displacement)
        result['pixel_sensitivity_m'] = sensitivity
        if sensitivity > _MAX_PIXEL_SENSITIVITY_M:
            reasons.append('excessive_pixel_sensitivity')
    if reasons:
        result['reason'] = '; '.join(reasons)
    else:
        result.update(accepted=True, xyz=[float(value) for value in point])
    return result
