"""Conditional two-view circular-rim estimates from current RGB selections.

No object poses, depth, task names, archived pixels or task-specific dimensions
are inputs. A fitted circle is not proof of an inner opening, grasp or success.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .regions import _on_plane, edges

MAX_TILT = np.radians(20)


def _basis(normal):
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("Rim normal must contain three finite coordinates")
    if abs(np.linalg.norm(normal) - 1) > 1e-5 or normal[2] < np.cos(MAX_TILT):
        raise ValueError(
            "Only near-upward circular rims within 20 degrees are supported"
        )
    u = np.cross([0.0, 1.0, 0.0], normal)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u)))


def rim_pixels(camera, center, radius, normal):
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Rim center must contain three finite coordinates")
    if not np.isfinite(radius) or not 0.005 <= radius <= 0.2:
        raise ValueError("Rim radius must be 0.005..0.2 m")
    angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    world = center + radius * np.c_[np.cos(angles), np.sin(angles)] @ _basis(normal).T
    local = np.c_[world, np.ones(len(world))] @ camera.view[:3].T
    if (local[:, 2] <= 0).any():
        raise ValueError("Rim boundary is behind a selected camera")
    projected = local @ camera.k.T
    return projected[:, :2] / projected[:, 2, None]


def rim_edge_support(image, camera, center, radius, normal):
    if image.shape != (camera.height, camera.width, 3):
        raise ValueError("RGB dimensions disagree with camera calibration")
    pixels = rim_pixels(camera, center, radius, normal)
    if (
        (pixels < 2).any()
        or (pixels[:, 0] >= camera.width - 2).any()
        or (pixels[:, 1] >= camera.height - 2).any()
    ):
        raise ValueError("Rim boundary is clipped in a selected view")
    distance = cv2.distanceTransform(255 - edges(image), cv2.DIST_L2, 5)
    samples = cv2.remap(
        distance,
        pixels[:, 0].astype(np.float32)[None],
        pixels[:, 1].astype(np.float32)[None],
        cv2.INTER_LINEAR,
    )[0]
    support = float(np.mean(samples <= 3))
    if support < 0.8:
        raise ValueError("Current RGB does not support the fitted rim boundary")
    return support


def _selected(camera, values):
    pixels = np.asarray(values, dtype=float)
    if (
        pixels.ndim != 2
        or pixels.shape[1] != 2
        or not 12 <= len(pixels) <= 48
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("Select 12..48 finite rim pixels in each view")
    if (
        (pixels < 2).any()
        or (pixels[:, 0] >= camera.width - 2).any()
        or (pixels[:, 1] >= camera.height - 2).any()
    ):
        raise ValueError("Selected rim pixels are outside calibrated image bounds")
    distances = np.linalg.norm(pixels[:, None] - pixels[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 2:
        raise ValueError("Rim selections must be distinct and at least 2 pixels apart")
    return pixels


def _coverage(camera, pixels, center, basis):
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.k).T
    rays = rays @ camera.optical[:3, :3].T
    normal = np.cross(basis[:, 0], basis[:, 1])
    denominator = rays @ normal
    if (np.abs(denominator) < 1e-5).any():
        raise ValueError("Rim rays are nearly parallel to its estimated plane")
    distance = np.dot(center - camera.optical[:3, 3], normal) / denominator
    if (distance <= 0).any():
        raise ValueError("Rim plane lies behind a selected camera")
    local = (camera.optical[:3, 3] + distance[:, None] * rays - center) @ basis
    angles = np.sort(np.mod(np.arctan2(local[:, 1], local[:, 0]), 2 * np.pi))
    gap = float(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max())
    if gap > 2 * np.pi / 3:
        raise ValueError("Rim selection does not cover enough of the circle")
    return float(np.degrees(gap))


def fit_rim(first, pixels_a, second, pixels_b, table):
    """Fit an oriented circle; RGB edge checks must additionally pass per view."""
    cameras = (first, second)
    pixels = [
        _selected(c, p) for c, p in zip(cameras, (pixels_a, pixels_b), strict=True)
    ]
    if not np.isfinite(table):
        raise ValueError("A finite public table height is required")
    if np.linalg.norm(first.optical[:3, 3] - second.optical[:3, 3]) < 0.03:
        raise ValueError("Rim fit requires a camera baseline of at least 3 cm")
    upper = min(table + 0.5, *(c.optical[2, 3] - 0.02 for c in cameras))
    if upper <= table + 0.01:
        raise ValueError("Selected cameras must be above the rim")
    homogeneous = [np.c_[p, np.ones(len(p))] for p in pixels]

    def residual(value):
        basis = Rotation.from_euler("XY", value[4:]).as_matrix()[:, :2]
        plane = np.vstack((np.column_stack((value[3] * basis, value[:3])), [0, 0, 1]))
        errors = []
        for camera, p in zip(cameras, homogeneous, strict=True):
            try:
                inverse = np.linalg.inv(camera.projection @ plane)
            except np.linalg.LinAlgError:
                return np.full(sum(map(len, pixels)), 1e6)
            conic = inverse.T @ np.diag([1.0, 1.0, -1.0]) @ inverse
            qp = p @ conic
            # Sampson distance in original calibrated pixel coordinates.
            errors.append(
                np.sum(qp * p, axis=1)
                / np.maximum(2 * np.linalg.norm(qp[:, :2], axis=1), 1e-12)
            )
        return np.concatenate(errors)

    results = []
    for z in np.linspace(table + 0.005, upper - 0.001, 4):
        try:
            xyz = np.concatenate([_on_plane(c, p, z) for c, p in zip(cameras, pixels)])
        except ValueError:
            continue
        center = xyz[:, :2].mean(axis=0)
        radius = np.clip(
            np.linalg.norm(xyz[:, :2] - center, axis=1).mean(), 0.008, 0.19
        )
        for tilt in ((0, 0), (0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15)):
            result = least_squares(
                residual,
                [*np.clip(center, -0.7, 0.7), z, radius, *tilt],
                bounds=(
                    [-0.75, -0.75, table, 0.005, -MAX_TILT, -MAX_TILT],
                    [0.75, 0.75, upper, 0.2, MAX_TILT, MAX_TILT],
                ),
                max_nfev=100,
            )
            if result.success and np.isfinite(result.fun).all():
                results.append(result)
    if not results:
        raise ValueError("Rim reconstruction did not converge")
    best = min(results, key=lambda r: np.linalg.norm(r.fun))
    center, radius = best.x[:3], float(best.x[3])
    rotation = Rotation.from_euler("XY", best.x[4:]).as_matrix()
    normal = rotation[:, 2]
    basis = _basis(normal)
    if best.active_mask.any():
        raise ValueError("Rim fit is constrained by workspace or orientation bounds")
    if center[2] - radius * np.linalg.norm(normal[:2]) < table:
        raise ValueError("Estimated rim crosses the public table plane")
    coverage, reprojection = [], []
    for camera, selected in zip(cameras, pixels, strict=True):
        coverage.append(_coverage(camera, selected, center, basis))
        polygon = rim_pixels(camera, center, radius, normal).astype(np.float32)
        distances = [
            abs(cv2.pointPolygonTest(polygon, tuple(point), True)) for point in selected
        ]
        reprojection.append(float(np.sqrt(np.mean(np.square(distances)))))
    if max(reprojection) > 2.5:
        raise ValueError("Two rim selections disagree in RGB reprojection")
    singular = np.linalg.svd(best.jac, compute_uv=False)
    if singular[-1] < 1e-5 or singular[0] / singular[-1] > 1e5:
        raise ValueError("Rim geometry is poorly constrained by these views")
    # Independent 2-pixel selection noise, not calibration/systematic error.
    inverse = np.linalg.pinv(best.jac)
    covariance = 4 * inverse @ inverse.T
    uncertainty = float(2 * np.sqrt(np.trace(covariance[:4, :4])))
    angular_uncertainty = float(2 * np.sqrt(np.trace(covariance[4:, 4:])))
    if uncertainty > min(0.012, radius * 0.3) or angular_uncertainty > 0.15:
        raise ValueError("Rim measurement uncertainty is too large")
    rms = float(np.sqrt(np.mean(best.fun**2)))
    for alternative in results:
        if np.sqrt(np.mean(alternative.fun**2)) > rms + 0.25:
            continue
        if np.linalg.norm(alternative.x[:4] - best.x[:4]) > uncertainty:
            raise ValueError("Two-view rim reconstruction has ambiguous solutions")
    return {
        "position": center.tolist(),
        "radius_m": radius,
        "normal": normal.tolist(),
        "method": "rgb_oriented_rim_fit",
        "reprojection_error_px": reprojection,
        "largest_unobserved_arc_degrees": coverage,
        "selection_uncertainty_m": uncertainty,
        "orientation_uncertainty_rad": angular_uncertainty,
        "sensor_ground_truth": False,
        "assumptions": [
            "Selections identify the SAME circular physical boundary in both views",
            "Rim normal is within 20 degrees of world-up; the rim is above the table",
            "Uncertainty assumes independent 2-pixel errors and omits calibration bias",
            "Boundary agreement does not certify the inner opening or object identity",
            "A fitted rim does not prove clearance, containment, retention or task success",
        ],
    }
