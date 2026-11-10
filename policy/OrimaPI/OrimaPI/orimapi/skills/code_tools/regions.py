"""Current-RGB outline proposals and conditional two-view horizontal circles."""

import cv2
import numpy as np
from scipy.optimize import least_squares


def crop_bounds(image, box):
    value = np.asarray(box)
    if value.shape != (4,) or value.dtype.kind not in "iu":
        raise ValueError("ROI must contain four integer original-image coordinates")
    x0, y0, x1, y1 = map(int, value)
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("ROI is outside the current image")
    if min(x1 - x0, y1 - y0) < 12:
        raise ValueError("ROI must be at least 12 pixels in each dimension")
    return x0, y0, x1, y1


def edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 100)


def outline_candidates(image, box):
    x0, y0, x1, y1 = crop_bounds(image, box)
    contours, _ = cv2.findContours(
        edges(image[y0:y1, x0:x1]), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    rows = []
    for contour in contours:
        points = contour[:, 0].astype(float)
        if len(points) < 24 or cv2.contourArea(contour) < 35:
            continue
        if (points.min(axis=0) < 2).any() or (
            points.max(axis=0) >= [x1 - x0 - 2, y1 - y0 - 2]
        ).any():
            continue
        center, diameters, angle = cv2.fitEllipse(contour)
        if min(diameters) < 7 or max(diameters) / min(diameters) > 5:
            continue
        radians = np.radians(angle)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        normalized = ((points - center) @ rotation) / (np.asarray(diameters) / 2)
        error = np.abs(np.linalg.norm(normalized, axis=1) - 1) * min(diameters) / 2
        if np.quantile(error, 0.9) > 2:
            continue
        if any(
            np.linalg.norm(np.asarray(row["ellipse_center_px"]) - center - [x0, y0]) < 2
            and np.max(np.abs(np.sort(row["diameters_px"]) - np.sort(diameters))) < 3
            for row in rows
        ):
            continue
        sample = points[np.linspace(0, len(points) - 1, 32).astype(int)] + [x0, y0]
        rows.append(
            {
                "points": sample.tolist(),
                "ellipse_center_px": (np.asarray(center) + [x0, y0]).tolist(),
                "diameters_px": list(map(float, diameters)),
                "edge_fit_error_px": float(np.quantile(error, 0.9)),
            }
        )
    return sorted(rows, key=lambda row: row["edge_fit_error_px"])[:8]


def _on_plane(camera, pixels, height):
    rays = (
        np.c_[pixels, np.ones(len(pixels))]
        @ np.linalg.inv(camera.k).T
        @ camera.optical[:3, :3].T
    )
    if (np.abs(rays[:, 2]) < 1e-5).any():
        raise ValueError("Outline rays nearly parallel to horizontal plane")
    distance = (height - camera.optical[2, 3]) / rays[:, 2]
    if (distance <= 0).any():
        raise ValueError("Horizontal circle is behind a selected camera")
    return camera.optical[:3, 3] + distance[:, None] * rays


def circle_pixels(camera, center, radius):
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = (
        np.asarray(center)
        + np.c_[radius * np.cos(angles), radius * np.sin(angles), np.zeros(128)]
    )
    return np.array([camera.project(point) for point in points])


def edge_support(image, camera, center, radius):
    pixels = circle_pixels(camera, center, radius)
    if (
        (pixels < 2).any()
        or (pixels[:, 0] >= image.shape[1] - 2).any()
        or (pixels[:, 1] >= image.shape[0] - 2).any()
    ):
        raise ValueError("Circle boundary clipped in a selected view")
    distance = cv2.distanceTransform(255 - edges(image), cv2.DIST_L2, 5)
    samples = cv2.remap(
        distance,
        pixels[:, 0].astype(np.float32)[None],
        pixels[:, 1].astype(np.float32)[None],
        cv2.INTER_LINEAR,
    )[0]
    support = float(np.mean(samples <= 3))
    if support < 0.8:
        raise ValueError("Current RGB no longer supports the fitted circle boundary")
    return support


def fit_circle(first, pixels_a, second, pixels_b, table):
    """Fit a shared circle to two unpaired outlines, never guessed metric scale."""
    cameras = (first, second)
    pixels = [np.asarray(p, dtype=float) for p in (pixels_a, pixels_b)]
    if any(
        p.ndim != 2
        or p.shape[1] != 2
        or not 12 <= len(p) <= 64
        or not np.isfinite(p).all()
        for p in pixels
    ):
        raise ValueError("Each outline needs 12..64 finite pixels")
    if np.linalg.norm(first.optical[:3, 3] - second.optical[:3, 3]) < 0.03:
        raise ValueError("Circle fit requires a camera baseline of at least 3 cm")
    upper = min(table + 0.5, *(camera.optical[2, 3] - 0.02 for camera in cameras))
    if upper <= table + 0.005:
        raise ValueError("Selected cameras must be above the horizontal circle")

    def residual(value, selections=pixels):
        return np.concatenate(
            [
                np.linalg.norm(
                    _on_plane(camera, p, value[2])[:, :2] - value[:2], axis=1
                )
                - value[3]
                for camera, p in zip(cameras, selections, strict=True)
            ]
        )

    results = []
    for z in np.linspace(table + 0.005, upper - 0.001, 4):
        xyz = np.concatenate(
            [_on_plane(c, p, z) for c, p in zip(cameras, pixels, strict=True)]
        )
        center = xyz[:, :2].mean(axis=0)
        radius = np.clip(
            np.linalg.norm(xyz[:, :2] - center, axis=1).mean(), 0.008, 0.19
        )
        result = least_squares(
            residual,
            [*np.clip(center, -0.7, 0.7), z, radius],
            bounds=([-0.75, -0.75, table, 0.005], [0.75, 0.75, upper, 0.2]),
            max_nfev=100,
        )
        if result.success:
            results.append(result)
    if not results:
        raise ValueError("Circle reconstruction did not converge")
    best = min(results, key=lambda row: np.linalg.norm(row.fun))
    center, radius = best.x[:3], float(best.x[3])
    reprojection = []
    for camera, selected in zip(cameras, pixels, strict=True):
        xyz = _on_plane(camera, selected, center[2])
        angles = np.sort(
            np.mod(np.arctan2(xyz[:, 1] - center[1], xyz[:, 0] - center[0]), 2 * np.pi)
        )
        if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() > 2 * np.pi / 3:
            raise ValueError("Outline does not cover enough of the circle")
        polygon = circle_pixels(camera, center, radius).astype(np.float32)
        distances = [
            abs(cv2.pointPolygonTest(polygon, tuple(point), True)) for point in selected
        ]
        reprojection.append(float(np.sqrt(np.mean(np.square(distances)))))
    if max(reprojection) > 2.5:
        raise ValueError("Two outlines disagree in RGB reprojection")
    singular = np.linalg.svd(best.jac, compute_uv=False)
    if singular[-1] < 1e-5 or singular[0] / singular[-1] > 1e5:
        raise ValueError("Circle depth is poorly constrained by these views")
    # Propagate 2-pixel independent selection errors through the fitted geometry.
    gradients = []
    for camera, p in zip(cameras, pixels, strict=True):
        gradients.append(
            np.stack(
                [
                    (
                        np.linalg.norm(
                            _on_plane(camera, p + np.eye(2)[axis] * 0.1, center[2])[
                                :, :2
                            ]
                            - center[:2],
                            axis=1,
                        )
                        - np.linalg.norm(
                            _on_plane(camera, p - np.eye(2)[axis] * 0.1, center[2])[
                                :, :2
                            ]
                            - center[:2],
                            axis=1,
                        )
                    )
                    / 0.2
                    for axis in range(2)
                ],
                axis=1,
            )
        )
    variance = 4 * np.square(np.concatenate(gradients)).sum(axis=1)
    inverse = np.linalg.pinv(best.jac)
    covariance = (inverse * variance) @ inverse.T
    uncertainty = float(2 * np.sqrt(np.trace(covariance[:3, :3]) + covariance[3, 3]))
    if uncertainty > min(0.012, radius * 0.3):
        raise ValueError("Circle measurement uncertainty is too large")
    return {
        "position": center.tolist(),
        "radius_m": radius,
        "method": "rgb_circle_fit",
        "reprojection_error_px": reprojection,
        "selection_uncertainty_m": uncertainty,
        "sensor_ground_truth": False,
        "assumptions": [
            "Both outlines select the same physical horizontal circular edge",
            "Calibration is accurate; uncertainty omits calibration error",
            "Geometric agreement does not establish an inner opening or containment",
        ],
    }
