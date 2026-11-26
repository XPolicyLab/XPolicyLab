"""Conditional sphere reconstruction from current RGB tangent silhouettes.

No object coordinates, radius, height, depth, segmentation or task names are
inputs. Different views need not select the same physical surface points.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from .regions import edges as grayscale_edges


def edges(image):
    """Original luminance OR color-channel gradients; no dilation or snapping.

    Equal-luminance colored boundaries are real RGB evidence that a grayscale
    conversion can discard. Keep the original blur/Canny thresholds, and keep
    every subsequent distance, model, sensitivity and motion check unchanged.
    Other geometry tools retain their existing edge extractor.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Sphere edges require original uint8 RGB")
    return np.maximum.reduce([
        grayscale_edges(image),
        *[cv2.Canny(cv2.GaussianBlur(channel, (3, 3), 0), 35, 100)
          for channel in cv2.split(image)],
    ])


class BoundarySupportError(ValueError):
    """The existing edge gate failed; diagnostics never repair the input."""

    def __init__(self, diagnostics):
        super().__init__("Selected sphere boundary lacks original RGB edge support")
        self.diagnostics = diagnostics


def boundary_diagnostics(distance, pixels):
    values = sample_distance(distance, pixels)
    unsupported = [int(i) for i in np.flatnonzero(values > 2)]
    has_edges = bool(np.any(distance == 0))
    examples = []
    height, width = distance.shape
    for index in sorted(unsupported, key=lambda i: (-float(values[i]), i))[:8]:
        x, y = pixels[index]
        x0, x1 = max(2, int(np.floor(x)) - 16), min(width - 2, int(np.ceil(x)) + 17)
        y0, y1 = max(2, int(np.floor(y)) - 16), min(height - 2, int(np.ceil(y)) + 17)
        yy, xx = np.nonzero(distance[y0:y1, x0:x1] == 0)
        points = np.c_[xx + x0, yy + y0]
        lengths = np.linalg.norm(points - [x, y], axis=1)
        proposals = []
        for candidate in np.argsort(lengths, kind="stable"):
            point = points[candidate]
            if lengths[candidate] > 16:
                break
            if any(np.linalg.norm(point - row["pixel"]) < 3 for row in proposals):
                continue
            proposals.append({
                "pixel": point.astype(int).tolist(),
                "euclidean_distance_px": float(lengths[candidate]),
                "accepted_boundary": False,
            })
            if len(proposals) == 3:
                break
        examples.append({
            "index": index, "selected_pixel": pixels[index].tolist(),
            "distance_transform_sample_px": float(values[index]) if has_edges else None,
            "nearby_edge_proposals": proposals,
        })
    supported = float(np.mean(values <= 2))
    return {
        "selected_count": len(pixels),
        "supported_fraction": supported,
        "passes_edge_fraction_gate": supported >= 0.9,
        "maximum_distance_px": float(values.max()) if has_edges else None,
        "edge_map_has_samples": has_edges,
        "unsupported_indices": unsupported,
        "unsupported_examples": examples,
        "point_distance_gate_px": 2,
        "required_supported_fraction": 0.9,
        "proposal_search_radius_px": 16,
        "scope": (
            "Current original-image luminance/R/G/B edge proximity ONLY. A nearby edge may be "
            "a shaft, reflection, occluder or background, not the sphere boundary. "
            "Proposals are not accepted points or geometry; inspect original RGB. "
            "A passing view need not be changed while its frame is unchanged."
        ),
    }


def selected(camera, values):
    pixels = np.asarray(values, dtype=float)
    if (
        pixels.ndim != 2 or pixels.shape[1] != 2 or not 8 <= len(pixels) <= 64
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("Select 8..64 finite sphere-boundary pixels per view")
    if (
        (pixels < 2).any() or (pixels[:, 0] >= camera.width - 2).any()
        or (pixels[:, 1] >= camera.height - 2).any()
    ):
        raise ValueError("Sphere selections must be inside the original image")
    distances = np.linalg.norm(pixels[:, None] - pixels[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 0.99:
        raise ValueError("Sphere selections must be distinct, at least one pixel apart")
    return pixels


def rays(camera, pixels):
    value = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.k).T
    return value / np.linalg.norm(value, axis=1)[:, None]


def cone(camera, pixels):
    directions = rays(camera, pixels)
    _, _, vh = np.linalg.svd(directions - directions.mean(axis=0))
    axis = vh[-1]
    if np.dot(axis, directions.mean(axis=0)) < 0:
        axis = -axis
    cosine = np.clip(np.mean(directions @ axis), -1, 1)
    return camera.optical[:3, :3] @ axis, np.sqrt(1 - cosine * cosine)


def basis(axis):
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def sphere_pixels(camera, center, radius, count=256):
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius):
        raise ValueError("Sphere parameters must be finite")
    if not 0.002 <= radius <= 0.15:
        raise ValueError("Sphere radius must be 0.002..0.15 m")
    c = (camera.view @ np.r_[center, 1])[:3]
    distance = np.linalg.norm(c)
    if distance <= radius or c[2] <= radius:
        raise ValueError("Sphere must be fully in front of a selected camera")
    axis = c / distance
    u, v = basis(axis)
    sine = radius / distance
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    directions = np.sqrt(1 - sine * sine) * axis + sine * (
        np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    )
    projected = directions @ camera.k.T
    return projected[:, :2] / projected[:, 2, None]


def coverage(camera, pixels, center):
    c = (camera.view @ np.r_[center, 1])[:3]
    u, v = basis(c / np.linalg.norm(c))
    directions = rays(camera, pixels)
    angles = np.sort(np.mod(np.arctan2(directions @ v, directions @ u), 2 * np.pi))
    largest_gap = float(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max())
    if largest_gap > np.pi / 2:
        raise ValueError("Sphere boundary selections must cover at least 270 degrees")
    return float(np.degrees(2 * np.pi - largest_gap))


def edge_distance(image, camera):
    if image.dtype != np.uint8 or image.shape != (camera.height, camera.width, 3):
        raise ValueError("Original RGB dimensions must match public camera calibration")
    return cv2.distanceTransform(255 - edges(image), cv2.DIST_L2, 5)


def sample_distance(distance, pixels):
    return cv2.remap(
        distance, pixels[:, 0].astype(np.float32)[None],
        pixels[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR,
    )[0]


def sphere_edge_support(image, camera, center, radius):
    distance = edge_distance(image, camera)
    pixels = sphere_pixels(camera, center, radius)
    inside = (
        (pixels[:, 0] >= 2) & (pixels[:, 0] < camera.width - 2)
        & (pixels[:, 1] >= 2) & (pixels[:, 1] < camera.height - 2)
    )
    if inside.mean() < 0.75:
        raise ValueError("Too much of the sphere silhouette is outside the current image")
    support = float(np.mean(sample_distance(distance, pixels[inside]) <= 2))
    if support < 0.85:
        raise ValueError("Current original RGB does not support the sphere silhouette")
    return {"visible_fraction": float(inside.mean()), "edge_support_fraction": support}


def fit_sphere(cameras, images, values, table):
    if not len(cameras) == len(images) == len(values) == 2:
        raise ValueError("Sphere reconstruction requires exactly two current views")
    pixels = [selected(c, p) for c, p in zip(cameras, values)]
    if not np.isfinite(table):
        raise ValueError("Finite public table height is required")
    origins = [c.optical[:3, 3] for c in cameras]
    if np.linalg.norm(origins[0] - origins[1]) < 0.03:
        raise ValueError("Sphere reconstruction requires a baseline of at least 3 cm")
    boundary_reports = [
        boundary_diagnostics(edge_distance(image, camera), p)
        for camera, image, p in zip(cameras, images, pixels)
    ]
    if any(not row["passes_edge_fraction_gate"] for row in boundary_reports):
        raise BoundarySupportError(boundary_reports)
    axes, sine = zip(*(cone(c, p) for c, p in zip(cameras, pixels)))
    parallax = float(np.arccos(np.clip(np.dot(*axes), -1, 1)))
    if parallax < np.radians(3):
        raise ValueError("Sphere viewing directions have insufficient parallax")
    projectors = [np.eye(3) - np.outer(axis, axis) for axis in axes]
    if np.linalg.cond(sum(projectors)) > 1e5:
        raise ValueError("Sphere viewing geometry is poorly conditioned")
    center = np.linalg.solve(
        sum(projectors), sum(a @ origin for a, origin in zip(projectors, origins))
    )
    radius = np.median([
        np.linalg.norm(center - origin) * s for origin, s in zip(origins, sine)
    ])
    inverse_k = [np.linalg.inv(c.k) for c in cameras]

    def residual(parameters, sets):
        center, radius = parameters[:3], parameters[3]
        results = []
        for camera, inverse, points in zip(cameras, inverse_k, sets):
            c = (camera.view @ np.r_[center, 1])[:3]
            conic = inverse.T @ (
                np.outer(c, c) - (c @ c - radius * radius) * np.eye(3)
            ) @ inverse
            p = np.c_[points, np.ones(len(points))]
            qp = p @ conic
            results.append(
                np.sum(qp * p, axis=1)
                / np.maximum(2 * np.linalg.norm(qp[:, :2], axis=1), 1e-12)
            )
        return np.concatenate(results)

    def solve(sets, initial):
        fit = least_squares(
            residual, initial, args=(sets,),
            bounds=([-np.inf] * 3 + [0.002], [np.inf] * 3 + [0.15]),
            max_nfev=80,
        )
        if not fit.success or fit.active_mask.any() or not np.isfinite(fit.x).all():
            raise ValueError("Sphere reconstruction did not converge away from bounds")
        return fit

    best = solve(pixels, np.r_[center, np.clip(radius, 0.0021, 0.149)])
    center, radius = best.x[:3], float(best.x[3])
    if any(not 0.05 <= np.linalg.norm(center - origin) <= 3 for origin in origins):
        raise ValueError("Sphere range is outside 0.05..3 m")
    if np.linalg.cond(best.jac) > 1e5:
        raise ValueError("Sphere geometry is poorly constrained")
    reprojection, angular_coverage, support = [], [], []
    for camera, image, p in zip(cameras, images, pixels):
        polygon = sphere_pixels(camera, center, radius).astype(np.float32)
        errors = np.array([abs(cv2.pointPolygonTest(polygon, tuple(q), True)) for q in p])
        rms = float(np.sqrt(np.mean(errors ** 2)))
        if rms > 1.5 or errors.max() > 3:
            raise ValueError("Selections do not agree with one spherical boundary")
        reprojection.append(rms)
        angular_coverage.append(coverage(camera, p, center))
        support.append(sphere_edge_support(image, camera, center, radius))
    # Independent 1px noise and 100 coherent two-view perturbation combinations.
    # These are sensitivity probes, NOT total error bounds or probabilities.
    inverse = np.linalg.pinv(best.jac)
    independent = float(2 * np.sqrt(np.trace(inverse @ inverse.T)))
    variations = []
    for camera, p in zip(cameras, pixels):
        image_center = sphere_pixels(camera, center, radius).mean(axis=0)
        radial = p - image_center
        norms = np.linalg.norm(radial, axis=1)
        if norms.min() < 1:
            raise ValueError("Sphere selection is too small or not a boundary")
        radial /= norms[:, None]
        variations.append([
            p + [np.cos(angle), np.sin(angle)]
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ] + [p + radial, p - radial])
    perturbed = np.array([
        solve([a, b], best.x).x for a in variations[0] for b in variations[1]
    ])
    delta = perturbed - best.x
    coherent = float(np.max(np.linalg.norm(delta[:, :3], axis=1) + abs(delta[:, 3])))
    sensitivity = max(independent, coherent)
    if sensitivity > min(0.005, radius * 0.5):
        raise ValueError("Sphere selection sensitivity is too large; obtain better current views")
    if center[2] - radius < table - sensitivity:
        raise ValueError("Fitted sphere materially crosses the public table plane")
    return {
        "position": center.tolist(), "radius_m": radius,
        "method": "rgb_sphere_fit", "reprojection_error_px": reprojection,
        "edge_evidence": "Original grayscale/R/G/B Canny union, same 3x3 blur and 35/100 thresholds; no dilation",
        "angular_coverage_degrees": angular_coverage, "support_by_view": support,
        "parallax_rad": parallax, "selection_sensitivity_m": sensitivity,
        "independent_1px_sensitivity_m": independent,
        "coherent_1px_sensitivity_m": coherent,
        "sampled_radius_range_m": [
            float(min(radius, perturbed[:, 3].min())),
            float(max(radius, perturbed[:, 3].max())),
        ],
        "sensor_ground_truth": False, "precision_verified": False,
        "assumptions": [
            "Both selections belong to the SAME approximately spherical surface.",
            "Selected pixels are tangent outer silhouettes, not seams, reflections or occluders.",
            "The center is a derived geometric point, not a trackable material feature.",
            "Sensitivity omits calibration/systematic shape bias and is not a confidence interval.",
            "A sphere fit does not prove contact, finger clearance, retention or native success.",
        ],
    }
