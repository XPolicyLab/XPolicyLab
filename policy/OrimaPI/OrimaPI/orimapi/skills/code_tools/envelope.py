"""Conditional visual hull from two complete RGB outlines.

No correspondence, symmetry, solid-material, contact or retention claim is made.
The caller must identify the same complete, unoccluded body in both images.
Neither simulator state nor guessed object dimensions enter the computation.
"""

import cv2
import numpy as np

from .regions import edges


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def polygon(image, camera, values):
    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise ValueError("Original RGB and calibrated image dimensions must agree")
    p = np.asarray(values, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or not 3 <= len(p) <= 64
        or not np.isfinite(p).all()
    ):
        raise ValueError("Use 3..64 finite ordered outline vertices")
    if (
        (p < 2).any()
        or (p[:, 0] >= camera.width - 2).any()
        or (p[:, 1] >= camera.height - 2).any()
    ):
        raise ValueError("Complete outlines must be inside image edges")
    if min(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)) < 1:
        raise ValueError("Outline vertices must be distinct")
    for i, (a, b) in enumerate(zip(p, np.roll(p, -1, axis=0))):
        for j in range(i + 2, len(p)):
            if i == 0 and j == len(p) - 1:
                continue
            c, d = p[j], p[(j + 1) % len(p)]
            if (
                np.maximum(np.minimum(a, b), np.minimum(c, d))
                > np.minimum(np.maximum(a, b), np.maximum(c, d))
            ).any():
                continue
            if (
                _cross(b - a, c - a) * _cross(b - a, d - a) <= 0
                and _cross(d - c, a - c) * _cross(d - c, b - c) <= 0
            ):
                raise ValueError("Outline is self-intersecting or self-touching")
    area = abs(cv2.contourArea(p.astype(np.float32)))
    if not 40 <= area <= image.shape[0] * image.shape[1] / 4:
        raise ValueError("Outline is too small or does not isolate one body")
    ends = np.roll(p, -1, axis=0)
    lengths = np.linalg.norm(ends - p, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distances = np.linspace(
        0, cumulative[-1], min(512, int(cumulative[-1]) + 1), endpoint=False
    )
    indices = np.minimum(
        np.searchsorted(cumulative, distances, side="right") - 1, len(p) - 1
    )
    fractions = (distances - cumulative[indices]) / lengths[indices]
    samples = p[indices] + fractions[:, None] * (ends[indices] - p[indices])
    edge_distance = cv2.distanceTransform(255 - edges(image), cv2.DIST_L2, 5)
    support = cv2.remap(
        edge_distance,
        samples[:, 0].astype(np.float32)[None],
        samples[:, 1].astype(np.float32)[None],
        cv2.INTER_LINEAR,
    )[0]
    fraction = float(np.mean(support <= 3))
    if fraction < 0.8:
        raise ValueError("Current RGB does not support the supplied complete boundary")
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 255)
    return mask, fraction


def grid(lower, upper, spacing, maximum):
    axes = [np.arange(a + spacing / 2, b, spacing) for a, b in zip(lower, upper)]
    if not all(len(a) for a in axes) or np.prod([len(a) for a in axes]) > maximum:
        raise ValueError("Outline geometry cannot fit the bounded voxel grid")
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def occupied(points, camera, mask):
    projected = points @ camera.projection[:, :3].T + camera.projection[:, 3]
    good = projected[:, 2] > 1e-6
    pixels = np.zeros((len(points), 2))
    np.divide(projected[:, :2], projected[:, 2, None], out=pixels, where=good[:, None])
    good &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height - 1)
    )
    result = np.zeros(len(points), bool)
    indices = np.rint(pixels[good]).astype(int)
    result[good] = mask[indices[:, 1], indices[:, 0]] > 0
    return result


def intersection(points, cameras, masks):
    for camera, mask in zip(cameras, masks):
        points = points[occupied(points, camera, mask)]
        if not len(points):
            break
    return points


def bounds(points, spacing):
    return np.array([points.min(0) - spacing / 2, points.max(0) + spacing / 2])


def measure(cameras, images, outlines, table):
    if not len(cameras) == len(images) == len(outlines) == 2:
        raise ValueError("Exactly two calibrated current views are required")
    if not np.isfinite(table):
        raise ValueError("A finite public support-plane elevation is required")
    if np.linalg.norm(cameras[0].optical[:3, 3] - cameras[1].optical[:3, 3]) < 0.03:
        raise ValueError("At least 3 cm camera baseline is required")
    checked = [polygon(image, c, p) for image, c, p in zip(images, cameras, outlines)]
    masks, support = [r[0] for r in checked], [r[1] for r in checked]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded = [cv2.dilate(mask, kernel) for mask in masks]
    # Generic public planning region, not task/object/seed-specific coordinates.
    lower, upper = np.array([-0.55, -0.4, table]), np.array(
        [0.55, 0.35, min(1.15, table + 0.3)]
    )
    coarse_step, fine_step = 0.008, 0.002
    coarse = intersection(grid(lower, upper, coarse_step, 800_000), cameras, expanded)
    if len(coarse) < 8:
        raise ValueError("No resolved supported body in the two selected outlines")
    rough = bounds(coarse, coarse_step)
    if (rough[1] - rough[0] > [0.35, 0.35, 0.3]).any():
        raise ValueError("Two-view body extent is insufficiently constrained")
    if (rough[0, :2] <= lower[:2] + coarse_step).any() or (
        rough[1] >= upper - coarse_step
    ).any():
        raise ValueError("Body envelope reaches the supported workspace boundary")
    fine = grid(
        np.maximum(lower, rough[0] - coarse_step),
        np.minimum(upper, rough[1] + coarse_step),
        fine_step,
        1_000_000,
    )
    outer = intersection(fine, cameras, expanded)
    normal = intersection(fine, cameras, masks)
    if len(normal) < 24:
        raise ValueError("Nominal silhouettes do not resolve a body volume")
    nominal_bounds, outer_bounds = bounds(normal, fine_step), bounds(outer, fine_step)
    if nominal_bounds[0, 2] > table + 0.015:
        raise ValueError("A visibly table-supported body is required")
    center = normal.mean(0)
    rays = [center - c.optical[:3, 3] for c in cameras]
    rays = [r / np.linalg.norm(r) for r in rays]
    parallax = float(np.arccos(np.clip(np.dot(*rays), -1, 1)))
    if not 0.1 <= parallax <= np.pi - 0.1:
        raise ValueError("Insufficient body-view parallax")
    sensitivity = float(np.max(np.abs(outer_bounds - nominal_bounds)))
    if sensitivity > 0.02:
        raise ValueError("Body bounds are too sensitive to a two-pixel boundary change")
    height = float(nominal_bounds[1, 2] - table)
    sections = []
    for fraction in (0.2, 0.3, 0.4, 0.5):
        z = table + fraction * height
        section = normal[np.abs(normal[:, 2] - z) <= fine_step]
        widened = outer[np.abs(outer[:, 2] - z) <= fine_step]
        if len(section) < 8 or len(widened) < 8:
            continue
        b, wide = bounds(section, fine_step), bounds(widened, fine_step)
        sections.append(
            {
                "height_above_support_m": float(z - table),
                "center_hypothesis": [*map(float, b.mean(0)[:2]), float(z)],
                "xy_spans_m": (b[1, :2] - b[0, :2]).tolist(),
                "expanded_xy_spans_m": (wide[1, :2] - wide[0, :2]).tolist(),
                "scope": "Visual-hull cross-section, not a measured solid contact surface",
            }
        )
    return {
        "method": "conditional_rgb_outline_visual_hull",
        "visual_hull_bounds_m": nominal_bounds.tolist(),
        "expanded_bounds_m": outer_bounds.tolist(),
        "centroid_hypothesis": center.tolist(),
        "height_upper_estimate_m": height,
        "sections": sections,
        "voxel_spacing_m": fine_step,
        "boundary_dilation_px": 2,
        "boundary_sensitivity_m": sensitivity,
        "parallax_rad": parallax,
        "rgb_edge_support": support,
        "sensor_ground_truth": False,
        "official_success": None,
        "assumptions": [
            "Both complete, unoccluded outlines enclose the SAME body on the public table",
            "Calibration is accurate; complete-boundary identity is supplied by the caller",
            "A visual hull can contain unobserved empty space and is not a solid-material model",
            "No head/body vertical-axis, circularity, uprightness or semantic identity is certified",
            "Boundary dilation is a sensitivity probe, not a total uncertainty/confidence bound",
            "Finite voxel sampling can miss thin parts; these are not certified collision bounds",
            "Section centers and spans are hypotheses, not TCP targets or contact/retention proof",
        ],
    }
