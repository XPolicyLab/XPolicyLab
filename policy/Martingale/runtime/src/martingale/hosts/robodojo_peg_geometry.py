# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations
from martingale import self_play_constants as sp

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import robodojo_sight as sight

_ANGLES = np.arange(48) * (2 * np.pi / 48)
_DIRECTIONS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)


def _mask_support(observation):
    """Projected silhouette support plus lower-bound flags at unknown edges."""
    rows, cols = np.nonzero(observation.mask)
    projected = np.c_[cols, rows] @ _DIRECTIONS.T
    target = projected.max(axis=0)
    unknown = observation.unknown
    touches_unknown = np.zeros(len(rows), dtype=bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = rows + dr, cols + dc
        inside = (rr >= 0) & (rr < unknown.shape[0]) & (cc >= 0) & (cc < unknown.shape[1])
        touches_unknown |= ~inside
        touches_unknown[inside] |= unknown[rr[inside], cc[inside]]
    clipped = np.any((projected >= target - .5) & touches_unknown[:, None], axis=0)
    return target, clipped


def triangulate(views, pixels, *, max_condition: float = 1e5) -> np.ndarray:
    """Find the least-squares intersection of calibrated pixel rays.

    ``pixels`` contains ``(row, column)`` pairs, one per view. The returned
    point minimizes summed squared perpendicular distance to the rays.
    Raises ``ValueError`` for mismatched, degenerate, or non-finite inputs.
    """
    views = list(views)
    pixels = list(pixels)
    if len(views) < 2 or len(views) != len(pixels):
        raise ValueError("Need matching pixels from at least two views")
    normal = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    for view, pixel in zip(views, pixels):
        pixel = np.asarray(pixel, dtype=float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("Each pixel must be a finite (row, column) pair")
        origin = np.asarray(view.pos, dtype=float)
        ray = np.asarray(view.rays(pixel[0:1], pixel[1:2]), dtype=float)
        if ray.shape == (1, 3):
            ray = ray[0]
        if origin.shape != (3,) or ray.shape != (3,) or not np.isfinite(ray).all():
            raise ValueError("View must provide finite XYZ origin and ray")
        length = np.linalg.norm(ray)
        if length <= 1e-12:
            raise ValueError("Pixel ray has zero length")
        ray = ray / length
        projector = np.eye(3) - np.outer(ray, ray)
        normal += projector
        rhs += projector @ origin
    if not np.isfinite(normal).all() or np.linalg.cond(normal) > max_condition:
        raise ValueError("Insufficient camera parallax")
    return np.linalg.solve(normal, rhs)


def cylinder_mesh(root, tip, radius, *, segments: int = 64):
    """Return vertices/faces for a closed finite-cylinder approximation."""
    root = np.asarray(root, dtype=float)
    tip = np.asarray(tip, dtype=float)
    radius = float(radius)
    if root.shape != (3,) or tip.shape != (3,) or not np.isfinite(root).all() or not np.isfinite(tip).all():
        raise ValueError("root and tip must be finite XYZ points")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if isinstance(segments, bool) or int(segments) != segments or segments < 3:
        raise ValueError("segments must be an integer of at least 3")
    axis = tip - root
    length = np.linalg.norm(axis)
    if length <= 1e-12:
        raise ValueError("root and tip must differ")
    axis /= length
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.arange(int(segments)) * (2 * np.pi / int(segments))
    ring = radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    vertices = np.concatenate([root + ring, tip + ring])
    faces = []
    for i in range(int(segments)):
        j = (i + 1) % int(segments)
        faces.extend(([i, j, segments + j], [i, segments + j, segments + i]))
    for i in range(1, int(segments) - 1):
        faces.extend(([0, i + 1, i], [segments, segments + i, segments + i + 1]))
    return vertices, np.asarray(faces, dtype=np.int64)


def crop_peg(view, mask, seed, occluded=None):
    """Keep the seeded shaft region and preserve mast/occlusion as unknown."""
    root, tip = np.asarray(seed, float)
    direction = tip - root
    length = np.linalg.norm(direction)
    if length < 12:
        raise ValueError("Peg is too foreshortened for a reliable fit")
    rows, cols = np.indices(mask.shape)
    pixels = np.stack([rows, cols], axis=-1)
    relative = pixels - root
    along = relative @ direction / length**2
    across = np.abs(relative @ np.array([-direction[1], direction[0]])) / length
    roi = (along > .22) & (along < sp.PEG_CROP_UPPER_FRACTION) & (across < max(7, .20 * length))
    unknown = ~roi
    if occluded is not None:
        occluded = np.asarray(occluded, bool)
        if occluded.shape != np.asarray(mask).shape:
            raise ValueError("occlusion mask shape must match the RGB mask")
        unknown |= occluded
    selected = np.asarray(mask, bool) & ~unknown
    if selected.sum() < 20:
        raise ValueError("Selected peg has too few visible pixels")
    return sight.Sight(view, selected, unknown)


def fit_peg(views, masks, seeds, occluded=None):
    """Fit a selected finite cylinder from per-view peg masks and root/tip pixels.

    Inputs must describe the same static peg in calibrated views. ``seeds`` are
    ``(root_rc, tip_rc)`` pairs in pixel row/column order. The fit is geometry
    from RGB masks and camera poses; its output does not certify insertion.
    """
    views, masks, seeds = list(views), list(masks), list(seeds)
    if len(views) < 2 or len(masks) != len(views) or len(seeds) != len(views):
        raise ValueError("Need matching masks and endpoint seeds from two or more views")
    if occluded is not None and len(occluded) != len(views):
        raise ValueError("occluded must have one entry per view")
    observations = [
        crop_peg(view, mask, seed, None if occluded is None else occluded[i])
        for i, (view, mask, seed) in enumerate(zip(views, masks, seeds))
    ]
    root = triangulate(views, [seed[0] for seed in seeds])
    tip = triangulate(views, [seed[1] for seed in seeds])
    length = np.linalg.norm(tip - root)
    if not .02 < length < .4:
        raise ValueError("Inconsistent endpoint correspondence")
    widths = []
    for observation, seed in zip(observations, seeds):
        rows, cols = np.nonzero(observation.mask)
        direction = np.asarray(seed[1], float) - np.asarray(seed[0], float)
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        offsets = (np.c_[rows, cols] - seed[0]) @ normal
        widths.append(.5 * np.ptp(offsets) * length / np.linalg.norm(direction))
    radius = float(np.clip(np.median(widths), *sp.PEG_RADIUS_INITIAL_BOUNDS_M))
    initial = np.r_[root, tip, radius]
    supports = [_mask_support(observation) for observation in observations]
    strips = []
    for observation, seed in zip(observations, seeds):
        rows, cols = np.nonzero(observation.mask)
        origin = np.asarray(seed[0], float)
        direction = np.asarray(seed[1], float) - origin
        length_px = np.linalg.norm(direction)
        direction /= length_px
        normal = np.array([-direction[1], direction[0]])
        pixels = np.c_[rows, cols] - origin
        along, across = pixels @ direction, pixels @ normal
        samples = []
        for fraction in np.linspace(*sp.PEG_AXIAL_SAMPLE_FRACTIONS, 12):
            at = fraction * length_px
            values = across[np.abs(along - at) < 1.5]
            if len(values) >= 3:
                samples.append((at, .5 * (values.min() + values.max())))
        strips.append((origin, direction, normal, samples))

    def residual(parameters):
        vertices, _ = cylinder_mesh(parameters[:3], parameters[3:6], parameters[6])
        parts = []
        for observation, (target, clipped) in zip(observations, supports):
            rows, cols, _ = observation.view.project(vertices)
            error = (np.c_[cols, rows] @ _DIRECTIONS.T).max(0) - target
            parts.append(np.where(clipped, np.minimum(error, 0), error))
        for observation, (origin, direction, normal, samples) in zip(observations, strips):
            rows, cols, _ = observation.view.project(np.array([parameters[:3], parameters[3:6]]))
            endpoints = np.c_[rows, cols] - origin
            along = endpoints @ direction
            across = endpoints @ normal
            parts.append(np.array([
                across[0] + (at - along[0]) * (across[1] - across[0])
                / (along[1] - along[0]) - observed
                for at, observed in samples
            ]))
        parts.append((parameters[:3] - root) * 5)
        return np.concatenate(parts)

    p0 = initial
    lower = np.r_[p0[:6] - .06, .001]
    upper = np.r_[p0[:6] + .06, .025]
    solution = least_squares(
        residual, p0, bounds=(lower, upper), x_scale="jac", loss="soft_l1",
        f_scale=1, max_nfev=150,
    )
    parameters = solution.x
    vertices, faces = cylinder_mesh(parameters[:3], parameters[3:6], parameters[6])
    scores = []
    for observation in observations:
        rows, cols, _ = observation.view.project(vertices)
        predicted = np.zeros(observation.mask.shape, np.uint8)
        cv2.fillConvexPoly(
            predicted, cv2.convexHull(np.rint(np.c_[cols, rows]).astype("int32")), 1
        )
        visible_predicted = predicted.astype(bool) & ~observation.unknown
        target = observation.mask
        scores.append(float((visible_predicted & target).sum() /
                            max(1, (visible_predicted | target).sum())))
    axis = parameters[3:6] - parameters[:3]
    axis /= np.linalg.norm(axis)
    support_rmse = float(np.sqrt(np.mean(residual(parameters)[:-3] ** 2)))
    fitted_root_shift = float(np.linalg.norm(parameters[:3] - root))
    fitted_tip_shift = float(np.linalg.norm(parameters[3:6] - tip))
    silhouette_fit_valid = bool(min(scores) > .65 and support_rmse < 2)
    endpoint_shift_warning = .010
    endpoint_uncertain = max(fitted_root_shift, fitted_tip_shift) > endpoint_shift_warning
    endpoint_geometry_valid = bool(silhouette_fit_valid and not endpoint_uncertain)
    return dict(
        valid=endpoint_geometry_valid,
        silhouette_fit_valid=silhouette_fit_valid,
        endpoint_geometry_valid=endpoint_geometry_valid,
        endpoint_uncertain=endpoint_uncertain,
        endpoint_shift_warning_m=endpoint_shift_warning,
        root_xyz=parameters[:3].tolist(), tip_xyz=parameters[3:6].tolist(),
        axis_xyz=axis.tolist(), radius_m=float(parameters[6]),
        length_m=float(np.linalg.norm(parameters[3:6] - parameters[:3])),
        per_view_iou=scores, support_rmse_px=support_rmse,
        baseline=dict(root_xyz=root.tolist(), tip_xyz=tip.tolist(), radius_m=radius),
        baseline_length_m=float(length),
        root_seed_shift_m=fitted_root_shift,
        tip_seed_shift_m=fitted_tip_shift,
        supporting_view_count=len(observations),
        quality_scope="silhouette_fit_only; occluded endpoint may be weakly constrained",
        insertion_clearance="not_evaluated",
        insertion_ready=False,
        note="Selected static peg fit from RGB masks and endpoint seeds; root is weakly constrained when hidden by the mast. Refit if the rack moves.",
    )


def fit_peg_landmarks(views, root_pixels, tip_pixels, *, radius_m: float | None = None):
    """Triangulate paired root/tip pixel landmarks into compact peg geometry.

    The landmarks must refer to the same visible physical endpoints in every
    view. Optional radius is a supplied hypothesis, not a visual measurement.
    This minimal fitter does not infer radius or certify insertion.
    """
    views = list(views)
    root = triangulate(views, root_pixels)
    tip = triangulate(views, tip_pixels)
    length = float(np.linalg.norm(tip - root))
    if not 0.02 < length < 0.4:
        raise ValueError("Inconsistent peg endpoint correspondence")
    result = {
        "kind": "peg_landmarks",
        "root_xyz": root.tolist(),
        "tip_xyz": tip.tolist(),
        "axis_xyz": ((tip - root) / length).tolist(),
        "length_m": length,
        "supporting_view_count": len(views),
        "radius_status": "not_estimated",
        "insertion_clearance": "not_evaluated",
    }
    if radius_m is not None:
        radius = float(radius_m)
        if not np.isfinite(radius) or not 0 < radius < 0.025:
            raise ValueError("radius_m must be positive and below 25 mm")
        result["radius_m"] = radius
        result["radius_status"] = "caller_supplied"
    return result
