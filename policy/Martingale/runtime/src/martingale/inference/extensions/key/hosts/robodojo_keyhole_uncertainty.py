# Copyright © 2026 Martingale. All Rights Reserved.
import itertools
import numpy as np
from scipy.spatial import ConvexHull

def _points(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or (not len(value)) or (not np.isfinite(value).all()):
        raise ValueError(name + ' must contain finite xyz samples.')
    return value

def _unit(value, name):
    value = np.asarray(value, dtype=float)
    lengths = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(lengths < 1e-10):
        raise ValueError(name + ' contains degenerate directions.')
    return value / lengths

def _pixels(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + ' has invalid pixel coordinates.')
    return value

def _variants(pixels):
    """Include the given input and every independent +/-1 pixel box corner."""
    pixels = np.asarray(pixels, dtype=float)
    offsets = np.asarray(list(itertools.product((-1.0, 1.0), repeat=pixels.size)))
    return np.concatenate([pixels[None], pixels + offsets.reshape((-1,) + pixels.shape)])

def _ray(view, pixel):
    ray = np.asarray(view.rays(np.array([pixel[1]]), np.array([pixel[0]])), dtype=float)
    if ray.shape != (1, 3):
        raise ValueError('A camera must return one xyz ray per pixel.')
    return _unit(ray[0], 'Camera ray')

def triangulated_samples(pixels, views):
    """Return xyz samples for one matched landmark in two or three cameras."""
    if not isinstance(pixels, dict) or len(pixels) not in (2, 3):
        raise ValueError('A landmark needs two or three calibrated views.')
    names = list(pixels)
    if any((name not in views or views[name] is None for name in names)):
        raise ValueError('Unavailable landmark camera.')
    given = _pixels([pixels[name] for name in names], (len(names), 2), 'Landmark')
    origins = _points([views[name].pos for name in names], 'Camera origins')
    reference = origins[0]
    result = []
    for sample in _variants(given):
        rays = np.array([_ray(views[name], pixel) for name, pixel in zip(names, sample)])
        projectors = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
        lhs = projectors.sum(axis=0)
        eigenvalues = np.linalg.eigvalsh(lhs)
        if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            raise ValueError('Landmark sensitivity includes degenerate ray geometry.')
        rhs = np.einsum('nij,nj->i', projectors, origins - reference)
        point = np.linalg.solve(lhs, rhs) + reference
        if not np.isfinite(point).all() or np.any(np.einsum('ni,ni->n', point - origins, rays) <= 0):
            raise ValueError('Landmark sensitivity includes a point behind a camera.')
        result.append(point.tolist())
    return result

def _on_plane(view, pixels, height):
    origin = _points([view.pos], 'Camera origin')[0]
    rays = np.array([_ray(view, pixel) for pixel in pixels])
    if np.any(np.abs(rays[:, 2]) < 1e-10):
        raise ValueError('Opening sensitivity includes a ray parallel to the rim.')
    distance = (height - origin[2]) / rays[:, 2]
    if np.any(distance <= 0):
        raise ValueError('Opening sensitivity includes a point behind a camera.')
    return origin + distance[:, None] * rays

def opening_samples(views, entry_pixels, axis_pixels, z, height_span, weights, slot_axis):
    """Sample the opening's weighted entry and directed in-plane slot axis.

The height alternatives are the fitted rim and +/- its reported sensitivity.
The entry weighting and axis averaging match keyhole.fit. Every axis sample
is sign-aligned to the measured directed slot_axis; this does not resolve an
ambiguous tooth side, which the caller must reject before using these samples.
"""
    if not isinstance(entry_pixels, dict) or len(entry_pixels) not in (2, 3):
        raise ValueError('Opening needs two or three entry views.')
    if not isinstance(axis_pixels, dict) or not 1 <= len(axis_pixels) <= 3:
        raise ValueError('Opening needs at least one resolved axis view.')
    names = list(entry_pixels)
    if any((name not in views or views[name] is None for name in names + list(axis_pixels))):
        raise ValueError('Unavailable opening camera.')
    z, height_span = (float(z), float(height_span))
    if not np.isfinite([z, height_span]).all() or height_span < 0:
        raise ValueError('Opening height sensitivity must be finite and nonnegative.')
    given = _pixels([entry_pixels[name] for name in names], (len(names), 2), 'Opening entry')
    weights = np.asarray([weights[name] for name in names] if isinstance(weights, dict) else weights, float)
    if weights.shape != (len(names),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Opening weights must be positive and match the entry views.')
    weights = weights / weights.sum()
    direction = _unit(_points([slot_axis], 'Slot axis')[0], 'Slot axis')
    if abs(direction[2]) > 1e-06:
        raise ValueError('Opening slot axis must lie in the horizontal rim plane.')
    heights = [z] if height_span == 0 else [z, z - height_span, z + height_span]
    entries, directions = ([], [])
    for height in heights:
        for sample in _variants(given):
            points = np.array([_on_plane(views[name], [pixel], height)[0] for name, pixel in zip(names, sample)])
            entries.append(np.sum(points * weights[:, None], axis=0).tolist())
        by_view = []
        for name, pixels in axis_pixels.items():
            pixels = _pixels(pixels, (2, 2), 'Opening axis')
            sample_directions = []
            for sample in _variants(pixels):
                endpoints = _on_plane(views[name], sample, height)
                axis = _unit(endpoints[1] - endpoints[0], 'Opening axis sample')
                if abs(axis @ direction) < 1e-08:
                    raise ValueError('Opening-axis sensitivity makes its directed side ambiguous.')
                if axis @ direction < 0:
                    axis = -axis
                sample_directions.append(axis)
            by_view.append(sample_directions)
        for combination in itertools.product(*by_view):
            directions.append(_unit(np.mean(combination, axis=0), 'Mean opening axis').tolist())
    return dict(entry_samples_xyz=entries, direction_samples_xyz=directions)

def _enclosing_radius(points):
    """Smallest circle containing a finite planar sample set, in metres."""
    points = np.unique(np.asarray(points, float), axis=0)
    if len(points) == 1:
        return 0.0
    pair_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    i, j = np.unravel_index(pair_distances.argmax(), pair_distances.shape)
    midpoint = (points[i] + points[j]) / 2
    radius = float(pair_distances[i, j] / 2)
    if np.linalg.norm(points - midpoint, axis=1).max() <= radius + 1e-12:
        return radius
    points = points[ConvexHull(points).vertices]
    best = float('inf')
    for a, b in itertools.combinations(points, 2):
        center = (a + b) / 2
        candidate = float(np.linalg.norm(a - b) / 2)
        if candidate < best and np.linalg.norm(points - center, axis=1).max() <= candidate + 1e-12:
            best = candidate
    for a, b, c in itertools.combinations(points, 3):
        matrix = np.stack([b - a, c - a])
        if abs(np.linalg.det(matrix)) < 1e-12 * np.linalg.norm(matrix) ** 2:
            continue
        offset = np.linalg.solve(2 * matrix, np.sum(matrix ** 2, axis=1))
        candidate = float(np.linalg.norm(offset))
        if candidate < best and np.linalg.norm(points - (a + offset), axis=1).max() <= candidate + 1e-12:
            best = candidate
    if not np.isfinite(best):
        raise ValueError('Opening entry uncertainty has no finite enclosing circle.')
    return best

def opening_accuracy(samples, entry, direction, normal=(0.0, 0.0, 1.0)):
    """Expose limits imposed by the opening before spending actions on a key.

    Minimum possible errors allow an ideal key at any position/direction and
    contain NO key annotation uncertainty. They are diagnostics, not substitute
    targets or evidence that final alignment is ready. Central margins report
    the error budget still available when matching the reported opening pose.
    """
    entry = _points([entry], 'Opening entry')[0]
    normal = _unit(_points([normal], 'Opening normal')[0], 'Opening normal')
    direction = _unit(_points([direction], 'Opening direction')[0], 'Opening direction')
    direction = _unit(direction - normal * (direction @ normal), 'Projected opening direction')
    cross = np.cross(normal, direction)
    entries = _points(samples['entry_samples_xyz'], 'Opening entry samples') - entry
    planar = np.column_stack([entries @ direction, entries @ cross])
    directions = _points(samples['direction_samples_xyz'], 'Opening direction samples')
    directions = _unit(directions - (directions @ normal)[:, None] * normal, 'Projected opening samples')
    angles = np.arctan2(directions @ cross, directions @ direction)
    ordered = np.sort(np.mod(angles, 2 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    minimum_twist = float(np.degrees((2 * np.pi - gaps.max()) / 2))
    spread = float(np.degrees(np.abs(angles).max()))
    reference_radius = float(np.linalg.norm(planar, axis=1).max())
    minimum_radius = _enclosing_radius(planar)
    reasons = []
    if minimum_twist > 8:
        reasons.append('Opening direction uncertainty alone exceeds the 8-degree twist gate even for an ideal key; improve the slot-edge view before key corrections.')
    if minimum_radius > 0.002:
        reasons.append('Opening entry uncertainty alone exceeds the 2-mm centering gate even for an ideal key; improve the entry views before key corrections.')
    return dict(direction_spread_deg=spread, minimum_possible_max_twist_deg=minimum_twist, entry_lateral_radius_m=reference_radius, minimum_possible_max_lateral_m=minimum_radius, height_radius_m=float(np.abs(entries @ normal).max()), remaining_twist_margin_deg=8 - spread, remaining_centering_margin_m=0.002 - reference_radius, irreducible_limits_exceeded=bool(reasons), reasons=reasons, note='Opening-only sensitivity. Minimum errors assume an ideal key; actual key error still adds to these limits. Small positive central margins may still be inadequate. Improve the observation when needed; these diagnostics do not authorize insertion or change the alignment gates.')

def summarize(tip_samples, shaft_samples, tooth_samples, entry_samples, direction_samples, normal):
    """Return worst sampled alignment errors, preserving shared-tip geometry."""
    tips = _points(tip_samples, 'Tip')
    shafts = _points(shaft_samples, 'Shaft')
    teeth = _points(tooth_samples, 'Tooth')
    entries = _points(entry_samples, 'Opening entry')
    directions = _points(direction_samples, 'Opening direction')
    normal = _unit(_points([normal], 'Opening normal')[0], 'Opening normal')
    directions = _unit(directions - (directions @ normal)[:, None] * normal, 'Projected opening direction')
    basis_x = directions[0]
    basis_y = np.cross(normal, basis_x)
    angles = np.sort(np.mod(np.arctan2(directions @ basis_y, directions @ basis_x), 2 * np.pi))
    circular_angles = np.r_[angles[-1] - 2 * np.pi, angles, angles[0] + 2 * np.pi]
    max_tilt = max_twist = max_lateral = 0.0
    for tip in tips:
        axes = _unit(shafts - tip, 'Sampled shaft')
        max_tilt = max(max_tilt, float(np.arccos(np.clip(axes @ normal, -1, 1)).max()))
        to_teeth = teeth - tip
        width = to_teeth[None, :, :] - np.einsum('ij,kj->ik', axes, to_teeth)[:, :, None] * axes[:, None, :]
        width = _unit(width, 'Sampled tooth')
        projected = width - (width @ normal)[:, :, None] * normal
        projected = _unit(projected.reshape(-1, 3), 'Projected sampled tooth')
        width_angles = np.arctan2(projected @ basis_y, projected @ basis_x)
        opposite = np.mod(width_angles + np.pi, 2 * np.pi)
        indices = np.searchsorted(circular_angles, opposite)
        closest = np.minimum(opposite - circular_angles[indices - 1], circular_angles[indices] - opposite)
        max_twist = max(max_twist, float((np.pi - closest).max()))
        offsets = tip - entries
        lateral = offsets - (offsets @ normal)[:, None] * normal
        max_lateral = max(max_lateral, float(np.linalg.norm(lateral, axis=1).max()))
    tip_heights, entry_heights = (tips @ normal, entries @ normal)
    return dict(max_tilt_deg=float(np.degrees(max_tilt)), max_twist_deg=float(np.degrees(max_twist)), max_lateral_m=max_lateral, min_above_m=float(tip_heights.min() - entry_heights.max()), max_above_m=float(tip_heights.max() - entry_heights.min()))

def summarize_widths(tip_samples, shaft_samples, width_samples, entry_samples, direction_samples, normal, *, shaft_axes=None):
    """Alignment envelope when transverse tooth edges supply width directions.

    Width samples must already include all shaft-axis and edge perturbations.
    Their positions along the shaft and tooth-face thickness do not enter.
    """
    tips = _points(tip_samples, 'Tip')
    shafts = _points(shaft_samples, 'Shaft') if shaft_axes is None else None
    entries = _points(entry_samples, 'Opening entry')
    normal = _unit(_points([normal], 'Opening normal')[0], 'Opening normal')
    widths = _points(width_samples, 'Blade width')
    widths = _unit(widths - (widths @ normal)[:, None] * normal, 'Projected blade width')
    directions = _points(direction_samples, 'Opening direction')
    directions = _unit(directions - (directions @ normal)[:, None] * normal, 'Projected opening direction')
    max_tilt = max_lateral = 0.0
    if shaft_axes is not None:
        axes = _unit(_points(shaft_axes, 'Shaft axes'), 'Shaft axes')
        max_tilt = float(np.degrees(np.arccos(np.clip(axes @ normal, -1, 1))).max())
    for tip in tips:
        if shafts is not None:
            axes = _unit(shafts - tip, 'Sampled shaft')
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(axes @ normal, -1, 1))).max()))
        offsets = tip - entries
        lateral = offsets - (offsets @ normal)[:, None] * normal
        max_lateral = max(max_lateral, float(np.linalg.norm(lateral, axis=1).max()))
    min_dot = 1.0
    for start in range(0, len(widths), 256):
        min_dot = min(min_dot, float((widths[start:start + 256] @ directions.T).min()))
    return dict(max_tilt_deg=max_tilt, max_twist_deg=float(np.degrees(np.arccos(np.clip(min_dot, -1, 1)))), max_lateral_m=max_lateral, min_above_m=float((tips @ normal).min() - (entries @ normal).max()), max_above_m=float((tips @ normal).max() - (entries @ normal).min()))
