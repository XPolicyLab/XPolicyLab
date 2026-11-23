# Copyright © 2026 Martingale. All Rights Reserved.
import itertools
import numpy as np

def _unit(values, name):
    values = np.asarray(values, float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm < 1e-10):
        raise ValueError(name + ' contains a nonfinite or zero direction.')
    return values / norm

def _shared_corners(origins, rays):
    """Closest common point for each pair of calibrated corner rays."""
    projections = np.eye(3) - rays[..., :, None] * rays[..., None, :]
    matrix = projections.sum(axis=1)
    target = np.einsum('nvij,vj->ni', projections, origins)
    return np.linalg.solve(matrix, target)

def _reconstruct(origins, common, other_rays, axes, strict=True):
    """All axis x pixel combinations; rows end in [view, xyz]."""
    denom = np.einsum('nvj,aj->anv', other_rays, axes)
    grazing = np.abs(denom) < 0.15
    if strict and np.any(grazing):
        raise ValueError('End-plane rays are too grazing; obtain elevated oblique views.')
    numer = np.einsum('nvj,aj->anv', common[:, None, :] - origins, axes)
    depth = numer / np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    if strict and np.any(depth <= 0):
        raise ValueError('An end-plane intersection is behind its camera.')
    other = origins + depth[..., None] * other_rays[None]
    sides = other - common[None, :, None, :]
    lengths = np.linalg.norm(sides, axis=-1)
    if strict and np.any(lengths < 0.0005):
        raise ValueError('The reconstructed end face has a degenerate side.')
    directions = sides / np.maximum(lengths[..., None], 1e-10)
    cosine = np.sum(directions[..., 0, :] * directions[..., 1, :], axis=-1)
    normals = np.cross(directions, axes[:, None, None, :])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-10)
    inward = np.stack([sides[..., 1, :], sides[..., 0, :]], axis=-2)
    signs = np.sum(normals * inward, axis=-1)
    normals *= np.where(signs > 0, -1.0, 1.0)[..., None]
    midpoints = common[None, :, None, :] + sides / 2
    sight = origins - midpoints
    sight /= np.maximum(np.linalg.norm(sight, axis=-1, keepdims=True), 1e-10)
    facing = np.sum(normals * sight, axis=-1)
    return dict(centers=other.mean(axis=-2), lengths=lengths, directions=directions, cosines=cosine, face_visibility=facing, depths=depth, valid=(~grazing & (depth > 0) & (lengths >= 0.0005)).all(axis=-1))

def estimate(views, end_rim_pixels, shaft_axis, shaft_samples=None):
    """Fit two adjacent *near* end-rim segments and propagate input error.

    end_rim_pixels maps exactly two camera names to two [column,row] endpoints
    each. The segments must be different adjacent edges of the rectangular
    insertion end. The far edge may be hidden behind the shaft. Endpoint order
    is arbitrary. Supply the shaft direction from longitudinal image lines or
    independent interior sections, not cross-view tip-rim midpoint matching.

    All four shared-corner assignments are tested. A valid pair has adequate
    parallax/reprojection, approximately perpendicular reconstructed rim sides,
    and both associated shaft faces point toward their observing camera. More
    than one valid mapping is rejected rather than choosing the lowest error.

    Every independent +/-1px corner of the eight endpoint coordinates is
    combined with every supplied shaft-axis sample. The returned samples are
    the central tip plus eight corners of the bounding box of ALL resulting
    centers. This is a conservative enclosure, not a probability estimate.
    No confidence is gained by treating annotations as independent repeats.
    """
    candidates = []

    def reject(reason):
        return dict(accepted=False, reasons=[reason], xyz=None, tip_xyz=None, samples_xyz=None, pixel_sensitivity_m=None, candidates=candidates, model='rectangular_end_rims')
    try:
        if not isinstance(end_rim_pixels, dict) or len(end_rim_pixels) != 2:
            raise ValueError('Use two different close views of adjacent shaft end rims.')
        axis = np.asarray(shaft_axis, float)
        if axis.shape != (3,):
            raise ValueError('Shaft axis must be one xyz direction.')
        axis = _unit(axis, 'Shaft axis')
        if shaft_samples is None:
            axes = axis[None]
        else:
            samples = np.asarray(shaft_samples, float)
            if samples.ndim != 2 or samples.shape[1] != 3 or (not len(samples)):
                raise ValueError('Shaft samples must be xyz directions.')
            axes = _unit(np.vstack([axis, samples]), 'Shaft samples')
        unique_axes = np.unique(axes, axis=0)
        unique_axes = unique_axes[np.linalg.norm(unique_axes - axis, axis=1) > 1e-14]
        axes = np.vstack([axis, unique_axes])
        names = list(end_rim_pixels)
        origins, pairs, reference_rays = ([], [], [])
        for name in names:
            if name not in views or views[name] is None:
                raise ValueError('Unavailable end-rim camera: ' + name)
            view = views[name]
            points = np.asarray(end_rim_pixels[name], float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError('Each end rim needs two finite [column,row] endpoints.')
            height, width = view.shape[:2]
            if np.any(points < 0) or np.any(points > [width - 1, height - 1]):
                raise ValueError('End-rim endpoints must be inside the original image.')
            if np.linalg.norm(points[1] - points[0]) < 4:
                raise ValueError('An end rim spans fewer than 4 pixels; obtain a closer view.')
            pairs.append(points)
            origins.append(np.asarray(view.pos, float))
            reference_rays.append(_unit(view.rays(points[:, 1], points[:, 0]), 'End-rim rays'))
        origins, pairs, reference_rays = map(np.asarray, (origins, pairs, reference_rays))
        baseline = float(np.linalg.norm(origins[1] - origins[0]))
        if baseline < 0.001:
            raise ValueError('The two end-rim cameras have insufficient baseline.')
        accepted = []
        for mapping in itertools.product((0, 1), repeat=2):
            candidate = dict(shared_endpoint_indices=dict(zip(names, mapping)), accepted=False, reasons=[])
            candidates.append(candidate)
            chosen = reference_rays[np.arange(2), mapping]
            parallax = float(np.degrees(np.arccos(np.clip(chosen[0] @ chosen[1], -1, 1))))
            candidate['ray_angle_deg'] = parallax
            if min(parallax, 180 - parallax) < 5:
                candidate['reasons'].append('Shared-corner rays have less than 5 degrees parallax.')
                continue
            corner = _shared_corners(origins, chosen[None])[0]
            errors, depths = ([], [])
            for i, name in enumerate(names):
                row, col, depth = views[name].project(corner[None])
                errors.append(float(np.linalg.norm([col[0] - pairs[i, mapping[i], 0], row[0] - pairs[i, mapping[i], 1]])))
                depths.append(float(depth[0]))
            candidate['reprojection_errors_px'] = errors
            if max(errors) > 3 or min(depths) <= 0:
                candidate['reasons'].append('Shared-corner reprojection is inconsistent.')
                continue
            try:
                other = reference_rays[np.arange(2), 1 - np.asarray(mapping)]
                geometry = _reconstruct(origins, corner[None], other[None], axis[None])
            except ValueError as exc:
                candidate['reasons'].append(str(exc))
                continue
            angle = float(np.degrees(np.arccos(np.clip(geometry['cosines'][0, 0], -1, 1))))
            facing = geometry['face_visibility'][0, 0]
            candidate.update(side_angle_deg=angle, side_lengths_m=geometry['lengths'][0, 0].tolist(), face_visibility=facing.tolist(), xyz=geometry['centers'][0, 0].tolist())
            if abs(angle - 90) > 15:
                candidate['reasons'].append('Adjacent end rims do not form a rectangular end face.')
            if min(facing) < 0.1:
                candidate['reasons'].append('An annotated rim belongs to a hidden or grazing shaft face.')
            if candidate['reasons']:
                continue
            candidate['accepted'] = True
            accepted.append((mapping, candidate, geometry))
        if not accepted:
            return reject('No consistent pair of visible adjacent rectangular shaft end rims.')
        if len(accepted) != 1:
            return reject('Several end-rim corner mappings are plausible; obtain a clearer distinct view.')
        mapping, candidate, reference_geometry = accepted[0]
        perturbations = np.asarray(list(itertools.product((-1.0, 1.0), repeat=8))).reshape(-1, 2, 2, 2)
        variants = np.concatenate([pairs[None], pairs + perturbations])
        rays = []
        for i, name in enumerate(names):
            points = variants[:, i].reshape(-1, 2)
            rays.append(_unit(views[name].rays(points[:, 1], points[:, 0]), 'Perturbed end-rim rays').reshape(-1, 2, 3))
        rays = np.stack(rays, axis=1)
        chosen = rays[:, np.arange(2), mapping]
        common = _shared_corners(origins, chosen)
        other = rays[:, np.arange(2), 1 - np.asarray(mapping)]
        geometry = _reconstruct(origins, common, other, axes)
        if np.any(geometry['face_visibility'] <= 0):
            return reject('One-pixel/shaft uncertainty makes end-rim visibility ambiguous.')
        for alternative in itertools.product((0, 1), repeat=2):
            if alternative == mapping:
                continue
            alt_chosen = rays[:, np.arange(2), alternative]
            alt_common = _shared_corners(origins, alt_chosen)
            plausible_pixels = np.ones(len(variants), dtype=bool)
            cosine = np.sum(alt_chosen[:, 0] * alt_chosen[:, 1], axis=-1)
            plausible_pixels &= np.abs(cosine) < np.cos(np.radians(5))
            for i, name in enumerate(names):
                row, col, depth = views[name].project(alt_common)
                error = np.linalg.norm(np.c_[col, row] - variants[:, i, alternative[i]], axis=-1)
                plausible_pixels &= (error <= 3) & (depth > 0)
            if not plausible_pixels.any():
                continue
            alt_other = rays[:, np.arange(2), 1 - np.asarray(alternative)]
            alt_geometry = _reconstruct(origins, alt_common, alt_other, axes, strict=False)
            plausible = alt_geometry['valid'] & plausible_pixels[None] & (np.abs(alt_geometry['cosines']) <= np.sin(np.radians(15))) & (alt_geometry['face_visibility'].min(axis=-1) >= 0.1)
            if plausible.any():
                return reject('One-pixel/shaft uncertainty admits another end-rim corner mapping; obtain a clearer distinct view.')
        centers = geometry['centers'].reshape(-1, 3)
        xyz = reference_geometry['centers'][0, 0]
        low, high = (centers.min(axis=0), centers.max(axis=0))
        corners = np.array(list(itertools.product(*zip(low, high))))
        sensitivity = float(np.linalg.norm(corners - xyz, axis=1).max())
        return dict(accepted=True, reasons=[], xyz=xyz.tolist(), tip_xyz=xyz.tolist(), samples_xyz=np.vstack([xyz, corners]).tolist(), pixel_sensitivity_m=sensitivity, model='rectangular_end_rims', mapping=candidate['shared_endpoint_indices'], side_lengths_m=candidate['side_lengths_m'], side_angle_deg=candidate['side_angle_deg'], reprojection_errors_px=candidate['reprojection_errors_px'], ray_angle_deg=candidate['ray_angle_deg'], baseline_m=baseline, candidates=candidates, sample_count=len(centers), uncertainty=dict(pixel_error_px=1.0, shaft_sample_count=len(axes), pixel_variant_count=len(variants), enclosure='axis_aligned_bounding_box', min_xyz=low.tolist(), max_xyz=high.tolist(), max_sample_displacement_m=float(np.linalg.norm(centers - xyz, axis=1).max())), note='Two adjacent near-end rims, not two projections of one midpoint. Rectangular shaft and correct endpoint identities must be checked visually. Samples conservatively enclose all independent +/-1 pixel endpoint corners crossed with the supplied shaft-axis samples; no key dimensions are assumed.')
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        return reject(str(exc))
