# Copyright © 2026 Martingale. All Rights Reserved.
import itertools
import numpy as np

def _unit(values, name):
    values = np.asarray(values, float)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(length < 1e-10):
        raise ValueError(name + ' contains a nonfinite or zero direction.')
    return values / length

def estimate(views, tip_edge_pairs, shaft_edge_pairs):
    """Return directed shaft-axis samples from two or three current views.

    Both mappings contain [[column,row],[column,row]] opposite endpoints of
    one face cross-section per camera. Near and proximal sections must lie on
    that same longitudinal face midline; the proximal section points toward
    the bow. Camera-to-camera axial stations and visible faces may differ.

    Independent +/-1-pixel midpoint errors on both line endpoints in every
    view are retained, conservatively covering +/-1-pixel errors of each rim
    endpoint. Perspective midpoint bias is not explicitly modelled; avoid a
    severely foreshortened face and inspect the actual selected endpoints.
    This measures direction only and cannot authorize insertion.
    """
    per_view = {}

    def reject(reason):
        return dict(accepted=False, reasons=[reason], shaft_axis_xyz=None, shaft_samples_xyz=None, per_view=per_view)
    try:
        if not isinstance(tip_edge_pairs, dict) or not isinstance(shaft_edge_pairs, dict) or set(tip_edge_pairs) != set(shaft_edge_pairs) or (len(tip_edge_pairs) not in (2, 3)):
            raise ValueError('Use the same two or three cameras for near-rim and proximal-face edge pairs.')
        names = list(tip_edge_pairs)
        corners = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4))).reshape(-1, 2, 2)
        plane_sets, delta_sets = ([], [])
        for name in names:
            if name not in views or views[name] is None:
                raise ValueError('Unavailable shaft camera: ' + name)
            view = views[name]
            height, width = view.shape[:2]
            pairs = np.asarray([tip_edge_pairs[name], shaft_edge_pairs[name]], float)
            if pairs.shape != (2, 2, 2) or not np.isfinite(pairs).all():
                raise ValueError('Each near and proximal face cross-section needs two finite pixel endpoints.')
            if np.any(pairs < 0) or np.any(pairs > [width - 1, height - 1]):
                raise ValueError('Every shaft endpoint must be inside the original image.')
            spans = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
            if np.any(spans < 2):
                raise ValueError('Both face cross-sections must span at least 2 pixels; obtain a clearer view.')
            midpoints = pairs.mean(axis=1)
            span = float(np.linalg.norm(midpoints[1] - midpoints[0]))
            per_view[name] = dict(longitudinal_span_px=span, face_cross_section_spans_px=spans.tolist(), midpoints=midpoints.tolist())
            if span < 20:
                raise ValueError('Shaft line in ' + name + ' spans fewer than 20 pixels; use a farther proximal cross-section or a closer view.')
            variants = np.concatenate([midpoints[None], midpoints + corners])
            flat = variants.reshape(-1, 2)
            rays = _unit(view.rays(flat[:, 1], flat[:, 0]), 'Shaft rays').reshape(-1, 2, 3)
            plane_sets.append(_unit(np.cross(rays[:, 0], rays[:, 1]), 'Shaft image planes'))
            delta_sets.append(rays[:, 1] - rays[:, 0])
        combinations = np.array(list(itertools.product(range(17), repeat=len(names))))
        planes = np.stack([plane_sets[i][combinations[:, i]] for i in range(len(names))], axis=1)
        deltas = np.stack([delta_sets[i][combinations[:, i]] for i in range(len(names))], axis=1)
        _, values, vh = np.linalg.svd(planes, full_matrices=True)
        axes = vh[:, -1, :]
        residual = values[:, 2] if len(names) == 3 else np.zeros(len(values))
        conditioning = (values[:, 1] - residual) / values[:, 0]
        minimum = float(conditioning.min())
        if minimum < 0.1:
            return reject('Shaft image planes do not resolve a stable common axis; obtain a supporting view from a different side.')
        signs = np.einsum('nij,nj->ni', deltas, axes)
        flip = np.sign(signs[:, 0])
        if np.any(signs * flip[:, None] <= 1e-10):
            return reject('Near-rim to proximal-face directions disagree between views; check each face line points toward the bow.')
        axes *= flip[:, None]
        central = axes[0]
        reference_residual = float(np.degrees(np.arcsin(np.clip(np.abs(planes[0] @ central), 0, 1))).max())
        if reference_residual > 3:
            return reject('Longitudinal face lines disagree by more than 3 degrees; check they follow the same shaft direction.')
        spread = float(np.degrees(np.arccos(np.clip(axes @ central, -1, 1))).max())
        return dict(accepted=True, reasons=[], shaft_axis_xyz=central.tolist(), shaft_samples_xyz=axes.tolist(), per_view=per_view, reference_conditioning=float(conditioning[0]), min_conditioning=minimum, reference_plane_residual_deg=reference_residual, sampled_spread_deg=spread, note='Direction from parallel longitudinal face midlines; different visible faces and axial stations are allowed across cameras. Samples include independent one-pixel line-endpoint errors in all views. Perspective midpoint bias and incorrect face identities remain unmodelled; this is not insertion readiness.')
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        return reject(str(exc))
