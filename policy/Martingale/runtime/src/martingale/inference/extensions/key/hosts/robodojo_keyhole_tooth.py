# Copyright © 2026 Martingale. All Rights Reserved.
import itertools
import numpy as np

def _unit(values, name):
    values = np.asarray(values, float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm < 1e-10):
        raise ValueError(name + ' contains a nonfinite or zero direction.')
    return values / norm

def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))

def estimate(views, tooth_edge_pixels, shaft_axis, shaft_samples):
    """Measure a directed width axis, with conservative pixel/shaft samples.

    tooth_edge_pixels maps each camera to [[root_col,root_row],
    [outer_col,outer_row]] on ONE straight transverse tooth edge, directed away
    from the shaft. A different parallel edge may be used in another camera.
    shaft_samples contains xyz DIRECTION vectors, not landmark positions.

    Given and independent +/-1 pixel endpoint corners are combined with
    every supplied shaft direction. Samples from each view are all retained;
    averaging views never shrinks the sensitivity envelope. Conditioning is
    sin(angle between image-plane normal and shaft); <0.15 is rejected because
    the perpendicular-edge solution approaches a geometric ambiguity. Final
    angular tolerance must still be checked by the caller against the slot.
    """
    per_view = {}

    def reject(reason):
        return dict(accepted=False, reasons=[reason], width_axis_xyz=None, width_samples_xyz=None, per_view=per_view)
    try:
        if not isinstance(tooth_edge_pixels, dict) or not 1 <= len(tooth_edge_pixels) <= 3:
            raise ValueError('Mark a transverse tooth edge in one to three current views.')
        axis = np.asarray(shaft_axis, float)
        samples = np.asarray(shaft_samples, float)
        if axis.shape != (3,) or samples.ndim != 2 or samples.shape[1] != 3 or (not len(samples)):
            raise ValueError('Shaft axis and samples must contain xyz directions.')
        axis = _unit(axis, 'Shaft axis')
        samples = _unit(np.vstack([axis, samples]), 'Shaft samples')
        corners = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4))).reshape(-1, 2, 2)
        reference_directions, all_samples = ([], [])
        for name, endpoints in tooth_edge_pixels.items():
            if name not in views or views[name] is None:
                raise ValueError('Unavailable tooth-edge camera: ' + name)
            points = np.asarray(endpoints, float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError('Each tooth edge needs two finite [column,row] endpoints.')
            view = views[name]
            height, width = view.shape[:2]
            if np.any(points < 0) or np.any(points > [width - 1, height - 1]):
                raise ValueError('Tooth-edge endpoints must be inside the original image.')
            length = float(np.linalg.norm(points[1] - points[0]))
            if length < 8:
                raise ValueError('Tooth edge spans fewer than 8 pixels; get a closer non-foreshortened view.')
            variants = np.concatenate([points[None], points + corners])
            flat = variants.reshape(-1, 2)
            rays = _unit(view.rays(flat[:, 1], flat[:, 0]), 'Tooth-edge rays').reshape(-1, 2, 3)
            planes = _unit(np.cross(rays[:, 0], rays[:, 1]), 'Tooth-edge image planes')
            widths = np.cross(planes[None, :, :], samples[:, None, :])
            conditioning = np.linalg.norm(widths, axis=-1)
            per_view[name] = dict(edge_length_px=length, reference_conditioning=float(conditioning[0, 0]), min_conditioning=float(conditioning.min()))
            if np.any(conditioning < 0.15):
                raise ValueError('Tooth edge in ' + name + ' does not resolve width about the shaft; use an elevated oblique view, not a view at the tooth edge height.')
            widths = widths / conditioning[:, :, None]
            signs = np.einsum('ijk,jk->ij', widths, rays[:, 1] - rays[:, 0])
            if np.any(np.abs(signs) < 1e-10):
                raise ValueError('Tooth edge in ' + name + ' is too foreshortened to resolve its direction.')
            widths *= np.where(signs >= 0, 1.0, -1.0)[:, :, None]
            central = widths[0, 0]
            reference_directions.append(central)
            all_samples.append(widths.reshape(-1, 3))
            pixel_spread = float(_angle(widths[0], central).max())
            per_view[name].update(width_axis_xyz=central.tolist(), sampled_spread_deg=float(_angle(widths, central).max()), pixel_only_spread_deg=pixel_spread, shaft_only_spread_deg=float(_angle(widths[:, 0], central).max()), suggested_edge_span_px=max(25, int(np.ceil(length * pixel_spread / 4.0))), preferred_conditioning=0.6, observation_guidance='For final alignment, use a close elevated oblique view with a broad exposed tooth edge. Aim for the suggested pixel span and conditioning at least 0.6, then remeasure; these are camera-placement hints, not acceptance guarantees. Shaft-only error needs clearer shaft views.')
        reference_directions = np.asarray(reference_directions)
        agreement = float(np.degrees(np.arccos(np.clip(reference_directions @ reference_directions.T, -1, 1))).max())
        if agreement > 8:
            return reject('Transverse tooth-edge directions disagree between views by more than 8 degrees.')
        central = _unit(reference_directions.mean(axis=0), 'Mean tooth direction')
        combined = np.vstack([central, *all_samples])
        return dict(accepted=True, reasons=[], width_axis_xyz=central.tolist(), width_samples_xyz=combined.tolist(), per_view=per_view, view_disagreement_deg=agreement, note='Uses straight tooth edges perpendicular to the shaft. Samples include one-pixel endpoint and shaft uncertainty, not a guarantee of feature identity. Check this direction against the measured slot before insertion.')
    except (ValueError, TypeError, IndexError) as exc:
        return reject(str(exc))
