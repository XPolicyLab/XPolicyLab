import cv2
import numpy as np

from skills.shared.perception.connected_image_regions.implementation import run as components


def hole_areas(mask):
    background = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    flooded = np.pad(background, 1, constant_values=1)
    cv2.floodFill(flooded, None, (0, 0), 2)
    holes = flooded[1:-1, 1:-1] == 1
    return sorted((item['area'] for item in components(holes, minimum_area=3)), reverse=True)


def run(observed, labels, outlines, ranking, raster_to_image=None, clipped=False):
    current, templates = np.asarray(observed), np.asarray(outlines)
    transform = np.eye(3) if raster_to_image is None else np.asarray(raster_to_image, dtype=float)
    if (current.ndim != 2 or not 1 <= current.size <= 307200 or current.dtype.kind not in 'buif'
            or not np.isfinite(current).all() or templates.ndim != 3 or len(templates) != len(labels)
            or len(ranking) != len(labels) or len(set(labels)) != len(labels) or not 2 <= len(labels) <= 64
            or {record['label'] for record in ranking} != set(labels) or type(clipped) is not bool
            or transform.shape != (3, 3) or not np.isfinite(transform).all() or np.linalg.cond(transform) > 1e9):
        raise ValueError('Compare only a complete independently ranked dictionary with current image holes')
    observed_holes = hole_areas(current)
    height, width = current.shape
    candidates = []
    for record in ranking[:3]:
        fitted = np.asarray(record['outline_to_observed'], dtype=float)
        if fitted.shape != (3, 3) or not np.isfinite(fitted).all() or not np.isfinite(record['similarity']):
            raise ValueError('A topology comparison needs a finite scale-free fitted image transform')
        projected = cv2.warpPerspective(templates[list(labels).index(record['label'])], transform @ fitted,
                                        (width, height), flags=cv2.INTER_NEAREST)
        holes = hole_areas(projected)
        candidates.append({'label': record['label'], 'hole_areas_pixels': holes,
                           'hole_count_matches': len(holes) == len(observed_holes)})
    runner_holes = candidates[1]['hole_areas_pixels']
    extra_holes = (observed_holes[len(runner_holes):] if len(observed_holes) > len(runner_holes)
                   else runner_holes[len(observed_holes):])
    resolves = (not clipped and ranking[0]['similarity'] >= .90 and np.count_nonzero(current) >= 140
                and candidates[0]['hole_count_matches'] and not candidates[1]['hole_count_matches']
                and min(extra_holes or [0]) >= 5)
    return {'observed_hole_areas_pixels': observed_holes, 'candidates': candidates,
            'resolves_close_candidates': bool(resolves), 'physical_dimensions_used': False,
            'interpretation': 'visible_hole_agreement_not_contact_or_hidden_occlusion_measurement'}
