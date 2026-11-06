import numpy as np
from scipy.optimize import linear_sum_assignment

from skills.shared.perception.rgb_surface_geometry.implementation import triangulate


CAMERAS = ('cam_left_wrist', 'cam_right_wrist')


def select_family(candidates, head_count, observed_planar_row=False):
    if not 2 <= head_count <= 8 or len(candidates) > 256 or type(observed_planar_row) is not bool:
        raise ValueError('A public surface family requires bounded region hypotheses')
    for candidate in candidates:
        if (not isinstance(candidate, dict) or type(candidate.get('head_index')) is not int
                or not 0 <= candidate['head_index'] < head_count
                or type(candidate.get('wrist_index')) is not int or not 0 <= candidate['wrist_index'] < 16
                or candidate.get('camera') not in CAMERAS):
            raise ValueError('Surface correspondence requires declared current camera region identities')
        point = np.asarray(candidate.get('point'), dtype=float)
        size, gap = float(candidate.get('size_m', np.nan)), float(candidate.get('ray_separation_m', np.nan))
        if (point.shape != (3,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 3
                or not .005 <= size <= .12 or not 0 <= gap <= .005):
            raise ValueError('Surface correspondence accepts only finite bounded visual measurements')
    families = []
    for seed in candidates:
        height, size = float(seed['point'][2]), float(seed['size_m'])
        chosen = []
        for camera in CAMERAS:
            allowed = [candidate for candidate in candidates if candidate['camera'] == camera
                       and abs(float(candidate['point'][2]) - height) <= .012
                       and .8 <= float(candidate['size_m']) / size <= 1.2]
            wrist_ids = sorted({candidate['wrist_index'] for candidate in allowed})
            if not wrist_ids:
                continue
            costs = np.full((head_count, len(wrist_ids) + head_count), 1000.)
            costs[:, len(wrist_ids):] = 5.
            options = {}
            for candidate in allowed:
                row, column = candidate['head_index'], wrist_ids.index(candidate['wrist_index'])
                cost = ((float(candidate['ray_separation_m']) / .005) ** 2
                        + ((float(candidate['point'][2]) - height) / .012) ** 2
                        + ((float(candidate['size_m']) / size - 1) / .2) ** 2)
                if cost < costs[row, column]:
                    costs[row, column], options[row, column] = cost, candidate
            rows, columns = linear_sum_assignment(costs)
            chosen.extend(options[row, column] for row, column in zip(rows, columns)
                          if (row, column) in options and costs[row, column] < 5)
        heads = sorted({candidate['head_index'] for candidate in chosen})
        views = sorted({candidate['camera'] for candidate in chosen})
        heights = np.asarray([candidate['point'][2] for candidate in chosen], dtype=float)
        sizes = np.asarray([candidate['size_m'] for candidate in chosen], dtype=float)
        if len(heads) < 2:
            continue
        partial_single_wrist = len(views) < 2 and len(heads) < head_count
        if partial_single_wrist and not (observed_planar_row
                and len(heads) >= max(3, head_count - 1)
                and np.ptp(heights) <= .002 and sizes.max() / sizes.min() <= 1.12
                and max(candidate['ray_separation_m'] for candidate in chosen) <= .0015):
            continue
        if np.ptp(heights) > .012 or sizes.max() / sizes.min() > 1.2:
            continue
        measured_row = ({'semantics': 'current_visible_planar_row_independent_head_wrist_pairs',
                         'matched_height_interval_m': [float(heights.min()), float(heights.max())],
                         'matched_radius_interval_m': [float(sizes.min()), float(sizes.max())],
                         'maximum_ray_separation_m': float(max(candidate['ray_separation_m'] for candidate in chosen))}
                        if partial_single_wrist else None)
        families.append({'height_m': float(np.median(heights)), 'size_m': float(np.median(sizes)),
            'observed_head_indices': heads, 'wrist_views': views,
            'quality': float(np.mean([candidate['ray_separation_m'] for candidate in chosen])),
            'rank': (len(heads), len(views)), 'partial_single_wrist': partial_single_wrist,
            'row_correspondence_evidence': measured_row})
    if not families:
        return None
    families.sort(key=lambda family: (-family['rank'][0], -family['rank'][1], family['quality']))
    best = families[0]
    ambiguity_height = .003 if best['partial_single_wrist'] else .015
    if any(family['rank'] == best['rank'] and abs(family['height_m'] - best['height_m']) > ambiguity_height
           and family['quality'] <= best['quality'] + .001 for family in families[1:]):
        return None
    return {name: value for name, value in best.items()
            if name not in ('rank', 'quality', 'partial_single_wrist') and value is not None}


def run(api, head_regions, detector, projector, support_plane_m, minimum_elevation_m=.005,
        maximum_elevation_m=.3, observed_planar_row=False):
    if (not 2 <= len(head_regions) <= 8 or not np.isfinite(support_plane_m)
            or not np.isfinite(minimum_elevation_m) or not np.isfinite(maximum_elevation_m)
            or not 0 <= minimum_elevation_m < maximum_elevation_m <= .3
            or type(observed_planar_row) is not bool):
        raise ValueError('A public family needs a measured support and bounded head regions')
    if observed_planar_row:
        from skills.shared.perception.observed_row_geometry.implementation import run as observed_rows
        centres = np.asarray([region['centroid'] for region in head_regions], dtype=float)
        widths = np.asarray([max(region['bbox'][2:]) for region in head_regions], dtype=float)
        _, _, directions = np.linalg.svd(centres - centres.mean(axis=0), full_matrices=False)
        rows = observed_rows(centres, widths, directions[0], len(head_regions))
        if len(rows) != 1 or widths.max() / widths.min() > 1.5:
            raise ValueError('Partial correspondence requires an independently visible complete head row')
    candidates = []
    for camera in CAMERAS:
        wrists = detector(api.image(camera))
        if len(wrists) > 16:
            raise ValueError('A public family cannot expand wrist detections without a budget')
        for head_index, head in enumerate(head_regions):
            for wrist_index, wrist in enumerate(wrists):
                estimate = triangulate(api, 'cam_head', head['centroid'], camera, wrist['centroid'])
                if estimate is None or not minimum_elevation_m < estimate['point'][2] - support_plane_m <= maximum_elevation_m:
                    continue
                surface = projector(api, 'cam_head', head, float(estimate['point'][2]))
                if surface is not None:
                    candidates.append(dict(estimate, head_index=head_index, wrist_index=wrist_index,
                        camera=camera, size_m=float(surface['radius_m'])))
    family = select_family(candidates, len(head_regions), observed_planar_row)
    if family is None:
        return None
    surfaces = [projector(api, 'cam_head', region, family['height_m']) for region in head_regions]
    if any(surface is None or not .8 <= surface['radius_m'] / family['size_m'] <= 1.2 for surface in surfaces):
        return None
    if family.get('row_correspondence_evidence') is not None:
        family['row_correspondence_evidence']['head_pixel_widths'] = widths.tolist()
    return dict(family, surfaces=surfaces)
