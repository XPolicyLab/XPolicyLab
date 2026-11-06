import cv2
import numpy as np

from skills.shared.perception.rgb_surface_geometry.implementation import camera_model
from skills.shared.perception.robot_mounting_reference.implementation import run as mounting_reference
from skills.shared.perception.spatial_patch_correspondence.implementation import measure as patch_measurement
from skills.shared.perception.spatial_patch_correspondence.implementation import rank as patch_rank


def image_fields(image, color_independent=False, antialiased=False):
    image = np.asarray(image)
    if (image.shape != (480, 640, 3) or image.dtype != np.uint8
            or type(color_independent) is not bool or type(antialiased) is not bool):
        raise ValueError('Align only current native public RGB views')
    reduced = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(reduced, cv2.COLOR_RGB2HSV)
    mask = np.asarray((hsv[..., 0] < 38) & (hsv[..., 1] > 45) & (hsv[..., 2] > 45), np.uint8)
    gray = cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if color_independent:
        high = gray - cv2.GaussianBlur(gray, (0, 0), 3.)
        energy = cv2.blur(high * high, (7, 7))
        mask = np.asarray((gray > 10.) & (gray < 252.) & (energy > .36), np.uint8)
    if antialiased:
        gray = cv2.GaussianBlur(gray, (0, 0), .5)
    return gray, cv2.erode(mask, np.ones((3, 3), np.uint8))


def plane_projection(api, camera, support_height):
    transform, focal = camera_model(api, camera)
    rotation = transform[:3, :3].T
    intrinsic = np.asarray([[focal / 2, 0, -160], [0, -focal / 2, -120], [0, 0, -1.]])
    world = np.column_stack((rotation[:, 0], rotation[:, 1],
                             rotation @ (np.asarray([0., 0., support_height]) - transform[:3, 3])))
    return intrinsic @ world


def ranked_correlation(source, destination, warp, robust=False, for_search=False):
    if type(robust) is not bool or type(for_search) is not bool:
        raise ValueError('Current texture robustness is an explicit boolean mode')
    source_gray, source_mask = source
    destination_gray, destination_mask = destination
    predicted = cv2.warpPerspective(source_gray, warp, (320, 240), flags=cv2.INTER_LINEAR)
    predicted_mask = cv2.warpPerspective(source_mask, warp, (320, 240), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(np.asarray((predicted_mask != 0) & (destination_mask != 0), np.uint8),
                     np.ones((5, 5), np.uint8)).astype(bool)
    if np.count_nonzero(valid) < 1500:
        return None
    predicted_high = predicted - cv2.GaussianBlur(predicted, (0, 0), 3.)
    destination_high = destination_gray - cv2.GaussianBlur(destination_gray, (0, 0), 3.)
    if robust:
        estimator = patch_rank if for_search else patch_measurement
        measured = estimator(predicted_high, destination_high, valid)
        return None if measured is None else (measured['correlation'],
            measured['current_matching_pixels'], measured['selection_score'])
    first, second = predicted_high[valid], destination_high[valid]
    first -= first.mean()
    second -= second.mean()
    if min(float(first.std()), float(second.std())) < .8:
        return None
    score = float(first @ second / np.sqrt((first @ first) * (second @ second)))
    return score, int(np.count_nonzero(valid)), score


def correlation(source, destination, warp, robust=False):
    measured = ranked_correlation(source, destination, warp, robust)
    return None if measured is None else measured[:2]


def qualified_alignments(records):
    accepted = [record for record in records if record['qualified']]
    if len(accepted) == 2 and np.ptp([record['height_estimate_m'] for record in accepted]) <= .006:
        return accepted, 'two_strong_current_camera_pairs'
    if len(accepted) != 1:
        return [], None
    strongest = accepted[0]
    if (strongest['correlation'] < .60 or strongest['height_uncertainty_m'] > .002
            or strongest['current_matching_pixels'] < 12000):
        return [], None
    corroborating = [record for record in records if record['camera'] != strongest['camera']
        and record.get('peak_unique') is True and record['correlation'] >= .20
        and record['height_uncertainty_m'] <= .002 and record['current_matching_pixels'] >= 6000
        and abs(record['height_estimate_m'] - strongest['height_estimate_m']) <= .004]
    return ([strongest], 'strong_current_pair_with_independent_camera_corroboration') if corroborating else ([], None)


def current_alignment(api, color_independent=False, antialiased=False):
    if type(color_independent) is not bool or type(antialiased) is not bool:
        raise ValueError('Current texture processing modes must be explicit booleans')
    if not api.own_robot_bases_available():
        return {'status': 'uncertain', 'plane': None, 'reason': 'own_camera_reference_unavailable'}
    mounting = mounting_reference(api.own_robot_bases())
    if mounting['status'] != 'observed':
        return {'status': 'uncertain', 'plane': None, 'reason': 'own_mounts_disagree'}
    cameras = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
    fields = {camera: image_fields(api.image(camera), color_independent, antialiased) for camera in cameras}
    coarse = mounting['height_m'] + np.linspace(-.15, .15, 31)
    records = []
    for camera in cameras[1:]:
        scores = []
        for candidate in coarse:
            source_projection = plane_projection(api, 'cam_head', float(candidate))
            destination_projection = plane_projection(api, camera, float(candidate))
            if abs(np.linalg.det(source_projection)) < 1e-6:
                continue
            observed = ranked_correlation(fields['cam_head'], fields[camera],
                                   destination_projection @ np.linalg.inv(source_projection),
                                   robust=color_independent, for_search=True)
            if observed is not None:
                scores.append((float(candidate), *observed))
        if not scores:
            continue
        best = max(scores, key=lambda value: value[3])
        fine = np.linspace(best[0] - .012, best[0] + .012, 25)
        fine_scores = []
        for candidate in fine:
            observed = ranked_correlation(fields['cam_head'], fields[camera],
                plane_projection(api, camera, float(candidate)) @ np.linalg.inv(
                    plane_projection(api, 'cam_head', float(candidate))), robust=color_independent)
            if observed is not None:
                fine_scores.append((float(candidate), *observed))
        if not fine_scores:
            continue
        best = max(fine_scores, key=lambda value: value[3])
        near = [value[0] for value in fine_scores if value[3] >= best[3] - .005]
        uncertainty = max(.001, (max(near) - min(near)) / 2)
        competing = [value for value in scores if abs(value[0] - best[0]) > .02 and value[3] >= best[3] - .01]
        peak_unique = not competing and fine[0] < best[0] < fine[-1]
        records.append({'camera': camera, 'height_estimate_m': best[0], 'correlation': best[1],
            'current_matching_pixels': best[2], 'height_uncertainty_m': uncertainty,
            'peak_unique': bool(peak_unique),
            'qualified': bool(best[1] >= .35 and uncertainty <= .006 and peak_unique)})
    accepted, qualification = qualified_alignments(records)
    qualified = bool(accepted)
    plane = None
    if qualified:
        plane = {'normal': [0., 0., 1.],
            'distance_m': float(np.mean([record['height_estimate_m'] for record in accepted])),
            'height_uncertainty_m': max(.002, max(record['height_uncertainty_m'] for record in accepted)),
            'semantics': 'horizontal_support_hypothesis_from_current_cross_view_texture_alignment'}
    return {'status': 'observed' if qualified else 'uncertain', 'plane': plane, 'camera_alignment': records,
            'qualification': qualification,
            'reason': None if qualified else 'independent_current_texture_alignment_not_qualified',
            'official_success_claimed': False}


def run(api):
    original = current_alignment(api)
    if original['status'] == 'observed' or original['reason'] != 'independent_current_texture_alignment_not_qualified':
        return original
    independent = current_alignment(api, color_independent=True)
    if independent['status'] == 'observed':
        return independent
    return current_alignment(api, color_independent=True, antialiased=True)
