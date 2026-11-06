import cv2
import numpy as np

from skills.shared.perception.colour_components.implementation import components, hue_distance
from skills.shared.perception.rgb_surface_geometry.implementation import FACE_HUES, face_on_plane, triangulate


def bright_faces(image, colour):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width = image.shape[:2]
    mask = (hsv[..., 1] >= 105) & (hsv[..., 2] >= 180)
    brightness = image.astype(float).mean(axis=2)
    found = []
    for region in components(mask, (0, 0, width, height)):
        column, row, box_width, box_height = region['bbox']
        if not (30 <= region['area'] <= height * width * 0.03
                and max(box_width, box_height) / max(1, min(box_width, box_height)) <= 2.2
                and column > 0 and row > 0 and column + box_width < width and row + box_height < height):
            continue
        hue = float(np.median(hsv[..., 0][region['mask']]))
        if min(FACE_HUES, key=lambda name: hue_distance(hue, FACE_HUES[name])) != colour:
            continue
        threshold = float(np.percentile(brightness[region['mask']], 90)) * 0.93
        rows, columns = np.where(region['mask'] & (brightness >= threshold))
        if len(rows) < 30:
            continue
        pixels = np.column_stack([columns, rows]).astype(float)
        found.append({'pixels': pixels, 'centroid': pixels.mean(axis=0), 'area': len(pixels),
                      'brightness_threshold': threshold, 'whole_area': region['area']})
    return sorted(found, key=lambda face: float(face['centroid'][0])) if len(found) <= 16 else []


def choose_consistent(candidates):
    candidates.sort(key=lambda value: value['ray_separation_m'])
    if not candidates:
        return None
    corroborated = []
    for candidate in candidates:
        cameras = set()
        for other in candidates:
            if np.linalg.norm(candidate['point'] - other['point']) <= 0.015:
                cameras.update(set(other.get('cameras', ())) & {'cam_left_wrist', 'cam_right_wrist'})
        if len(cameras) == 2:
            corroborated.append(candidate)
    if corroborated:
        chosen = corroborated[0]
        if any(np.linalg.norm(chosen['point'] - other['point']) > 0.015 for other in corroborated[1:]):
            return None
        return chosen
    chosen = candidates[0]
    for candidate in candidates[1:]:
        if (candidate['ray_separation_m'] <= chosen['ray_separation_m'] + 0.003
                and np.linalg.norm(candidate['point'] - chosen['point']) > 0.015):
            return None
    return chosen


def surface_for_instance(api, head_face, colour):
    candidates = []
    for camera in ('cam_left_wrist', 'cam_right_wrist'):
        for face in bright_faces(api.image(camera), colour):
            stereo = triangulate(api, 'cam_head', head_face['centroid'], camera, face['centroid'])
            if stereo is None:
                continue
            surface = face_on_plane(api, 'cam_head', head_face, float(stereo['point'][2]))
            if surface is not None:
                candidates.append(dict(stereo, **surface))
    return choose_consistent(candidates)


def contact_for_instance(api, head_region, regions):
    candidates = []
    for camera in ('cam_left_wrist', 'cam_right_wrist'):
        for region in regions(api.image(camera)):
            stereo = triangulate(api, 'cam_head', head_region['centroid'], camera, region['centroid'])
            if stereo is not None:
                candidates.append(stereo)
    return choose_consistent(candidates)
