"""Metric hypotheses from current RGB and a declared static X5 camera rig."""
from itertools import combinations, product
import math

import cv2
import numpy as np

from harness.execution import ExecutionError
from skills.shared.perception.colour_components.implementation import COLOURS, components, hue_distance


FACE_HUES = {'blue': 120, 'green': 60, 'red': 0, 'yellow': 30, 'orange': 15, 'cyan': 90}


def rotation_matrix(quaternion):
    value = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or norm < 1e-8:
        raise ExecutionError('Camera reconstruction requires a finite public robot orientation')
    scalar, first, second, third = value / norm
    return np.asarray([
        [1 - 2 * (second * second + third * third), 2 * (first * second - scalar * third), 2 * (first * third + scalar * second)],
        [2 * (first * second + scalar * third), 1 - 2 * (first * first + third * third), 2 * (second * third - scalar * first)],
        [2 * (first * third - scalar * second), 2 * (second * third + scalar * first), 1 - 2 * (first * first + second * second)],
    ])


def euler_matrix(angles):
    first, second, third = np.radians(angles)
    rotation_first = np.asarray([[1, 0, 0], [0, math.cos(first), -math.sin(first)], [0, math.sin(first), math.cos(first)]])
    rotation_second = np.asarray([[math.cos(second), 0, math.sin(second)], [0, 1, 0], [-math.sin(second), 0, math.cos(second)]])
    rotation_third = np.asarray([[math.cos(third), -math.sin(third), 0], [math.sin(third), math.cos(third), 0], [0, 0, 1]])
    return rotation_first @ rotation_second @ rotation_third


def fixed_transform(translation, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def camera_model(api, camera):
    image = api.image(camera)
    if image.shape != (480, 640, 3):
        raise ExecutionError('The declared static camera rig only supports native 640x480 RGB')
    if camera == 'cam_head':
        transform = fixed_transform([0.0, -0.41, 1.308], euler_matrix([30, 0, 0]))
        focal = 640 * 10.0 / 22.212
    elif camera in ('cam_left_wrist', 'cam_right_wrist'):
        side = camera.split('_')[1]
        pose = api.targets()[side + '_ee_pose']
        transform = fixed_transform(pose[:3], rotation_matrix(pose[3:]))
        transform = transform @ fixed_transform([0.057, 0, 0], euler_matrix([0, 0, -math.degrees(3.1416)]))
        transform = transform @ fixed_transform([-0.0275, 0, 0.05], euler_matrix([0, 0, 180]) @ euler_matrix([0, math.degrees(0.3491), 0]))
        transform = transform @ fixed_transform([0, 0, 0.001], euler_matrix([0, -80, -90]))
        focal = 640 * 13.0 / 20.955
    else:
        raise ExecutionError('A geometry tool cannot introduce another camera')
    return transform, focal


def pixel_rays(api, camera, pixels):
    transform, focal = camera_model(api, camera)
    points = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ExecutionError('A camera ray requires finite image measurements')
    if np.any(points < 0) or np.any(points[:, 0] >= 640) or np.any(points[:, 1] >= 480):
        raise ExecutionError('A camera ray must originate inside the current public image')
    rays = np.column_stack([(points[:, 0] - 320) / focal, -(points[:, 1] - 240) / focal, -np.ones(len(points))])
    directions = rays @ transform[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return transform[:3, 3].copy(), directions


def bright_face(image, colour):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] >= 105) & (hsv[..., 2] >= 180)
    height, width = image.shape[:2]
    found = [component for component in components(mask, (0, 0, width, height))
             if 30 <= component['area'] <= height * width * 0.03
             and max(component['bbox'][2:]) / max(1, min(component['bbox'][2:])) <= 2.2
             and min(FACE_HUES, key=lambda name: hue_distance(float(np.median(hsv[..., 0][component['mask']])), FACE_HUES[name])) == colour]
    found.sort(key=lambda component: component['area'], reverse=True)
    if not found or len(found) > 1 and found[1]['area'] >= found[0]['area'] * 0.15:
        return None
    component = found[0]
    brightness = image.astype(float).mean(axis=2)
    threshold = float(np.percentile(brightness[component['mask']], 90)) * 0.93
    face = component['mask'] & (brightness >= threshold)
    rows, columns = np.where(face)
    if len(rows) < 30:
        return None
    pixels = np.column_stack([columns, rows]).astype(float)
    return {'pixels': pixels, 'centroid': pixels.mean(axis=0), 'area': len(pixels),
            'brightness_threshold': threshold, 'whole_area': component['area']}


def triangulate(api, first_camera, first_pixel, second_camera, second_pixel):
    first_origin, first_rays = pixel_rays(api, first_camera, [first_pixel])
    second_origin, second_rays = pixel_rays(api, second_camera, [second_pixel])
    first_direction, second_direction = first_rays[0], second_rays[0]
    angle = math.acos(float(np.clip(abs(first_direction @ second_direction), 0, 1)))
    if np.linalg.norm(first_origin - second_origin) < 0.02 or angle < math.radians(8):
        return None
    matrix = np.column_stack([first_direction, -second_direction])
    distances = np.linalg.lstsq(matrix, second_origin - first_origin, rcond=None)[0]
    if min(distances) <= 0:
        return None
    first_point = first_origin + first_direction * distances[0]
    second_point = second_origin + second_direction * distances[1]
    residual = float(np.linalg.norm(first_point - second_point))
    if residual > 0.004:
        return None
    return {'point': (first_point + second_point) * 0.5, 'ray_separation_m': residual,
            'ray_angle_radians': angle, 'cameras': [first_camera, second_camera]}


def triangulated_height_interval(api, first_camera, first_pixel, second_camera, second_pixel, estimator=None):
    estimator = triangulate if estimator is None else estimator
    first_pixel, second_pixel = np.asarray(first_pixel, dtype=float), np.asarray(second_pixel, dtype=float)
    if (first_pixel.shape != (2,) or second_pixel.shape != (2,)
            or not np.isfinite(first_pixel).all() or not np.isfinite(second_pixel).all()):
        raise ExecutionError('A feature interval requires two finite current public pixels')
    current = estimator(api, first_camera, first_pixel, second_camera, second_pixel)
    if current is None:
        return None
    current_point = np.asarray(current['point'], dtype=float)
    if current_point.shape != (3,) or not np.isfinite(current_point).all():
        return None
    heights = [float(current_point[2])]
    for first_offset in product((-.5, .5), repeat=2):
        for second_offset in product((-.5, .5), repeat=2):
            try:
                estimate = estimator(api, first_camera, first_pixel + first_offset,
                                     second_camera, second_pixel + second_offset)
            except ExecutionError:
                return None
            if estimate is None:
                return None
            point = np.asarray(estimate['point'], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                return None
            heights.append(float(point[2]))
    return min(heights), max(heights)


def face_on_plane(api, camera, face, height):
    origin, directions = pixel_rays(api, camera, face['pixels'])
    if np.any(np.abs(directions[:, 2]) < 1e-6):
        return None
    distances = (height - origin[2]) / directions[:, 2]
    if min(distances) <= 0:
        return None
    points = origin + directions * distances[:, None]
    rectangle = cv2.minAreaRect(points[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rectangle).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 0.01 or max(lengths) > 0.10 or max(lengths) / min(lengths) > 1.6:
        return None
    direction = edges[int(np.argmax(lengths))]
    yaw = (math.atan2(direction[1], direction[0]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return {'point': np.asarray([*rectangle[0], height]), 'edge_m': float(np.median(lengths)),
            'yaw_radians': yaw, 'face_area_pixels': face['area']}


def scene_geometry(api, colours, remembered=None):
    colours = tuple(dict.fromkeys(colours))
    if not 1 <= len(colours) <= len(COLOURS) or any(colour not in COLOURS for colour in colours):
        raise ExecutionError('A geometry request must select bounded declared colours')
    remembered = remembered or {}
    faces = {colour: {camera: bright_face(api.image(camera), colour)
                     for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')} for colour in colours}
    stereo = {}
    for colour, views in faces.items():
        candidates = []
        for first, second in combinations(views, 2):
            if views[first] is None or views[second] is None:
                continue
            estimate = triangulate(api, first, views[first]['centroid'], second, views[second]['centroid'])
            if estimate is not None and views['cam_head'] is not None:
                measured = face_on_plane(api, 'cam_head', views['cam_head'], float(estimate['point'][2]))
                if measured is not None:
                    candidates.append(dict(measured, stereo=estimate))
        if candidates:
            stereo[colour] = min(candidates, key=lambda value: value['stereo']['ray_separation_m'])
    plane = remembered.get('support_plane_m')
    if plane is None:
        anchors = [value for colour, value in stereo.items() if colour not in remembered.get('raised_colours', {})]
        if not anchors:
            return {'status': 'uncertain', 'reason': 'No consistent current multi-view coloured surface', 'targets': {}}
        plane = float(np.median([value['point'][2] - value['edge_m'] for value in anchors]))
    if type(plane) not in (int, float) or not math.isfinite(plane):
        raise ExecutionError('A remembered support plane must be a finite RGB-derived hypothesis')
    targets = {}
    for colour in colours:
        if colour in stereo:
            targets[colour] = stereo[colour]
            continue
        face = faces[colour]['cam_head']
        if face is None:
            continue
        base = remembered.get('raised_colours', {}).get(colour, plane)
        height = base
        measured = None
        for iteration in range(6):
            measured = face_on_plane(api, 'cam_head', face, height)
            if measured is None:
                break
            height = base + measured['edge_m']
        if measured is not None:
            targets[colour] = dict(measured, stereo=None)
    return {'status': 'observed' if targets else 'uncertain', 'support_plane_m': plane,
            'targets': targets, 'calibration': 'declared_static_x5_rig',
            'interpretation': 'Current RGB cube-face and same-episode support hypotheses, not scene ground truth.'}
