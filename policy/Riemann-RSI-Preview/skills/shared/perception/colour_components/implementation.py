"""Public RGB components, extracted from the mature block-alignment perception."""
from typing import Any

import cv2
import numpy as np


COLOURS = {'blue': 112.0, 'green': 60.0, 'red': 0.0,
           'yellow': 28.0, 'orange': 14.0, 'cyan': 88.0}


def hue_distance(value, reference):
    return np.abs((value - reference + 90.0) % 180.0 - 90.0)


def components(mask, roi):
    column_start, row_start, column_end, row_end = roi
    cropped = np.zeros(mask.shape, dtype=np.uint8)
    cropped[row_start:row_end, column_start:column_end] = np.asarray(
        mask[row_start:row_end, column_start:column_end], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped, 8)
    result: list[dict[str, Any]] = []
    for index in range(1, count):
        column, row, width, height, area = [int(value) for value in stats[index]]
        rows, columns = np.where(labels == index)
        result.append({'area': area, 'bbox': (column, row, width, height),
            'centroid': np.asarray(centroids[index], dtype=np.float64),
            'pixels': np.stack([columns, rows], axis=1).astype(np.float64),
            'mask': labels == index})
    return result


def rgb_image(observation, camera):
    raw = observation['vision'][camera]
    image = np.asarray(raw['color'] if isinstance(raw, dict) else raw)
    if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1:
        image = image * 255
    if not np.isfinite(image).all():
        raise ValueError('A skill requires finite public RGB')
    return np.clip(image[..., :3], 0, 255).astype(np.uint8)


def colour_components(image, colour):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hue_distance(hsv[..., 0].astype(float), COLOURS[colour]) <= 11)
            & (hsv[..., 1] >= 75) & (hsv[..., 2] >= 45)).astype(np.uint8) * 255
    height, width = image.shape[:2]
    minimum_area = max(20, int(height * width * 0.0004))
    found = components(mask, (0, 0, width, height))
    return sorted((item for item in found if item['area'] >= minimum_area),
                  key=lambda item: item['area'], reverse=True)[:6]


def image_geometry(component, image_shape):
    height, width = image_shape[:2]
    column, row, box_width, box_height = component['bbox']
    return {'centroid_image': (component['centroid'] / [width, height]).tolist(),
            'box_image': [column / width, row / height,
                          box_width / width, box_height / height],
            'area_fraction': component['area'] / (width * height)}


def run(observation, request, previous=None):
    image = rgb_image(observation, request['camera'])
    found = colour_components(image, request['colour'])
    return {'kind': 'colour_components', 'status': 'observed' if found else 'not_observed',
            'camera': request['camera'], 'colour': request['colour'],
            'components': [image_geometry(item, image.shape) for item in found],
            'interpretation': 'Colour regions in the current image; not object identity or world poses.'}
