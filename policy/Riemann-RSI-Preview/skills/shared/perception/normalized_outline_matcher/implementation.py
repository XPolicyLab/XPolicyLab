import math
from functools import lru_cache

import cv2
import numpy as np


RASTER_SIZE = 128
RASTER_PADDING = 24
MINIMUM_SIMILARITY = .885
MINIMUM_MARGIN = .012
DEFAULT_ANGLES = tuple(range(-90, 91, 3))


def normalize_with_transform(mask):
    value = np.asarray(mask)
    if (value.ndim != 2 or not 1 <= value.size <= 307200 or value.dtype.kind not in 'buif'
            or not np.isfinite(value).all()):
        raise ValueError('Normalize only a bounded finite currently observed silhouette')
    binary = (value > 0).astype(np.uint8)
    pixels = cv2.findNonZero(binary)
    if pixels is None or len(pixels) < 6:
        return None, None
    column, row, width, height = cv2.boundingRect(pixels)
    crop = binary[row:row + height, column:column + width]
    scale = (RASTER_SIZE - 2 * RASTER_PADDING) / max(width, height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((RASTER_SIZE, RASTER_SIZE), np.uint8)
    left, top = (RASTER_SIZE - resized_width) // 2, (RASTER_SIZE - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    horizontal, vertical = resized_width / width, resized_height / height
    transform = np.asarray([[horizontal, 0., left - column * horizontal],
                            [0., vertical, top - row * vertical], [0., 0., 1.]])
    return canvas, transform


def normalize(mask):
    return normalize_with_transform(mask)[0]


@lru_cache(maxsize=2)
def dictionary_bank(labels, shape, dtype, data, angles):
    templates = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    bank = []
    for label, outline in zip(labels, templates):
        original, template_transform = normalize_with_transform(outline)
        if original is None:
            raise ValueError('The declared visual dictionary contains an empty identity template')
        rotations = []
        for angle in angles:
            rotation = cv2.getRotationMatrix2D((RASTER_SIZE / 2, RASTER_SIZE / 2), -float(angle), 1.)
            rotated, rotated_transform = normalize_with_transform(
                cv2.warpAffine(original, rotation, (RASTER_SIZE, RASTER_SIZE), flags=cv2.INTER_NEAREST))
            if rotated is not None:
                transform = rotated_transform @ np.vstack([rotation, [0., 0., 1.]]) @ template_transform
                rotated.setflags(write=False)
                transform.setflags(write=False)
                rotations.append((angle, rotated, transform))
        bank.append((label, tuple(rotations)))
    return tuple(bank)


def run(observed, labels, outlines, angles=DEFAULT_ANGLES):
    labels = tuple(labels)
    templates = np.asarray(outlines)
    angles = tuple(angles)
    if (not 2 <= len(labels) <= 64 or len(set(labels)) != len(labels)
            or any(type(label) not in (int, str) for label in labels)
            or templates.ndim != 3 or len(templates) != len(labels)
            or templates.dtype.kind not in 'buif' or not np.isfinite(templates).all()
            or not 1 <= templates.shape[1] * templates.shape[2] <= 307200
            or not 1 <= len(angles) <= 121 or len(set(angles)) != len(angles)
            or any(type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) > 180 for angle in angles)):
        raise ValueError('Rank a complete bounded visual dictionary without inventing or conditioning an identity')
    current, observed_transform = normalize_with_transform(observed)
    empty = {'status': 'uncertain', 'label': None, 'yaw_degrees': None, 'similarity': 0., 'margin': 0.,
             'ranking': [], 'score_semantics': 'outline_similarity_not_probability', 'physical_dimensions_used': False}
    if current is None:
        return empty
    padded = np.pad(current, 5)
    inverse_observed = np.linalg.inv(observed_transform)
    ranked = []
    bank = dictionary_bank(labels, templates.shape, templates.dtype.str, templates.tobytes(), angles)
    for label, rotations in bank:
        best = None
        for angle, rotated, template_transform in rotations:
            scores = cv2.matchTemplate(padded, rotated, cv2.TM_CCORR_NORMED)
            maxima = cv2.minMaxLoc(scores)
            similarity = float(np.clip(maxima[1], 0., 1.))
            if best is None or similarity > best['similarity']:
                displacement = np.eye(3)
                displacement[:2, 2] = np.asarray(maxima[3], dtype=float) - 5
                observed_from_template = inverse_observed @ displacement @ template_transform
                best = {'label': label, 'yaw_degrees': float(angle), 'similarity': similarity,
                        'outline_to_observed': observed_from_template.tolist(),
                        'transform_space': 'current_outline_raster_not_metric_pose'}
        if best is not None:
            ranked.append(best)
    ranked.sort(key=lambda record: -record['similarity'])
    if len(ranked) != len(labels):
        return empty
    similarity = ranked[0]['similarity']
    margin = similarity - ranked[1]['similarity']
    confirmed = similarity >= MINIMUM_SIMILARITY and margin >= MINIMUM_MARGIN
    return dict(empty, status='confirmed' if confirmed else 'uncertain',
        label=ranked[0]['label'] if confirmed else None,
        yaw_degrees=ranked[0]['yaw_degrees'] if confirmed else None,
        similarity=similarity, margin=margin, ranking=ranked)
