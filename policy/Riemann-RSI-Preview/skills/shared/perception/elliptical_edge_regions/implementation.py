import cv2
import numpy as np


def run(image):
    image = np.asarray(image)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError('Read ellipse boundaries only in current native public RGB')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(gray, 20, 45), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found = []
    for contour in sorted(contours, key=len, reverse=True)[:512]:
        if not 30 <= len(contour) <= 6000:
            continue
        center, axes, angle = cv2.fitEllipse(contour)
        center, axes = np.asarray(center), np.asarray(axes)
        if (not np.isfinite(center).all() or not np.isfinite(axes).all()
                or min(axes) < 10 or max(axes) > 96 or max(axes) / min(axes) > 3):
            continue
        pixels = contour.reshape(-1, 2)
        column, row, width, height = cv2.boundingRect(contour)
        if column <= 0 or row <= 0 or column + width >= 640 or row + height >= 480:
            continue
        rotation = cv2.getRotationMatrix2D((0., 0.), angle, 1.)[:, :2]
        normalized = (pixels - center) @ rotation.T / (axes / 2)
        error = float(np.quantile(np.abs(np.linalg.norm(normalized, axis=1) - 1), .85))
        bins = np.floor((np.arctan2(normalized[:, 1], normalized[:, 0]) + np.pi) * 12 / np.pi).astype(int) % 24
        if error > .12 or len(set(bins)) < 18:
            continue
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(contour), 1)
        area = int(np.count_nonzero(mask))
        if not 120 <= area <= image.shape[0] * image.shape[1] * .03:
            continue
        found.append({'area': area, 'bbox': (column, row, width, height), 'centroid': center,
                      'pixels': pixels.astype(float), 'mask': mask.astype(bool), 'boundary_error': error})
    selected = []
    for region in sorted(found, key=lambda value: value['boundary_error']):
        diameter = min(region['bbox'][2:])
        if any(np.linalg.norm(region['centroid'] - existing['centroid']) < .18 * diameter
               and .8 <= region['area'] / existing['area'] <= 1.25 for existing in selected):
            continue
        selected.append(region)
    return selected[:64]
