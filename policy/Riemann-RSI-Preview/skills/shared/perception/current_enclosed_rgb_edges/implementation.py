import cv2
import numpy as np

from skills.shared.perception.current_vessel_regions.implementation import boundary


def run(image, region):
    image = np.asarray(image)
    outline = np.asarray(region.get('outline'), dtype=float)
    if (image.shape != (480, 640, 3) or image.dtype != np.uint8 or outline.shape != (96, 2)
            or not np.isfinite(outline).all() or region.get('clipped') is not False
            or np.any(outline <= 1) or np.any(outline[:, 0] >= 638) or np.any(outline[:, 1] >= 478)):
        return []
    contained = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(contained, [np.rint(outline).astype(np.int32)], 1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    samples = gray[contained > 0]
    if len(samples) < 80 or np.ptp(samples) < 18:
        return []
    threshold = cv2.threshold(samples.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    dark = ((gray <= threshold) & (contained > 0)).astype(np.uint8)
    edge = (cv2.Canny(gray, 40, 110) > 0).astype(np.uint8) * contained
    outputs = []
    parent_area = float(np.count_nonzero(contained))
    for mask in (dark, edge):
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
            area = float(cv2.contourArea(contour))
            if len(contour) < 12 or not 12 <= area <= .85 * parent_area:
                continue
            coordinates = contour.reshape(-1, 2)
            bounds = cv2.boundingRect(coordinates)
            if (min(bounds[2:]) < 3 or np.any(coordinates <= 1)
                    or np.any(coordinates[:, 0] >= 638) or np.any(coordinates[:, 1] >= 478)):
                continue
            inner = np.zeros_like(contained)
            cv2.drawContours(inner, [contour], -1, 1, cv2.FILLED)
            if np.count_nonzero(inner & (1 - contained)) > .01 * area:
                continue
            center = np.mean(coordinates, axis=0)
            if any(np.linalg.norm(center - np.asarray(item['centroid'])) < 3
                   and abs(area - item['area_pixels']) < .15 * area for item in outputs):
                continue
            outputs.append({'outline': boundary(coordinates), 'centroid': center.tolist(), 'bbox': list(bounds),
                'area_pixels': area, 'source': 'current_public_enclosed_rgb_edges', 'clipped': False})
    return sorted(outputs, key=lambda value: value['area_pixels'], reverse=True)[:8]
