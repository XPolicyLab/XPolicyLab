import cv2
import numpy as np


def boundary(contour, count=96):
    vertices = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(vertices) < 3:
        return None
    closed = np.vstack((vertices, vertices[0]))
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    if distances[-1] <= 1.:
        return None
    locations = np.linspace(0., distances[-1], count, endpoint=False)
    return np.column_stack([np.interp(locations, distances, closed[:, axis]) for axis in range(2)])


def run(image, appearance):
    image = np.asarray(image)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8 or appearance not in ('yellow_cup', 'light_vase', 'dark_vase'):
        raise ValueError('Vessel hypotheses use only declared current native public RGB')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if appearance == 'yellow_cup':
        selected = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 80) & (hsv[..., 2] > 110)
    elif appearance == 'light_vase':
        selected = (hsv[..., 1] < 70) & (hsv[..., 2] > 155)
    else:
        selected = (((hsv[..., 2] < 95) & (hsv[..., 1] < 180)) |
                    ((hsv[..., 0] >= 60) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 60) & (hsv[..., 2] < 210)))
    mask = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    found = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 80 <= area <= 480 * 640 * .18:
            continue
        bbox = cv2.boundingRect(contour)
        if min(bbox[2:]) < 6 or max(bbox[2:]) / min(bbox[2:]) > 5:
            continue
        moments = cv2.moments(contour)
        outline = boundary(cv2.convexHull(contour))
        if not moments['m00'] or outline is None:
            continue
        selected_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(selected_mask, [contour], -1, 1, cv2.FILLED)
        found.append({'outline': outline, 'bbox': bbox, 'area': area,
            'centroid': [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']],
            'appearance_rgb': np.median(image[selected_mask.astype(bool)], axis=0).tolist(),
            'source': 'whole_current_rgb_component', 'appearance': appearance})
    return sorted(found, key=lambda item: item['area'], reverse=True)[:8]
