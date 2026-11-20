"""Match an observed RGB instance near its earlier visible footprint.

This tool receives arrays only. It estimates appearance persistence, not grasp,
contact, object pose ground truth or task completion. Ambiguity is UNKNOWN.
"""
import cv2
import numpy as np


def _image(value, shape=None):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('An explicit uint8 three-channel public image is required')
    if shape is not None and value.shape != shape:
        raise ValueError('Public image dimensions changed')
    return np.ascontiguousarray(value)


def _search(reference, image):
    image = _image(image, reference['image_shape'])
    x0, y0, x1, y1 = reference['bbox']
    margin = reference['margin']
    left, top = max(0, x0-margin), max(0, y0-margin)
    right, bottom = min(image.shape[1], x1+margin), min(image.shape[0], y1+margin)
    losses = cv2.matchTemplate(image[top:bottom, left:right], reference['template'],
                              cv2.TM_SQDIFF_NORMED, mask=reference['mask'])
    finite = np.isfinite(losses)
    if not finite.any():
        return None
    losses = np.where(finite, losses, np.inf)
    row, column = np.unravel_index(np.argmin(losses), losses.shape)
    return float(np.clip(1-losses[row, column], 0, 1)), (left+int(column), top+int(row))


def capture(initial, background, mask, *, margin=14):
    initial = _image(initial)
    background = _image(background, initial.shape)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != initial.shape[:2]:
        raise ValueError('An explicit Boolean observed instance mask is required')
    if type(margin) is not int or not 0 <= margin <= 64:
        raise ValueError('Local search margin must be a bounded pixel count')
    yy, xx = np.where(mask)
    if len(xx) < 20:
        return None
    x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
    reference = {'image_shape': initial.shape, 'bbox': (x0,y0,x1,y1), 'margin': margin,
                 'template': initial[y0:y1,x0:x1].copy(),
                 'mask': np.ascontiguousarray(mask[y0:y1,x0:x1].astype(np.uint8)*255)}
    empty = _search(reference, background)
    if empty is None or empty[0] >= .80:
        return None  # A patch resembling the empty table cannot identify an object.
    reference['empty_similarity'] = empty[0]
    return reference


def match(reference, current):
    if reference is None:
        return {'state':'unknown', 'reason':'source_template_not_distinct_from_background'}
    found = _search(reference, current)
    if found is None:
        return {'state':'unknown', 'reason':'nonfinite_template_match'}
    similarity, origin = found
    empty = reference['empty_similarity']
    excess = (similarity-empty)/(1-empty)
    report = {'state':'unknown', 'reason':'source_appearance_not_reconfirmed',
              'similarity':similarity, 'empty_similarity':empty,
              'foreground_similarity':float(excess)}
    # Both absolute agreement and separation from an empty-table control are
    # required. These are RGB confidence bounds, never physical width gates.
    if similarity < .90 or excess < .80:
        return report
    x, y = origin
    h, w = reference['mask'].shape
    return {**report, 'state':'observed', 'reason':'distinct_source_appearance_reconfirmed',
            'bbox':[x,y,x+w,y+h],
            'translation':[x-reference['bbox'][0], y-reference['bbox'][1]]}


def matched_mask(reference, observation):
    if observation.get('state') != 'observed':
        raise ValueError('An observed template match is required')
    x0,y0,x1,y1 = observation['bbox']
    mask = np.zeros(reference['image_shape'][:2], dtype=bool)
    mask[y0:y1,x0:x1] = reference['mask'] > 0
    return mask
