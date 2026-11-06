"""Current RGB appearance proposals for a changed static background."""
from __future__ import annotations
import cv2
import numpy as np


def row_median_candidates(rgb, roi, patch_size=64, threshold=26,
                          min_area=70, max_area=7000):
    """Propose compact departures from the current row's dominant RGB color.

    Proposals are deliberately unlabelled. The caller must use its original
    classifier and confidence threshold; background edges/shadows also propose
    components. No reset layout, object pose, depth or stored empty image is used.
    """
    image = np.asarray(rgb, dtype=np.uint8)[..., :3]
    background = np.median(image, axis=1, keepdims=True)
    mask = (np.max(np.abs(image.astype(np.float32)-background), axis=2)>threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask[~np.asarray(roi, dtype=bool)] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = image.shape[:2]
    half = patch_size//2
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if not min_area <= area <= max_area:
            continue
        if x <= 2 or y <= 2 or x+w >= width-2 or y+h >= height-2:
            continue
        u, v = centroids[index]
        left, top = int(round(u))-half, int(round(v))-half
        if min(left, top)<0 or left+patch_size>width or top+patch_size>height:
            continue
        rows, cols = np.where(labels==index)
        top_row = int(rows.min())
        candidates.append({
            'centroid': np.asarray([u, v], dtype=np.float64),
            'area': area, 'bbox': (x, y, w, h),
            'pixels': np.stack([cols, rows], axis=1).astype(np.float64),
            'top_uv': np.asarray([float(cols[rows<=top_row+2].mean()), float(top_row)]),
            'patch': image[top:top+patch_size, left:left+patch_size].copy(),
            'appearance_fallback': True,
        })
    return candidates


def prefer_primary_detections(detections, radius=20):
    """Keep the mature silhouette when both paths classified the same object."""
    primary = [d for d in detections if not d.get('appearance_fallback')]
    return [d for d in detections if not d.get('appearance_fallback') or not any(
        p['category']==d['category'] and
        np.linalg.norm(p['centroid']-d['centroid'])<radius for p in primary)]
