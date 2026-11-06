import cv2
import numpy as np

def blobs(mask, min_area=20):
    (n, labels, stats, cents) = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        (vs, us) = np.nonzero(labels == i)
        out.append({'area': area, 'uv': np.array(cents[i]), 'bbox': tuple((int(x) for x in stats[i, :4])), 'pix': np.stack([us, vs], axis=1)})
    out.sort(key=lambda b: -b['area'])
    return out
