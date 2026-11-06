import cv2
import numpy as np


def run(mask, minimum_area=20, maximum_regions=64):
    value = np.asarray(mask)
    if (value.ndim != 2 or not 1 <= value.size <= 307200 or value.dtype.kind not in 'buif'
            or not np.isfinite(value).all() or type(minimum_area) is not int or minimum_area < 1
            or type(maximum_regions) is not int or not 1 <= maximum_regions <= 128):
        raise ValueError('Read only bounded current-image connected regions')
    binary = (value > 0).astype(np.uint8)
    count, labels, statistics, centers = cv2.connectedComponentsWithStats(binary, 8)
    identities = sorted((index for index in range(1, count) if statistics[index, cv2.CC_STAT_AREA] >= minimum_area),
                        key=lambda index: (-int(statistics[index, cv2.CC_STAT_AREA]), index))[:maximum_regions]
    found = []
    for identity in identities:
        column, row, width, height, area = (int(value) for value in statistics[identity])
        region = labels == identity
        rows, columns = np.where(region)
        found.append({'mask': region, 'pixels': np.column_stack([columns, rows]),
                      'bbox': (column, row, width, height), 'area': area, 'centroid': centers[identity]})
    return found
