import numpy as np


def run(current, previous, region):
    current, region = np.asarray(current), np.asarray(region)
    if current.shape != (480, 640, 3) or current.dtype != np.uint8 or region.shape != current.shape[:2] or region.dtype != bool:
        raise ValueError('Compare native current public RGB only inside an explicit current-image boolean region')
    if not np.any(region):
        raise ValueError('Public image stability requires a nonempty independently projected region')
    if previous is None:
        return None
    previous = np.asarray(previous)
    if previous.shape != current.shape or previous.dtype != current.dtype:
        raise ValueError('Previous public RGB must use the same native camera and format')
    changed = np.any(np.abs(current.astype(np.int16) - previous.astype(np.int16)) > 15, axis=2)
    return int(np.count_nonzero(changed & region))
