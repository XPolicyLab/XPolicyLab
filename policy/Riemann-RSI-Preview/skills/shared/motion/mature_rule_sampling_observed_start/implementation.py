import math
import numpy as np

def speed_up_controls(positions, speedup):
    dense = np.asarray(positions, dtype=float)
    if dense.ndim != 2 or dense.shape[1] != 6 or (not 1 <= len(dense) <= 20000) or (not np.isfinite(dense).all()) or isinstance(speedup, bool) or (not math.isfinite(float(speedup))) or (not 0.1 <= float(speedup) <= 4):
        raise ValueError('Original rule sampling requires a finite six-joint path and bounded speedup')
    if speedup <= 1 or len(dense) <= 2:
        return dense.copy()
    count = max(2, int(math.ceil(len(dense) / speedup)))
    indices = np.unique(np.rint(np.linspace(0, len(dense) - 1, count)).astype(np.int64))
    return dense[indices].copy()
