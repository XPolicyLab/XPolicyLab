"""Camera-only metadata permitted across the review-stmem observation boundary."""
from collections.abc import Mapping

import numpy as np


def project_calibration(view):
    """Validate and copy only same-frame camera geometry; never forward extra fields."""
    meta = view.get("stmem_calibration")
    if not isinstance(meta, Mapping) or meta.get("same_observation") is not True:
        raise ValueError("review-stmem requires same-observation camera calibration")
    if (meta.get("frame_id") != "env_relative_world" or meta.get("units") != "m"
            or meta.get("convention") != "opencv"):
        raise ValueError("invalid review-stmem camera frame, units or convention")
    k = np.asarray(meta.get("K"), dtype=float)
    t = np.asarray(meta.get("world_from_camera"), dtype=float)
    shape = meta.get("rgb_shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("invalid calibration RGB shape")
    color = view.get("color")
    if isinstance(color, np.ndarray) and tuple(shape) != color.shape[:2]:
        raise ValueError("camera calibration does not match the current RGB shape")
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("invalid camera intrinsics")
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError("invalid camera rigid transform")
    return {"K": k.tolist(), "world_from_camera": t.tolist(), "rgb_shape": list(shape),
            "frame_id": "env_relative_world", "units": "m", "convention": "opencv",
            "same_observation": True}
