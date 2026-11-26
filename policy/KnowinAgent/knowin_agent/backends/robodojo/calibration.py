"""Optical camera calibration read from the native RoboDojo camera manager.

Uses the Isaac Sim camera itself, not the camera housing Xform the native
``get_camera_extrinsics`` reports: ``get_view_matrix_ros`` is world-to-optical
(+X right, +Y down, +Z forward), so its inverse is the camera-to-world pose of
the optical frame. Anything that does not validate raises; nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpticalCalibration:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    size: tuple[int, int]


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _call_cpu(fn: Callable) -> Any:
    try:
        return fn(device="cpu")
    except TypeError:
        return fn()


def validate_intrinsics(k: np.ndarray, size: tuple[int, int], name: str) -> np.ndarray:
    width, height = size
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise CalibrationError(f"{name}: intrinsics must be a finite 3x3 matrix, got shape {k.shape}")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationError(f"{name}: intrinsics are not a pinhole matrix: {k.tolist()}")
    if not (0 < k[0, 2] < width and 0 < k[1, 2] < height):
        raise CalibrationError(f"{name}: principal point {k[:2, 2].tolist()} outside a {width}x{height} image")
    return k


def validate_rigid(t: np.ndarray, name: str) -> np.ndarray:
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise CalibrationError(f"{name}: transform must be a finite 4x4 matrix, got shape {t.shape}")
    r = t[:3, :3]
    if not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationError(f"{name}: transform bottom row is {t[3].tolist()}")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-3):
        raise CalibrationError(f"{name}: rotation is not orthonormal and right-handed")
    return t


def camera_calibration(
    camera: Any, rgb_shape: Sequence[int], name: str = "camera", env_origin: Optional[Sequence[float]] = None
) -> OpticalCalibration:
    """Calibration of one Isaac Sim camera, checked against the RGB image it produced."""
    width, height = (int(v) for v in camera.get_resolution())
    if tuple(rgb_shape[:2]) != (height, width):
        raise CalibrationError(f"{name}: camera resolution {width}x{height} disagrees with rgb shape {tuple(rgb_shape)}")
    k = validate_intrinsics(to_numpy(_call_cpu(camera.get_intrinsics_matrix)), (width, height), name)
    view = validate_rigid(to_numpy(_call_cpu(camera.get_view_matrix_ros)), name)
    extrinsics = np.linalg.inv(view)
    if env_origin is not None:
        origin = to_numpy(env_origin).reshape(-1)[:3]
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise CalibrationError(f"{name}: invalid env origin {origin}")
        extrinsics[:3, 3] -= origin
    return OpticalCalibration(k, validate_rigid(extrinsics, name), (width, height))


def calibrate_cameras(
    camera_manager: Any,
    rgb_by_name: Mapping[str, np.ndarray],
    env_idx: int = 0,
    env_origin: Optional[Sequence[float]] = None,
) -> dict[str, OpticalCalibration]:
    """Calibration for every native camera name in ``rgb_by_name``; all must be found."""
    names = list(camera_manager.camera_names[env_idx])
    missing = [name for name in rgb_by_name if name not in names]
    if missing:
        raise CalibrationError(f"cameras {missing} not in the native camera manager ({names})")
    return {
        name: camera_calibration(
            camera_manager.cameras[env_idx][names.index(name)], np.shape(rgb), name, env_origin
        )
        for name, rgb in rgb_by_name.items()
    }
