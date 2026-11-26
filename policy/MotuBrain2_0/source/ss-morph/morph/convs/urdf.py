"""NumPy-only URDF forward kinematics conversion."""

from __future__ import annotations

import importlib
from importlib import metadata
import sys
from types import ModuleType
from typing import Any

import numpy as np
import torch

from morph.convs import geometry
from morph.types.fields import UrdfEEF


def qpos_to_eef(value: np.ndarray, field: UrdfEEF) -> np.ndarray:
    """Compute EEF poses with Pinocchio.

    Pinocchio is CPU-only and non-differentiable in this integration, so a
    tensor input is rejected instead of silently round-tripping through NumPy.
    """
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "UrdfEEF only supports NumPy arrays; convert explicitly at the "
            "application boundary when gradients and device placement do not "
            "matter"
        )
    pin = import_pinocchio()
    qpos = np.asarray(value, dtype=np.float64)
    original_dtype = np.asarray(value).dtype
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2:
        raise ValueError(f"UrdfEEF qpos must have shape [T, D], got {qpos.shape}")
    model = pin.buildModelFromUrdf(field.urdf_path)
    data = model.createData()
    frame_id = model.getFrameId(field.frame)
    if frame_id >= len(model.frames):
        raise ValueError(f"URDF frame {field.frame!r} not found in {field.urdf_path}")
    frame_transform = _pin_se3_from_matrix(pin, field.eef_tf)
    xyz = np.empty((qpos.shape[0], 3), dtype=np.float64)
    rotation = np.empty((qpos.shape[0], 3, 3), dtype=np.float64)
    for index, row in enumerate(qpos):
        q = _pin_qpos(pin, model, row, field.joint_names)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[frame_id] * frame_transform
        xyz[index] = pose.translation
        rotation[index] = pose.rotation
    result = geometry.concatenate(
        [xyz, geometry.mat_to_rot(rotation, field.input_rot)],
        axis=-1,
    )
    dtype = original_dtype if np.issubdtype(original_dtype, np.floating) else np.float32
    return result.astype(dtype, copy=False)


def _pin_se3_from_matrix(pin: Any, matrix: object | None) -> Any:
    if matrix is None:
        return pin.SE3.Identity()
    array = np.asarray(matrix, dtype=np.float64)
    return pin.SE3(array[:3, :3], array[:3, 3])


def _pin_qpos(
    pin: Any,
    model: Any,
    row: np.ndarray,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    if row.shape[0] == model.nq and not joint_names:
        return row
    if not joint_names:
        raise ValueError(
            f"UrdfEEF qpos dim {row.shape[0]} does not match model.nq "
            f"{model.nq}; configure joint_names"
        )
    if len(joint_names) != row.shape[0]:
        raise ValueError(
            f"UrdfEEF joint_names length {len(joint_names)} != qpos dim "
            f"{row.shape[0]}"
        )
    q = np.asarray(pin.neutral(model), dtype=np.float64).copy()
    for item, joint_name in zip(row, joint_names, strict=True):
        if not model.existJointName(joint_name):
            raise ValueError(f"URDF joint {joint_name!r} not found")
        joint = model.joints[model.getJointId(joint_name)]
        if joint.nq == 1:
            q[joint.idx_q] = item
        elif joint.nq == 2 and joint.nv == 1:
            q[joint.idx_q] = np.cos(item)
            q[joint.idx_q + 1] = np.sin(item)
        else:
            raise ValueError(
                f"URDF joint {joint_name!r} has nq={joint.nq}, "
                f"nv={joint.nv}; only scalar and continuous joints are supported"
            )
    return q


def import_pinocchio() -> ModuleType:
    """Import robotics Pinocchio while avoiding its namesake plugin."""
    pin = _try_import_pinocchio()
    if _is_robotics_pinocchio(pin):
        return pin
    site_packages = _pin_cmeel_site_packages()
    if site_packages is not None:
        sys.path = [path for path in sys.path if path != site_packages]
        sys.path.insert(0, site_packages)
    loaded = sys.modules.get("pinocchio")
    if loaded is pin and not _is_robotics_pinocchio(loaded):
        sys.modules.pop("pinocchio", None)
    importlib.invalidate_caches()
    pin = _try_import_pinocchio()
    if _is_robotics_pinocchio(pin):
        return pin
    origin = getattr(pin, "__file__", None) if pin is not None else None
    raise ImportError(
        "Pinocchio robotics bindings are required; "
        f"imported pinocchio from {origin!r}"
    )


def _try_import_pinocchio() -> ModuleType | None:
    try:
        return importlib.import_module("pinocchio")
    except ImportError:
        return None


def _is_robotics_pinocchio(module: ModuleType | None) -> bool:
    return bool(
        module is not None
        and hasattr(module, "buildModelFromUrdf")
        and hasattr(module, "SE3")
    )


def _pin_cmeel_site_packages() -> str | None:
    try:
        distribution = metadata.distribution("pin")
    except metadata.PackageNotFoundError:
        return None
    for file in distribution.files or ():
        if file.parts[-2:] == ("pinocchio", "__init__.py"):
            return str(distribution.locate_file(file.parent.parent))
    return None
