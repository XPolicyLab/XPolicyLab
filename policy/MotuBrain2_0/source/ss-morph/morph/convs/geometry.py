"""Backend-preserving geometry conversions for NumPy and PyTorch arrays."""

from __future__ import annotations

from typing import overload

import numpy as np
import torch

from morph.types.fields import Array
from morph.types.fields import ArrayT
from morph.types.fields import ROT_DIM
from morph.types.fields import RotRepr

ROTATION_UNIT_ATOL = 1e-4


def is_tensor(value: object) -> bool:
    """Return whether ``value`` uses the PyTorch backend."""
    return isinstance(value, torch.Tensor)


def _float64(value: Array) -> Array:
    if is_tensor(value):
        return value.to(torch.float64)
    return np.asarray(value, dtype=np.float64)


def _stack(values: list[Array], axis: int) -> Array:
    if is_tensor(values[0]):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)


def concatenate(values: list[Array] | tuple[Array, ...], axis: int = -1) -> Array:
    """Concatenate arrays without changing their backend."""
    if not values:
        raise ValueError("cannot concatenate an empty sequence")
    validate_same_backend(values)
    if is_tensor(values[0]):
        return torch.cat(values, dim=axis)
    return np.concatenate(values, axis=axis)


def _norm(value: Array, axis: int, keepdims: bool = False) -> Array:
    if is_tensor(value):
        return torch.linalg.vector_norm(value, dim=axis, keepdim=keepdims)
    return np.linalg.norm(value, axis=axis, keepdims=keepdims)


def _clip_min(value: Array, minimum: float) -> Array:
    if is_tensor(value):
        return torch.clamp(value, min=minimum)
    return np.clip(value, minimum, None)


def _clip(value: Array, low: float, high: float) -> Array:
    if is_tensor(value):
        return torch.clamp(value, min=low, max=high)
    return np.clip(value, low, high)


def _where(condition: Array, left: Array, right: Array) -> Array:
    if is_tensor(condition):
        return torch.where(condition, left, right)
    return np.where(condition, left, right)


def _zeros_like_shape(value: Array, shape: tuple[int, ...]) -> Array:
    if is_tensor(value):
        return torch.zeros(shape, dtype=torch.float64, device=value.device)
    return np.zeros(shape, dtype=np.float64)


def _ones_like(value: Array) -> Array:
    if is_tensor(value):
        return torch.ones_like(value)
    return np.ones_like(value)


def as_like(value: object, like: Array, *, dtype: object | None = None) -> Array:
    """Materialize auxiliary values on the backend of ``like``.

    This function is only for constants and metadata. Public conversion
    functions never route the primary input through the other backend.
    """
    if is_tensor(like):
        target_dtype = like.dtype if dtype is None else dtype
        return torch.as_tensor(value, dtype=target_dtype, device=like.device)
    target_dtype = like.dtype if dtype is None else dtype
    return np.asarray(value, dtype=target_dtype)


def zeros(like: Array, shape: tuple[int, ...]) -> Array:
    """Return zeros with ``like`` backend, dtype, and device."""
    if is_tensor(like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def bool_array(value: object, like: Array) -> Array:
    """Return a boolean array on ``like`` backend and device."""
    if is_tensor(like):
        return torch.as_tensor(value, dtype=torch.bool, device=like.device)
    return np.asarray(value, dtype=np.bool_)


def validate_same_backend(values: list[Array] | tuple[Array, ...]) -> None:
    """Require all arrays to use one backend and, for tensors, one device."""
    if not values:
        return
    tensor_backend = is_tensor(values[0])
    for value in values[1:]:
        if is_tensor(value) != tensor_backend:
            raise TypeError("NumPy arrays and PyTorch tensors cannot be mixed")
    if tensor_backend:
        devices = {value.device for value in values}
        if len(devices) != 1:
            raise ValueError(f"all tensors must use one device, got {devices}")


def cast_like(value: Array, like: Array) -> Array:
    """Cast a result to ``like`` dtype while preserving its backend."""
    if is_tensor(value):
        dtype = like.dtype if like.is_floating_point() else torch.float32
        return value.to(dtype=dtype)
    dtype = like.dtype if np.issubdtype(like.dtype, np.floating) else np.float32
    return value.astype(dtype, copy=False)


def _validate_rotation(value: Array, representation: RotRepr) -> None:
    if representation in ("xyzw", "wxyz"):
        norm = _norm(value, axis=-1)
        if is_tensor(norm):
            invalid = (~torch.isfinite(norm)) | (
                torch.abs(norm - 1.0) > ROTATION_UNIT_ATOL
            )
            if torch.any(invalid).item():
                first = torch.nonzero(invalid, as_tuple=False)[0]
                index = tuple(int(item) for item in first)
                number = float(norm[index].item())
                raise ValueError(
                    "quaternion must have unit norm before conversion; "
                    f"first invalid batch index={index}, norm={number}, "
                    f"atol={ROTATION_UNIT_ATOL}"
                )
        else:
            invalid = ~np.isfinite(norm) | (np.abs(norm - 1.0) > ROTATION_UNIT_ATOL)
            if np.any(invalid):
                index = tuple(int(item) for item in np.argwhere(invalid)[0])
                raise ValueError(
                    "quaternion must have unit norm before conversion; "
                    f"first invalid batch index={index}, "
                    f"norm={float(norm[index])}, atol={ROTATION_UNIT_ATOL}"
                )
    elif representation == "rot6d":
        axis0 = value[..., :3]
        axis1 = value[..., 3:]
        norm0 = _norm(axis0, axis=-1)
        norm1 = _norm(axis1, axis=-1)
        dot = (
            (axis0 * axis1).sum(dim=-1)
            if is_tensor(value)
            else np.sum(
                axis0 * axis1,
                axis=-1,
            )
        )
        if is_tensor(value):
            invalid = (
                ~torch.isfinite(norm0)
                | ~torch.isfinite(norm1)
                | ~torch.isfinite(dot)
                | (torch.abs(norm0 - 1.0) > ROTATION_UNIT_ATOL)
                | (torch.abs(norm1 - 1.0) > ROTATION_UNIT_ATOL)
                | (torch.abs(dot) > ROTATION_UNIT_ATOL)
            )
            if torch.any(invalid).item():
                first = torch.nonzero(invalid, as_tuple=False)[0]
                index = tuple(int(item) for item in first)
                raise ValueError(
                    "rot6d basis columns must be unit length and orthogonal; "
                    f"first invalid batch index={index}"
                )
        else:
            invalid = (
                ~np.isfinite(norm0)
                | ~np.isfinite(norm1)
                | ~np.isfinite(dot)
                | (np.abs(norm0 - 1.0) > ROTATION_UNIT_ATOL)
                | (np.abs(norm1 - 1.0) > ROTATION_UNIT_ATOL)
                | (np.abs(dot) > ROTATION_UNIT_ATOL)
            )
            if np.any(invalid):
                index = tuple(int(item) for item in np.argwhere(invalid)[0])
                raise ValueError(
                    "rot6d basis columns must be unit length and orthogonal; "
                    f"first invalid batch index={index}"
                )


def _quat_xyzw_to_mat(quaternion: Array) -> Array:
    quaternion = _float64(quaternion)
    norm = _norm(quaternion, axis=-1, keepdims=True)
    if is_tensor(norm):
        too_small = torch.any(norm < 1e-12).item()
    else:
        too_small = bool(np.any(norm < 1e-12))
    if too_small:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = [quaternion[..., index] / norm[..., 0] for index in range(4)]
    values = [
        x * x - y * y - z * z + w * w,
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        -x * x + y * y - z * z + w * w,
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        -x * x - y * y + z * z + w * w,
    ]
    return _stack(values, axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def _copysign(value: Array, sign: Array) -> Array:
    if is_tensor(value):
        return torch.copysign(value, sign)
    return np.copysign(value, sign)


def _sqrt(value: Array) -> Array:
    if is_tensor(value):
        return torch.sqrt(value)
    return np.sqrt(value)


def _mat_to_quat_xyzw(matrix: Array) -> Array:
    matrix = _float64(matrix)
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    if is_tensor(matrix):
        clamp = lambda value: torch.clamp(value, min=0.0)
    else:
        clamp = lambda value: np.clip(value, 0.0, None)
    x = 0.5 * _copysign(
        _sqrt(clamp(1 + m00 - m11 - m22)),
        matrix[..., 2, 1] - matrix[..., 1, 2],
    )
    y = 0.5 * _copysign(
        _sqrt(clamp(1 - m00 + m11 - m22)),
        matrix[..., 0, 2] - matrix[..., 2, 0],
    )
    z = 0.5 * _copysign(
        _sqrt(clamp(1 - m00 - m11 + m22)),
        matrix[..., 1, 0] - matrix[..., 0, 1],
    )
    w = 0.5 * _sqrt(clamp(1 + m00 + m11 + m22))
    quaternion = _stack([x, y, z, w], axis=-1)
    quaternion = quaternion / _clip_min(
        _norm(quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    sign = _where(
        quaternion[..., 3:4] < 0,
        -_ones_like(quaternion[..., 3:4]),
        _ones_like(quaternion[..., 3:4]),
    )
    return quaternion * sign


def _rpy_to_mat(rpy: Array) -> Array:
    rpy = _float64(rpy)
    roll, pitch, yaw = [rpy[..., index] for index in range(3)]
    if is_tensor(rpy):
        sin, cos = torch.sin, torch.cos
    else:
        sin, cos = np.sin, np.cos
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    values = [
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    ]
    return _stack(values, axis=-1).reshape(rpy.shape[:-1] + (3, 3))


def _identity(like: Array, shape: tuple[int, ...]) -> Array:
    if is_tensor(like):
        identity = torch.eye(3, dtype=torch.float64, device=like.device)
        return identity.expand(shape + (3, 3)).clone()
    return np.broadcast_to(
        np.eye(3, dtype=np.float64),
        shape + (3, 3),
    ).copy()


def _axis_rotation_matrix(axis: str, angle: Array) -> Array:
    if is_tensor(angle):
        cos, sin = torch.cos, torch.sin
    else:
        cos, sin = np.cos, np.sin
    c = cos(angle)
    s = sin(angle)
    result = _identity(angle, angle.shape)
    if axis == "x":
        result[..., 1, 1] = c
        result[..., 1, 2] = -s
        result[..., 2, 1] = s
        result[..., 2, 2] = c
    elif axis == "y":
        result[..., 0, 0] = c
        result[..., 0, 2] = s
        result[..., 2, 0] = -s
        result[..., 2, 2] = c
    elif axis == "z":
        result[..., 0, 0] = c
        result[..., 0, 1] = -s
        result[..., 1, 0] = s
        result[..., 1, 1] = c
    else:
        raise ValueError(f"unknown Euler axis: {axis!r}")
    return result


def euler_to_mat(angles: ArrayT, order: str = "xyz") -> ArrayT:
    """Convert intrinsic Euler angles using the configured axis order."""
    normalized_order = order.lower()
    if not normalized_order or any(axis not in "xyz" for axis in normalized_order):
        raise ValueError(f"invalid Euler order: {order!r}")
    if angles.shape[-1] != len(normalized_order):
        raise ValueError(
            f"Euler angle dim {angles.shape[-1]} does not match order "
            f"{normalized_order!r}"
        )
    value = _float64(angles)
    result = _identity(value, value.shape[:-1])
    for index, axis in enumerate(normalized_order):
        result = result @ _axis_rotation_matrix(axis, value[..., index])
    return result


def _mat_to_rpy(matrix: Array) -> Array:
    matrix = _float64(matrix)
    if is_tensor(matrix):
        atan2, sqrt = torch.atan2, torch.sqrt
    else:
        atan2, sqrt = np.arctan2, np.sqrt
    pitch = atan2(
        -matrix[..., 2, 0],
        sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2),
    )
    yaw = atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    roll = atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    return _stack([roll, pitch, yaw], axis=-1)


def _rvec_to_mat(rvec: Array) -> Array:
    rvec = _float64(rvec)
    if not is_tensor(rvec):
        flat = rvec.reshape(-1, 3)
        result = np.empty((flat.shape[0], 3, 3), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for index, vector in enumerate(flat):
            angle = float(np.linalg.norm(vector))
            if angle < 1e-12:
                result[index] = identity
                continue
            x, y, z = vector / angle
            skew = np.asarray(
                [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
                dtype=np.float64,
            )
            result[index] = (
                identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
            )
        return result.reshape(rvec.shape[:-1] + (3, 3))
    angle = _norm(rvec, axis=-1, keepdims=True)
    half_angle = angle / 2
    if is_tensor(rvec):
        sin, cos = torch.sin, torch.cos
    else:
        sin, cos = np.sin, np.cos
    coefficient = _where(
        angle < 1e-6,
        0.5 - angle**2 / 48,
        sin(half_angle) / _clip_min(angle, 1e-30),
    )
    quaternion = concatenate([coefficient * rvec, cos(half_angle)], axis=-1)
    return _quat_xyzw_to_mat(quaternion)


def _mat_to_rvec(matrix: Array) -> Array:
    quaternion = _mat_to_quat_xyzw(matrix)
    xyz = quaternion[..., :3]
    w = quaternion[..., 3]
    xyz_norm = _norm(xyz, axis=-1)
    if is_tensor(matrix):
        atan2 = torch.atan2
    else:
        atan2 = np.arctan2
    angle = 2 * atan2(xyz_norm, w)
    factor = _where(
        xyz_norm < 1e-8,
        2 / _clip_min(w, 1e-30),
        angle / _clip_min(xyz_norm, 1e-30),
    )
    return factor[..., None] * xyz


def _rot6d_to_mat(rot6d: Array) -> Array:
    rot6d = _float64(rot6d)
    first = rot6d[..., :3]
    second = rot6d[..., 3:]
    basis1 = first / _clip_min(_norm(first, axis=-1, keepdims=True), 1e-8)
    projection = (
        (basis1 * second).sum(dim=-1, keepdim=True)
        if is_tensor(rot6d)
        else np.sum(basis1 * second, axis=-1, keepdims=True)
    )
    basis2 = second - projection * basis1
    basis2 = basis2 / _clip_min(_norm(basis2, axis=-1, keepdims=True), 1e-8)
    basis3 = (
        torch.linalg.cross(basis1, basis2)
        if is_tensor(rot6d)
        else np.cross(basis1, basis2, axis=-1)
    )
    return _stack([basis1, basis2, basis3], axis=-1)


@overload
def rot_to_mat(
    rotation: np.ndarray,
    representation: RotRepr,
    *,
    validate: bool = True,
) -> np.ndarray: ...


@overload
def rot_to_mat(
    rotation: torch.Tensor,
    representation: RotRepr,
    *,
    validate: bool = True,
) -> torch.Tensor: ...


def rot_to_mat(
    rotation: ArrayT,
    representation: RotRepr,
    *,
    validate: bool = True,
) -> ArrayT:
    """Convert a rotation representation without changing array backend."""
    if rotation.shape[-1] != ROT_DIM[representation]:
        raise ValueError(
            f"rotation dim {rotation.shape[-1]} does not match "
            f"{representation!r} dim {ROT_DIM[representation]}"
        )
    if validate:
        _validate_rotation(rotation, representation)
    if representation == "rot6d":
        result = _rot6d_to_mat(rotation)
    elif representation == "xyzw":
        result = _quat_xyzw_to_mat(rotation)
    elif representation == "wxyz":
        result = _quat_xyzw_to_mat(rotation[..., [1, 2, 3, 0]])
    elif representation == "rpy":
        result = _rpy_to_mat(rotation)
    elif representation == "rvec":
        result = _rvec_to_mat(rotation)
    else:
        raise ValueError(f"unknown rotation representation: {representation!r}")
    return result


@overload
def mat_to_rot(matrix: np.ndarray, representation: RotRepr) -> np.ndarray: ...


@overload
def mat_to_rot(matrix: torch.Tensor, representation: RotRepr) -> torch.Tensor: ...


def mat_to_rot(matrix: ArrayT, representation: RotRepr) -> ArrayT:
    """Convert rotation matrices without changing array backend."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end with (3, 3), got {matrix.shape}")
    if representation == "rot6d":
        return concatenate(
            [matrix[..., :, 0], matrix[..., :, 1]],
            axis=-1,
        )
    if representation == "xyzw":
        return _mat_to_quat_xyzw(matrix)
    if representation == "wxyz":
        return _mat_to_quat_xyzw(matrix)[..., [3, 0, 1, 2]]
    if representation == "rpy":
        return _mat_to_rpy(matrix)
    if representation == "rvec":
        return _mat_to_rvec(matrix)
    raise ValueError(f"unknown rotation representation: {representation!r}")


def pose_to_matrix(pose: ArrayT, rotation: RotRepr) -> ArrayT:
    """Convert xyz plus rotation values to homogeneous pose matrices."""
    expected_dim = 3 + ROT_DIM[rotation]
    if pose.shape[-1] != expected_dim:
        raise ValueError(
            f"pose dim {pose.shape[-1]} does not match expected {expected_dim}"
        )
    rotation_matrix = rot_to_mat(pose[..., 3:], rotation)
    upper = concatenate(
        [rotation_matrix, _float64(pose[..., :3])[..., None]],
        axis=-1,
    )
    bottom = _zeros_like_shape(pose, pose.shape[:-1] + (1, 4))
    bottom[..., 0, 3] = 1
    return concatenate([upper, bottom], axis=-2)


def matrix_to_pose(matrix: ArrayT, rotation: RotRepr) -> ArrayT:
    """Convert homogeneous pose matrices to xyz plus rotation values."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"pose matrix must end with (4, 4), got {matrix.shape}")
    return concatenate(
        [matrix[..., :3, 3], mat_to_rot(matrix[..., :3, :3], rotation)],
        axis=-1,
    )


def matrix_from_rotation_translation(
    rotation: ArrayT,
    translation: ArrayT,
) -> ArrayT:
    """Build homogeneous matrices from broadcastable rotation and translation."""
    validate_same_backend((rotation, translation))
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end with (3, 3), got " f"{rotation.shape}")
    if translation.shape[-1:] != (3,):
        raise ValueError(f"translation must end with (3,), got {translation.shape}")
    batch_shape = np.broadcast_shapes(
        rotation.shape[:-2],
        translation.shape[:-1],
    )
    rotation64 = _float64(rotation)
    translation64 = _float64(translation)
    if is_tensor(rotation64):
        rotation64 = torch.broadcast_to(rotation64, batch_shape + (3, 3))
        translation64 = torch.broadcast_to(translation64, batch_shape + (3,))
    else:
        rotation64 = np.broadcast_to(rotation64, batch_shape + (3, 3))
        translation64 = np.broadcast_to(translation64, batch_shape + (3,))
    result = _zeros_like_shape(rotation64, batch_shape + (4, 4))
    result[..., :3, :3] = rotation64
    result[..., :3, 3] = translation64
    result[..., 3, 3] = 1.0
    return result


def broadcast_vector(reference: ArrayT, target: Array) -> ArrayT:
    """Insert time-like axes so a vector broadcasts to ``target``."""
    while reference.ndim < target.ndim:
        if is_tensor(reference):
            reference = reference.unsqueeze(-2)
        else:
            reference = np.expand_dims(reference, axis=-2)
    return reference


def broadcast_matrix(reference: ArrayT, target: Array) -> ArrayT:
    """Insert time-like axes so a matrix broadcasts to ``target``."""
    while reference.ndim < target.ndim:
        if is_tensor(reference):
            reference = reference.unsqueeze(-3)
        else:
            reference = np.expand_dims(reference, axis=-3)
    return reference


def inverse(matrix: ArrayT) -> ArrayT:
    """Invert a matrix batch without changing backend."""
    if is_tensor(matrix):
        return torch.linalg.inv(matrix)
    return np.linalg.inv(matrix)


def inverse_rigid(matrix: ArrayT) -> ArrayT:
    """Invert homogeneous rigid transforms without changing array backend."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"rigid transform must end with (4, 4), got {matrix.shape}")
    rotation = matrix[..., :3, :3]
    inverse_rotation = (
        rotation.transpose(-1, -2)
        if is_tensor(rotation)
        else np.swapaxes(rotation, -1, -2)
    )
    inverse_translation = -(inverse_rotation @ matrix[..., :3, 3, None])[..., 0]
    return matrix_from_rotation_translation(
        inverse_rotation,
        inverse_translation,
    )


def compose_transforms(left: ArrayT, right: ArrayT) -> ArrayT:
    """Compose broadcastable homogeneous transforms on one array backend."""
    validate_same_backend((left, right))
    if left.shape[-2:] != (4, 4) or right.shape[-2:] != (4, 4):
        raise ValueError(
            "homogeneous transforms must end with (4, 4), got "
            f"{left.shape} and {right.shape}"
        )
    return left @ right


def rotation_angle_between(
    left: ArrayT,
    right: ArrayT,
    *,
    degrees: bool = False,
) -> ArrayT:
    """Return the geodesic angle between batches of rotation matrices."""
    validate_same_backend((left, right))
    if left.shape[-2:] != (3, 3) or right.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation matrices must end with (3, 3), got "
            f"{left.shape} and {right.shape}"
        )
    left = _float64(left)
    right = _float64(right)
    left_transpose = (
        left.transpose(-1, -2) if is_tensor(left) else np.swapaxes(left, -1, -2)
    )
    relative = left_transpose @ right
    if is_tensor(relative):
        trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        return torch.rad2deg(angle) if degrees else angle
    trace = np.trace(relative, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.rad2deg(angle) if degrees else angle


def apply_transform(pose: ArrayT, transform: object) -> ArrayT:
    """Right-multiply a homogeneous transform on the input backend."""
    return pose @ as_like(transform, pose, dtype=pose.dtype)


def ensure_2d(value: ArrayT) -> ArrayT:
    """Return ``value`` as ``[T, D]`` while retaining its backend."""
    if value.ndim == 1:
        return value.reshape(-1, 1)
    if value.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {value.shape}")
    return value


def quat_xyzw_to_mat(quaternion: ArrayT) -> ArrayT:
    """Convert unit xyzw quaternions to rotation matrices."""
    return rot_to_mat(quaternion, "xyzw")


def mat_to_quat_xyzw(matrix: ArrayT) -> ArrayT:
    """Convert rotation matrices to hemisphere-canonical xyzw quaternions."""
    return mat_to_rot(matrix, "xyzw")


def rot6d_to_rotmat(rotation: ArrayT) -> ArrayT:
    """Convert valid 6D rotations to matrices."""
    return rot_to_mat(rotation, "rot6d")


def rotmat_to_rot6d(matrix: ArrayT) -> ArrayT:
    """Convert rotation matrices to first-two-column 6D rotations."""
    return mat_to_rot(matrix, "rot6d")


def true_runs(mask: Array) -> list[tuple[int, int]]:
    """Return half-open runs where a one-dimensional mask is true."""
    if mask.ndim != 1:
        raise ValueError(f"true_runs expects a 1D mask, got {mask.shape}")
    if isinstance(mask, torch.Tensor):
        values = mask.detach().cpu().tolist()
    else:
        values = mask.tolist()
    runs = []
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        start = index
        while index < len(values) and values[index]:
            index += 1
        runs.append((start, index))
    return runs


def axis_stats(values: Array) -> dict[str, float]:
    """Summarize one numeric axis with robust quantiles."""
    if isinstance(values, torch.Tensor):
        array = values.detach().to(torch.float64).cpu().numpy()
    else:
        array = values.astype(np.float64, copy=False)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p01": float(np.quantile(array, 0.01)),
        "p99": float(np.quantile(array, 0.99)),
    }
