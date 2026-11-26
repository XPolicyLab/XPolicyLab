"""Timestamp interpolation preserving NumPy or PyTorch backends."""

from __future__ import annotations

import numpy as np
import torch

from morph.convs import geometry
from morph.types.fields import ArrayT
from morph.types.fields import EEF
from morph.types.fields import Qpos
from morph.types.fields import UrdfEEF


def interpolate_action(
    value: ArrayT,
    source_ts: ArrayT,
    target_ts: ArrayT,
    field: Qpos | EEF,
    *,
    periodic: bool = False,
) -> ArrayT:
    """Interpolate a raw action source without changing array backend."""
    geometry.validate_same_backend((value, source_ts, target_ts))
    if isinstance(field, (Qpos, UrdfEEF)):
        return _interpolate_qpos(value, source_ts, target_ts, periodic)
    if isinstance(field, EEF):
        return _interpolate_eef(value, source_ts, target_ts, field)
    raise ValueError("unsupported interpolation field: " f"{type(field).__name__}")


def _interpolate_qpos(
    value: ArrayT,
    source_ts: ArrayT,
    target_ts: ArrayT,
    periodic: bool,
) -> ArrayT:
    array = geometry.ensure_2d(value)
    if periodic:
        array = _unwrap(array)
    output = _linear(array, source_ts, target_ts)
    if periodic:
        output = (output + np.pi) % (2.0 * np.pi) - np.pi
    return geometry.cast_like(output, value)


def _interpolate_eef(
    value: ArrayT,
    source_ts: ArrayT,
    target_ts: ArrayT,
    field: EEF,
) -> ArrayT:
    array = geometry.ensure_2d(value)
    xyz = _linear(array[..., :3], source_ts, target_ts)
    quaternion = geometry.mat_to_rot(
        geometry.rot_to_mat(array[..., 3:], field.input_rot, validate=True),
        "xyzw",
    )
    interpolated = _slerp(source_ts, quaternion, target_ts)
    rotation = geometry.mat_to_rot(
        geometry.rot_to_mat(interpolated, "xyzw"),
        field.input_rot,
    )
    return geometry.cast_like(
        geometry.concatenate([xyz, rotation], axis=-1),
        value,
    )


def _linear(value: ArrayT, source_ts: ArrayT, target_ts: ArrayT) -> ArrayT:
    if source_ts.ndim != 1 or target_ts.ndim != 1:
        raise ValueError("source_ts and target_ts must be one-dimensional")
    if value.shape[0] != source_ts.shape[0]:
        raise ValueError(
            f"value length {value.shape[0]} != timestamp length "
            f"{source_ts.shape[0]}"
        )
    right = _searchsorted(source_ts, target_ts)
    right = _clamp_indices(right, 1, source_ts.shape[0] - 1)
    left = right - 1
    t0 = source_ts[left]
    t1 = source_ts[right]
    if isinstance(value, torch.Tensor):
        denominator = torch.clamp(t1 - t0, min=1e-12)
        weight = torch.clamp((target_ts - t0) / denominator, 0.0, 1.0)
    else:
        denominator = np.clip(t1 - t0, 1e-12, None)
        weight = np.clip((target_ts - t0) / denominator, 0.0, 1.0)
    return value[left] + weight[..., None] * (value[right] - value[left])


def _slerp(
    source_ts: ArrayT,
    quaternion: ArrayT,
    target_ts: ArrayT,
) -> ArrayT:
    right = _clamp_indices(
        _searchsorted(source_ts, target_ts),
        1,
        source_ts.shape[0] - 1,
    )
    left = right - 1
    t0 = source_ts[left]
    t1 = source_ts[right]
    q0 = _normalize_quaternion(quaternion[left])
    q1 = _normalize_quaternion(quaternion[right])
    if isinstance(quaternion, torch.Tensor):
        weight = torch.clamp(
            (target_ts - t0) / torch.clamp(t1 - t0, min=1e-12),
            0.0,
            1.0,
        )
        dot = (q0 * q1).sum(dim=-1)
        q1 = torch.where(dot[..., None] < 0, -q1, q1)
        dot = torch.clamp(torch.abs(dot), 0.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        near = sin_theta < 1e-6
        safe = torch.where(near, torch.ones_like(sin_theta), sin_theta)
        w0 = torch.where(
            near,
            1.0 - weight,
            torch.sin((1.0 - weight) * theta) / safe,
        )
        w1 = torch.where(
            near,
            weight,
            torch.sin(weight * theta) / safe,
        )
    else:
        weight = np.clip(
            (target_ts - t0) / np.clip(t1 - t0, 1e-12, None),
            0.0,
            1.0,
        )
        dot = np.sum(q0 * q1, axis=-1)
        q1 = np.where(dot[..., None] < 0, -q1, q1)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        near = sin_theta < 1e-6
        safe = np.where(near, 1.0, sin_theta)
        w0 = np.where(
            near,
            1.0 - weight,
            np.sin((1.0 - weight) * theta) / safe,
        )
        w1 = np.where(
            near,
            weight,
            np.sin(weight * theta) / safe,
        )
    return _normalize_quaternion(w0[..., None] * q0 + w1[..., None] * q1)


def _normalize_quaternion(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value / torch.clamp(norm, min=1e-12)
    return value / np.clip(
        np.linalg.norm(value, axis=-1, keepdims=True),
        1e-12,
        None,
    )


def _searchsorted(source: ArrayT, target: ArrayT) -> ArrayT:
    if isinstance(source, torch.Tensor):
        return torch.searchsorted(source, target, right=True)
    return np.searchsorted(source, target, side="right")


def _clamp_indices(value: ArrayT, low: int, high: int) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=low, max=high)
    return np.clip(value, low, high)


def _unwrap(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        difference = value[1:] - value[:-1]
        wrapped = (difference + np.pi) % (2 * np.pi) - np.pi
        correction = wrapped - difference
        correction = torch.where(
            torch.abs(difference) < np.pi,
            torch.zeros_like(correction),
            correction,
        )
        return torch.cat(
            [value[:1], value[1:] + torch.cumsum(correction, dim=0)],
            dim=0,
        )
    return np.unwrap(value, axis=0)
