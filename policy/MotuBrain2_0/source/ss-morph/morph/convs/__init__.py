"""Stateless action conversions for NumPy arrays and PyTorch tensors."""

from __future__ import annotations

import numpy as np
import torch

from morph.convs import geometry
from morph.convs import image
from morph.convs import interpolation
from morph.convs import landmark
from morph.convs import mask
from morph.convs import normalization
from morph.convs import urdf
from morph.types.fields import ActionField
from morph.types.fields import Array
from morph.types.fields import ArrayT
from morph.types.fields import BinaryQpos
from morph.types.fields import Chassis
from morph.types.fields import EEF
from morph.types.fields import Padding
from morph.types.fields import Qpos


def convert_field(
    value: ArrayT,
    source: ActionField,
    target: ActionField,
) -> ArrayT:
    """Convert between two semantic field representations."""
    if isinstance(source, Qpos) and isinstance(target, Qpos):
        return value
    if isinstance(source, EEF) and isinstance(target, EEF):
        pose = geometry.pose_to_matrix(value, source.input_rot)
        if target.eef_tf is not None:
            pose = geometry.apply_transform(pose, target.eef_tf)
        result = geometry.matrix_to_pose(pose, target.output_rot)
        return geometry.cast_like(result, value)
    raise TypeError(
        "cannot convert action field: "
        f"{type(source).__name__} -> {type(target).__name__}"
    )


def _validate_last_dim(value: Array, expected: int, label: str) -> None:
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(
            f"{label} must end with dimension {expected}, got {value.shape}"
        )


def _reference(
    field: ActionField,
    action: Array,
    ref_state: Array | None,
) -> Array | None:
    if field.model_repr == "absolute":
        return None
    if ref_state is None:
        raise ValueError(f"relative field {field.name!r} requires ref_state")
    _validate_last_dim(
        ref_state,
        field.input_dim,
        f"ref_state {field.name!r}",
    )
    geometry.validate_same_backend((action, ref_state))
    return geometry.broadcast_vector(ref_state, action)


def _numeric(
    value: object,
    *,
    dim: int,
    like: Array,
    label: str,
) -> Array | float:
    if isinstance(value, tuple):
        if len(value) != dim:
            raise ValueError(f"{label} length {len(value)} does not match {dim}")
        return geometry.as_like(value, like)
    return value


def encode_field(
    field: ActionField,
    action: Array,
    *,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> Array:
    """Convert one semantic field from input to model representation."""
    _validate_last_dim(action, field.input_dim, f"field {field.name!r}")
    if isinstance(field, Padding):
        return geometry.zeros(action, action.shape)
    if isinstance(field, BinaryQpos):
        use_max = action > field.threshold
        if field.invert:
            use_max = ~use_max
        positive = geometry.as_like(1.0, action)
        negative = geometry.as_like(-1.0, action)
        if isinstance(action, torch.Tensor):
            return torch.where(use_max, positive, negative)
        return np.where(use_max, positive, negative)
    if isinstance(field, Qpos):
        output = _clip_field(action, field.clip)
        reference = _reference(field, output, ref_state)
        if reference is not None:
            output = output - reference
        trans = _numeric(
            field.trans,
            dim=field.dim,
            like=output,
            label="trans",
        )
        scale = _numeric(
            field.scale,
            dim=field.dim,
            like=output,
            label="scale",
        )
        return (output - trans) / scale
    if isinstance(field, EEF):
        return _encode_eef(field, action, ref_state, ref_frame)
    if isinstance(field, Chassis):
        reference = _reference(field, action, ref_state)
        if reference is None:
            raise RuntimeError("relative chassis reference validation failed")
        delta_x = action[..., 0] - reference[..., 0]
        delta_y = action[..., 1] - reference[..., 1]
        yaw = reference[..., 2]
        delta_yaw = action[..., 2] - yaw
        sin, cos, atan2 = _trig(action)
        return geometry.concatenate(
            [
                (cos(yaw) * delta_x + sin(yaw) * delta_y)[..., None],
                (-sin(yaw) * delta_x + cos(yaw) * delta_y)[..., None],
                atan2(sin(delta_yaw), cos(delta_yaw))[..., None],
            ],
            axis=-1,
        )
    raise TypeError(f"unsupported action field: {type(field).__name__}")


def decode_field(
    field: ActionField,
    action: Array,
    *,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> Array:
    """Convert one field from model to semantic output representation."""
    _validate_last_dim(action, field.model_dim, f"field {field.name!r}")
    if isinstance(field, Padding):
        return geometry.zeros(action, action.shape)
    if isinstance(field, BinaryQpos):
        use_max = action > 0
        if field.invert:
            use_max = ~use_max
        high = geometry.as_like(field.max_value, action)
        low = geometry.as_like(field.min_value, action)
        if isinstance(action, torch.Tensor):
            return torch.where(use_max, high, low)
        return np.where(use_max, high, low)
    if isinstance(field, Qpos):
        trans = _numeric(
            field.trans,
            dim=field.dim,
            like=action,
            label="trans",
        )
        scale = _numeric(
            field.scale,
            dim=field.dim,
            like=action,
            label="scale",
        )
        output = action * scale + trans
        reference = _reference(field, output, ref_state)
        if reference is not None:
            output = output + reference
        return _clip_field(output, field.clip)
    if isinstance(field, EEF):
        return _decode_eef(field, action, ref_state, ref_frame)
    if isinstance(field, Chassis):
        reference = _reference(field, action, ref_state)
        if reference is None:
            raise RuntimeError("relative chassis reference validation failed")
        yaw = reference[..., 2]
        output_yaw = yaw + action[..., 2]
        sin, cos, atan2 = _trig(action)
        return geometry.concatenate(
            [
                (
                    reference[..., 0]
                    + cos(yaw) * action[..., 0]
                    - sin(yaw) * action[..., 1]
                )[..., None],
                (
                    reference[..., 1]
                    + sin(yaw) * action[..., 0]
                    + cos(yaw) * action[..., 1]
                )[..., None],
                atan2(sin(output_yaw), cos(output_yaw))[..., None],
            ],
            axis=-1,
        )
    raise TypeError(f"unsupported action field: {type(field).__name__}")


def _encode_eef(
    field: EEF,
    action: Array,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> Array:
    pose = geometry.pose_to_matrix(action, field.input_rot)
    if field.eef_tf is not None:
        pose = geometry.apply_transform(pose, field.eef_tf)
    if field.model_repr == "absolute":
        return geometry.cast_like(
            geometry.matrix_to_pose(pose, field.model_rot),
            action,
        )
    reference = _reference(field, action, ref_state)
    if reference is None:
        raise RuntimeError("relative EEF reference validation failed")
    reference_pose = geometry.pose_to_matrix(reference, field.input_rot)
    if field.eef_tf is not None:
        reference_pose = geometry.apply_transform(
            reference_pose,
            field.eef_tf,
        )
    reference_pose = geometry.broadcast_matrix(reference_pose, pose)
    action_xyz = pose[..., :3, 3]
    action_rot = pose[..., :3, :3]
    reference_xyz = reference_pose[..., :3, 3]
    reference_rot = reference_pose[..., :3, :3]
    if ref_frame is None:
        xyz = action_xyz - reference_xyz
        rotation = reference_rot.swapaxes(-1, -2) @ action_rot
    else:
        frame = geometry.broadcast_matrix(
            geometry.as_like(ref_frame, pose, dtype=pose.dtype),
            pose,
        )
        frame_from_base = frame[..., :3, :3].swapaxes(-1, -2)
        frame_from_eef = frame_from_base @ reference_rot
        delta_rot = reference_rot.swapaxes(-1, -2) @ action_rot
        rotation = frame_from_eef @ delta_rot @ frame_from_eef.swapaxes(-1, -2)
        local = reference_rot.swapaxes(-1, -2) @ (action_xyz - reference_xyz)[..., None]
        xyz = (frame_from_eef @ local)[..., 0]
    return geometry.cast_like(
        geometry.concatenate(
            [xyz, geometry.mat_to_rot(rotation, field.model_rot)],
            axis=-1,
        ),
        action,
    )


def _decode_eef(
    field: EEF,
    action: Array,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> Array:
    xyz = action[..., :3]
    rotation = geometry.rot_to_mat(
        action[..., 3:],
        field.model_rot,
        validate=False,
    )
    xyz = geometry.as_like(xyz, rotation, dtype=rotation.dtype)
    if field.model_repr == "relative":
        if ref_state is None:
            raise ValueError(f"relative field {field.name!r} requires ref_state")
        reference = geometry.broadcast_vector(ref_state, action)
        reference_pose = geometry.pose_to_matrix(reference, field.input_rot)
        if field.eef_tf is not None:
            reference_pose = geometry.apply_transform(
                reference_pose,
                field.eef_tf,
            )
        reference_xyz = reference_pose[..., :3, 3]
        reference_rot = reference_pose[..., :3, :3]
        if ref_frame is None:
            xyz = xyz + reference_xyz
            rotation = reference_rot @ rotation
        else:
            frame = geometry.broadcast_matrix(
                geometry.as_like(
                    ref_frame,
                    reference_pose,
                    dtype=reference_pose.dtype,
                ),
                reference_pose,
            )
            frame_from_base = frame[..., :3, :3].swapaxes(-1, -2)
            frame_from_eef = frame_from_base @ reference_rot
            eef_from_frame = frame_from_eef.swapaxes(-1, -2)
            rotation = reference_rot @ eef_from_frame @ rotation @ frame_from_eef
            xyz = (
                reference_xyz
                + (reference_rot @ eef_from_frame @ xyz[..., None])[..., 0]
            )
    pose = _pose_matrix(xyz, rotation)
    if field.eef_tf is not None:
        inverse_transform = geometry.inverse(
            geometry.as_like(field.eef_tf, pose, dtype=pose.dtype)
        )
        pose = geometry.apply_transform(pose, inverse_transform)
    return geometry.cast_like(
        geometry.matrix_to_pose(pose, field.output_rot),
        action,
    )


def _pose_matrix(xyz: Array, rotation: Array) -> Array:
    upper = geometry.concatenate([rotation, xyz[..., None]], axis=-1)
    bottom = geometry.zeros(xyz, xyz.shape[:-1] + (1, 4))
    bottom[..., 0, 3] = 1
    return geometry.concatenate([upper, bottom], axis=-2)


def _clip_field(
    value: Array,
    bounds: tuple[float, float] | None,
) -> Array:
    if bounds is None:
        return value
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=bounds[0], max=bounds[1])
    return np.clip(value, bounds[0], bounds[1])


def _trig(value: Array):
    if isinstance(value, torch.Tensor):
        return torch.sin, torch.cos, torch.atan2
    return np.sin, np.cos, np.arctan2


__all__ = [
    "convert_field",
    "decode_field",
    "encode_field",
    "geometry",
    "image",
    "interpolation",
    "landmark",
    "mask",
    "normalization",
    "urdf",
]
