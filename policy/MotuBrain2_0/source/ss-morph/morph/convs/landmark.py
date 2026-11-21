"""Landmark-to-EEF conversion for NumPy and PyTorch arrays."""

from __future__ import annotations

import numpy as np
import torch

from morph.convs import geometry
from morph.types.fields import ArrayT
from morph.types.fields import LandmarkEEF


def landmark_to_eef(value: ArrayT, field: LandmarkEEF) -> ArrayT:
    """Derive an EEF pose while preserving the input array backend."""
    if value.ndim != 3 or value.shape[-1] != 7:
        raise ValueError("LandmarkEEF expects shape [T, N, 7], " f"got {value.shape}")
    required = max((field.wrist_index, *field.center_indices))
    if value.shape[1] <= required:
        raise ValueError(
            f"LandmarkEEF has {value.shape[1]} landmarks but index "
            f"{required} is required"
        )
    xyz = value[..., :3]
    rotation = geometry.rot_to_mat(
        value[..., 3:],
        field.source_rotation,
        validate=True,
    )
    wrist_xyz = xyz[:, field.wrist_index]
    if isinstance(value, torch.Tensor):
        center_xyz = xyz[:, list(field.center_indices)].mean(dim=1)
    else:
        center_xyz = xyz[:, field.center_indices].mean(axis=1)
    eef_xyz = wrist_xyz + field.center_alpha * (center_xyz - wrist_xyz)
    if field.position_filter_window is not None:
        eef_xyz = _savgol(
            eef_xyz,
            field.position_filter_window,
            field.position_filter_polyorder,
        )
    eef_rotation = rotation[:, field.wrist_index]
    if field.eef_tf is not None:
        pose = _pose_matrix(eef_xyz, eef_rotation)
        pose = geometry.apply_transform(pose, field.eef_tf)
        eef_xyz = pose[..., :3, 3]
        eef_rotation = pose[..., :3, :3]
    result = geometry.concatenate(
        [eef_xyz, geometry.mat_to_rot(eef_rotation, field.input_rot)],
        axis=-1,
    )
    return geometry.cast_like(result, value)


def _pose_matrix(xyz: ArrayT, rotation: ArrayT) -> ArrayT:
    upper = geometry.concatenate([rotation, xyz[..., None]], axis=-1)
    bottom = geometry.zeros(xyz, xyz.shape[:-1] + (1, 4))
    bottom[..., 0, 3] = 1
    return geometry.concatenate([upper, bottom], axis=-2)


def _savgol(value: ArrayT, window: int, polyorder: int) -> ArrayT:
    """Apply SciPy-compatible Savitzky-Golay interpolation boundaries."""
    if value.shape[0] < window:
        raise ValueError(
            f"filter window {window} exceeds sequence length {value.shape[0]}"
        )
    half = window // 2
    coordinates = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(coordinates, N=polyorder + 1, increasing=True)
    center_weights = np.linalg.pinv(design)[0]
    if isinstance(value, torch.Tensor):
        weights = torch.as_tensor(
            center_weights,
            dtype=value.dtype,
            device=value.device,
        )
        windows = value.unfold(0, window, 1).movedim(-1, -2)
        middle = (windows * weights[:, None]).sum(dim=-2)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(
            value,
            window_shape=window,
            axis=0,
        ).swapaxes(-1, -2)
        middle = np.sum(windows * center_weights[:, None], axis=-2)

    left = _fit_edges(value[:window], np.arange(-half, 0), polyorder)
    right = _fit_edges(
        value[-window:],
        np.arange(window - half, window),
        polyorder,
    )
    return geometry.concatenate([left, middle, right], axis=0)


def _fit_edges(value: ArrayT, positions: np.ndarray, polyorder: int) -> ArrayT:
    x = np.arange(value.shape[0], dtype=np.float64)
    design = np.vander(x, N=polyorder + 1, increasing=True)
    evaluate = np.vander(positions, N=polyorder + 1, increasing=True)
    weights = evaluate @ np.linalg.pinv(design)
    if isinstance(value, torch.Tensor):
        tensor = torch.as_tensor(weights, dtype=value.dtype, device=value.device)
        return tensor @ value
    return weights @ value
