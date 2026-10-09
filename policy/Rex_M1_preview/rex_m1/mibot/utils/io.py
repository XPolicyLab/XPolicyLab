# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ACTION_DIM = 60
STATE_DIM = 60
ACTION_EPS = 1e-6

ACTION_PARTS = (
    ("left_ee_pos", slice(0, 3)),
    ("left_ee_aa", slice(3, 6)),
    ("left_gripper", slice(6, 7)),
    ("right_ee_pos", slice(8, 11)),
    ("right_ee_aa", slice(11, 14)),
    ("right_gripper", slice(14, 15)),
    ("waist", slice(16, 17)),
    ("base", slice(17, 20)),
)


def get_value(data, path):
    for key in path.split("."):
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def resize_image(
    image: Image.Image,
    factor: int = 32,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 160000,
) -> Image.Image:
    width, height = image.size
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = math.floor(height / scale / factor) * factor
        new_width = math.floor(width / scale / factor) * factor
    elif new_height * new_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_height = math.ceil(height * scale / factor) * factor
        new_width = math.ceil(width * scale / factor) * factor

    return image.resize((new_width, new_height))


def _axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]

    if rot00 >= rot11 and rot00 >= rot22:
        vx = np.sqrt(max((rot00 + 1) / 2, 0))
        if vx > 1e-8:
            vy = rotm[0, 1] / (2 * vx)
            vz = rotm[0, 2] / (2 * vx)
        else:
            vy = vz = 0.0
        axis = np.array([vx, vy, vz])
    elif rot11 >= rot22:
        vy = np.sqrt(max((rot11 + 1) / 2, 0))
        if vy > 1e-8:
            vx = rotm[0, 1] / (2 * vy)
            vz = rotm[1, 2] / (2 * vy)
        else:
            vx = vz = 0.0
        axis = np.array([vx, vy, vz])
    else:
        vz = np.sqrt(max((rot22 + 1) / 2, 0))
        if vz > 1e-8:
            vx = rotm[0, 2] / (2 * vz)
            vy = rotm[1, 2] / (2 * vz)
        else:
            vx = vy = 0.0
        axis = np.array([vx, vy, vz])

    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))

    axis_angle = np.zeros((rotms.shape[0], 3))
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        for i in np.where(near_pi)[0]:
            axis_angle[i] = _axis_from_pi(rotms[i]) * theta[i]

    return axis_angle


def aa2rotm(axis_angle) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / (angle + 1e-10)
    x, y, z = axis.tolist()
    axis_hat = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    eye = np.identity(3, dtype=np.float32)
    return eye + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * axis_hat @ axis_hat


def validate_stats(mean: Sequence[Sequence[float]], std: Sequence[Sequence[float]], action_length: int):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (action_length, ACTION_DIM):
        raise ValueError(f"mean expected shape {(action_length, ACTION_DIM)}, got {mean.shape}")
    if std.shape != (action_length, ACTION_DIM):
        raise ValueError(f"std expected shape {(action_length, ACTION_DIM)}, got {std.shape}")
    return mean, std


def validate_quantiles(q01: Sequence[Sequence[float]], q99: Sequence[Sequence[float]]):
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    if q01.ndim != 2 or q99.ndim != 2:
        raise ValueError(f"q01 and q99 must be two-dimensional, got {q01.shape} and {q99.shape}")
    if q01.shape != q99.shape:
        raise ValueError(f"q01 and q99 must have the same shape, got {q01.shape} and {q99.shape}")
    if q01.shape != (1, STATE_DIM):
        raise ValueError(f"q01 and q99 expected shape {(1, STATE_DIM)}, got {q01.shape}")
    padding = (q01 == 0.0) & (q99 == 0.0)
    if np.any((q99 <= q01) & ~padding):
        raise ValueError("q99 must be greater than q01 outside zero-padded dimensions")
    return q01, q99


def build_action_mask(action_length: int, temporal_mask=None) -> np.ndarray:
    temporal = np.ones(action_length, dtype=np.int32) if temporal_mask is None else np.asarray(temporal_mask, dtype=np.int32)
    mask = np.zeros((action_length, ACTION_DIM), dtype=np.int32)
    for _, slc in ACTION_PARTS:
        mask[:, slc] = temporal[:, None]
    return mask


def compose_action(
    left_ee_pos,
    left_ee_aa,
    left_gripper,
    right_ee_pos,
    right_ee_aa,
    right_gripper,
    waist,
    base,
    action_length: int,
) -> np.ndarray:
    action = np.zeros((action_length, ACTION_DIM), dtype=np.float32)
    values = (left_ee_pos, left_ee_aa, left_gripper, right_ee_pos, right_ee_aa, right_gripper, waist, base)
    for (_, slc), value in zip(ACTION_PARTS, values):
        action[:, slc] = np.asarray(value, dtype=np.float32)
    return action


def compose_state(left_gripper, left_joint, right_gripper, right_joint) -> np.ndarray:
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    left_joint = np.asarray(left_joint, dtype=np.float32).reshape(1, -1)
    right_joint = np.asarray(right_joint, dtype=np.float32).reshape(1, -1)
    if left_joint.shape[1] > 7 or right_joint.shape[1] > 7:
        raise ValueError("arm joint state cannot exceed 7 dimensions")
    state[:, : left_joint.shape[1]] = left_joint
    state[:, 7:8] = np.asarray(left_gripper, dtype=np.float32).reshape(1, 1)
    state[:, 8 : 8 + right_joint.shape[1]] = right_joint
    state[:, 15:16] = np.asarray(right_gripper, dtype=np.float32).reshape(1, 1)
    return state


def normalize_action(action, mean, std) -> np.ndarray:
    return (action - mean) / (std + ACTION_EPS)


def normalize_action_parts(values, mean, std):
    return tuple(
        (np.asarray(value) - mean[:, slc]) / (std[:, slc] + ACTION_EPS)
        for (_, slc), value in zip(ACTION_PARTS, values)
    )


def normalize_quantile(value, q01, q99) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != q01.shape:
        raise ValueError(f"state and quantiles must have the same shape, got {value.shape} and {q01.shape}")
    normalized = np.zeros_like(value)
    valid = q99 > q01
    adjusted_q99 = q99[valid] + ACTION_EPS
    normalized[valid] = (value[valid] - q01[valid]) / (adjusted_q99 - q01[valid])
    normalized[valid] = 2.0 * normalized[valid] - 1.0
    return np.clip(normalized, -1.0, 1.0)


def denormalize_action(action, mean, std):
    return action * (std + ACTION_EPS) + mean


def split_action(action) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    return {name: action[..., slc] for name, slc in ACTION_PARTS}


def recover_action(action, robot_state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    parts = split_action(action)
    targets = {}

    for side in ("left", "right"):
        rotm = np.asarray(robot_state[f"{side}_ee_rotm"], dtype=np.float32).reshape(3, 3)
        pos = np.asarray(robot_state[f"{side}_ee_pos"], dtype=np.float32).reshape(3)
        gripper = np.asarray(robot_state[f"{side}_gripper_pos"], dtype=np.float32).reshape(1)
        targets[f"{side}_ee_pos"] = (pos[None] + parts[f"{side}_ee_pos"] @ rotm.T).astype(np.float32)
        targets[f"{side}_ee_rotm"] = np.stack([rotm @ aa2rotm(delta) for delta in parts[f"{side}_ee_aa"]], axis=0).astype(np.float32)
        targets[f"{side}_gripper_pos"] = (gripper[None] + parts[f"{side}_gripper"]).astype(np.float32)

    if "waist_pos" in robot_state:
        waist = np.asarray(robot_state["waist_pos"], dtype=np.float32).reshape(1)
        targets["waist_pos"] = (waist[None] + parts["waist"]).astype(np.float32)
    targets["base_vel"] = parts["base"].astype(np.float32)

    return targets
