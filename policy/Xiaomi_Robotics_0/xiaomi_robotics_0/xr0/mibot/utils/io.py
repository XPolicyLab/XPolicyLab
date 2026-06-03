# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ACTION_DIM = 32
STATE_DIM = 32
ACTION_EPS = 1e-6

ACTION_PARTS = (
    ("left_ee_pos", slice(0, 3)),
    ("left_ee_aa", slice(3, 6)),
    ("left_gripper", slice(6, 7)),
    ("left_joint", slice(7, 13)),
    ("right_ee_pos", slice(14, 17)),
    ("right_ee_aa", slice(17, 20)),
    ("right_gripper", slice(20, 21)),
    ("right_joint", slice(21, 27)),
)


def get_value(data, path):
    for key in path.split("."):
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def resize_image(image: Image.Image, factor: int = 32, min_pixels: int = 32 * 32, max_pixels: int = 90000) -> Image.Image:
    width, height = image.size
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)
    elif new_height * new_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_height = max(factor, math.ceil(height * scale / factor) * factor)
        new_width = max(factor, math.ceil(width * scale / factor) * factor)

    return image.resize((new_width, new_height))


def _axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]

    if rot00 >= rot11 and rot00 >= rot22:
        axis = np.array([np.sqrt(max((rot00 + 1.0) / 2.0, 0.0)), 0.0, 0.0], dtype=np.float32)
        if axis[0] > 1e-8:
            axis[1] = rotm[0, 1] / (2.0 * axis[0])
            axis[2] = rotm[0, 2] / (2.0 * axis[0])
    elif rot11 >= rot22:
        axis = np.array([0.0, np.sqrt(max((rot11 + 1.0) / 2.0, 0.0)), 0.0], dtype=np.float32)
        if axis[1] > 1e-8:
            axis[0] = rotm[0, 1] / (2.0 * axis[1])
            axis[2] = rotm[1, 2] / (2.0 * axis[1])
    else:
        axis = np.array([0.0, 0.0, np.sqrt(max((rot22 + 1.0) / 2.0, 0.0))], dtype=np.float32)
        if axis[2] > 1e-8:
            axis[0] = rotm[0, 2] / (2.0 * axis[2])
            axis[1] = rotm[1, 2] / (2.0 * axis[2])

    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis / norm


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))

    axis_angle = np.zeros((rotms.shape[0], 3), dtype=np.float32)
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
    left_joint,
    right_ee_pos,
    right_ee_aa,
    right_gripper,
    right_joint,
    action_length: int,
) -> np.ndarray:
    values = (left_ee_pos, left_ee_aa, left_gripper, left_joint, right_ee_pos, right_ee_aa, right_gripper, right_joint)
    action = np.zeros((action_length, ACTION_DIM), dtype=np.float32)
    for (_, slc), value in zip(ACTION_PARTS, values):
        action[:, slc] = np.asarray(value, dtype=np.float32)
    return action


def compose_state(left_gripper, left_joint, right_gripper, right_joint) -> np.ndarray:
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    for slc, value in (
        (slice(6, 7), left_gripper),
        (slice(7, 13), left_joint),
        (slice(20, 21), right_gripper),
        (slice(21, 27), right_joint),
    ):
        state[:, slc] = np.asarray(value, dtype=np.float32).reshape(1, slc.stop - slc.start)
    return state


def normalize_action(action, mean, std) -> np.ndarray:
    return (action - mean) / (std + ACTION_EPS)


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
        joint = np.asarray(robot_state[f"{side}_arm_joint"], dtype=np.float32).reshape(6)

        targets[f"{side}_ee_pos"] = (pos[None] + parts[f"{side}_ee_pos"] @ rotm.T).astype(np.float32)
        targets[f"{side}_ee_rotm"] = np.stack([rotm @ aa2rotm(delta) for delta in parts[f"{side}_ee_aa"]], axis=0).astype(np.float32)
        targets[f"{side}_gripper_pos"] = (gripper[None] + parts[f"{side}_gripper"]).astype(np.float32)
        targets[f"{side}_arm_joint"] = (joint[None] + parts[f"{side}_joint"]).astype(np.float32)

    return targets
