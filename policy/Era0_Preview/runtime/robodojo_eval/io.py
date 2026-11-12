# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboDojo entry point for Hy-VLA evaluation.

This module mirrors ``robotwin_eval.deploy_policy`` but accepts RoboDojo-style
observations:

* ``vision/{cam_head,cam_left_wrist,cam_right_wrist}/color``
* ``state/{left,right}_ee_pose`` as xyz + quat_wxyz
* ``state/{left,right}_ee_joint_state`` as gripper scalars
* optional ``instruction`` / ``prompt``
"""

from __future__ import annotations

from typing import Any

import numpy as np


CAM_HEAD = ["cam_head", "head_camera", "cam_high", "top_camera"]
CAM_LEFT = ["cam_left_wrist", "left_camera", "left_wrist"]
CAM_RIGHT = ["cam_right_wrist", "right_camera", "right_wrist"]


def _extract_image(obs: dict[str, Any], candidates: list[str]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in candidates:
        if name not in vision:
            continue
        entry = vision[name]
        if isinstance(entry, dict):
            for key in ("color", "rgb", "colors"):
                if key in entry:
                    return _to_hwc_uint8_rgb(entry[key])
        else:
            return _to_hwc_uint8_rgb(entry)
    raise KeyError(f"No image for any of {candidates}; have {list(vision.keys())}")


def _to_hwc_uint8_rgb(img: Any) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _to_chw_float(img: np.ndarray) -> np.ndarray:
    return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def encode_obs(observation: dict[str, Any], instruction: str | None = None) -> dict[str, Any]:
    """Pack a RoboDojo observation into the Hy-VLA batch format."""
    head = _extract_image(observation, CAM_HEAD)
    left = _extract_image(observation, CAM_LEFT)
    right = _extract_image(observation, CAM_RIGHT)

    state = observation.get("state", {})
    left_pose = np.asarray(state["left_ee_pose"], dtype=np.float32)
    right_pose = np.asarray(state["right_ee_pose"], dtype=np.float32)
    left_gripper = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
    right_gripper = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
    state16 = np.concatenate(
        [
            left_pose[:3],
            left_pose[3:7],
            [left_gripper],
            right_pose[:3],
            right_pose[3:7],
            [right_gripper],
        ]
    ).astype(np.float32)

    task = instruction or observation.get("instruction") or observation.get("prompt") or ""
    return {
        "observation.images.top_head": _to_chw_float(head),
        "observation.images.hand_left": _to_chw_float(left),
        "observation.images.hand_right": _to_chw_float(right),
        "observation.state": _pad_state(state16[np.newaxis, :], max_state_dim=32),
        "task": [task],
        "raw_images.top_head": head,
        "raw_images.hand_left": left,
        "raw_images.hand_right": right,
    }


from ..hy_vla.utils.transform_utils import convert_frame_robo_to_umi, convert_PosQuat2PosRotationMatrix_batch

L_POS, L_QUAT, L_GRIP = slice(0, 3), slice(3, 7), 7
R_POS, R_QUAT, R_GRIP = slice(8, 11), slice(11, 15), 15


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def _convert_pose_robo_dojo(
    eepose16_wxyz: np.ndarray,
    qpos_mean: np.ndarray,
    qpos_std: np.ndarray,
    *,
    umi_coord_frame: bool,
    umi_gripper_space: bool,
) -> np.ndarray:
    """Encode a 16-d RoboDojo EE state into normalized Hy-VLA state space."""
    e = eepose16_wxyz.copy()
    e[3:7] = eepose16_wxyz[[4, 5, 6, 3]]
    e[11:15] = eepose16_wxyz[[12, 13, 14, 11]]
    if umi_coord_frame:
        e = convert_frame_robo_to_umi(
            e[None, :], convert_gripper=umi_gripper_space,
        )[0]
    ee_prop = convert_PosQuat2PosRotationMatrix_batch(e[None, :], quat_order="xyzw")[0]
    return ((ee_prop - qpos_mean) / (qpos_std + 1e-8))[None, ...]


def action16_to_robodojo_dict(action_wxyz: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one 16-d dual-arm EE action to RoboDojo's action dict."""
    row = np.asarray(action_wxyz, dtype=np.float32)
    if row.shape != (16,):
        raise ValueError(f"Expected a 16-d action vector, got shape {row.shape}")
    return {
        "left_ee_pose": np.concatenate([row[L_POS], row[L_QUAT]]).astype(np.float32),
        "right_ee_pose": np.concatenate([row[R_POS], row[R_QUAT]]).astype(np.float32),
        "left_ee_joint_state": np.array([row[L_GRIP]], dtype=np.float32),
        "right_ee_joint_state": np.array([row[R_GRIP]], dtype=np.float32),
    }


def action_chunk_to_robodojo_dicts(chunk16_wxyz: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Convert a ``(T, 16)`` action chunk to RoboDojo action dictionaries."""
    return [action16_to_robodojo_dict(row) for row in np.asarray(chunk16_wxyz)]
