# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map RLinf env observations to XPolicyLab observation dicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_hwc_uint8(image: Any) -> np.ndarray:
    image = _as_numpy(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC or CHW image, got shape {image.shape}.")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        max_value = float(image.max()) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _batch_size(env_obs: dict[str, Any]) -> int:
    for key in ("main_images", "states", "task_descriptions"):
        value = env_obs.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return len(value)
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
    raise ValueError("Cannot infer batch size from env_obs.")


def _instruction_at(env_obs: dict[str, Any], index: int) -> str:
    prompts = env_obs.get("task_descriptions")
    if prompts is None:
        return ""
    if isinstance(prompts, (list, tuple)):
        return str(prompts[index]) if index < len(prompts) else ""
    return str(prompts)


def _wrist_image(wrist_images: Any, batch_index: int, wrist_index: int):
    if wrist_images is None:
        return None
    if isinstance(wrist_images, (list, tuple)):
        item = wrist_images[batch_index]
        if torch.is_tensor(item) or isinstance(item, np.ndarray):
            item = _as_numpy(item)
            if item.ndim >= 4:
                return item[wrist_index] if item.shape[0] > wrist_index else None
            return item if wrist_index == 0 else None
        if isinstance(item, (list, tuple)):
            return item[wrist_index] if wrist_index < len(item) else None
        return None
    array = _as_numpy(wrist_images)
    if array.ndim == 5:
        return array[batch_index, wrist_index]
    if array.ndim == 4 and wrist_index == 0:
        return array[batch_index]
    return None


def env_obs_to_xpolicy_obs_list(
    env_obs: dict[str, Any],
    *,
    action_type: str,
    robot_action_dim_info: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Convert an RLinf rollout ``env_obs`` batch into XPolicyLab obs dicts."""
    from XPolicyLab.utils.process_data import unpack_robot_state

    batch_size = _batch_size(env_obs)
    main_images = env_obs.get("main_images")
    wrist_images = env_obs.get("wrist_images")
    states = env_obs.get("states")
    if states is None:
        raise KeyError("env_obs must contain 'states'.")
    states = _as_numpy(states)

    obs_list: list[dict[str, Any]] = []
    for index in range(batch_size):
        vision: dict[str, Any] = {}
        if main_images is not None:
            main = (
                main_images[index]
                if isinstance(main_images, (list, tuple))
                else _as_numpy(main_images)[index]
            )
            vision["cam_high"] = {"color": _as_hwc_uint8(main)}

        left = _wrist_image(wrist_images, index, 0)
        if left is not None:
            vision["cam_left_wrist"] = {"color": _as_hwc_uint8(left)}
        right = _wrist_image(wrist_images, index, 1)
        if right is not None:
            vision["cam_right_wrist"] = {"color": _as_hwc_uint8(right)}

        packed_state = np.asarray(states[index], dtype=np.float32)
        if robot_action_dim_info is None:
            raise ValueError(
                "XPolicyLab rollout requires env_cfg_type so state can be unpacked "
                "into left/right joint keys. Set actor.model.xpolicylab.env_cfg_type."
            )
        state = unpack_robot_state(
            packed_state,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        )
        if isinstance(state, list):
            state = state[0]

        obs_list.append(
            {
                "vision": vision,
                "state": state,
                "instruction": _instruction_at(env_obs, index),
                "env_idx": index,
            }
        )
    return obs_list


def pack_action_steps(
    action_lists: list[list[dict[str, Any]]],
    *,
    action_type: str,
    robot_action_dim_info: dict[str, Any] | None,
) -> np.ndarray:
    """Pack XPolicyLab ``get_action_batch`` output into ``[B, H, D]``."""
    from XPolicyLab.utils.process_data import pack_robot_state

    if robot_action_dim_info is None:
        raise ValueError(
            "Cannot pack XPolicyLab actions without robot_action_dim_info / env_cfg_type."
        )
    packed_batch = []
    horizon = None
    for steps in action_lists:
        if not steps:
            raise ValueError(
                "XPolicyLab get_action_batch returned an empty action list."
            )
        rows = [
            pack_robot_state(
                {"state": step},
                action_type,
                robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
            for step in steps
        ]
        chunk = np.stack(rows, axis=0)
        if horizon is None:
            horizon = chunk.shape[0]
        elif chunk.shape[0] != horizon:
            raise ValueError(
                "Inconsistent action horizon in a batch: "
                f"{horizon} vs {chunk.shape[0]}."
            )
        packed_batch.append(chunk)
    return np.stack(packed_batch, axis=0)
