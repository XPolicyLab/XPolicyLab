import dataclasses
import functools
import math
from pathlib import Path
from typing import ClassVar

import einops
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_xyzw_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension to be 4, got shape {quat.shape}.")
    return R.from_quat(quat.reshape(-1, 4)).as_euler("xyz", degrees=False).reshape(quat.shape[:-1] + (3,))


@dataclasses.dataclass(frozen=True)
class AgibotBetaInputs(transforms.DataTransformFn):
    """Inputs for the AgibotBeta policy.
    """

    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (120 = closed, 35 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        # Assume that all three-view images always exist.
        images = {
            "base_0_rgb": _parse_image(data["images"]["cam_high"]),
            "left_wrist_0_rgb": _parse_image(data["images"]["cam_left_wrist"]),
            "right_wrist_0_rgb": _parse_image(data["images"]["cam_right_wrist"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        grippers = np.clip(np.asarray(data['grippers']), 35, 120)
        if self.unified_gripper:
            grippers = (120 - grippers) / (120 - 35)

        state_gripper = grippers[0]

        if self.use_eef_action:
            state_eef_orientation = np.asarray(data['state_eef_orientation'])
            state_eef_euler = _quat_xyzw_to_euler_xyz(state_eef_orientation)
            state_eef_position = np.asarray(data['state_eef_position'])
            state = np.concatenate(
                [
                    state_eef_position[0],  # left eef xyz
                    state_eef_euler[0],  # left eef rpy
                    state_gripper[:1],  # left gripper
                    state_eef_position[1],  # right eef xyz
                    state_eef_euler[1],  # right eef rpy
                    state_gripper[1:2],  # right gripper
                ],
                axis=-1,
            )
        else:
            state_joint = np.asarray(data['state_joint'])
            state = np.concatenate(
                [
                    state_joint[:7],  # left joint angles
                    state_gripper[:1],  # left gripper
                    state_joint[7:],  # right joint angles
                    state_gripper[1:2],  # right gripper
                ],
                axis=-1,
            )

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions_eef_position" in data or "actions_joint" in data:
            if self.use_eef_action:
                actions_eef_orientation = np.asarray(data['actions_eef_orientation'])
                actions_eef_euler = _quat_xyzw_to_euler_xyz(actions_eef_orientation)
                actions_eef_position = np.asarray(data['actions_eef_position'])
                actions = np.concatenate(
                    [
                        actions_eef_position[:, 0],  # left eef xyz
                        actions_eef_euler[:, 0],  # left eef rpy
                        grippers[:, :1],  # left gripper
                        actions_eef_position[:, 1],  # right eef xyz
                        actions_eef_euler[:, 1],  # right eef rpy
                        grippers[:, 1:2],  # right gripper
                    ],
                    axis=-1,
                )
            else:
                actions_joint = np.asarray(data['actions_joint'])
                actions = np.concatenate(
                    [
                        actions_joint[:, :7],  # left joint angles
                        grippers[:, :1],  # left gripper
                        actions_joint[:, 7:],  # right joint angles
                        grippers[:, 1:2],  # right gripper
                    ],
                    axis=-1,
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"].split(" | ")[0]  # Only use the instruction part

        return inputs


@dataclasses.dataclass(frozen=True)
class AgibotBetaOutputs(transforms.DataTransformFn):
    """Outputs for the AgibotBeta policy."""

    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (120 = closed, 35 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        if self.use_eef_action:
            actions = np.asarray(data["actions"][:, :14])
            if self.unified_gripper:
                actions[:, 6] = 120 - actions[:, 6] * (120 - 35)
                actions[:, 13] = 120 - actions[:, 13] * (120 - 35)
        else:
            actions = np.asarray(data["actions"][:, :16])
            if self.unified_gripper:
                actions[:, 7] = 120 - actions[:, 7] * (120 - 35)
                actions[:, 15] = 120 - actions[:, 15] * (120 - 35)

        return {"actions": actions}
