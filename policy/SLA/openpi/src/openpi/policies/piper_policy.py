import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def quat_xyzw_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension to be 4, got shape {quat.shape}.")
    return Rotation.from_quat(quat.reshape(-1, 4)).as_euler("xyz", degrees=False).reshape((*quat.shape[:-1], 3))


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for the Piper policy."""

    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (100 = closed, 0 = open).
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

        if self.use_eef_action:
            state = np.asarray(data['eef_state'])
            gripper_state = np.clip(state[[7, 15]], 0, 100)
            if self.unified_gripper:
                gripper_state = (100 - gripper_state) / (100 - 0)
            state = np.concatenate(
                [
                    state[0:3],  # left eef xyz
                    quat_xyzw_to_euler_xyz(state[3:7]),  # left eef rpy
                    gripper_state[:1],  # left gripper
                    state[8:11],  # right eef xyz
                    quat_xyzw_to_euler_xyz(state[11:15]),  # right eef rpy
                    gripper_state[1:2],  # right gripper
                ],
                axis=-1,
            )
        else:
            state = np.asarray(data['joint_state'])
            gripper_state = np.clip(state[[6, 13]], 0, 100)
            if self.unified_gripper:
                gripper_state = (100 - gripper_state) / 100
            state = np.concatenate(
                [
                    state[:6],  # left joint angles
                    gripper_state[:1],  # left gripper
                    state[7:13],  # right joint angles
                    gripper_state[1:2],  # right gripper
                ],
                axis=-1,
            )

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "joint_actions" in data or "eef_actions" in data:
            if self.use_eef_action:
                actions = np.asarray(data['eef_actions'])
                gripper_actions = np.clip(actions[:, [7, 15]], 0, 100)
                if self.unified_gripper:
                    gripper_actions = (100 - gripper_actions) / (100 - 0)
                actions = np.concatenate(
                    [
                        actions[:, 0:3],  # left eef xyz
                        quat_xyzw_to_euler_xyz(actions[:, 3:7]),  # left eef rpy
                        gripper_actions[:, :1],  # left gripper
                        actions[:, 8:11],  # right eef xyz
                        quat_xyzw_to_euler_xyz(actions[:, 11:15]),  # right eef rpy
                        gripper_actions[:, 1:2],  # right gripper
                    ],
                    axis=-1,
                )
            else:
                actions = np.asarray(data['joint_actions'])
                gripper_actions = np.clip(actions[:, [6, 13]], 0, 100)
                if self.unified_gripper:
                    gripper_actions = (100 - gripper_actions) / 100
                actions = np.concatenate(
                    [
                        actions[:, :6],  # left joint angles
                        gripper_actions[:, :1],  # left gripper
                        actions[:, 7:13],  # right joint angles
                        gripper_actions[:, 1:2],  # right gripper
                    ],
                    axis=-1,
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for the Piper policy."""

    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (100 = closed, 0 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][:, :14], copy=True)
        if self.unified_gripper:
            actions[:, 6] = 100 - actions[:, 6] * 100
            actions[:, 13] = 100 - actions[:, 13] * 100

        if not self.use_eef_action:
            return {"actions": actions}

        return {
            "actions": np.concatenate(
                [
                    actions[:, 0:3],
                    Rotation.from_euler("xyz", actions[:, 3:6]).as_quat().astype(actions.dtype, copy=False),
                    actions[:, 6:7],
                    actions[:, 7:10],
                    Rotation.from_euler("xyz", actions[:, 10:13]).as_quat().astype(actions.dtype, copy=False),
                    actions[:, 13:14],
                ],
                axis=-1,
            )
        }
