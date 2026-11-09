import dataclasses
import random

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


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    # Convert gripper state/action to 0 = closed, 1 = open.
    # RoboCasa stores state as finger joint positions and action as (1 = closed, -1 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["observation/state"])
        eef_rotvec = Rotation.from_quat(raw_state[10:14]).as_rotvec()
        # RoboSuite's PandaGripper XML defines nominal joint ranges [0, 0.04] and [-0.04, 0].
        gripper_state = np.clip(raw_state[14] - raw_state[15], 0.0, 0.08)
        if self.unified_gripper:
            gripper_state = gripper_state / 0.08

        # Randomly select one of the two agentview cameras to use as the base image. 
        # The other agentview camera is not used in this policy.
        agentview_key = random.choice(["observation/agentview_left", "observation/agentview_right"])
        base_image = _parse_image(data[agentview_key])
        wrist_image = _parse_image(data["observation/eye_in_hand"])

        inputs = {
            "state": np.concatenate(
                [
                    raw_state[7:10],  # eef xyz
                    eef_rotvec,
                    gripper_state[None],
                ],
                axis=-1,
            ).astype(np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"])
            gripper_actions = raw_actions[..., 11:12]
            if self.unified_gripper:
                gripper_actions = (1.0 - gripper_actions) / 2.0
            inputs["actions"] = np.concatenate(
                [
                    # RoboSuite OSC maps normalized position commands [-1, 1] to EEF deltas [-0.05, 0.05] meters.
                    raw_actions[..., 5:8] * 0.05,
                    # RoboSuite OSC maps normalized rotation commands [-1, 1] to rot-vector deltas [-0.5, 0.5] rads.
                    raw_actions[..., 8:11] * 0.5,
                    gripper_actions,
                ],
                axis=-1,
            ).astype(np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "index" in data:
            inputs["index"] = data["index"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (1 = closed, -1 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        model_actions = np.asarray(data["actions"])[..., :7]
        # Reconstruct the RoboCasa 12D action prefix. 
        # NoMovement uses zero base and torso commands (dims 0:4) and arm control mode -1 (dim 4).
        prefix = np.zeros((*model_actions.shape[:-1], 5), dtype=model_actions.dtype)
        prefix[..., 4] = -1.0
        gripper_actions = model_actions[..., 6:7]
        if self.unified_gripper:
            gripper_actions = 1.0 - 2.0 * gripper_actions
        actions = np.concatenate(
            [
                prefix,
                # Invert the RoboSuite OSC physical position scale (meters).
                model_actions[..., :3] / 0.05,
                # Invert the RoboSuite OSC physical rotation scale (radians).
                model_actions[..., 3:6] / 0.5,
                gripper_actions,
            ],
            axis=-1,
        )
        return {"actions": actions}
