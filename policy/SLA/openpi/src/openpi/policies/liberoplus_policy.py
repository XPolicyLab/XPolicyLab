import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoPlusInputs(transforms.DataTransformFn):
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (1 = closed, -1 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["observation/state"])
        # Panda finger joints have nominal ranges [0, 0.04] and [-0.04, 0], for a maximum width of 0.08 m.
        gripper_state = np.clip(raw_state[..., 6] - raw_state[..., 7], 0.0, 0.08)
        if self.unified_gripper:
            gripper_state = gripper_state / 0.08

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.concatenate([raw_state[..., :6], gripper_state[..., None]], axis=-1).astype(np.float32),
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
            gripper_actions = raw_actions[..., 6:7]
            if self.unified_gripper:
                gripper_actions = (1.0 - gripper_actions) / 2.0
            inputs["actions"] = np.concatenate(
                [
                    # robosuite 1.4.1 OSC_POSE maps normalized position commands to EEF deltas in meters.
                    raw_actions[..., :3] * 0.05,
                    # OSC_POSE maps normalized axis-angle commands to rotation-vector deltas in radians.
                    raw_actions[..., 3:6] * 0.5,
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
class LiberoPlusOutputs(transforms.DataTransformFn):
    """Converts physical actions back to LIBERO OSC commands."""

    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        model_actions = np.asarray(data["actions"])[..., :7]
        gripper_actions = model_actions[..., 6:7]
        if self.unified_gripper:
            gripper_actions = 1.0 - 2.0 * gripper_actions
        actions = np.concatenate(
            [
                model_actions[..., :3] / 0.05,
                model_actions[..., 3:6] / 0.5,
                gripper_actions,
            ],
            axis=-1,
        )
        return {"actions": actions}
