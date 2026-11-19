import dataclasses
import random

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MolmoActInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (1 = closed, 0 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"])
        gripper_state = raw_state[..., 6:7]
        if self.unified_gripper:
            gripper_state = 1 - np.clip(gripper_state, 0, 1)
        state = np.concatenate([raw_state[..., :6], gripper_state], axis=-1)

        main_image = _parse_image(data["images"][random.choice(("primary", "secondary"))])
        wrist_image = _parse_image(data["images"]["wrist"])
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (main_image, wrist_image, np.zeros_like(main_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (main_image, np.zeros_like(main_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"])
            gripper_actions = raw_actions[..., 6:7]
            if self.unified_gripper:
                gripper_actions = 1 - np.clip(gripper_actions, 0, 1)
            inputs["actions"] = np.concatenate([raw_actions[..., :6], gripper_actions], axis=-1)

        inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MolmoActOutputs(transforms.DataTransformFn):
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (1 = closed, 0 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][..., :7], copy=True)
        if self.unified_gripper:
            actions[..., -1] = 1 - actions[..., -1]
        return {"actions": actions}
