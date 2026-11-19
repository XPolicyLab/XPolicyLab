import dataclasses

import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


@dataclasses.dataclass(frozen=True)
class FrankaHkustInputs(transforms.DataTransformFn):
    """Builds absolute EEF targets from recorded states, keeping the original gripper commands."""

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["observation/state"])
        current_state = raw_state[0] if raw_state.ndim == 2 else raw_state
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": np.concatenate([current_state[9:15], current_state[7:8]], axis=-1).astype(np.float32),
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
            if raw_state.ndim != 2 or raw_actions.shape != (len(raw_state), 7):
                raise ValueError("Training requires equally long state (H, 15) and action (H, 7) sequences.")
            # Only the gripper label comes from action; EEF pose targets come from recorded states.
            inputs["actions"] = np.concatenate([raw_state[:, 9:15], raw_actions[:, 6:7]], axis=-1).astype(np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "index" in data:
            inputs["index"] = data["index"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaHkustOutputs(transforms.DataTransformFn):
    """Returns absolute xyz/RPY targets and the unchanged gripper command, without padding."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
