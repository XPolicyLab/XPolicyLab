"""Policy transforms for Tianji Marvin + dual Wuji Hand (Gen1).

Adapted from wuji-openpi. Action layout (54D):
  left arm (7) + left hand (20) + right arm (7) + right hand (20)

Expected LeRobot fields:
- observation.state: 54 dims
- action: 54 dims
- observation.images.cam_left_wrist / cam_right_wrist
- observation.images.stereo_right (or cam_high via DataConfig repack)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Dual-arm + dual Wuji Hand Gen1
WUJI_ACTION_DIM = 54


def make_wuji_example() -> dict:
    """Creates a random input example for the Wuji policy."""
    return {
        "observation/state": np.random.rand(WUJI_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the object with both hands",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WujiInputs(transforms.DataTransformFn):
    """Convert Wuji dataset / runtime inputs to Pi0 model format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WujiOutputs(transforms.DataTransformFn):
    """Trim padded model actions back to the Wuji action dimension."""

    action_dim: int = WUJI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
