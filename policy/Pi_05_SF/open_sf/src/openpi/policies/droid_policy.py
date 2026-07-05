import dataclasses

import einops
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def convert_extrinsics_to_matrix(extrinsics):
    """input extrinsics is a 6-dim vector: [x, y, z, roll, pitch, yaw]"""
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    pos = extrinsics[0:3] # translation
    rot_mat = R.from_euler("xyz", extrinsics[3:6]).as_matrix() # rotation

    # Make homogenous transformation matrix
    cam_to_base_extrinsics_matrix = np.eye(4)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    # As DROID uses a different coordinate system from the geometry model, we need to apply an extra transformation
    extra_op_matrix = np.diag([-1, 1, 1, 1])
    cam_to_base_with_extra_op_matrix = cam_to_base_extrinsics_matrix @ extra_op_matrix
    return cam_to_base_with_extra_op_matrix[:3, :]


def convert_intrinsics_to_matrix(intrinsics):
    """input extrinsics is a 4-dim vector: [fx, cx, fy, cy]"""
    fx, cx, fy, cy = intrinsics
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OmniDroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Randomly sample one of the two exterior images who have extrinsics in DROID during training.
        # Note: only train with one at a time.
        # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
        # TODO: shall we use both two exterior_images during training?
        suffixes = ["1_left", "2_left"]
        omni_indics = [
            suffix
            for suffix in suffixes
            if np.abs(data[f"extrinsics/exterior_image_{suffix}"]).sum() > 0
            and np.abs(data[f"intrinsics/exterior_image_{suffix}"]).sum() > 0
        ]
        omni_index = random.choice(omni_indics) if omni_indics else random.choice(suffixes)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data[f"observation/exterior_image_{omni_index}"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        inputs["camera_param"] = {
            f"{names[0]}_extrinsics": convert_extrinsics_to_matrix(data[f"extrinsics/exterior_image_{omni_index}"]),
            f"{names[0]}_intrinsics": convert_intrinsics_to_matrix(data[f"intrinsics/exterior_image_{omni_index}"]),
            f"{names[1]}_extrinsics": np.zeros([3, 4], dtype=np.float32),
            f"{names[1]}_intrinsics": np.zeros([3, 3], dtype=np.float32),
            f"{names[2]}_extrinsics": np.zeros([3, 4], dtype=np.float32),
            f"{names[2]}_intrinsics": np.zeros([3, 3], dtype=np.float32),
        }
        inputs["camera_param_mask"] = dict(
            zip(names, (np.True_ if omni_indics else np.False_, np.False_, np.False_), strict=True)
        )

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        valid_prompt = [
            res
            for p in [
                data["language_instruction"],
                data["language_instruction_2"],
                data["language_instruction_3"],
            ]
            if (res := (p.decode("utf-8") if isinstance(p, bytes) else p)) is not None
        ]
        inputs["prompt"] = random.choice(valid_prompt)

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
