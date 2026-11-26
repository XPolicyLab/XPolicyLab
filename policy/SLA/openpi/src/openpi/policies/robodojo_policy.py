import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms

_SOURCE_TO_TARGET = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_TARGET_TO_SOURCE = _SOURCE_TO_TARGET.T
_ROBOT_BASE_MIDPOINT = np.array([0.0, -0.45, 0.765])  # The midpoint of the arm base in the source coordinate frame.


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _arm_y_offset(base_distance: float, *, is_left: bool) -> float:
    if base_distance < 0:
        raise ValueError(f"Arm base distance must be non-negative, got {base_distance}.")
    return (0.5 if is_left else -0.5) * base_distance


class _RobodojoInputs(transforms.DataTransformFn):
    def _to_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, data: dict) -> dict:
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
        state = np.asarray(data["state"])
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.concatenate(
                [
                    self._to_target_pose(state[..., :8], is_left=True),
                    self._to_target_pose(state[..., 8:16], is_left=False),
                ],
                axis=-1,
            ),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = np.concatenate(
                [
                    self._to_target_pose(actions[..., :8], is_left=True),
                    self._to_target_pose(actions[..., 8:16], is_left=False),
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobodojoSimInputs(_RobodojoInputs):
    def _to_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        """Align the simulator's axes and world origin with the model's robot frame."""
        # Formerly source_to_target_pose: convert simulator poses into the model frame.
        del is_left
        position = np.einsum("ij,...j->...i", _SOURCE_TO_TARGET, pose[..., :3] - _ROBOT_BASE_MIDPOINT)
        xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1)
        source_rotation = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
        target_rotation = np.einsum("ij,...jk,kl->...il", _SOURCE_TO_TARGET, source_rotation, _TARGET_TO_SOURCE)
        rpy = Rotation.from_matrix(target_rotation.reshape(-1, 3, 3)).as_euler("xyz").reshape(*pose.shape[:-1], 3)
        return np.concatenate([position, rpy, pose[..., 7:8]], axis=-1)


@dataclasses.dataclass(frozen=True)
class RobodojoRealInputs(_RobodojoInputs):
    arm_base_distance: float = 0.600

    def _to_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        """Merge the two arm-local origins into one robot frame; their axes already match."""
        # Formerly local_arm_to_target_pose: shift each arm pose to the shared midpoint origin.
        position = pose[..., :3] + np.array([0.0, _arm_y_offset(self.arm_base_distance, is_left=is_left), 0.0])
        xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1)
        rpy = Rotation.from_quat(xyzw.reshape(-1, 4)).as_euler("xyz").reshape(*pose.shape[:-1], 3)
        gripper = np.clip(pose[..., 7:8], 0.0, 1.0)
        return np.concatenate([position, rpy, gripper], axis=-1)


class _RobodojoOutputs(transforms.DataTransformFn):

    def _from_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[..., :14]
        return {
            "actions": np.concatenate(
                [
                    self._from_target_pose(actions[..., :7], is_left=True),
                    self._from_target_pose(actions[..., 7:14], is_left=False),
                ],
                axis=-1,
            )
        }


@dataclasses.dataclass(frozen=True)
class RobodojoSimOutputs(_RobodojoOutputs):
    def _from_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        # Formerly target_to_source_pose: convert model-frame actions back to simulator coordinates.
        del is_left
        position = np.einsum("ij,...j->...i", _TARGET_TO_SOURCE, pose[..., :3]) + _ROBOT_BASE_MIDPOINT
        target_rotation = Rotation.from_euler("xyz", pose[..., 3:6].reshape(-1, 3)).as_matrix().reshape(
            *pose.shape[:-1], 3, 3
        )
        source_rotation = np.einsum("ij,...jk,kl->...il", _TARGET_TO_SOURCE, target_rotation, _SOURCE_TO_TARGET)
        xyzw = Rotation.from_matrix(source_rotation.reshape(-1, 3, 3)).as_quat().reshape(*pose.shape[:-1], 4)
        quaternion = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
        return np.concatenate([position, quaternion, pose[..., 6:7]], axis=-1)


@dataclasses.dataclass(frozen=True)
class RobodojoRealOutputs(_RobodojoOutputs):
    arm_base_distance: float = 0.600

    def _from_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        # Formerly target_to_local_arm_pose: shift shared-frame actions back to the arm-local origin.
        position = pose[..., :3] - np.array([0.0, _arm_y_offset(self.arm_base_distance, is_left=is_left), 0.0])
        xyzw = Rotation.from_euler("xyz", pose[..., 3:6].reshape(-1, 3)).as_quat().reshape(*pose.shape[:-1], 4)
        quaternion = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
        gripper = np.clip(pose[..., 6:7], 0.0, 1.0)
        return np.concatenate([position, quaternion, gripper], axis=-1)
