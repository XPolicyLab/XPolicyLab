import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms

_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# Rotation from the simulator world frame (source) to the policy frame (target).
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


def source_to_target_pose(pose: np.ndarray) -> np.ndarray:
    """Simulation pose [xyz, wxyz, gripper] in the world frame -> [xyz, rpy, gripper] in the policy frame.

    Only the world frame is rebased (the rotation is left-multiplied), which is the correct transform for an
    EEF pose matrix (tool -> world); the tool's own axes stay as they are.
    """
    position = np.einsum("ij,...j->...i", _SOURCE_TO_TARGET, pose[..., :3] - _ROBOT_BASE_MIDPOINT)
    xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1)
    source_rotation = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    target_rotation = np.einsum("ij,...jk->...ik", _SOURCE_TO_TARGET, source_rotation)
    rpy = Rotation.from_matrix(target_rotation.reshape(-1, 3, 3)).as_euler("xyz").reshape(*pose.shape[:-1], 3)
    return np.concatenate([position, rpy, pose[..., 7:8]], axis=-1)


def target_to_source_pose(pose: np.ndarray) -> np.ndarray:
    """Inverse of `source_to_target_pose`."""
    position = np.einsum("ij,...j->...i", _TARGET_TO_SOURCE, pose[..., :3]) + _ROBOT_BASE_MIDPOINT
    target_rotation = Rotation.from_euler(
        "xyz",
        pose[..., 3:6].reshape(-1, 3)
    ).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    source_rotation = np.einsum("ij,...jk->...ik", _TARGET_TO_SOURCE, target_rotation)
    xyzw = Rotation.from_matrix(source_rotation.reshape(-1, 3, 3)).as_quat().reshape(*pose.shape[:-1], 4)
    quaternion = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
    return np.concatenate([position, quaternion, pose[..., 6:7]], axis=-1)


def _arm_y_offset(base_distance: float, *, is_left: bool) -> float:
    if base_distance < 0:
        raise ValueError(f"Arm base distance must be non-negative, got {base_distance}.")
    return (0.5 if is_left else -0.5) * base_distance


def local_arm_to_target_pose(pose: np.ndarray, *, is_left: bool, arm_base_distance: float) -> np.ndarray:
    """Merge the two arm-local origins into one robot frame; their axes already match."""
    position = pose[..., :3] + np.array([0.0, _arm_y_offset(arm_base_distance, is_left=is_left), 0.0])
    xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1)
    rpy = Rotation.from_quat(xyzw.reshape(-1, 4)).as_euler("xyz").reshape(*pose.shape[:-1], 3)
    gripper = np.clip(pose[..., 7:8], 0.0, 1.0)
    return np.concatenate([position, rpy, gripper], axis=-1)


def target_to_local_arm_pose(pose: np.ndarray, *, is_left: bool, arm_base_distance: float) -> np.ndarray:
    """Inverse of `local_arm_to_target_pose`: move each arm back to its own base origin."""
    position = pose[..., :3] - np.array([0.0, _arm_y_offset(arm_base_distance, is_left=is_left), 0.0])
    xyzw = Rotation.from_euler("xyz", pose[..., 3:6].reshape(-1, 3)).as_quat().reshape(*pose.shape[:-1], 4)
    quaternion = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
    gripper = np.clip(pose[..., 6:7], 0.0, 1.0)
    return np.concatenate([position, quaternion, gripper], axis=-1)


@dataclasses.dataclass(frozen=True)
class RobodojoInputs(transforms.DataTransformFn):
    # Distance between the two arm bases in meters. None means simulation data, where both arms share one world
    # frame (`source_to_target_pose`). A value means real-robot data, where each arm's pose is relative to its
    # own base (`local_arm_to_target_pose`).
    arm_base_distance: float | None = None

    def _to_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        if self.arm_base_distance is None:
            return source_to_target_pose(pose)
        return local_arm_to_target_pose(pose, is_left=is_left, arm_base_distance=self.arm_base_distance)

    def __call__(self, data: dict) -> dict:
        # `delta_timestamps` stacks [*extra frames, current frame] onto every camera it covers.
        # Extra frames come out camera-major: all of cam_high's, then cam_left_wrist's, etc.
        extra_images = []
        current = {}
        for name in _CAMERAS:
            frames = np.asarray(data["images"][name])
            if frames.ndim == 4:
                extra_images += [_parse_image(frame) for frame in frames[:-1]]
                frames = frames[-1]
            current[name] = _parse_image(frames)

        images = {
            "base_0_rgb": current["cam_high"],
            "left_wrist_0_rgb": current["cam_left_wrist"],
            "right_wrist_0_rgb": current["cam_right_wrist"],
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

        if extra_images:
            inputs["extra_images"] = np.stack(extra_images)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobodojoOutputs(transforms.DataTransformFn):
    # Must match the `RobodojoInputs` the checkpoint was trained with.
    arm_base_distance: float | None = None

    def _from_target_pose(self, pose: np.ndarray, *, is_left: bool) -> np.ndarray:
        if self.arm_base_distance is None:
            return target_to_source_pose(pose)
        return target_to_local_arm_pose(pose, is_left=is_left, arm_base_distance=self.arm_base_distance)

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
