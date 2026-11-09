import dataclasses
import functools
import math
from pathlib import Path
from typing import ClassVar

import einops
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from openpi import transforms

_ARM_JOINT_DIM = 6
_ARM_STATE_DIM = 7
_PIPER_URDF_PATH = Path(__file__).resolve().parents[3] / "assets" / "embodiedments" / "piper_description.urdf"
_PIPER_LOCKED_JOINTS = ("joint7", "joint8")
_PIPER_EE_JOINT = "joint6"
_PIPER_EE_FRAME = "ee"
_INTERNDATA_ARM_BASE_DISTANCE = 0.612
_INTERNDATA_LIFT2_ARM_BASE_DISTANCE = 0.5


def make_interndata_split_aloha_example() -> dict:
    """Creates a random input example for the Interndata policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class InterndataSplitAlohaInputs(transforms.DataTransformFn):
    """Inputs for the Interndata policy.

    When EEF actions are enabled, the frame origin is midway between the arm bases.
    Its axes are +x forward, +y left, and +z up.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (0 = closed, 0.1 = open).
    unified_gripper: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that the base image always exists.
        base_image = _parse_image(in_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        # Add wrist images, replacing missing cameras with black images.
        for output_key, input_key in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if input_key in in_images:
                images[output_key] = _parse_image(in_images[input_key])
                image_masks[output_key] = np.True_
            else:
                images[output_key] = np.zeros_like(base_image)
                image_masks[output_key] = np.False_

        if self.unified_gripper:
            data["state"][[6, 13]] = 10 * np.clip(data["state"][[6, 13]], 0.0, 0.1)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": joint_state_to_eef_state(data["state"]) if self.use_eef_action else data["state"],
        }

        # Actions are only available during training.
        if "actions_left_joint" in data:
            actions = np.concatenate(
                [
                    data["actions_left_joint"],
                    data["actions_left_gripper"][..., None],
                    data["actions_right_joint"],
                    data["actions_right_gripper"][..., None],
                ],
                axis=-1,
            )
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)

            if self.unified_gripper:
                actions[:, [6, 13]] = 10 * np.clip(actions[:, [6, 13]], 0.0, 0.1)

            inputs["actions"] = joint_actions_to_eef_actions(actions) if self.use_eef_action else actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class InterndataSplitAlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Interndata Split Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (0 = closed, 0.1 = open).
    unified_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        if self.unified_gripper:
            actions[:, [6, 13]] = 0.1 * actions[:, [6, 13]]
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


@dataclasses.dataclass(frozen=True)
class InterndataLift2Inputs(transforms.DataTransformFn):
    """Inputs for lift2 data that already contains EEF robot poses.

    Poses use a shared origin midway between the two arm bases. The axes match
    Split Aloha (+x forward, +y left, +z up).
    """

    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (0 = closed, 0.1 = open).
    unified_gripper: bool = True

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        base_image = _parse_image(in_images["cam_high"])
        left_arm_offset = np.array(
            [0.0, 0.5 * _INTERNDATA_LIFT2_ARM_BASE_DISTANCE, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for output_key, input_key in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if input_key in in_images:
                images[output_key] = _parse_image(in_images[input_key])
                image_masks[output_key] = np.True_
            else:
                images[output_key] = np.zeros_like(base_image)
                image_masks[output_key] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.concatenate(
                [
                    _format_eef_arm(
                        np.asarray(data["state_left_eef"]) + left_arm_offset,
                        data["state_left_gripper"],
                        unified_gripper=self.unified_gripper,
                    ),
                    _format_eef_arm(
                        np.asarray(data["state_right_eef"]) - left_arm_offset,
                        data["state_right_gripper"],
                        unified_gripper=self.unified_gripper,
                    ),
                ],
                axis=-1,
            ),
        }
        if "actions_left_eef" in data:
            inputs["actions"] = np.concatenate(
                [
                    _format_eef_arm(
                        np.asarray(data["actions_left_eef"]) + left_arm_offset,
                        data["actions_left_gripper"],
                        unified_gripper=self.unified_gripper,
                    ),
                    _format_eef_arm(
                        np.asarray(data["actions_right_eef"]) - left_arm_offset,
                        data["actions_right_gripper"],
                        unified_gripper=self.unified_gripper,
                    ),
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class InterndataLift2Outputs(transforms.DataTransformFn):
    unified_gripper: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14]).copy()
        actions[:, 1] -= 0.5 * _INTERNDATA_LIFT2_ARM_BASE_DISTANCE
        actions[:, 8] += 0.5 * _INTERNDATA_LIFT2_ARM_BASE_DISTANCE
        left = _eef_rpy_to_wxyz(actions[:, :6])
        right = _eef_rpy_to_wxyz(actions[:, 7:13])
        left_gripper = actions[:, 6:7]
        right_gripper = actions[:, 13:14]
        if self.unified_gripper:
            left_gripper = 0.1 * np.clip(left_gripper, 0.0, 1.0)
            right_gripper = 0.1 * np.clip(right_gripper, 0.0, 1.0)
        return {"actions": np.concatenate([left, left_gripper, right, right_gripper], axis=-1)}


@dataclasses.dataclass(frozen=True)
class InterndataFrankaInputs(transforms.DataTransformFn):
    """Inputs for single-arm Franka data that already contains an EEF pose."""

    # Convert gripper values to a unified representation (0 = closed, 1 = open). Original values is (0 = closed, 0.1 = open).
    unified_gripper: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["images"]["cam_high"])
        hand_image = _parse_image(data["images"]["cam_hand"])
        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": hand_image,
                "right_wrist_0_rgb": np.zeros_like(hand_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": _format_eef_arm(
                data["state_eef"],
                data["state_gripper"],
                unified_gripper=self.unified_gripper,
            ),
        }
        if "actions_eef" in data:
            inputs["actions"] = _format_eef_arm(
                data["actions_eef"],
                data["actions_gripper"],
                unified_gripper=self.unified_gripper,
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class InterndataFrankaOutputs(transforms.DataTransformFn):
    unified_gripper: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7])
        pose = _eef_rpy_to_wxyz(actions[:, :6])
        gripper = actions[:, 6:7]
        if self.unified_gripper:
            gripper = 0.1 * np.clip(gripper, 0.0, 1.0)
        return {"actions": np.concatenate([pose, gripper], axis=-1)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are n from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.concatenate(
        [
            data["state_left_joint"],
            data["state_left_gripper"][None],
            data["state_right_joint"],
            data["state_right_gripper"][None],
        ],
        axis=-1,
    )
    data["state"] = _decode_state(state, adapt_to_pi=adapt_to_pi)
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions


class PiperPinocchioKinematics:
    def __init__(self, urdf_path: Path):
        model = pin.buildModelFromUrdf(str(urdf_path))
        reduced_joint_ids = [model.getJointId(joint_name) for joint_name in _PIPER_LOCKED_JOINTS]
        self.model = pin.buildReducedModel(
            model,
            reduced_joint_ids,
            np.zeros(model.nq),
        )
        ee_joint_id = self.model.getJointId(_PIPER_EE_JOINT)
        ee_parent_frame_id = self.model.getFrameId(_PIPER_EE_JOINT)
        self.ee_frame_id = self.model.addFrame(
            pin.Frame(
                _PIPER_EE_FRAME,
                ee_joint_id,
                ee_parent_frame_id,
                pin.SE3(np.eye(3), np.zeros(3)),
                pin.FrameType.OP_FRAME,
            )
        )
        self.ee_joint_id = ee_joint_id

    def create_data(self):
        return self.model.createData()

    def se3_to_pose(self, transform: pin.SE3) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(transform.translation),
                _rotation_matrix_to_rpy(np.asarray(transform.rotation)),
            ]
        )

    def forward_kinematics_se3(self, joints: np.ndarray, data=None) -> pin.SE3:
        data = self.create_data() if data is None else data
        pin.framesForwardKinematics(self.model, data, joints)
        placement = data.oMf[self.ee_frame_id]
        return pin.SE3(np.array(placement.rotation), np.array(placement.translation))


def _rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    if rotation[2, 0] < -1 + 1e-4:
        pitch = math.pi / 2
        yaw = 0.0
        roll = math.atan2(rotation[0, 1], rotation[1, 1])
    elif rotation[2, 0] > 1 - 1e-4:
        pitch = -math.pi / 2
        yaw = 0.0
        roll = -math.atan2(rotation[0, 1], rotation[1, 1])
    else:
        pitch = math.atan2(-rotation[2, 0], math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
        yaw = math.atan2(rotation[1, 0] / math.cos(pitch), rotation[0, 0] / math.cos(pitch))
        roll = math.atan2(rotation[2, 1] / math.cos(pitch), rotation[2, 2] / math.cos(pitch))

    angles = np.array([roll, pitch, yaw])
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


@functools.lru_cache(maxsize=1)
def _piper_kinematics() -> PiperPinocchioKinematics:
    return PiperPinocchioKinematics(_PIPER_URDF_PATH)


def _interndata_arm_base_offset(*, is_left: bool) -> np.ndarray:
    # The Piper URDF is single-arm, so FK returns the EEF pose in each arm's
    # own base frame. Shift positions into the common frame whose origin is the
    # midpoint between the two arm bases.
    y_offset = 0.5 * _INTERNDATA_ARM_BASE_DISTANCE
    return np.array([0.0, y_offset if is_left else -y_offset, 0.0])


def _joint_arm_to_eef_arm(arm_state: np.ndarray, *, is_left: bool) -> np.ndarray:
    kinematics = _piper_kinematics()
    eef_se3 = kinematics.forward_kinematics_se3(arm_state[:_ARM_JOINT_DIM])
    eef_se3 = pin.SE3(
        np.asarray(eef_se3.rotation),
        np.asarray(eef_se3.translation) + _interndata_arm_base_offset(is_left=is_left),
    )
    eef_pose = kinematics.se3_to_pose(eef_se3)
    return np.concatenate([eef_pose, arm_state[[_ARM_JOINT_DIM]]])


def joint_state_to_eef_state(state: np.ndarray) -> np.ndarray:
    left = _joint_arm_to_eef_arm(state[:_ARM_STATE_DIM], is_left=True)
    right = _joint_arm_to_eef_arm(state[_ARM_STATE_DIM:], is_left=False)
    return np.concatenate([left, right])


def joint_actions_to_eef_actions(actions: np.ndarray) -> np.ndarray:
    return np.stack([joint_state_to_eef_state(action) for action in actions], axis=0)


def _eef_rpy_to_wxyz(pose: np.ndarray) -> np.ndarray:
    """Convert model pose [xyz, rpy] to robot-pose [xyz, quaternion(wxyz)]."""
    pose = np.asarray(pose)
    if pose.shape[-1] != 6:
        raise ValueError(f"Expected EEF pose with 6 values, got {pose.shape}")
    quat_xyzw = Rotation.from_euler("xyz", pose[..., 3:6].reshape(-1, 3)).as_quat()
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]].reshape((*pose.shape[:-1], 4))
    return np.concatenate([pose[..., :3], quat_wxyz], axis=-1)


def _format_eef_arm(pose: np.ndarray, gripper: np.ndarray, *, unified_gripper: bool) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected EEF pose with 7 values, got {pose.shape}")

    # Dataset poses use [xyz, quaternion(wxyz)]; SciPy expects xyzw.
    quat_xyzw = pose[..., 3:7][..., [1, 2, 3, 0]]
    rpy = Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_euler("xyz").reshape((*pose.shape[:-1], 3))

    gripper = np.asarray(gripper)
    if unified_gripper:
        gripper = 10.0 * np.clip(gripper, 0.0, 0.1)
    if gripper.ndim and gripper.shape[-1] == 1:
        gripper = gripper[..., 0]
    return np.concatenate([pose[..., :3], rpy, gripper[..., None]], axis=-1)
