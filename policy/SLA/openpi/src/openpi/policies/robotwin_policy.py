import dataclasses
import functools
import math
from pathlib import Path
from typing import ClassVar

import einops
import numpy as np
import pinocchio as pin

from openpi import transforms

_ARM_JOINT_DIM = 6
_ARM_STATE_DIM = 7
_ROBOTWIN_URDF_PATH = Path(__file__).resolve().parents[3] / "assets" / "embodiedments" / "arx5_description_isaac.urdf"
_LEFT_ARM_JOINTS = ("fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6")
_RIGHT_ARM_JOINTS = ("fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6")
_LEFT_ROOT_JOINT = "fl_joint1"
_RIGHT_ROOT_JOINT = "fr_joint1"
_LEFT_EE_JOINT = "fl_joint6"
_RIGHT_EE_JOINT = "fr_joint6"
_ROBOTWIN_EE_FRAME = "ee"
_ROBOTWIN_EE_AXIS_ALIGNMENT = np.eye(3)
_CAMERA_PARAM_ZEROS = {
    "extrinsics": np.zeros((3, 4), dtype=np.float32),
    "intrinsics": np.zeros((3, 3), dtype=np.float32),
}


def make_robotwin_example() -> dict:
    """Creates a random input example for the robotwin policy."""
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


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    """Inputs for the Robotwin policy.

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

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }
        camera_name_map = {
            "base_0_rgb": "cam_high",
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_
            camera_name_map[dest] = source

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": joint_state_to_eef_state(data["state"]) if self.use_eef_action else data["state"],
        }

        # camera parameters are only available during training.
        if "camera_param" in data:
            camera_param = {}
            camera_param_mask = {}
            for dest, source in camera_name_map.items():
                if f"{source}_extrinsics" in data["camera_param"] and f"{source}_intrinsics" in data["camera_param"]:
                    camera_param[f"{dest}_extrinsics"] = np.asarray(data["camera_param"][f"{source}_extrinsics"])
                    camera_param[f"{dest}_intrinsics"] = np.asarray(data["camera_param"][f"{source}_intrinsics"])
                    camera_param_mask[dest] = np.True_
            inputs["camera_param"] = camera_param
            inputs["camera_param_mask"] = camera_param_mask

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = joint_actions_to_eef_actions(actions) if self.use_eef_action else actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    """Outputs for the Robotwin policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    # Use EEF instead of joint as action and state representations.
    use_eef_action: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


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
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
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


class RobotwinPinocchioKinematics:
    def __init__(self, urdf_path: Path, *, arm_joint_names: tuple[str, ...], ee_joint_name: str):
        model = pin.buildModelFromUrdf(str(urdf_path))
        reduced_joint_ids = [model.getJointId(joint_name) for joint_name in model.names[1:] if joint_name not in arm_joint_names]
        self.model = pin.buildReducedModel(
            model,
            reduced_joint_ids,
            np.zeros(model.nq),
        )
        ee_joint_id = self.model.getJointId(ee_joint_name)
        ee_parent_frame_id = self.model.getFrameId(ee_joint_name)
        self.ee_frame_id = self.model.addFrame(
            pin.Frame(
                _ROBOTWIN_EE_FRAME,
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


def _align_robotwin_eef_rotation(rotation: np.ndarray) -> np.ndarray:
    # Use the raw joint6 frame convention. At zero joints this is approximately
    # +x forward, +y left, +z up in the Robotwin base/world frame.
    return rotation @ _ROBOTWIN_EE_AXIS_ALIGNMENT


@functools.lru_cache(maxsize=1)
def _robotwin_common_eef_origin() -> np.ndarray:
    # Common frame origin: midpoint between the two root revolute joints.
    model = pin.buildModelFromUrdf(str(_ROBOTWIN_URDF_PATH))
    data = model.createData()
    pin.framesForwardKinematics(model, data, np.zeros(model.nq))
    left_root = np.asarray(data.oMf[model.getFrameId(_LEFT_ROOT_JOINT)].translation)
    right_root = np.asarray(data.oMf[model.getFrameId(_RIGHT_ROOT_JOINT)].translation)
    return 0.5 * (left_root + right_root)


@functools.lru_cache(maxsize=1)
def _left_arm_kinematics() -> RobotwinPinocchioKinematics:
    return RobotwinPinocchioKinematics(
        _ROBOTWIN_URDF_PATH,
        arm_joint_names=_LEFT_ARM_JOINTS,
        ee_joint_name=_LEFT_EE_JOINT,
    )


@functools.lru_cache(maxsize=1)
def _right_arm_kinematics() -> RobotwinPinocchioKinematics:
    return RobotwinPinocchioKinematics(
        _ROBOTWIN_URDF_PATH,
        arm_joint_names=_RIGHT_ARM_JOINTS,
        ee_joint_name=_RIGHT_EE_JOINT,
    )


def _joint_arm_to_eef_arm(arm_state: np.ndarray, *, kinematics: RobotwinPinocchioKinematics) -> np.ndarray:
    eef_se3 = kinematics.forward_kinematics_se3(arm_state[:_ARM_JOINT_DIM])
    aligned_eef_se3 = pin.SE3(
        _align_robotwin_eef_rotation(np.asarray(eef_se3.rotation)),
        np.asarray(eef_se3.translation) - _robotwin_common_eef_origin(),
    )
    eef_pose = kinematics.se3_to_pose(aligned_eef_se3)
    return np.concatenate([eef_pose, arm_state[[_ARM_JOINT_DIM]]])


def joint_state_to_eef_state(state: np.ndarray) -> np.ndarray:
    left = _joint_arm_to_eef_arm(state[:_ARM_STATE_DIM], kinematics=_left_arm_kinematics())
    right = _joint_arm_to_eef_arm(state[_ARM_STATE_DIM:], kinematics=_right_arm_kinematics())
    return np.concatenate([left, right])


def joint_actions_to_eef_actions(actions: np.ndarray) -> np.ndarray:
    return np.stack([joint_state_to_eef_state(action) for action in actions], axis=0)
