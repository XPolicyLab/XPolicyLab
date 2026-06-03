# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from ABot.dataloader.gr00t_lerobot.datasets import ModalityConfig
from ABot.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from ABot.dataloader.gr00t_lerobot.transform.concat import (
    ConcatTransform,
    ConcatStateActionTransform,
    ConcatDeltaChunkTransform,
)
from ABot.dataloader.gr00t_lerobot.transform.padding import (
    BimanualPadTransform,
    BimanualPadAndGripperPadTransform,
)
from ABot.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
    StateActionDeltaTransform,
    StateActionDeltaOxeAugeTransform,
)
from ABot.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass

class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image_compress",
        "video.wrist_image_compress",
    ]
    
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(10))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                "action.x": "min_max",
                "action.y": "min_max",
                "action.z": "min_max",
                "action.roll": "min_max",
                "action.pitch": "min_max",
                "action.yaw": "min_max",
            },
        ),

            ConcatStateActionTransform(
            state_concat_order=self.state_keys,
            action_concat_order=self.action_keys,
            ),

            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class AgilexDataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",  
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class OXESingleCameraDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
    ]
    state_keys = [
        "state.single_arm_eef_position",
        "state.single_arm_eef_orientation",
        "state.single_arm_gripper_range"
    ]
    action_keys = [
        "action.single_arm_eef_position",
        "action.single_arm_eef_orientation",
        "action.single_arm_gripper_mode"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(17))
    # rotation_delta_specs = "euler_delta_sub"
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="cubic"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.0,
            ),
            VideoToNumpy(apply_to=self.video_keys),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"single_arm_eef_position"},
                rotation_keys={"single_arm_eef_orientation"},
                no_delta_keys={"single_arm_gripper_mode"},
                no_delta_align="drop_last",
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.single_arm_eef_position": "q99",
                    "state.single_arm_eef_orientation": "q99",
                    "state.single_arm_gripper_range": "binary",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.single_arm_eef_position": "q99",
                    "action.single_arm_eef_orientation": "q99",
                    "action.single_arm_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),

        ]

        return ComposedModalityTransform(transforms=transforms)

class OXEDualCameraDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_single_wrist_rgb_compress"
    ]
    state_keys = [
        "state.single_arm_eef_position",
        "state.single_arm_eef_orientation",
        "state.single_arm_gripper_range"
    ]
    action_keys = [
        "action.single_arm_eef_position",
        "action.single_arm_eef_orientation",
        "action.single_arm_gripper_mode"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(17))
    # rotation_delta_specs = "euler_delta_sub"
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"single_arm_eef_position"},
                rotation_keys={"single_arm_eef_orientation"},
                no_delta_keys={"single_arm_gripper_mode"},
                no_delta_align="drop_last",
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.single_arm_eef_position": "q99",
                    "state.single_arm_eef_orientation": "q99",
                    "state.single_arm_gripper_range": "binary",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.single_arm_eef_position": "q99",
                    "action.single_arm_eef_orientation": "q99",
                    "action.single_arm_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),    
        ]

        return ComposedModalityTransform(transforms=transforms)

class OxeAugeDataConfig:
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(17))
    # rotation_delta_specs = "euler_delta_sub"
    rotation_delta_specs = "axis_angle_delta_rel"

    def __init__(self, robot_key=None):
        if robot_key == "original":
            self.robot_type = ""
        else:
            self.robot_type = f"{robot_key}_"
        
        self.video_keys = [f"video.{self.robot_type}cam_high_rgb_compress"]
        self.state_keys = [f"state.{self.robot_type}single_arm_eef_position", f"state.{self.robot_type}single_arm_eef_orientation"]
        self.action_keys = [f"action.{self.robot_type}single_arm_eef_position", f"action.{self.robot_type}single_arm_eef_orientation"]


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={f"{self.robot_type}single_arm_eef_position"},
                rotation_keys={f"{self.robot_type}single_arm_eef_orientation"},
                rotation_delta_specs=self.rotation_delta_specs,
                no_delta_keys=set(),
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaOxeAugeTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    f"state.{self.robot_type}single_arm_eef_position": "q99",
                    f"state.{self.robot_type}single_arm_eef_orientation": "min_max",
                },
                target_rotations={
                    f"state.{self.robot_type}single_arm_eef_orientation": "axis_angle",
                },
            ),

            StateActionDeltaOxeAugeTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    f"action.{self.robot_type}single_arm_eef_position": "q99",
                    f"action.{self.robot_type}single_arm_eef_orientation": "q99",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadAndGripperPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
                gripper_pad_value_state=-1.0,
                gripper_pad_value_action=-1.0,
            )

        ]

        return ComposedModalityTransform(transforms=transforms)

class RobocoinDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_left_wrist_rgb_compress",
        "video.cam_right_wrist_rgb_compress",
    ]
    state_keys = [
        "state.left_arm_eef_position",
        "state.left_arm_eef_orientation",
        "state.left_gripper_range",
        "state.right_arm_eef_position",
        "state.right_arm_eef_orientation",
        "state.right_gripper_range",
    ]

    action_keys = [
        "action.left_arm_eef_position",
        "action.left_arm_eef_orientation",
        "action.left_gripper_mode",
        "action.right_arm_eef_position",
        "action.right_arm_eef_orientation",
        "action.right_gripper_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"left_arm_eef_position", "right_arm_eef_position"},
                rotation_keys={"left_arm_eef_orientation", "right_arm_eef_orientation"},
                no_delta_keys={"left_gripper_mode", "right_gripper_mode"},
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_eef_position": "q99",
                    "state.left_arm_eef_orientation": "q99",
                    "state.left_gripper_range": "q99",
                    "state.right_arm_eef_position": "q99",
                    "state.right_arm_eef_orientation": "q99",
                    "state.right_gripper_range": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.left_arm_eef_position": "q99",
                    "action.left_arm_eef_orientation": "q99",
                    "action.left_gripper_mode": "binary",
                    "action.right_arm_eef_position": "q99",
                    "action.right_arm_eef_orientation": "q99",
                    "action.right_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class RobocoinDexterousDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_left_wrist_rgb_compress",
        "video.cam_right_wrist_rgb_compress",
    ]
    state_keys = [
        "state.left_arm_eef_position",
        "state.left_arm_eef_orientation",
        "state.right_arm_eef_position",
        "state.right_arm_eef_orientation",
    ]

    action_keys = [
        "action.left_arm_eef_position",
        "action.left_arm_eef_orientation",
        "action.right_arm_eef_position",
        "action.right_arm_eef_orientation",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"left_arm_eef_position", "right_arm_eef_position"},
                rotation_keys={"left_arm_eef_orientation", "right_arm_eef_orientation"},
                no_delta_keys=set(),
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_eef_position": "q99",
                    "state.left_arm_eef_orientation": "q99",
                    "state.right_arm_eef_position": "q99",
                    "state.right_arm_eef_orientation": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.left_arm_eef_position": "q99",
                    "action.left_arm_eef_orientation": "q99",
                    "action.right_arm_eef_position": "q99",
                    "action.right_arm_eef_orientation": "q99",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadAndGripperPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
                gripper_pad_value_state=-1.0,
                gripper_pad_value_action=-1.0,
            )
        ]

        return ComposedModalityTransform(transforms=transforms)

class AgibotWorldDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_left_wrist_rgb_compress",
        "video.cam_right_wrist_rgb_compress",
    ]
    state_keys = [
        "state.left_arm_eef_position",
        "state.left_arm_eef_orientation",
        "state.left_gripper_range",
        "state.right_arm_eef_position",
        "state.right_arm_eef_orientation",
        "state.right_gripper_range",
    ]

    action_keys = [
        "action.left_arm_eef_position",
        "action.left_arm_eef_orientation",
        "action.left_gripper_mode",
        "action.right_arm_eef_position",
        "action.right_arm_eef_orientation",
        "action.right_gripper_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"left_arm_eef_position", "right_arm_eef_position"},
                rotation_keys={"left_arm_eef_orientation", "right_arm_eef_orientation"},
                no_delta_keys={"left_gripper_mode", "right_gripper_mode"},
                rotation_delta_specs=self.rotation_delta_specs,
                quat_order_in="xyzw",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_eef_position": "q99",
                    "state.left_arm_eef_orientation": "q99",
                    "state.left_gripper_range": "q99",
                    "state.right_arm_eef_position": "q99",
                    "state.right_arm_eef_orientation": "q99",
                    "state.right_gripper_range": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.left_arm_eef_position": "q99",
                    "action.left_arm_eef_orientation": "q99",
                    "action.left_gripper_mode": "binary",
                    "action.right_arm_eef_position": "q99",
                    "action.right_arm_eef_orientation": "q99",
                    "action.right_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class GalaxeaDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_left_wrist_rgb_compress",
        "video.cam_right_wrist_rgb_compress",
    ]
    state_keys = [
        "state.left_arm_eef_position",
        "state.left_arm_eef_orientation",
        "state.left_gripper_range",
        "state.right_arm_eef_position",
        "state.right_arm_eef_orientation",
        "state.right_gripper_range",
    ]

    action_keys = [
        "action.left_arm_eef_position",
        "action.left_arm_eef_orientation",
        "action.left_gripper_mode",
        "action.right_arm_eef_position",
        "action.right_arm_eef_orientation",
        "action.right_gripper_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"left_arm_eef_position", "right_arm_eef_position"},
                rotation_keys={"left_arm_eef_orientation", "right_arm_eef_orientation"},
                no_delta_keys={"left_gripper_mode", "right_gripper_mode"},
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="xyzw",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_eef_position": "q99",
                    "state.left_arm_eef_orientation": "q99",
                    "state.left_gripper_range": "q99",
                    "state.right_arm_eef_position": "q99",
                    "state.right_arm_eef_orientation": "q99",
                    "state.right_gripper_range": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.left_arm_eef_position": "q99",
                    "action.left_arm_eef_orientation": "q99",
                    "action.left_gripper_mode": "binary",
                    "action.right_arm_eef_position": "q99",
                    "action.right_arm_eef_orientation": "q99",
                    "action.right_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class InternDualDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_left_wrist_rgb_compress",
        "video.cam_right_wrist_rgb_compress",
    ]
    state_keys = [
        "state.left_arm_eef_position",
        "state.left_arm_eef_orientation",
        "state.left_gripper_range",
        "state.right_arm_eef_position",
        "state.right_arm_eef_orientation",
        "state.right_gripper_range",
    ]

    action_keys = [
        "action.left_arm_eef_position",
        "action.left_arm_eef_orientation",
        "action.left_gripper_mode",
        "action.right_arm_eef_position",
        "action.right_arm_eef_orientation",
        "action.right_gripper_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"left_arm_eef_position", "right_arm_eef_position"},
                rotation_keys={"left_arm_eef_orientation", "right_arm_eef_orientation"},
                no_delta_keys={"left_gripper_mode", "right_gripper_mode"},
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_eef_position": "q99",
                    "state.left_arm_eef_orientation": "q99",
                    "state.left_gripper_range": "q99",
                    "state.right_arm_eef_position": "q99",
                    "state.right_arm_eef_orientation": "q99",
                    "state.right_gripper_range": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.left_arm_eef_position": "q99",
                    "action.left_arm_eef_orientation": "q99",
                    "action.left_gripper_mode": "binary",
                    "action.right_arm_eef_position": "q99",
                    "action.right_arm_eef_orientation": "q99",
                    "action.right_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class InternSingleDataConfig:
    video_keys = [
        "video.cam_high_rgb_compress",
        "video.cam_single_wrist_rgb_compress"
    ]
    state_keys = [
        "state.single_arm_eef_position",
        "state.single_arm_eef_orientation",
        "state.single_arm_gripper_range"
    ]
    action_keys = [
        "action.single_arm_eef_position",
        "action.single_arm_eef_orientation",
        "action.single_arm_gripper_mode"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(17)) # action_chunk_size = n - 1 
    rotation_delta_specs = "axis_angle_delta_rel"

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionToTensor(apply_to=self.state_keys),

            ConcatDeltaChunkTransform(
                action_keys=self.action_keys,
                position_keys={"single_arm_eef_position"},
                rotation_keys={"single_arm_eef_orientation"},
                no_delta_keys={"single_arm_gripper_mode"},
                rotation_delta_specs=self.rotation_delta_specs,
                euler_convention="XYZ",
                quat_order_in="wxyz",
            ),

            StateActionDeltaTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.single_arm_eef_position": "q99",
                    "state.single_arm_eef_orientation": "q99",
                    "state.single_arm_gripper_range": "q99",
                },
            ),

            StateActionDeltaTransform(
                apply_to=self.action_keys,
                rotation_delta_specs=self.rotation_delta_specs,
                normalization_modes={
                    "action.single_arm_eef_position": "q99",
                    "action.single_arm_eef_orientation": "q99",
                    "action.single_arm_gripper_mode": "binary",
                },
            ),

            ConcatStateActionTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # padding transforms
            BimanualPadTransform(
                arm_state_dim=7,
                arm_action_dim=7,
                max_state_dim=14,
                max_action_dim=14,
                single_arm_placement="right",
                pad_value_state=0.0,
                pad_value_action=0.0,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

ROBOT_TYPE_CONFIG_MAP = {
    # OXE
    "oxe_franka_austin": OXEDualCameraDataConfig(),
    "oxe_google_robot_bc": OXESingleCameraDataConfig(),
    "oxe_ur5_berkeley": OXEDualCameraDataConfig(),
    "oxe_franka_berkeley": OXEDualCameraDataConfig(),
    "oxe_fanuc_mate_berkeley": OXEDualCameraDataConfig(),
    "oxe_widowx_bridge": OXESingleCameraDataConfig(),
    "oxe_google_robot_fractal": OXESingleCameraDataConfig(),
    "oxe_hello_stretch_cmu": OXESingleCameraDataConfig(),
    "oxe_dlr_edan": OXESingleCameraDataConfig(),
    "oxe_franka_fmb": OXEDualCameraDataConfig(),
    "oxe_franka_furniture": OXEDualCameraDataConfig(),
    "oxe_jaco2": OXEDualCameraDataConfig(),
    "oxe_kuka_iiwa": OXESingleCameraDataConfig(),
    "oxe_xarm_language": OXESingleCameraDataConfig(),
    "oxe_franka_nyu": OXESingleCameraDataConfig(),
    "oxe_franka_stanford": OXEDualCameraDataConfig(),
    "oxe_franka_taco": OXEDualCameraDataConfig(),
    "oxe_franka_droid": OXEDualCameraDataConfig(),

    # RoboCoin
    "AgiBot-g1": RobocoinDataConfig(),
    "alpha_bot_2": RobocoinDataConfig(),
    "Cobot_Magic": RobocoinDataConfig(),
    "Galbot_g1": RobocoinDataConfig(),
    "R1_Lite": RobocoinDataConfig(),
    "RMC-AIDA-L": RobocoinDataConfig(),
    "Split_aloha": RobocoinDataConfig(),
    "Tianqin_A2": RobocoinDataConfig(),

    # RoboCoin Dexterous
    "Airbot_MMK2": RobocoinDexterousDataConfig(),
    "Unitree_G1": RobocoinDexterousDataConfig(),
    "leju_robot": RobocoinDexterousDataConfig(),

    # AgiBotWorld
    "AgiBotWorld-g1": AgibotWorldDataConfig(),

    # Galaxea
    "Galaxea_R1_Lite": GalaxeaDataConfig(),

    # OXE-Auge
    "oxe_auge_original": OxeAugeDataConfig(robot_key="original"),
    "oxe_auge_google_robot": OxeAugeDataConfig(robot_key="google_robot"),
    "oxe_auge_jaco": OxeAugeDataConfig(robot_key="jaco"),
    "oxe_auge_kinova3": OxeAugeDataConfig(robot_key="kinova3"),
    "oxe_auge_kuka_iiwa": OxeAugeDataConfig(robot_key="kuka_iiwa"),
    "oxe_auge_panda": OxeAugeDataConfig(robot_key="panda"),
    "oxe_auge_sawyer": OxeAugeDataConfig(robot_key="sawyer"),
    "oxe_auge_widowX": OxeAugeDataConfig(robot_key="widowX"),
    "oxe_auge_xarm7": OxeAugeDataConfig(robot_key="xarm7"),
    "oxe_auge_ur5e": OxeAugeDataConfig(robot_key="ur5e"),

    # InternData-A1
    "intern_franka": InternSingleDataConfig(),
    "intern_split_aloha": InternDualDataConfig(),
    "intern_lift2": InternDualDataConfig(),
    "intern_genie1": InternDualDataConfig(),

    # Libero
    "libero_franka": Libero4in1DataConfig(),

    # RoboTwin
    "robotwin": AgilexDataConfig(),

    # RoboCase
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
}

