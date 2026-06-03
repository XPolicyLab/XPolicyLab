# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from typing import Optional
from BeingH.utils.schema import RotationType
from BeingH.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from BeingH.dataset.transform.concat import ConcatTransform
from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from BeingH.utils.constants import TARGET_STATE_ROTATION_TYPE, TARGET_ACTION_ROTATION_TYPE, TARGET_STATE_ROTATION_DIM, TARGET_ACTION_ROTATION_DIM, AGIBOT_ABS_OR_RELA


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column (exclusive)")
    absolute: bool = True

    rotation_type: Optional[RotationType] = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = Field(True, description="Whether the data is continuous (floating point)")


class BaseDataConfig(ABC):
    def __init__(self, embodiment_tag, use_fixed_view, max_view_num, 
                obs_indices=[0], action_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices
        self.action_indices = action_indices

    @abstractmethod
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """
        Define how to extract and name new modalities from raw Parquet columns.
        Returns: {'modality.key': ModalityDef(...), ...}
        """
        pass

    def get_sampling_indices(self) -> Dict[str, List[int]]:
        """Define sampling indices"""
        sampling_map = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        """
        Define a complete, ordered data transformation pipeline.
        Returns a ComposedModalityTransform object.
        """
        pass

    def add_video_modality(self, modalities):
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
            # rand_view_num = random.randint(1, len(self.VIDEO_SOURCE_COLUMNS))
            # video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), rand_view_num)
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), max_view_num)
   
        for video_key in video_keys:
            modalities[video_key] = ModalityDef(source_column=self.VIDEO_SOURCE_COLUMNS[video_key], start=0, end=0)

        return modalities


class LiberoOriginDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.top_view']
    VIDEO_SOURCE_COLUMNS = {'video.top_view': 'observation.images.image'}
    STATE_KEYS = ['state.state']
    ACTION_KEYS = ['action.action']

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {'state.state': 'min_max'} 
    action_normalization_modes = {'action.action': 'min_max'}

    state_action_type = {'state.state': "7-d absolute state (xyz,roll,pitch,yaw,pad) + 1-d gripper pos", 
                         'action.action': "6-d relative action (xyz,roll,pitch,yaw) + 1-d gripper pos"
                        }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),
            'state.state': ModalityDef(source_column='observation.state', start=0, end=8),
            'action.action': ModalityDef(source_column='action', start=0, end=7, absolute=False),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoNoNormDataConfig(LiberoOriginDataConfig):
    VIDEO_KEYS = ['video.top_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.top_view': 'observation.images.image',
        'video.wrist_view': 'observation.images.wrist_image',
    }
    STATE_KEYS = ['state.eef_position', 'state.eef_rotation', 'state.libero_gripper_position']
    ACTION_KEYS = ['action.eef_position', 'action.eef_rotation', 'action.gripper_position']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':     (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.libero_gripper_position': (44, 46),

        'action.eef_position':    (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position':(18, 19),
    }

    state_normalization_modes = {
    }
    
    action_normalization_modes = {
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.libero_gripper_position': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='observation.state', start=3, end=6, rotation_type="axis_angle"),
            'state.libero_gripper_position': ModalityDef(source_column='observation.state', start=6, end=8),

            'action.eef_position': ModalityDef(source_column='action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='action', start=6, end=7),
        }
        modalities = self.add_video_modality(modalities)

        return modalities


class RobocasaHumanDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.left_view', 'video.right_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.left_view': 'observation.images.left_view',
        'video.right_view': 'observation.images.right_view',
        'video.wrist_view': 'observation.images.wrist_view',
    }
    STATE_KEYS = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    ACTION_KEYS = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
        "action.base_motion",
        "action.control_mode",
    ]

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':  (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.gripper_qpos': (44, 46),
        'state.base_position': (70, 73),
        'state.base_rotation': (73, 76),

        'action.eef_position': (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position': (18, 19),
        'action.base_motion': (70, 74),
        'action.control_mode': (74, 75),
    }

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {} 
    # action_normalization_modes = {}

    action_normalization_modes = {
        # "action.end_effector_position": "min_max",
        # "action.end_effector_rotation": "min_max",
        "action.gripper_position": "binary",
        # "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.gripper_qpos': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='world_abs_state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='world_abs_state', start=3, end=6, rotation_type="axis_angle"),
            'state.gripper_qpos': ModalityDef(source_column='world_abs_state', start=6, end=8),
            'state.base_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.base_rotation': ModalityDef(source_column='observation.state', start=3, end=7, rotation_type="quaternion"),

            'action.eef_position': ModalityDef(source_column='world_delta_action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='world_delta_action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='world_delta_action', start=6, end=7),
            'action.base_motion': ModalityDef(source_column='action', start=7, end=11, absolute=False),
            'action.control_mode': ModalityDef(source_column='action', start=11, end=12),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={
                    # "state.eef_rotation": TARGET_STATE_ROTATION_TYPE,
                    "state.base_rotation": TARGET_STATE_ROTATION_TYPE
                },
                # normalization_modes=self.action_normalization_modes,
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                # target_rotations={"action.eef_rotation": TARGET_ACTION_ROTATION_TYPE},
                normalization_modes=self.action_normalization_modes,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class RoboTwinQposDataConfig(BaseDataConfig):
    """
    Data config for RoboTwin with aloha-agilex dual-arm robot.

    Action / State: 14-dim absolute qpos
        [left_arm_j1..j6 (0:6), left_gripper (6),
         right_arm_j1..j6 (7:13), right_gripper (13)]

    Unified Action Space mapping:
        Right arm joints   → dims 50-55  (arm_joint_position slot, 6-DoF)
        Right gripper      → dim  18     (gripper_position slot)
        Left  arm joints   → dims 57-62  (left_arm_joint_position slot, 6-DoF)
        Left  gripper      → dim  19     (left_gripper_position slot)
    """

    VIDEO_KEYS = [
        'video.head_view',
        'video.right_wrist_view',
        'video.left_wrist_view',
    ]
    VIDEO_SOURCE_COLUMNS = {
        'video.head_view':        'observation.images.head_camera',
        'video.right_wrist_view': 'observation.images.right_camera',
        'video.left_wrist_view':  'observation.images.left_camera',
    }
    STATE_KEYS = [
        'state.left_arm_joint_position',
        'state.left_gripper_position',
        'state.right_arm_joint_position',
        'state.right_gripper_position',
    ]
    ACTION_KEYS = [
        'action.left_arm_joint_position',
        'action.left_gripper_position',
        'action.right_arm_joint_position',
        'action.right_gripper_position',
    ]
    LANGUAGE_KEYS = ['language.instruction']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        # State
        'state.right_arm_joint_position': (50, 56),
        'state.right_gripper_position':   (18, 19),
        'state.left_arm_joint_position':  (57, 63),
        'state.left_gripper_position':    (19, 20),
        # Action
        'action.right_arm_joint_position': (50, 56),
        'action.right_gripper_position':   (18, 19),
        'action.left_arm_joint_position':  (57, 63),
        'action.left_gripper_position':    (19, 20),
    }

    state_normalization_modes = {}
    action_normalization_modes = {}

    def get_feature_meta(self):
        return {
            'state.left_arm_joint_position': ("6-d absolute left arm joint position", 6),
            'state.left_gripper_position': ("1-d left gripper position", 1),
            'state.right_arm_joint_position': ("6-d absolute right arm joint position", 6),
            'state.right_gripper_position': ("1-d right gripper position", 1),
            'action.left_arm_joint_position': ("6-d absolute left arm joint position", 6),
            'action.left_gripper_position': ("1-d left gripper position", 1),
            'action.right_arm_joint_position': ("6-d absolute right arm joint position", 6),
            'action.right_gripper_position': ("1-d right gripper position", 1),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            'language.instruction': ModalityDef(
                source_column='task_index', start=0, end=0),

            # State: sliced from 'observation.state' (14-dim)
            'state.left_arm_joint_position':  ModalityDef(
                source_column='observation.state', start=0, end=6),
            'state.left_gripper_position':    ModalityDef(
                source_column='observation.state', start=6, end=7),
            'state.right_arm_joint_position': ModalityDef(
                source_column='observation.state', start=7, end=13),
            'state.right_gripper_position':   ModalityDef(
                source_column='observation.state', start=13, end=14),

            # Action: sliced from 'action' (14-dim)
            'action.left_arm_joint_position':  ModalityDef(
                source_column='action', start=0, end=6, absolute=True),
            'action.left_gripper_position':    ModalityDef(
                source_column='action', start=6, end=7, absolute=True),
            'action.right_arm_joint_position': ModalityDef(
                source_column='action', start=7, end=13, absolute=True),
            'action.right_gripper_position':   ModalityDef(
                source_column='action', start=13, end=14, absolute=True),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboDojoQposDataConfig(RoboTwinQposDataConfig):
    """
    RoboDojo / XPolicyLab LeRobot (14-dim joint, cam_high / cam_*_wrist cameras).
    Same qpos layout as RoboTwinQposDataConfig; only video column names differ.
    """

    VIDEO_SOURCE_COLUMNS = {
        'video.head_view': 'observation.images.cam_high',
        'video.right_wrist_view': 'observation.images.cam_right_wrist',
        'video.left_wrist_view': 'observation.images.cam_left_wrist',
    }


class RoboTwinFrankaQposDataConfig(BaseDataConfig):
    """
    Data config for RoboTwin with franka-panda dual-arm robot.

    Action / State: 16-dim absolute qpos
        [left_arm_j1..j7 (0:7), left_gripper (7),
         right_arm_j1..j7 (8:15), right_gripper (15)]

    Unified Action Space mapping:
        Right arm joints   → dims 50-56  (arm_joint_position slot, 7-DoF)
        Right gripper      → dim  18     (gripper_position slot)
        Left  arm joints   → dims 57-63  (left_arm_joint_position slot, 7-DoF)
        Left  gripper      → dim  19     (left_gripper_position slot)
    """

    VIDEO_KEYS = [
        'video.head_view',
        'video.right_wrist_view',
        'video.left_wrist_view',
    ]
    VIDEO_SOURCE_COLUMNS = {
        'video.head_view':        'observation.images.head_camera',
        'video.right_wrist_view': 'observation.images.right_camera',
        'video.left_wrist_view':  'observation.images.left_camera',
    }
    STATE_KEYS = [
        'state.left_arm_joint_position',
        'state.left_gripper_position',
        'state.right_arm_joint_position',
        'state.right_gripper_position',
    ]
    ACTION_KEYS = [
        'action.left_arm_joint_position',
        'action.left_gripper_position',
        'action.right_arm_joint_position',
        'action.right_gripper_position',
    ]
    LANGUAGE_KEYS = ['language.instruction']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        # State
        'state.right_arm_joint_position': (50, 57),
        'state.right_gripper_position':   (18, 19),
        'state.left_arm_joint_position':  (57, 64),
        'state.left_gripper_position':    (19, 20),
        # Action
        'action.right_arm_joint_position': (50, 57),
        'action.right_gripper_position':   (18, 19),
        'action.left_arm_joint_position':  (57, 64),
        'action.left_gripper_position':    (19, 20),
    }

    state_normalization_modes = {}
    action_normalization_modes = {}

    def get_feature_meta(self):
        return {
            'state.left_arm_joint_position': ("7-d absolute left arm joint position", 7),
            'state.left_gripper_position': ("1-d left gripper position", 1),
            'state.right_arm_joint_position': ("7-d absolute right arm joint position", 7),
            'state.right_gripper_position': ("1-d right gripper position", 1),
            'action.left_arm_joint_position': ("7-d absolute left arm joint position", 7),
            'action.left_gripper_position': ("1-d left gripper position", 1),
            'action.right_arm_joint_position': ("7-d absolute right arm joint position", 7),
            'action.right_gripper_position': ("1-d right gripper position", 1),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            'language.instruction': ModalityDef(
                source_column='task_index', start=0, end=0),

            # State: sliced from 'observation.state' (16-dim)
            'state.left_arm_joint_position':  ModalityDef(
                source_column='observation.state', start=0, end=7),
            'state.left_gripper_position':    ModalityDef(
                source_column='observation.state', start=7, end=8),
            'state.right_arm_joint_position': ModalityDef(
                source_column='observation.state', start=8, end=15),
            'state.right_gripper_position':   ModalityDef(
                source_column='observation.state', start=15, end=16),

            # Action: sliced from 'action' (16-dim)
            'action.left_arm_joint_position':  ModalityDef(
                source_column='action', start=0, end=7, absolute=True),
            'action.left_gripper_position':    ModalityDef(
                source_column='action', start=7, end=8, absolute=True),
            'action.right_arm_joint_position': ModalityDef(
                source_column='action', start=8, end=15, absolute=True),
            'action.right_gripper_position':   ModalityDef(
                source_column='action', start=15, end=16, absolute=True),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboTwinEEFDataConfig(BaseDataConfig):
    VIDEO_KEYS = [
        'video.head_view',
        'video.right_wrist_view',
        'video.left_wrist_view',
    ]
    VIDEO_SOURCE_COLUMNS = {
        'video.head_view': 'observation.images.head_camera',
        'video.right_wrist_view': 'observation.images.right_camera',
        'video.left_wrist_view': 'observation.images.left_camera',
    }
    STATE_KEYS = [
        'state.left_eef_position',
        'state.left_eef_rotation',
        'state.left_gripper_position',
        'state.right_eef_position',
        'state.right_eef_rotation',
        'state.right_gripper_position',
    ]
    ACTION_KEYS = [
        'action.left_eef_position',
        'action.left_eef_rotation',
        'action.left_gripper_position',
        'action.right_eef_position',
        'action.right_eef_rotation',
        'action.right_gripper_position',
    ]
    LANGUAGE_KEYS = ['language.instruction']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.right_eef_position': (0, 3),
        'state.right_eef_rotation': (3, 6),
        'state.left_eef_position': (7, 10),
        'state.left_eef_rotation': (10, 13),
        'state.right_gripper_position': (18, 19),
        'state.left_gripper_position': (19, 20),

        'action.right_eef_position': (0, 3),
        'action.right_eef_rotation': (3, 6),
        'action.left_eef_position': (7, 10),
        'action.left_eef_rotation': (10, 13),
        'action.right_gripper_position': (18, 19),
        'action.left_gripper_position': (19, 20),
    }

    state_normalization_modes = {
        'state.left_gripper_position': 'binary',
        'state.right_gripper_position': 'binary',
    }
    action_normalization_modes = {
        'action.left_gripper_position': 'binary',
        'action.right_gripper_position': 'binary',
    }

    def get_feature_meta(self):
        return {
            'state.left_eef_position': ("3-d absolute left eef position (xyz)", 3),
            'state.left_eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute left eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
            'state.left_gripper_position': ("1-d left gripper position", 1),
            'state.right_eef_position': ("3-d absolute right eef position (xyz)", 3),
            'state.right_eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute right eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
            'state.right_gripper_position': ("1-d right gripper position", 1),
            'action.left_eef_position': ("3-d absolute left eef position (xyz)", 3),
            'action.left_eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d absolute left eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
            'action.left_gripper_position': ("1-d left gripper position", 1),
            'action.right_eef_position': ("3-d absolute right eef position (xyz)", 3),
            'action.right_eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d absolute right eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
            'action.right_gripper_position': ("1-d right gripper position", 1),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.left_eef_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.left_eef_rotation': ModalityDef(source_column='observation.state', start=3, end=7, rotation_type='quaternion'),
            'state.left_gripper_position': ModalityDef(source_column='observation.state', start=7, end=8),
            'state.right_eef_position': ModalityDef(source_column='observation.state', start=8, end=11),
            'state.right_eef_rotation': ModalityDef(source_column='observation.state', start=11, end=15, rotation_type='quaternion'),
            'state.right_gripper_position': ModalityDef(source_column='observation.state', start=15, end=16),

            'action.left_eef_position': ModalityDef(source_column='action', start=0, end=3, absolute=True),
            'action.left_eef_rotation': ModalityDef(source_column='action', start=3, end=7, absolute=True, rotation_type='quaternion'),
            'action.left_gripper_position': ModalityDef(source_column='action', start=7, end=8, absolute=True),
            'action.right_eef_position': ModalityDef(source_column='action', start=8, end=11, absolute=True),
            'action.right_eef_rotation': ModalityDef(source_column='action', start=11, end=15, absolute=True, rotation_type='quaternion'),
            'action.right_gripper_position': ModalityDef(source_column='action', start=15, end=16, absolute=True),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes,
                target_rotations={
                    'state.left_eef_rotation': TARGET_STATE_ROTATION_TYPE,
                    'state.right_eef_rotation': TARGET_STATE_ROTATION_TYPE,
                },
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes,
                target_rotations={
                    'action.left_eef_rotation': TARGET_ACTION_ROTATION_TYPE,
                    'action.right_eef_rotation': TARGET_ACTION_ROTATION_TYPE,
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "libero_nonorm": LiberoNoNormDataConfig,
    "robocasa_human": RobocasaHumanDataConfig,
    # RoboTwin: aloha-agilex / arx-x5 / piper / ur5 均为 14-dim 双臂 6+1+6+1，共用同一 config
    "robotwin_qpos": RoboTwinQposDataConfig,
    "robotwin_eef": RoboTwinEEFDataConfig,
    "robotwin_qpos_arx": RoboTwinQposDataConfig,
    "robotwin_qpos_piper": RoboTwinQposDataConfig,
    "robotwin_qpos_ur5": RoboTwinQposDataConfig,
    
    "robodojo_qpos": RoboDojoQposDataConfig, # robodojo与RoboTwin一致，所以继承了RoboTwin的config
    # franka-panda: 16-dim 双臂 7+1+7+1
    "robotwin_qpos_franka": RoboTwinFrankaQposDataConfig,
}
