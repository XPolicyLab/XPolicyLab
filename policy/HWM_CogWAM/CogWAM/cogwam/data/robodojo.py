"""RoboDojo simulation data registration.

The released LeRobot v2.1 dataset stores one 14-D vector in the order
``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.  Keep that
order explicit here: other dual-arm ARX-X5 configs concatenate the same groups
in a different order and therefore must not be reused.

Upstream discovered this file through an ``importlib`` scan over
``examples/*/train_files/data_registry/``, merging whatever benchmarks happened
to be present into three mutable global registries.  This release reproduces a
single recipe, so the registries are hard-wired here and imported directly.
"""

from __future__ import annotations

from typing import ClassVar

from cogwam.data.lerobot.datasets import ModalityConfig
from cogwam.data.lerobot.embodiment_tags import EmbodimentTag
from cogwam.data.lerobot.transform.base import ComposedModalityTransform
from cogwam.data.lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboDojoArxX5DataConfig:
    """Three source views, one tri-view composite, state, and a 16-step chunk."""

    embodiment_tag: ClassVar[EmbodimentTag] = EmbodimentTag.NEW_EMBODIMENT

    video_keys: ClassVar[list[str]] = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys: ClassVar[list[str]] = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys: ClassVar[list[str]] = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    state_key_dims: ClassVar[dict[str, int]] = {
        "state.left_joints": 6,
        "state.left_gripper": 1,
        "state.right_joints": 6,
        "state.right_gripper": 1,
    }
    action_key_dims: ClassVar[dict[str, int]] = {
        "action.left_joints": 6,
        "action.left_gripper": 1,
        "action.right_joints": 6,
        "action.right_gripper": 1,
    }
    modality_key_ranges: ClassVar[dict[str, dict[str, tuple[int, int]]]] = {
        "state": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
        "action": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
    }
    language_keys: ClassVar[list[str]] = ["annotation.human.action.task_description"]
    observation_indices: ClassVar[list[int]] = [0]
    action_indices: ClassVar[list[int]] = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # Match the data ABI exactly: normalize every continuous state and
        # action component with (x - mean) / (std + 1e-8), clipped to [-5, 5].
        # The simulated grippers remain continuous; do not threshold them as
        # binary joints.  The policy server reuses this same transform at
        # inference, so action denormalization stays checkpoint-consistent.
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "fastwam_zscore" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "fastwam_zscore" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "robodojo_arx_x5": RoboDojoArxX5DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    robot_type: getattr(config, "embodiment_tag", EmbodimentTag.NEW_EMBODIMENT)
    for robot_type, config in ROBOT_TYPE_CONFIG_MAP.items()
}

DATASET_NAMED_MIXTURES = {
    # Every row in the language-v2 dataset has both planning annotations.
    # Keep this pointed at language_v2: changing the alias changes the sampled
    # trajectories and the dataset statistics even when text loss is disabled,
    # so it would no longer reproduce this checkpoint family.
    "robodojo_v21_language": [
        ("RoboDojo_lerobot_v21_language_v2", 1.0, "robodojo_arx_x5"),
    ],
}


__all__ = [
    "DATASET_NAMED_MIXTURES",
    "ROBOT_TYPE_CONFIG_MAP",
    "ROBOT_TYPE_TO_EMBODIMENT_TAG",
    "RoboDojoArxX5DataConfig",
]
