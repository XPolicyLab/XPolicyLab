"""GR00T modality config generated for XPolicyLab data."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


xpolicylab_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=['left_arm', 'left_gripper', 'right_arm', 'right_gripper'],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=['left_arm', 'left_gripper', 'right_arm', 'right_gripper'],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(xpolicylab_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
