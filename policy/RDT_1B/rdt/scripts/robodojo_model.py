"""RoboDojo deployment model; skips ALOHA gripper rescaling used in agilex_model.py."""

import os

import torch

from scripts.agilex_model import (
    AGILEX_STATE_INDICES,
    RoboticDiffusionTransformerModel as _BaseRoboticDiffusionTransformerModel,
)


class RoboticDiffusionTransformerModel(_BaseRoboticDiffusionTransformerModel):
    def _format_joint_to_state(self, joints):
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state[:, :, AGILEX_STATE_INDICES] = joints
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        return action[:, :, AGILEX_STATE_INDICES]


def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)
    return model
