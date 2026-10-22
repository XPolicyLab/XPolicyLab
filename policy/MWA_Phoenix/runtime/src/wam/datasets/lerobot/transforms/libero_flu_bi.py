"""Opt-in FLU world / identity end-effector convention for LIBERO OSC."""
from __future__ import annotations

import torch

from .libero import LiberoRight80NoFrameTransform, _matrix_to_rotation_6d, _rotation_6d_to_matrix
from .libero_ab import LiberoRight80ABTransform
from ..utils.rotation import axis_angle_to_matrix

LIBERO_FLU_A = ((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.))
LIBERO_FLU_B = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
LIBERO_FLU_BI_CONTRACT = {
    "version": "libero_osc_flu_b_identity_v1",
    "world_from_source": LIBERO_FLU_A,
    "source_ee_from_canonical_ee": LIBERO_FLU_B,
    "controller_scale": (.05, .05, .05, .5, .5, .5),
    "state": "A_p__A_R_B__column_first_rot6d",
    "action": "world_delta_xyz_m__canonical_local_rotvec_rad__gripper_open01",
    "temporal_pairing": "action_t_with_state_t_no_shift",
}


class LiberoRight80FLUBIdentityTransform(LiberoRight80NoFrameTransform):
    rotation_semantics = LIBERO_FLU_BI_CONTRACT["action"]
    _paired_ee = staticmethod(LiberoRight80ABTransform._paired_ee)

    def encode_commands(self, commands, source_ee):
        self._require_last_dim(commands, 7, "LIBERO commands")
        self._require_last_dim(source_ee, 6, "source EE")
        ee = self._paired_ee(commands, source_ee)
        a = commands.new_tensor(LIBERO_FLU_A)
        scaled = commands[..., :6].clamp(-1, 1) * commands.new_tensor(LIBERO_FLU_BI_CONTRACT["controller_scale"])
        r = axis_angle_to_matrix(ee[..., 3:])
        local = (r.transpose(-1, -2) @ scaled[..., 3:, None]).squeeze(-1)
        return torch.cat((scaled[..., :3] @ a.T, local, commands[..., 6:7]), -1)

    def decode_commands(self, canonical, source_ee):
        """Inverse physical conversion only; simulator clipping is separate."""
        self._require_last_dim(canonical, 7, "canonical action")
        self._require_last_dim(source_ee, 6, "source EE")
        ee = self._paired_ee(canonical, source_ee)
        a = canonical.new_tensor(LIBERO_FLU_A)
        r = axis_angle_to_matrix(ee[..., 3:])
        world = (r @ canonical[..., 3:6, None]).squeeze(-1)
        physical = torch.cat((canonical[..., :3] @ a, world), -1)
        return torch.cat((physical / canonical.new_tensor(LIBERO_FLU_BI_CONTRACT["controller_scale"]), canonical[..., 6:7]), -1)

    def decode_simulator_commands(self, canonical, source_ee):
        """Call at each execution step with the latest measured source EE."""
        command = self.decode_commands(canonical, source_ee)
        return torch.cat((command[..., :6].clamp(-1, 1), 1 - 2 * command[..., 6:7].clamp(0, 1)), -1)

    def forward(self, batch):
        ee = batch["state"][self.state_ee_key]
        if "action" in batch:
            batch["action"][self.action_key] = self.encode_commands(batch["action"][self.action_key], ee)
        batch = super().forward(batch)
        a = ee.new_tensor(LIBERO_FLU_A)
        r = a @ axis_angle_to_matrix(ee[..., 3:])
        batch["state"][self.state_ee_key] = torch.cat((ee[..., :3] @ a.T, _matrix_to_rotation_6d(r)), -1)
        return batch

    def backward(self, batch):
        if "state" not in batch:
            raise ValueError("Frame decoding requires paired current state")
        ee = batch["state"][self.state_ee_key]
        self._require_last_dim(ee, 9, "canonical EE")
        a = ee.new_tensor(LIBERO_FLU_A)
        r = a.T @ _rotation_6d_to_matrix(ee[..., 3:])
        batch["state"][self.state_ee_key] = torch.cat((ee[..., :3] @ a, _matrix_to_rotation_6d(r)), -1)
        batch = super().backward(batch)
        if "action" in batch:
            batch["action"][self.action_key] = self.decode_commands(batch["action"][self.action_key], batch["state"][self.state_ee_key])
        return batch
