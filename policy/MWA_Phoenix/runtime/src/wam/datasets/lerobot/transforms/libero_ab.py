"""LIBERO OSC commands and poses in the Intern Franka canonical frame.

Source world axes are +X right, +Y forward, +Z up (user-confirmed).
Commands are scaled world-frame OSC deltas, not measured next-state deltas.
"""

from __future__ import annotations

import torch

from .libero import (
    LiberoRight80NoFrameTransform, _matrix_to_rotation_6d, _rotation_6d_to_matrix,
)
from ..utils.rotation import axis_angle_to_matrix


LIBERO_A = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
INTERN_FRANKA_B = ((0., 0., -1.), (0., 1., 0.), (1., 0., 0.))
LIBERO_AB_CONTRACT = {
    "version": "libero_osc_ab_v1",
    "world_from_source": LIBERO_A,
    "source_ee_from_canonical_ee": INTERN_FRANKA_B,
    "controller_scale": (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
    "state": "A_p__A_R_B__column_first_rot6d",
    "action": "world_delta_xyz_m__canonical_local_rotvec_rad__gripper_open01",
    "temporal_pairing": "action_t_with_state_t_no_shift",
}


class LiberoRight80ABTransform(LiberoRight80NoFrameTransform):
    rotation_semantics = LIBERO_AB_CONTRACT["action"]

    @staticmethod
    def _paired_ee(action, ee):
        if action.shape[:-1] == ee.shape[:-1]:
            return ee
        if (action.ndim >= 2 and action.ndim == ee.ndim
                and action.shape[:-2] == ee.shape[:-2]
                and ee.shape[-2] == action.shape[-2] + 1):
            return ee[..., :-1, :]
        raise ValueError("Each action needs its own current EE orientation; "
                         f"cannot broadcast {tuple(ee.shape)} over {tuple(action.shape)}")

    def encode_commands(self, commands, source_ee):
        self._require_last_dim(commands, 7, "LIBERO command")
        self._require_last_dim(source_ee, 6, "source EE")
        ee = self._paired_ee(commands, source_ee)
        a, b = commands.new_tensor(LIBERO_A), commands.new_tensor(INTERN_FRANKA_B)
        scaled = commands[..., :6].clamp(-1, 1) * commands.new_tensor(
            LIBERO_AB_CONTRACT["controller_scale"])
        r = axis_angle_to_matrix(ee[..., 3:])
        # Log(B.T R.T Exp(w_world) R B) = B.T R.T w_world, since |w| < pi.
        local = (b.T @ r.transpose(-1, -2) @ scaled[..., 3:, None]).squeeze(-1)
        return torch.cat((scaled[..., :3] @ a.T, local, commands[..., 6:7]), -1)

    def decode_commands(self, canonical, source_ee):
        self._require_last_dim(canonical, 7, "canonical action")
        self._require_last_dim(source_ee, 6, "source EE")
        ee = self._paired_ee(canonical, source_ee)
        a, b = canonical.new_tensor(LIBERO_A), canonical.new_tensor(INTERN_FRANKA_B)
        r = axis_angle_to_matrix(ee[..., 3:])
        world = (r @ b @ canonical[..., 3:6, None]).squeeze(-1)
        physical = torch.cat((canonical[..., :3] @ a, world), -1)
        command = physical / canonical.new_tensor(LIBERO_AB_CONTRACT["controller_scale"])
        return torch.cat((command, canonical[..., 6:7]), -1)

    def forward(self, batch):
        ee = batch["state"][self.state_ee_key]
        self._require_last_dim(ee, 6, "source EE")
        if "action" in batch:
            batch["action"][self.action_key] = self.encode_commands(
                batch["action"][self.action_key], ee)
        batch = super().forward(batch)
        a, b = ee.new_tensor(LIBERO_A), ee.new_tensor(INTERN_FRANKA_B)
        r = a @ axis_angle_to_matrix(ee[..., 3:]) @ b
        batch["state"][self.state_ee_key] = torch.cat(
            (ee[..., :3] @ a.T, _matrix_to_rotation_6d(r)), -1)
        return batch

    def backward(self, batch):
        if "state" not in batch:
            raise ValueError("LIBERO AB decoding requires the current source EE state")
        ee = batch["state"][self.state_ee_key]
        self._require_last_dim(ee, 9, "canonical EE")
        a, b = ee.new_tensor(LIBERO_A), ee.new_tensor(INTERN_FRANKA_B)
        r = a.T @ _rotation_6d_to_matrix(ee[..., 3:]) @ b.T
        # The inherited inverse only changes the representation and gripper.
        batch["state"][self.state_ee_key] = torch.cat(
            (ee[..., :3] @ a, _matrix_to_rotation_6d(r)), -1)
        batch = super().backward(batch)
        if "action" in batch:
            batch["action"][self.action_key] = self.decode_commands(
                batch["action"][self.action_key], batch["state"][self.state_ee_key])
        return batch
