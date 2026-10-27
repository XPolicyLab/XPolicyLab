"""LIBERO adapter for the canonical right-arm 80-D representation.

The 7-D action remains in LIBERO's original world-delta convention. Only the
absolute end-effector state rotation is reparameterized from axis-angle to a
column-first 6-D rotation before scattering into the canonical 80-D slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.rotation import axis_angle_to_matrix, matrix_to_axis_angle


def _matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    if int(rotation_6d.shape[-1]) != 6:
        raise ValueError(
            f"rotation_6d must have last dimension 6, got {tuple(rotation_6d.shape)}"
        )
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    first = F.normalize(first, dim=-1)
    second = F.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class LiberoRight80NoFrameTransform:
    """Map LIBERO fields to canonical slots without a frame conversion."""

    rotation_semantics = "raw_libero_world_delta_no_frame_conversion"

    def __init__(
        self,
        *,
        action_key: str = "default",
        state_joint_key: str = "joint",
        state_ee_key: str = "ee",
        state_gripper_key: str = "gripper",
    ) -> None:
        self.action_key = str(action_key)
        self.state_joint_key = str(state_joint_key)
        self.state_ee_key = str(state_ee_key)
        self.state_gripper_key = str(state_gripper_key)

    @staticmethod
    def _require_last_dim(value: torch.Tensor, dim: int, name: str) -> None:
        if not isinstance(value, torch.Tensor) or int(value.shape[-1]) != int(dim):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"{name} must have last dimension {dim}, got {shape}")

    def forward(self, batch):
        if "action" in batch:
            action = batch["action"][self.action_key]
            self._require_last_dim(action, 7, "LIBERO action")

        state = batch["state"]
        self._require_last_dim(
            state[self.state_joint_key], 7, "LIBERO joint state"
        )
        ee = state[self.state_ee_key]
        self._require_last_dim(ee, 6, "LIBERO EE state")
        state[self.state_ee_key] = torch.cat(
            [ee[..., :3], _matrix_to_rotation_6d(axis_angle_to_matrix(ee[..., 3:6]))],
            dim=-1,
        )
        gripper = state[self.state_gripper_key]
        self._require_last_dim(gripper, 2, "LIBERO gripper state")
        state[self.state_gripper_key] = 0.5 * (
            gripper[..., 0:1] - gripper[..., 1:2]
        )
        return batch

    def backward(self, batch):
        if "state" in batch:
            state = batch["state"]
            ee = state[self.state_ee_key]
            self._require_last_dim(ee, 9, "canonical EE state")
            state[self.state_ee_key] = torch.cat(
                [
                    ee[..., :3],
                    matrix_to_axis_angle(_rotation_6d_to_matrix(ee[..., 3:9])),
                ],
                dim=-1,
            )
            aperture = state[self.state_gripper_key]
            self._require_last_dim(aperture, 1, "canonical gripper state")
            state[self.state_gripper_key] = torch.cat(
                [aperture, -aperture], dim=-1
            )
        return batch
