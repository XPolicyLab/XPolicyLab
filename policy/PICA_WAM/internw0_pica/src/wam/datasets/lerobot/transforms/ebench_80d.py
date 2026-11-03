"""Ebench-specific transforms for the canonical 80-D robot layout.

Ebench stores each parallel gripper as two non-negative finger displacements:
``[left_finger, right_finger]``.  The canonical gripper slot represents the
physical jaw aperture, so the two displacements must be summed rather than
averaged.  The inverse transform splits the aperture equally between the two
fingers for open-loop post-processing.

Ebench end-effector poses are stored per arm as ``xyz + quaternion(wxyz)``
and concatenated ``[left, right]`` into one 14-D field.  Following the LIBERO
80-D adapter, no coordinate-frame conversion is applied: positions are kept in
Ebench's native frame and only the rotation parameterization is changed to
match the canonical slots (state ``xyz + rot6d``, action ``dxyz + rotvec``).
"""

from __future__ import annotations

import torch

from ..utils.rotation import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    matrix_to_rotation_6d_columns,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_columns_to_matrix,
    standardize_quaternion,
)


class EbenchParallelGripperTransform:
    """Convert Ebench's two-finger-per-gripper representation to apertures."""

    def __init__(
        self,
        *,
        action_key: str = "gripper",
        state_key: str = "gripper",
    ) -> None:
        self.action_key = str(action_key)
        self.state_key = str(state_key)

    @staticmethod
    def _require_last_dim(value: torch.Tensor, dim: int, name: str) -> None:
        if not isinstance(value, torch.Tensor) or int(value.shape[-1]) != int(dim):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"{name} must have last dimension {dim}, got {shape}")

    @classmethod
    def _fingers_to_apertures(cls, value: torch.Tensor, name: str) -> torch.Tensor:
        cls._require_last_dim(value, 4, name)
        return torch.stack(
            (
                value[..., 0] + value[..., 1],
                value[..., 2] + value[..., 3],
            ),
            dim=-1,
        )

    @classmethod
    def _apertures_to_fingers(cls, value: torch.Tensor, name: str) -> torch.Tensor:
        cls._require_last_dim(value, 2, name)
        half = value * 0.5
        return torch.stack(
            (half[..., 0], half[..., 0], half[..., 1], half[..., 1]),
            dim=-1,
        )

    def forward(self, batch):
        if "action" in batch:
            action = batch["action"]
            action[self.action_key] = self._fingers_to_apertures(
                action[self.action_key], "Ebench action gripper"
            )

        state = batch["state"]
        state[self.state_key] = self._fingers_to_apertures(
            state[self.state_key], "Ebench state gripper"
        )
        return batch

    def backward(self, batch):
        if "action" in batch:
            action = batch["action"]
            action[self.action_key] = self._apertures_to_fingers(
                action[self.action_key], "canonical action gripper"
            )

        if "state" in batch:
            state = batch["state"]
            state[self.state_key] = self._apertures_to_fingers(
                state[self.state_key], "canonical state gripper"
            )
        return batch


class EbenchDualArmEEPoseTransform:
    """Reparameterize Ebench dual-arm EE poses for the canonical 80-D slots.

    Forward (per arm, applied to ``[left, right]`` concatenated fields):
      * state  ``xyz + quat(wxyz)`` (7) -> ``xyz + rot6d`` (9), column-first
        rot6d as in the mix pretraining space.
      * action ``dxyz + dquat(wxyz)`` (7) -> ``dxyz + rotvec`` (6).

    Backward inverts both so open-loop eval / deployment recovers Ebench's
    native quaternion fields.  Positions are never frame-transformed.
    """

    rotation_semantics = "raw_ebench_pose_no_frame_conversion"

    NATIVE_POSE_DIM = 7
    STATE_POSE_DIM = 9
    ACTION_POSE_DIM = 6
    NUM_ARMS = 2

    def __init__(
        self,
        *,
        action_key: str | None = "ee",
        state_key: str = "ee",
    ) -> None:
        self.action_key = None if action_key is None else str(action_key)
        self.state_key = str(state_key)

    @staticmethod
    def _require_last_dim(value: torch.Tensor, dim: int, name: str) -> None:
        if not isinstance(value, torch.Tensor) or int(value.shape[-1]) != int(dim):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"{name} must have last dimension {dim}, got {shape}")

    @classmethod
    def _split_arms(cls, value: torch.Tensor, pose_dim: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            value[..., i * pose_dim : (i + 1) * pose_dim] for i in range(cls.NUM_ARMS)
        )

    # ---- state: quat(wxyz) <-> rot6d --------------------------------------
    @classmethod
    def _state_forward(cls, pose: torch.Tensor) -> torch.Tensor:
        quat = standardize_quaternion(pose[..., 3:7])
        rot6d = matrix_to_rotation_6d_columns(quaternion_to_matrix(quat))
        return torch.cat([pose[..., :3], rot6d], dim=-1)

    @classmethod
    def _state_backward(cls, pose: torch.Tensor) -> torch.Tensor:
        quat = matrix_to_quaternion(rotation_6d_columns_to_matrix(pose[..., 3:9]))
        return torch.cat([pose[..., :3], quat], dim=-1)

    # ---- action: dquat(wxyz) <-> rotvec -----------------------------------
    @classmethod
    def _action_forward(cls, pose: torch.Tensor) -> torch.Tensor:
        quat = standardize_quaternion(pose[..., 3:7])
        return torch.cat([pose[..., :3], quaternion_to_axis_angle(quat)], dim=-1)

    @classmethod
    def _action_backward(cls, pose: torch.Tensor) -> torch.Tensor:
        quat = axis_angle_to_quaternion(pose[..., 3:6])
        return torch.cat([pose[..., :3], quat], dim=-1)

    def forward(self, batch):
        if self.action_key is not None and "action" in batch:
            action = batch["action"][self.action_key]
            self._require_last_dim(
                action, self.NUM_ARMS * self.NATIVE_POSE_DIM, "Ebench action ee_pose_delta"
            )
            batch["action"][self.action_key] = torch.cat(
                [self._action_forward(arm) for arm in self._split_arms(action, self.NATIVE_POSE_DIM)],
                dim=-1,
            )

        state = batch["state"][self.state_key]
        self._require_last_dim(
            state, self.NUM_ARMS * self.NATIVE_POSE_DIM, "Ebench state ee_pose"
        )
        batch["state"][self.state_key] = torch.cat(
            [self._state_forward(arm) for arm in self._split_arms(state, self.NATIVE_POSE_DIM)],
            dim=-1,
        )
        return batch

    def backward(self, batch):
        if self.action_key is not None and "action" in batch:
            action = batch["action"][self.action_key]
            self._require_last_dim(
                action, self.NUM_ARMS * self.ACTION_POSE_DIM, "canonical action ee"
            )
            batch["action"][self.action_key] = torch.cat(
                [self._action_backward(arm) for arm in self._split_arms(action, self.ACTION_POSE_DIM)],
                dim=-1,
            )

        if "state" in batch:
            state = batch["state"][self.state_key]
            self._require_last_dim(
                state, self.NUM_ARMS * self.STATE_POSE_DIM, "canonical state ee"
            )
            batch["state"][self.state_key] = torch.cat(
                [self._state_backward(arm) for arm in self._split_arms(state, self.STATE_POSE_DIM)],
                dim=-1,
            )
        return batch
