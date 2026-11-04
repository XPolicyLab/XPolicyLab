from typing import List, Dict, Sequence

import torch
from flexwam.datasets.lerobot.utils.rotation import rotation_6d_to_matrix, matrix_to_rotation_6d


class MaskedRelativeJointTransform:
    def __init__(self, keys: List[str], mask: Sequence[bool] | Dict[str, Sequence[bool]]):
        self.keys = keys
        self.mask = mask

    def _get_mask(self, key: str, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask[key] if isinstance(self.mask, dict) else self.mask
        mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
        if mask.ndim != 1 or mask.shape[0] != x.shape[-1]:
            raise ValueError(
                f"MaskedRelativeJointTransform mask for key {key!r} must have length {x.shape[-1]}, "
                f"got shape {tuple(mask.shape)}."
            )
        return mask

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch

        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            mask = self._get_mask(k, action).view(*([1] * (action.ndim - 1)), -1)
            batch["action"][k] = torch.where(mask, action - state, action)

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            mask = self._get_mask(k, action).view(*([1] * (action.ndim - 1)), -1)
            batch["action"][k] = torch.where(mask, action + state, action)

        return batch


class FlexivPose6DDeltaTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch

        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            batch["action"][k] = self._forward(action, state)

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            batch["action"][k] = self._backward(action, state)

        return batch

    @staticmethod
    def _check_pose6d(name: str, pose: torch.Tensor):
        if pose.shape[-1] != 10:
            raise ValueError(f"{name} must be 10D [xyz, rot6D, gripper], got shape {tuple(pose.shape)}.")

    @classmethod
    def _forward(cls, pose: torch.Tensor, base_pose: torch.Tensor):
        cls._check_pose6d("pose", pose)
        cls._check_pose6d("base_pose", base_pose)
        delta_pos = pose[..., 0:3] - base_pose[..., 0:3]
        pose_rot = rotation_6d_to_matrix(pose[..., 3:9])
        base_rot = rotation_6d_to_matrix(base_pose[..., 3:9])
        delta_rot = base_rot.transpose(-1, -2) @ pose_rot
        delta_rot6d = matrix_to_rotation_6d(delta_rot)
        gripper = pose[..., 9:10]
        return torch.cat([delta_pos, delta_rot6d, gripper], dim=-1)

    @classmethod
    def _backward(cls, pose: torch.Tensor, base_pose: torch.Tensor):
        cls._check_pose6d("pose", pose)
        cls._check_pose6d("base_pose", base_pose)
        abs_pos = pose[..., 0:3] + base_pose[..., 0:3]
        delta_rot = rotation_6d_to_matrix(pose[..., 3:9])
        base_rot = rotation_6d_to_matrix(base_pose[..., 3:9])
        abs_rot = base_rot @ delta_rot
        abs_rot6d = matrix_to_rotation_6d(abs_rot)
        gripper = pose[..., 9:10]
        return torch.cat([abs_pos, abs_rot6d, gripper], dim=-1)


class BimanualPose6DDeltaTransform:
    """Two 10D Flexiv-style arms concatenated on the last dim (20D total)."""

    ARM_DIM = 10

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            batch["action"][k] = self._map_arms(action, state, FlexivPose6DDeltaTransform._forward)
        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k][..., :1, :]
            batch["action"][k] = self._map_arms(action, state, FlexivPose6DDeltaTransform._backward)
        return batch

    @classmethod
    def _map_arms(cls, action: torch.Tensor, state: torch.Tensor, fn):
        if action.shape[-1] != cls.ARM_DIM * 2:
            raise ValueError(
                f"BimanualPose6DDeltaTransform expects last dim {cls.ARM_DIM * 2}, got {action.shape[-1]}."
            )
        left = fn(action[..., : cls.ARM_DIM], state[..., : cls.ARM_DIM])
        right = fn(action[..., cls.ARM_DIM :], state[..., cls.ARM_DIM :])
        return torch.cat([left, right], dim=-1)
