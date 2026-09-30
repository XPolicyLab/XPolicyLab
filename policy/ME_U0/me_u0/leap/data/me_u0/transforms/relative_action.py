from typing import List, Dict

import torch

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r




class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch

        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]

        return batch


class NormalizedOpenToCloseTransform:
    """Convert ``0=closed, 1=open`` into ``0=open, 1=closed``."""

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        for group in ("state", "action"):
            if group not in batch:
                continue
            for key in self.keys:
                batch[group][key] = 1.0 - batch[group][key]
        return batch

    def backward(self, batch: Dict):
        return self.forward(batch)


class AgiBotParallelGripperCloseStateTransform:
    """Express AgiBot's measured parallel-gripper state as close position.

    Raw state is a two-sided millimetre coordinate while raw action is a
    normalized close target.  This applies the fixed dataset calibration so
    the state uses the same ``0=open, 1=closed`` direction and approximate
    range as action.  It applies only to the state; action gripper targets are
    already in that representation.
    """

    # Empirical q01/q99 calibration from all 160,399 parallel-gripper episodes
    # selected by the joint reader.  The canonical metadata record is
    # ``leap/configs/data/meta/agibot_beta.yaml`` in the state signal entry
    # ``effectors.position.training_close_coordinate``.  Native state increases
    # from the open q01 endpoint toward the closed q99 endpoint.
    NATIVE_Q01_MM = (34.53746828571428, 34.658451428571425)
    NATIVE_Q99_MM = (122.27198314285715, 122.26707746031747)

    def __init__(self, key: str = "gripper", side: int | None = None):
        self.key = key
        if side not in (None, 0, 1):
            raise ValueError("side must be None, 0 (left), or 1 (right)")
        self.side = side

    def _parameters(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_q01 = torch.as_tensor(
            AgiBotParallelGripperCloseStateTransform.NATIVE_Q01_MM,
            dtype=value.dtype,
            device=value.device,
        )
        state_q99 = torch.as_tensor(
            AgiBotParallelGripperCloseStateTransform.NATIVE_Q99_MM,
            dtype=value.dtype,
            device=value.device,
        )
        if self.side is None:
            if value.shape[-1] != 2:
                raise ValueError(
                    f"combined AgiBot gripper state must have width 2, got {value.shape}"
                )
        else:
            if value.shape[-1] != 1:
                raise ValueError(
                    f"split AgiBot gripper state must have width 1, got {value.shape}"
                )
            state_q01 = state_q01[self.side : self.side + 1]
            state_q99 = state_q99[self.side : self.side + 1]
        return state_q01, state_q99

    def forward(self, batch: Dict):
        value = batch["state"][self.key]
        state_q01, state_q99 = self._parameters(value)
        batch["state"][self.key] = (
            (value - state_q01) / (state_q99 - state_q01)
        ).clamp(0.0, 1.0)
        return batch

    def backward(self, batch: Dict):
        value = batch["state"][self.key]
        state_q01, state_q99 = self._parameters(value)
        batch["state"][self.key] = (
            state_q01 + value * (state_q99 - state_q01)
        )
        return batch
