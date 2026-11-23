"""Static per-config action pad mask: train a fixed subset of the action head's channels.

``StaticActionDimPad(total_dim, pad_ids)`` marks ``pad_ids`` in ``action_dim_is_pad`` on every
sample; ``ScatterToChannels`` ORs it into its own mask, and the model then zeroes those channels
on the noisy action input and the target and drops them from the action loss
(``_mask_padded_action_dims`` / ``_action_mse_per_step``). The RoboDojo
``0930_..._ABS_JOINTONLY_...`` run uses it with ``pad_ids = 0..17 + [26, 33]``, so only the 14
joint + gripper channels are trained and the EEF block is not.

Reconstructed for deploy on 2026-10-01: the training-side class was never published. What it did
is read off that checkpoint's weights against the pretrained checkpoint it started from:

* action_encoder input columns and action head output rows at ``pad_ids`` equal
  ``0.9678 * init`` to bf16 precision (relative residual 0.0017), i.e. zero gradient for the whole
  run with only decoupled weight decay acting. The other channels moved (residual 0.06-0.5).
* proprio_encoder columns 0..17 trained exactly like the ABS run's (residual ~0.1), and so did the
  norm-skipped rot6d columns, which a raw zero would have frozen. So the STATE is untouched: this
  transform only sets the action mask.

Action values are left as they are. The model never reads them at a padded channel.
"""

from typing import Dict, Iterable

import torch


class StaticActionDimPad:
    def __init__(self, total_dim: int, pad_ids: Iterable[int]) -> None:
        self.total_dim = int(total_dim)
        self.pad_ids = sorted({int(i) for i in pad_ids})
        if not self.pad_ids or self.pad_ids[0] < 0 or self.pad_ids[-1] >= self.total_dim:
            raise ValueError(f"StaticActionDimPad: pad_ids {self.pad_ids} must lie in [0, {self.total_dim})")

    def set_shape_meta(self, shape_meta) -> None:
        pass

    def mask(self) -> torch.Tensor:
        m = torch.zeros(self.total_dim, dtype=torch.bool)
        m[self.pad_ids] = True
        return m

    def forward(self, batch: Dict) -> Dict:
        if "action" not in batch:
            return batch
        width = batch["action"]["default"].shape[-1]
        if width != self.total_dim:
            raise ValueError(f"StaticActionDimPad: action width {width} != total_dim {self.total_dim}")
        mask = self.mask()
        prior = batch.get("action_dim_is_pad", None)
        if prior is not None:
            mask = mask | prior.to(dtype=torch.bool, device=mask.device)
        batch["action_dim_is_pad"] = mask
        return batch

    def backward(self, batch: Dict) -> Dict:
        return batch
