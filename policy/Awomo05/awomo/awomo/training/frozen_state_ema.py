# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete DiT EMA with exact snapshots of the frozen state encoder."""

import torch
from awomo.awomo.training.ema import EMAManager


class CompleteFrozenStateEMA(EMAManager):
    """Track trainable EMA tensors and exact snapshots of frozen parameters."""

    def __init__(self, existing: EMAManager, model: torch.nn.Module) -> None:
        self.decay = existing.decay
        self._shadow = existing._shadow
        self.frozen_names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
        assert self.frozen_names == {"proprio_encoder.weight", "proprio_encoder.bias"}
        assert not self.frozen_names.intersection(self._shadow)
        for name, parameter in model.named_parameters():
            if name in self.frozen_names:
                self._shadow[name] = parameter.detach().clone()
        assert set(self._shadow) == {name for name, _ in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update trainable DiT EMA; leave frozen state tensors unchanged."""
        assert set(self._shadow) == {name for name, _ in model.named_parameters()}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert name not in self.frozen_names
                self._shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
