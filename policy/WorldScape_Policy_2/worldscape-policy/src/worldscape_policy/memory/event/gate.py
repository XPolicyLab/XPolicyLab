from __future__ import annotations

import torch
from torch import Tensor

from worldscape_policy.memory.event.history_compressor import LatentCoTCore


class MemoryGate:
    """Apply the legacy learned gate and scaled residual fusion."""

    def __call__(
        self,
        core: LatentCoTCore,
        current_tokens: Tensor,
        memory_readout: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = core.gate(torch.cat([current_tokens, memory_readout], dim=-1))
        gated_readout = gate * memory_readout
        fused = current_tokens + (core.residual_scale * gated_readout)
        return fused, gate


