from __future__ import annotations

import math

import torch
from torch import Tensor

from worldscape_policy.memory.event.history_compressor import LatentCoTCore


class MemoryRetriever:
    """Run the legacy projected scaled-dot-product memory read."""

    def __call__(
        self,
        core: LatentCoTCore,
        current_tokens: Tensor,
        memory_bank: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        memory_bank = core.value_proj(memory_bank)
        query = core.query_proj(current_tokens)
        logits = torch.matmul(query, memory_bank.transpose(-1, -2))
        logits = logits / math.sqrt(core.context_dim)
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        attention = torch.softmax(logits, dim=-1)
        memory_readout = torch.matmul(attention, memory_bank)
        memory_readout = core.dropout(memory_readout)
        return memory_readout, memory_bank, attention


