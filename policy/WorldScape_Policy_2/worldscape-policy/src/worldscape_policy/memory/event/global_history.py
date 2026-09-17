from __future__ import annotations

import torch
from torch import Tensor

from worldscape_policy.memory.event.history_compressor import (
    CompressedHistory,
    LatentCoTCore,
)


class GlobalHistoryBuilder:
    """Build the legacy goal slots and complete compressed history bank."""

    def __call__(
        self,
        core: LatentCoTCore,
        history: CompressedHistory,
        task_embeddings: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        valid_float = history.valid_mask.unsqueeze(-1).to(history.bank.dtype)
        history_mean = (history.bank * valid_float).sum(dim=1)
        history_mean = history_mean / valid_float.sum(dim=1).clamp(min=1.0)
        if task_embeddings is not None:
            task = task_embeddings
            if task.dim() == 3:
                task = task.mean(dim=1)
            task = task.to(device=history_mean.device, dtype=history_mean.dtype)
            goal_anchor = core.goal_fuser(torch.cat([task, history_mean], dim=-1))
        else:
            goal_anchor = history_mean
        goal_slots = core.goal_proj(
            core._expand_from_anchor(goal_anchor, core.goal_slots)
        )
        return goal_slots, history.bank


