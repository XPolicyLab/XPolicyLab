from __future__ import annotations

from torch import Tensor

from worldscape_policy.memory.event.history_compressor import LatentCoTCore


class LocalActiveSelector:
    """Select and project the most recent active history slots."""

    def __call__(self, core: LatentCoTCore, raw_history_tokens: Tensor, history_mask: Tensor | None) -> Tensor:
        return core.active_proj(core._select_active_tokens(raw_history_tokens, history_mask))
