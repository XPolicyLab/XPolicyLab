from __future__ import annotations

from torch import Tensor

from worldscape_policy.memory.event.history_compressor import LatentCoTCore


class EventBoundarySelector:
    """Select legacy recent-active and high-change event-boundary slots."""

    def __call__(
        self,
        core: LatentCoTCore,
        raw_history_tokens: Tensor,
        history_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        active_slots = core.active_proj(
            core._select_active_tokens(raw_history_tokens, history_mask)
        )
        done_slots, done_valid = core._select_done_tokens(
            raw_history_tokens, history_mask
        )
        return active_slots, core.done_proj(done_slots), done_valid


