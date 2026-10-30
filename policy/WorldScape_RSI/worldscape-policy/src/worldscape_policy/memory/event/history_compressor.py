from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class LatentCoTCore(Protocol):
    """Numerical surface used from the legacy LatentCoT memory oracle."""

    context_dim: int
    goal_slots: int
    active_slots: int
    residual_scale: float
    norm: torch.nn.Module
    goal_fuser: torch.nn.Module
    goal_proj: torch.nn.Module
    active_proj: torch.nn.Module
    done_proj: torch.nn.Module
    value_proj: torch.nn.Module
    query_proj: torch.nn.Module
    gate: torch.nn.Module
    dropout: torch.nn.Module

    def _pool_history_tokens(
        self,
        history_tokens: Tensor,
        history_planning_tokens: Tensor | None = None,
    ) -> Tensor: ...

    def _raw_history_tokens(
        self,
        history_tokens: Tensor,
        history_planning_tokens: Tensor | None = None,
    ) -> Tensor: ...

    def _select_active_tokens(
        self,
        hist_tokens: Tensor,
        hist_mask: Tensor | None,
    ) -> Tensor: ...

    def _select_done_tokens(
        self,
        hist_tokens: Tensor,
        hist_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...

    def _expand_from_anchor(self, anchor: Tensor, n_slots: int) -> Tensor: ...


@dataclass(frozen=True)
class CompressedHistory:
    tokens: Tensor
    raw_tokens: Tensor
    bank: Tensor
    valid_mask: Tensor


class HistoryCompressor:
    """Adapt legacy perception-gist compression without changing its math."""

    def __call__(
        self,
        core: LatentCoTCore,
        history_tokens: Tensor,
        history_planning_tokens: Tensor | None,
        history_mask: Tensor | None,
    ) -> CompressedHistory:
        tokens = core.norm(
            core._pool_history_tokens(history_tokens, history_planning_tokens)
        )
        raw_tokens = core.norm(
            core._raw_history_tokens(history_tokens, history_planning_tokens)
        )
        batch_size, history_length, tokens_per_step, dim = tokens.shape
        bank = tokens.reshape(batch_size, history_length * tokens_per_step, dim)
        if history_mask is None:
            valid_mask = torch.ones(
                (batch_size, history_length * tokens_per_step),
                device=tokens.device,
                dtype=torch.bool,
            )
        else:
            valid_mask = history_mask.to(device=tokens.device, dtype=torch.bool)
            valid_mask = valid_mask.unsqueeze(-1).expand(
                -1, -1, tokens_per_step
            )
            valid_mask = valid_mask.reshape(
                batch_size, history_length * tokens_per_step
            )
        return CompressedHistory(tokens, raw_tokens, bank, valid_mask)


