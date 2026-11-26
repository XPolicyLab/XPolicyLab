from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from worldscape_policy.memory.event.event_boundary import EventBoundarySelector
from worldscape_policy.memory.event.gate import MemoryGate
from worldscape_policy.memory.event.global_history import GlobalHistoryBuilder
from worldscape_policy.memory.event.history_compressor import HistoryCompressor
from worldscape_policy.memory.event.local_active import LocalActiveSelector
from worldscape_policy.memory.event.retriever import MemoryRetriever
from worldscape_policy.types import EventMemoryState


class LatentCoTMemory(nn.Module):
    """Build goal/active/done latent slots from history and fuse into current tokens."""

    def __init__(
        self,
        context_dim: int,
        goal_slots: int = 1,
        active_slots: int = 4,
        done_slots: int = 8,
        done_min_gap: int = 1,
        perception_gist_tokens: int = 8,
        residual_scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.goal_slots = max(1, int(goal_slots))
        self.active_slots = max(1, int(active_slots))
        self.done_slots = max(1, int(done_slots))
        self.done_min_gap = max(1, int(done_min_gap))
        self.perception_gist_tokens = max(1, int(perception_gist_tokens))
        self.residual_scale = float(max(0.0, residual_scale))

        self.goal_fuser = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.goal_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.active_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.done_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.value_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.query_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.gist_queries = nn.Parameter(
            torch.randn(self.perception_gist_tokens, context_dim) / math.sqrt(context_dim)
        )
        self.gist_query_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.gist_key_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.gist_value_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.gist_out_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(context_dim)
        self.token_norm = nn.LayerNorm(context_dim)

    def _pool_perception_gist(self, perception_tokens: torch.Tensor) -> torch.Tensor:
        # [B, H, L, D] -> [B, H, M, D]
        bsz, hist_len, token_len, dim = perception_tokens.shape
        flat = self.token_norm(perception_tokens.reshape(bsz * hist_len, token_len, dim))
        queries = self.gist_queries.to(device=flat.device, dtype=flat.dtype)
        queries = queries.unsqueeze(0).expand(flat.shape[0], -1, -1)
        q = self.gist_query_proj(queries)
        k = self.gist_key_proj(flat)
        v = self.gist_value_proj(flat)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.context_dim)
        attn = torch.softmax(logits, dim=-1)
        gist = torch.matmul(attn, v)
        gist = self.gist_out_proj(gist)
        return gist.reshape(bsz, hist_len, self.perception_gist_tokens, dim)

    def _as_history_token_grid(self, tokens: torch.Tensor, pool_perception: bool) -> torch.Tensor:
        # [B, H, L, D] -> [B, H, M/L, D], [B, H, D] -> [B, H, 1, D], [B, D] -> [B, 1, 1, D]
        if tokens.dim() == 4:
            if pool_perception:
                return self._pool_perception_gist(tokens)
            return tokens
        if tokens.dim() == 3:
            return tokens.unsqueeze(2)
        if tokens.dim() == 2:
            return tokens.unsqueeze(1).unsqueeze(2)
        raise ValueError(f"Unsupported history token shape: {tuple(tokens.shape)}")

    def _pool_history_tokens(
        self,
        history_tokens: torch.Tensor,
        history_planning_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Per history chunk: perception tokens -> M gist tokens; planning tokens are appended unchanged.
        # Output is [B, H, T, D], where T = M + planning_num_tokens when planning history is present.
        hist_tokens = self._as_history_token_grid(history_tokens, pool_perception=history_tokens.dim() == 4)
        if history_planning_tokens is None or history_planning_tokens.numel() == 0:
            return hist_tokens
        planning_tokens = self._as_history_token_grid(history_planning_tokens, pool_perception=False)
        if hist_tokens.shape[:2] != planning_tokens.shape[:2]:
            return hist_tokens
        return torch.cat([hist_tokens, planning_tokens], dim=2)

    def _raw_history_tokens(
        self,
        history_tokens: torch.Tensor,
        history_planning_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Preserve the original perception token count for active/done shortcuts.
        hist_tokens = self._as_history_token_grid(history_tokens, pool_perception=False)
        if history_planning_tokens is None or history_planning_tokens.numel() == 0:
            return hist_tokens
        planning_tokens = self._as_history_token_grid(history_planning_tokens, pool_perception=False)
        if hist_tokens.shape[:2] != planning_tokens.shape[:2]:
            return hist_tokens
        return torch.cat([hist_tokens, planning_tokens], dim=2)

    def _pool_history_steps(self, history_tokens: torch.Tensor) -> torch.Tensor:
        # [B, H, T, D] -> [B, H, D]
        if history_tokens.dim() == 4:
            return history_tokens.mean(dim=2)
        if history_tokens.dim() == 3:
            return history_tokens
        if history_tokens.dim() == 2:
            return history_tokens.unsqueeze(1)
        raise ValueError(f"Unsupported history_tokens shape: {tuple(history_tokens.shape)}")

    def _expand_from_anchor(self, anchor: torch.Tensor, n_slots: int) -> torch.Tensor:
        # anchor: [B, D] -> [B, n_slots, D]
        return anchor.unsqueeze(1).expand(-1, n_slots, -1)

    def _select_active(self, hist_steps: torch.Tensor, hist_mask: torch.Tensor | None) -> torch.Tensor:
        # Most recent steps as active slots.
        if hist_mask is None:
            valid_len = torch.full(
                (hist_steps.shape[0],),
                fill_value=hist_steps.shape[1],
                device=hist_steps.device,
                dtype=torch.long,
            )
        else:
            valid_len = hist_mask.long().sum(dim=1)

        out = []
        for b in range(hist_steps.shape[0]):
            n = int(max(1, valid_len[b].item()))
            cur = hist_steps[b, :n]
            tail = cur[-self.active_slots :]
            if tail.shape[0] < self.active_slots:
                pad = tail[-1:].expand(self.active_slots - tail.shape[0], -1)
                tail = torch.cat([pad, tail], dim=0)
            out.append(tail)
        return torch.stack(out, dim=0)

    def _select_active_tokens(
        self,
        hist_tokens: torch.Tensor,
        hist_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Most recent steps as active slots, preserving each step's gist/planning tokens.
        hist_steps = hist_tokens.mean(dim=2)
        selected = self._select_active(hist_steps, hist_mask)  # [B, A, D]
        out = []
        for b in range(hist_tokens.shape[0]):
            if hist_mask is None:
                n = hist_tokens.shape[1]
            else:
                n = int(max(1, hist_mask[b].long().sum().item()))
            cur = hist_tokens[b, :n]
            tail = cur[-self.active_slots :]
            if tail.shape[0] < self.active_slots:
                pad = tail[-1:].expand(self.active_slots - tail.shape[0], -1, -1)
                tail = torch.cat([pad, tail], dim=0)
            out.append(tail.reshape(-1, hist_tokens.shape[-1]))
        active_tokens = torch.stack(out, dim=0)
        if active_tokens.shape[1] == selected.shape[1]:
            return selected
        return active_tokens

    def _select_done(
        self,
        hist_steps: torch.Tensor,
        hist_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Select high-change steps by cosine change score (1 - cos(z_t, z_{t-1})).
        if hist_mask is None:
            valid_len = torch.full(
                (hist_steps.shape[0],),
                fill_value=hist_steps.shape[1],
                device=hist_steps.device,
                dtype=torch.long,
            )
        else:
            valid_len = hist_mask.long().sum(dim=1)

        out = []
        valid_out = []
        for b in range(hist_steps.shape[0]):
            n = int(max(1, valid_len[b].item()))
            cur = hist_steps[b, :n]  # [n, D]
            if n == 1:
                done = torch.zeros((self.done_slots, cur.shape[-1]), device=cur.device, dtype=cur.dtype)
                done[0] = cur[-1]
                valid = torch.zeros((self.done_slots,), device=cur.device, dtype=torch.bool)
                valid[0] = True
                out.append(done)
                valid_out.append(valid)
                continue
            cur_prev = cur[:-1]
            cur_next = cur[1:]
            cos = torch.nn.functional.cosine_similarity(cur_next, cur_prev, dim=-1)  # [n-1]
            change_score = 1.0 - cos
            order = torch.argsort(change_score, descending=True) + 1  # map back to cur indices
            chosen = []
            for idx in order.tolist():
                if len(chosen) >= self.done_slots:
                    break
                if any(abs(int(idx) - int(prev)) <= self.done_min_gap for prev in chosen):
                    continue
                chosen.append(int(idx))
            done = torch.zeros((self.done_slots, cur.shape[-1]), device=cur.device, dtype=cur.dtype)
            valid = torch.zeros((self.done_slots,), device=cur.device, dtype=torch.bool)
            for i, idx in enumerate(chosen):
                done[i] = cur[idx]
                valid[i] = True
            out.append(done)
            valid_out.append(valid)
        return torch.stack(out, dim=0), torch.stack(valid_out, dim=0)

    def _select_done_tokens(
        self,
        hist_tokens: torch.Tensor,
        hist_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Select high-change steps, preserving each selected step's gist/planning tokens.
        hist_steps = hist_tokens.mean(dim=2)

        if hist_mask is None:
            valid_len = torch.full(
                (hist_steps.shape[0],),
                fill_value=hist_steps.shape[1],
                device=hist_steps.device,
                dtype=torch.long,
            )
        else:
            valid_len = hist_mask.long().sum(dim=1)

        out = []
        valid_out = []
        tokens_per_step = hist_tokens.shape[2]
        for b in range(hist_steps.shape[0]):
            n = int(max(1, valid_len[b].item()))
            cur_steps = hist_steps[b, :n]
            cur_tokens = hist_tokens[b, :n]
            if n == 1:
                chosen = [0]
            else:
                cos = torch.nn.functional.cosine_similarity(cur_steps[1:], cur_steps[:-1], dim=-1)
                order = torch.argsort(1.0 - cos, descending=True) + 1
                chosen = []
                for idx in order.tolist():
                    if len(chosen) >= self.done_slots:
                        break
                    if any(abs(int(idx) - int(prev)) <= self.done_min_gap for prev in chosen):
                        continue
                    chosen.append(int(idx))

            done = torch.zeros(
                (self.done_slots, tokens_per_step, hist_tokens.shape[-1]),
                device=hist_tokens.device,
                dtype=hist_tokens.dtype,
            )
            valid = torch.zeros((self.done_slots, tokens_per_step), device=hist_tokens.device, dtype=torch.bool)
            for i, idx in enumerate(chosen[: self.done_slots]):
                done[i] = cur_tokens[idx]
                valid[i] = True
            out.append(done.reshape(self.done_slots * tokens_per_step, hist_tokens.shape[-1]))
            valid_out.append(valid.reshape(self.done_slots * tokens_per_step))
        return torch.stack(out, dim=0), torch.stack(valid_out, dim=0)

    def forward(
        self,
        current_tokens: torch.Tensor,
        history_tokens: torch.Tensor | None = None,
        history_planning_tokens: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        task_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if history_tokens is None or history_tokens.numel() == 0:
            return current_tokens, {}
        hist_tokens = self._pool_history_tokens(history_tokens, history_planning_tokens)  # [B, H, T, D]
        hist_tokens = self.norm(hist_tokens)
        raw_hist_tokens = self._raw_history_tokens(history_tokens, history_planning_tokens)
        raw_hist_tokens = self.norm(raw_hist_tokens)
        bsz, hist_len, tokens_per_step, dim = hist_tokens.shape
        history_bank = hist_tokens.reshape(bsz, hist_len * tokens_per_step, dim)
        if history_mask is None:
            history_valid_mask = torch.ones(
                (bsz, hist_len * tokens_per_step),
                device=hist_tokens.device,
                dtype=torch.bool,
            )
        else:
            history_valid_mask = history_mask.to(device=hist_tokens.device, dtype=torch.bool)
            history_valid_mask = history_valid_mask.unsqueeze(-1).expand(-1, -1, tokens_per_step)
            history_valid_mask = history_valid_mask.reshape(bsz, hist_len * tokens_per_step)

        history_valid_float = history_valid_mask.unsqueeze(-1).to(history_bank.dtype)
        hist_mean = (history_bank * history_valid_float).sum(dim=1)
        hist_mean = hist_mean / history_valid_float.sum(dim=1).clamp(min=1.0)  # [B, D]
        if task_embeddings is not None:
            task_emb = task_embeddings
            if task_emb.dim() == 3:
                task_emb = task_emb.mean(dim=1)
            task_emb = task_emb.to(device=hist_mean.device, dtype=hist_mean.dtype)
            goal_anchor = self.goal_fuser(torch.cat([task_emb, hist_mean], dim=-1))
        else:
            goal_anchor = hist_mean

        goal_slots = self.goal_proj(self._expand_from_anchor(goal_anchor, self.goal_slots))
        active_slots = self.active_proj(self._select_active_tokens(raw_hist_tokens, history_mask))
        done_slots, done_valid = self._select_done_tokens(raw_hist_tokens, history_mask)
        done_slots = self.done_proj(done_slots)
        memory_bank = torch.cat([goal_slots, active_slots, done_slots, history_bank], dim=1)  # [B, S, D]
        memory_bank = self.value_proj(memory_bank)
        valid_mask = torch.cat(
            [
                torch.ones(
                    (bsz, self.goal_slots + active_slots.shape[1]),
                    device=history_valid_mask.device,
                    dtype=torch.bool,
                ),
                done_valid,
                history_valid_mask,
            ],
            dim=1,
        )  # [B, S]

        query = self.query_proj(current_tokens)  # [B, N, D]
        logits = torch.matmul(query, memory_bank.transpose(-1, -2)) / math.sqrt(self.context_dim)
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(logits, dim=-1)
        memory_readout = torch.matmul(attn, memory_bank)  # [B, N, D]
        memory_readout = self.dropout(memory_readout)

        gate = self.gate(torch.cat([current_tokens, memory_readout], dim=-1))
        gated_readout = gate * memory_readout
        fused = current_tokens + (self.residual_scale * gated_readout)
        return fused, {
            "memory_gate": gate,
            "memory_readout": memory_readout,
            "memory_bank": memory_bank,
            "memory_attention": attn,
            "memory_valid_mask": valid_mask,
        }


class EventMemoryFusion(LatentCoTMemory):
    """Compose event-memory stages while preserving legacy parameters and math.

    Subclassing keeps every converted checkpoint key directly under
    ``event_memory.*``. The collaborators are stateless and therefore add no
    registered module or parameter names.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history_compressor = HistoryCompressor()
        self.global_history_builder = GlobalHistoryBuilder()
        self.event_boundary_selector = EventBoundarySelector()
        self.memory_retriever = MemoryRetriever()
        self.memory_gate = MemoryGate()
        self.local_active_selector = LocalActiveSelector()

    def forward(
        self,
        current_tokens: Tensor,
        history_tokens: Tensor | None = None,
        history_planning_tokens: Tensor | None = None,
        history_mask: Tensor | None = None,
        task_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        if history_tokens is None or history_tokens.numel() == 0:
            return current_tokens, {}

        history = self.history_compressor(
            self,
            history_tokens,
            history_planning_tokens,
            history_mask,
        )
        goal_slots, history_bank = self.global_history_builder(
            self,
            history,
            task_embeddings,
        )
        active_slots, done_slots, done_valid = self.event_boundary_selector(
            self,
            history.raw_tokens,
            history_mask,
        )
        memory_bank = torch.cat(
            [goal_slots, active_slots, done_slots, history_bank],
            dim=1,
        )
        batch_size = history.tokens.shape[0]
        valid_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, self.goal_slots + active_slots.shape[1]),
                    device=history.valid_mask.device,
                    dtype=torch.bool,
                ),
                done_valid,
                history.valid_mask,
            ],
            dim=1,
        )

        memory_readout, memory_bank, attention = self.memory_retriever(
            self,
            current_tokens,
            memory_bank,
            valid_mask,
        )
        fused, gate = self.memory_gate(
            self,
            current_tokens,
            memory_readout,
        )
        return fused, {
            "memory_gate": gate,
            "memory_readout": memory_readout,
            "memory_bank": memory_bank,
            "memory_attention": attention,
            "memory_valid_mask": valid_mask,
        }


class EventMemoryManager:
    """Own Auto event-history candidate, promotion, and cache transactions."""

    def __init__(
        self,
        *,
        max_history_steps: int,
        detach_inference_history: bool = True,
    ) -> None:
        if max_history_steps <= 0:
            raise ValueError("max_history_steps must be positive")
        self.max_history_steps = int(max_history_steps)
        self.detach_inference_history = bool(detach_inference_history)

    def begin(
        self,
        previous: EventMemoryState | None,
        *,
        prompt_signature: tuple[str, ...],
        has_planning: bool,
        training: bool,
    ) -> EventMemoryState:
        state = previous or EventMemoryState()
        if (
            not training
            and state.prompt_signature is not None
            and state.prompt_signature != prompt_signature
        ):
            state = EventMemoryState()
        previous_has_planning = (
            state.planning_tokens is not None
            or state.pending_planning_tokens is not None
        )
        previous_has_history = (
            state.valid_mask is not None or state.pending_valid_mask is not None
        )
        if previous_has_history and previous_has_planning != has_planning:
            raise ValueError(
                "Planning-token presence must remain consistent within an Auto episode"
            )
        return state

    def stage(
        self,
        previous: EventMemoryState,
        perception: Tensor,
        planning: Tensor | None,
        *,
        training: bool,
    ) -> EventMemoryState:
        if training:
            return self._append_committed(previous, perception, planning)
        return self._append_pending(previous, perception, planning)

    @staticmethod
    def cached_conditions(
        previous: EventMemoryState,
        positive: Tensor,
        negative: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if previous.cached_cross_attention_tokens is None:
            return positive, negative
        return (
            previous.cached_cross_attention_tokens,
            previous.cached_negative_cross_attention_tokens
            if previous.cached_negative_cross_attention_tokens is not None
            else previous.cached_cross_attention_tokens,
        )

    @staticmethod
    def cache_conditions(
        state: EventMemoryState,
        *,
        positive: Tensor,
        negative: Tensor,
        prompt_signature: tuple[str, ...],
    ) -> EventMemoryState:
        if state.cached_cross_attention_tokens is None:
            state = replace(
                state,
                cached_cross_attention_tokens=positive.detach(),
                cached_negative_cross_attention_tokens=negative.detach(),
                prompt_signature=prompt_signature,
            )
        elif state.prompt_signature is None:
            state = replace(state, prompt_signature=prompt_signature)
        return state

    def promote_pending(
        self,
        previous: EventMemoryState | None,
    ) -> EventMemoryState | None:
        if previous is None or previous.pending_valid_mask is None:
            return previous
        return EventMemoryState(
            perception_tokens=self._cat_trim(
                previous.perception_tokens,
                previous.pending_perception_tokens,
            ),
            planning_tokens=self._cat_trim(
                previous.planning_tokens,
                previous.pending_planning_tokens,
            ),
            valid_mask=self._cat_trim(
                previous.valid_mask,
                previous.pending_valid_mask,
            ),
            prompt_signature=previous.prompt_signature,
        )

    def _append_committed(
        self,
        previous: EventMemoryState,
        perception: Tensor,
        planning: Tensor | None,
    ) -> EventMemoryState:
        perception = perception.unsqueeze(1)
        planning = planning.unsqueeze(1) if planning is not None else None
        valid = torch.ones(
            perception.shape[:2],
            device=perception.device,
            dtype=torch.bool,
        )
        return EventMemoryState(
            perception_tokens=self._cat_trim(previous.perception_tokens, perception),
            planning_tokens=self._cat_trim(previous.planning_tokens, planning),
            valid_mask=self._cat_trim(previous.valid_mask, valid),
            pending_perception_tokens=previous.pending_perception_tokens,
            pending_planning_tokens=previous.pending_planning_tokens,
            pending_valid_mask=previous.pending_valid_mask,
            cached_cross_attention_tokens=previous.cached_cross_attention_tokens,
            cached_negative_cross_attention_tokens=(
                previous.cached_negative_cross_attention_tokens
            ),
            prompt_signature=previous.prompt_signature,
        )

    def _append_pending(
        self,
        previous: EventMemoryState,
        perception: Tensor,
        planning: Tensor | None,
    ) -> EventMemoryState:
        if self.detach_inference_history:
            perception = perception.detach()
            planning = planning.detach() if planning is not None else None
        perception = perception.unsqueeze(1)
        planning = planning.unsqueeze(1) if planning is not None else None
        valid = torch.ones(
            perception.shape[:2],
            device=perception.device,
            dtype=torch.bool,
        )
        return EventMemoryState(
            perception_tokens=previous.perception_tokens,
            planning_tokens=previous.planning_tokens,
            valid_mask=previous.valid_mask,
            pending_perception_tokens=self._cat_trim(
                previous.pending_perception_tokens, perception
            ),
            pending_planning_tokens=self._cat_trim(
                previous.pending_planning_tokens, planning
            ),
            pending_valid_mask=self._cat_trim(previous.pending_valid_mask, valid),
            cached_cross_attention_tokens=previous.cached_cross_attention_tokens,
            cached_negative_cross_attention_tokens=(
                previous.cached_negative_cross_attention_tokens
            ),
            prompt_signature=previous.prompt_signature,
        )

    def _cat_trim(
        self,
        previous: Tensor | None,
        current: Tensor | None,
    ) -> Tensor | None:
        if current is None:
            return previous
        value = current if previous is None else torch.cat([previous, current], dim=1)
        return value[:, -self.max_history_steps :]
