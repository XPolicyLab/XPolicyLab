"""Dual-stream action expert with optional future-representation alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from mmabc.models.attention import (
    FAR,
    NEAR,
    STREAM_ID,
    build_attention_mask,
    sdpa,
    split_action_future_masks,
)

ACTION_STREAMS = ("manip", "aux")


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding, projected to the stream width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        angles = t.float()[:, None] * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class ActionEncoder(nn.Module):
    """Noisy action chunk + flow time -> stream tokens."""

    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden)
        self.fuse = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, actions: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(actions)
        t = time_emb[:, None, :].expand(-1, x.shape[1], -1)
        x = self.fuse(torch.cat([x, t], dim=-1))
        return self.out(nn.functional.silu(x))


class StateEncoder(nn.Module):
    """Proprioceptive slice plus its validity mask -> one token.

    Feeding the mask alongside the values is what makes "this dimension is
    absent" distinguishable from "this dimension is genuinely zero". Without it
    a zero-padded slot and a joint resting at zero look identical.
    """

    def __init__(self, state_dim: int, hidden: int, inner: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, inner), nn.SiLU(), nn.Linear(inner, hidden)
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, mask], dim=-1))[:, None, :]


class ContextKV(nn.Module):
    """Keys and values for the read-only context, for one block.

    Lives on the expert rather than inside the block, because it is called once
    per step outside the block loop and its result is reused by every noise
    draw. FSDP unshards a unit's parameters in that unit's own pre-forward
    hook, so a module invoked outside the loop needs to *be* a unit; see
    ``_context_modules`` in ``mmabc/train/fsdp.py``, which is what makes this
    placement safe.
    """

    def __init__(self, context_dim: int, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.norm = nn.LayerNorm(context_dim)
        self.proj = nn.Linear(context_dim, 2 * hidden, bias=False)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = context.to(self.proj.weight.dtype)
        B, S, _ = context.shape
        k, v = self.proj(self.norm(context)).chunk(2, dim=-1)
        shape = lambda t: t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return shape(k), shape(v)


class ContextEncoder(nn.Module):
    """Builds every block's context keys and values, once per step."""

    def __init__(
        self,
        context_dim: int,
        hidden: int,
        num_heads: int,
        num_blocks: int,
        injection_blocks: tuple[int, ...],
        context_mode: str = "deepstack",
    ) -> None:
        super().__init__()
        # deepstack injects earlier taps; per_block supplies one context tap per block.
        if context_mode not in ("deepstack", "per_block"):
            raise ValueError(f"unknown context_mode {context_mode!r}")
        self.context_mode = context_mode
        self.num_blocks = num_blocks
        self.injection_blocks = tuple(injection_blocks)
        self.deepstack_in = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_dim))
                for _ in self.injection_blocks
            ]
        )
        # Zero-initialised so injection starts as a no-op and cannot perturb the
        # pretrained context before the gate has learnt anything.
        self.deepstack_gate = nn.ParameterList(
            [nn.Parameter(torch.zeros(context_dim)) for _ in self.injection_blocks]
        )
        self.context_kv = nn.ModuleList(
            [ContextKV(context_dim, hidden, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, memory: list[torch.Tensor]) -> list:
        """Per-block keys and values, with the DeepStack taps folded in.

        The context does not depend on the noisy action input, so every noise
        draw in a step sees exactly the same keys and values. A tap modifies
        the context before the block it is assigned to, and that modification
        persists for the remaining blocks.
        """
        if self.context_mode == "per_block":
            # Block i reads VLM tap i directly; no shared base, no residual
            # injection. Each tap has the same sequence length, so the mask's
            # context_len (cache[0][0].shape[2]) is still valid for every block.
            if len(memory) != len(self.context_kv):
                raise ValueError(
                    f"per_block context expects {len(self.context_kv)} taps "
                    f"(one per block) but got {len(memory)}"
                )
            return [ctx_kv(memory[i]) for i, ctx_kv in enumerate(self.context_kv)]

        context = memory[-1]
        inject = {block: i for i, block in enumerate(self.injection_blocks)}

        cache = []
        for i, ctx_kv in enumerate(self.context_kv):
            if i in inject:
                k = inject[i]
                if k < len(memory) - 1:
                    module = self.deepstack_in[k]
                    dtype = module[1].weight.dtype
                    projected = module(memory[k].to(dtype))
                    gate = self.deepstack_gate[k].to(projected.dtype)
                    context = context.to(projected.dtype) + projected * gate
            cache.append(ctx_kv(context))
        return cache


class JointBlock(nn.Module):
    """Per-stream projections and feed-forward, one shared attention.

    The context enters as precomputed keys and values appended to this block's
    own. It gets no query, no output projection and no feed-forward.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        ffn: int,
        streams: tuple[str, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.streams = streams
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        if hidden % num_heads:
            raise ValueError("hidden must be divisible by num_heads")

        self.norm1 = nn.ModuleDict({s: nn.LayerNorm(hidden) for s in streams})
        self.qkv = nn.ModuleDict({s: nn.Linear(hidden, 3 * hidden, bias=False) for s in streams})
        self.proj = nn.ModuleDict({s: nn.Linear(hidden, hidden, bias=False) for s in streams})
        self.norm2 = nn.ModuleDict({s: nn.LayerNorm(hidden) for s in streams})
        self.ffn = nn.ModuleDict({s: SwiGLU(hidden, ffn) for s in streams})
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        parts: dict[str, torch.Tensor],
        mask: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        order = [s for s in self.streams if s in parts]
        qs, ks, vs = [], [], []
        for s in order:
            parts[s] = parts[s].to(self.qkv[s].weight.dtype)
            x = self.norm1[s](parts[s])
            B, N, _ = x.shape
            q, k, v = self.qkv[s](x).chunk(3, dim=-1)
            shape = lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            qs.append(shape(q))
            ks.append(shape(k))
            vs.append(shape(v))

        q_all = torch.cat(qs, dim=2)
        k_all = torch.cat(ks, dim=2)
        v_all = torch.cat(vs, dim=2)
        drop = self.dropout.p if self.training else 0.0

        # Actions and the future bank are mutually invisible. Running them as
        # one SDPA would still score every dead key. Split so each side only
        # pays for keys it can actually read, then share the same context.
        if "future" in order:
            n_act = sum(parts[s].shape[1] for s in order if s != "future")
            n_fut = parts["future"].shape[1]
            mask_act, mask_fut = split_action_future_masks(mask, n_act, n_fut)
            k_ctx = v_ctx = None
            if context_kv is not None:
                k_ctx, v_ctx = context_kv
            k_act = torch.cat([k_all[:, :, :n_act], k_ctx], dim=2) if k_ctx is not None else k_all[:, :, :n_act]
            v_act = torch.cat([v_all[:, :, :n_act], v_ctx], dim=2) if v_ctx is not None else v_all[:, :, :n_act]
            k_fut = torch.cat([k_all[:, :, n_act:], k_ctx], dim=2) if k_ctx is not None else k_all[:, :, n_act:]
            v_fut = torch.cat([v_all[:, :, n_act:], v_ctx], dim=2) if v_ctx is not None else v_all[:, :, n_act:]
            attn = torch.cat(
                [
                    sdpa(q_all[:, :, :n_act], k_act, v_act, mask_act, dropout_p=drop),
                    sdpa(q_all[:, :, n_act:], k_fut, v_fut, mask_fut, dropout_p=drop),
                ],
                dim=2,
            )
        else:
            if context_kv is not None:
                k_all = torch.cat([k_all, context_kv[0]], dim=2)
                v_all = torch.cat([v_all, context_kv[1]], dim=2)
            attn = sdpa(q_all, k_all, v_all, mask, dropout_p=drop)

        out: dict[str, torch.Tensor] = {}
        start = 0
        for s in order:
            n = parts[s].shape[1]
            chunk = attn[:, :, start : start + n]
            start += n
            B = chunk.shape[0]
            merged = chunk.transpose(1, 2).reshape(B, n, -1)
            h = parts[s] + self.dropout(self.proj[s](merged))
            out[s] = h + self.dropout(self.ffn[s](self.norm2[s](h)))
        return out


@dataclass
class ExpertOutput:
    predictions: dict[str, torch.Tensor]  # stream -> (B, chunk, slice_width)
    future_near: torch.Tensor | None
    future_far: torch.Tensor | None


class MMABCJointExpert(nn.Module):
    """Dual-head flow-matching action expert over a near/far split chunk."""

    def __init__(
        self,
        *,
        head_slices: dict[str, tuple[int, int]],
        chunk_size: int,
        near_steps: int,
        hidden: int = 1024,
        num_blocks: int = 16,
        num_heads: int = 16,
        ffn: int = 4096,
        dropout: float = 0.1,
        context_dim: int = 2560,
        injection_blocks: tuple[int, ...] = (0, 1, 2),
        context_mode: str = "deepstack",
        state_dim: int = 80,
        future_tokens: int = 192,
        future_dim: int | None = 2048,
        decoder_hidden: int = 1024,
    ) -> None:
        super().__init__()
        if not 0 < near_steps <= chunk_size:
            raise ValueError(f"near_steps {near_steps} outside chunk {chunk_size}")
        self.head_slices = {k: tuple(v) for k, v in head_slices.items()}
        self.chunk_size = chunk_size
        self.near_steps = near_steps
        self.hidden = hidden
        self.injection_blocks = tuple(injection_blocks)
        self.future_tokens = future_tokens
        self.future_enabled = future_dim is not None

        streams = list(ACTION_STREAMS)
        if self.future_enabled:
            streams.append("future")
        self.stream_names = tuple(streams)

        self.time_embed = TimestepEmbedding(hidden)
        self.action_encoders = nn.ModuleDict(
            {
                name: ActionEncoder(hi - lo, hidden)
                for name, (lo, hi) in self.head_slices.items()
            }
        )
        self.state_encoders = nn.ModuleDict(
            {name: StateEncoder(state_dim, hidden) for name in self.head_slices}
        )
        self.pos_embed = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, chunk_size + 1, hidden))
                for name in self.head_slices
            }
        )
        # Tells a token which side of the execute/replan boundary it is on. The
        # two segments share all weights, so this is the only thing that
        # distinguishes "will be executed" from "will probably be revised".
        self.segment_embed = nn.Parameter(torch.zeros(2, hidden))

        self.blocks = nn.ModuleList(
            [
                JointBlock(hidden, num_heads, ffn, self.stream_names, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.context = ContextEncoder(
            context_dim, hidden, num_heads, num_blocks, self.injection_blocks,
            context_mode=context_mode,
        )

        self.decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, decoder_hidden),
                    nn.GELU(),
                    nn.Linear(decoder_hidden, hi - lo),
                )
                for name, (lo, hi) in self.head_slices.items()
            }
        )

        if self.future_enabled:
            # Two banks: one predicting the frame at the end of the near
            # segment, one at the end of the far segment.
            self.future_queries = nn.Parameter(torch.zeros(1, 2 * future_tokens, hidden))
            nn.init.normal_(self.future_queries, std=0.02)
            self.future_decoder = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, future_dim)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.pos_embed.values():
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.segment_embed, std=0.02)
        for dec in self.decoders.values():
            # Small but non-zero: exact zeros make grad_input = W^T grad_out
            # identically zero, so nothing upstream trains until the decoder
            # has moved off zero.
            nn.init.normal_(dec[-1].weight, std=1e-3)
            nn.init.zeros_(dec[-1].bias)

    def _token_layout(self, *, include_future: bool, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream and segment id per query token, in concatenation order."""
        streams: list[int] = []
        segments: list[int] = []
        for name in ACTION_STREAMS:
            # The state token describes the present, so it belongs to `near`;
            # both segments may read it.
            streams.append(STREAM_ID[name])
            segments.append(NEAR)
            for step in range(self.chunk_size):
                streams.append(STREAM_ID[name])
                segments.append(NEAR if step < self.near_steps else FAR)
        if include_future:
            streams += [STREAM_ID["future"]] * (2 * self.future_tokens)
            segments += [NEAR] * self.future_tokens + [FAR] * self.future_tokens
        return (
            torch.tensor(streams, dtype=torch.long, device=device),
            torch.tensor(segments, dtype=torch.long, device=device),
        )

    def _segment_bias(self, device, dtype) -> torch.Tensor:
        """(chunk + 1, hidden) segment embedding for one action stream."""
        idx = torch.full((self.chunk_size + 1,), NEAR, dtype=torch.long, device=device)
        idx[1 + self.near_steps :] = FAR
        return self.segment_embed.to(dtype)[idx]

    def forward(
        self,
        *,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        memory: list[torch.Tensor],
        context_valid: torch.Tensor,
        aux_active: torch.Tensor,
        want_future: bool = False,
        context_cache: list | None = None,
    ) -> ExpertOutput:
        """Args:
            noisy_actions: (B, chunk, 80) interpolated action chunk, model space.
            flow_time: (B,) flow-matching time in [0, 1].
            state: (B, 80) normalised proprioception.
            state_mask: (B, 80) 1.0 where the dim is structurally present.
            memory: backbone taps; the last is the context, earlier ones are
                DeepStack-injected into it.
            context_valid: (B, S) 1.0 for non-padding context tokens.
            aux_active: (B,) 1.0 where whole-body labels exist.
            want_future: build the future stream (training only).
            context_cache: reusable per-block context keys/values from
                :meth:`build_context_cache`. Supplying it is what makes several
                noise draws per step cheap."""
        device = noisy_actions.device
        time_emb = self.time_embed(flow_time)

        parts: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in self.head_slices.items():
            # Each encoder casts against its own weights; see the note in
            # JointBlock.context_keys_values.
            enc_dtype = self.action_encoders[name].proj.weight.dtype
            tokens = self.action_encoders[name](
                noisy_actions[..., lo:hi].to(enc_dtype), time_emb.to(enc_dtype)
            )
            state_token = self.state_encoders[name](
                state.to(enc_dtype), state_mask.to(enc_dtype)
            )
            seq = torch.cat([state_token, tokens.to(state_token.dtype)], dim=1)
            dtype = seq.dtype
            seg_bias = self._segment_bias(device, dtype)
            parts[name] = seq + self.pos_embed[name].to(dtype) + seg_bias[None]

        include_future = bool(want_future and self.future_enabled)
        if include_future:
            fq = self.future_queries
            parts["future"] = fq.expand(state.shape[0], -1, -1)

        cache = context_cache if context_cache is not None else self.build_context_cache(memory)
        stream_ids, segment_ids = self._token_layout(include_future=include_future, device=device)
        mask = build_attention_mask(
            stream_ids,
            segment_ids,
            aux_active=aux_active,
            context_len=cache[0][0].shape[2],
            context_valid=context_valid,
        )

        for block, ctx_kv in zip(self.blocks, cache):
            parts = block(parts, mask, ctx_kv)

        predictions = {name: self.decoders[name](parts[name][:, 1:]) for name in self.head_slices}
        future_near = future_far = None
        if include_future:
            decoded = self.future_decoder(parts["future"])
            future_near = decoded[:, : self.future_tokens]
            future_far = decoded[:, self.future_tokens :]
        return ExpertOutput(
            predictions=predictions, future_near=future_near, future_far=future_far
        )

    def build_context_cache(self, memory: list[torch.Tensor]) -> list:
        """Per-block context keys and values, computed once per step.

        Delegates to :class:`ContextEncoder` through ``__call__`` rather than
        doing the work here. Reaching the parameters from a plain method skips
        the pre-forward hook that makes them whole under FSDP; see that class
        for what going without it costs.
        """
        return self.context(memory)
