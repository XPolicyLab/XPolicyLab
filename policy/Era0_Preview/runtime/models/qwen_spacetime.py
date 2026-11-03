# ------------------------------------------------------------------------------
# MEM-style space-time attention for the Qwen3.5 native vision tower.
#
# Port of the Hy-Embodied MEM recipe
# (`Hy-Embodied-0.5-VLA/hy_vla/space_time_attention.py`) onto the
# `Qwen3_5VisionModel` used by the X-VLA `qwen3_5` backend. Design goals,
# in priority order:
#
#   * ZERO new parameters. Every wrapped submodule is adopted BY REFERENCE
#     (the wrapper stores `.norm1 / .attn / .norm2 / .mlp` pointing at the
#     unpatched `Qwen3_5VisionBlock`'s children). state_dict layout is
#     byte-for-byte unchanged, so:
#       - mixed11 warm-start checkpoints load with no missing keys;
#       - checkpoints saved after the patch still load into unpatched code;
#       - DeepSpeed / FSDP parameter sharding is unaffected.
#
#   * K=1 exact degeneration. When the number of frames per (batch, view)
#     group equals one, `SpaceTimeVisionBlock.forward` short-circuits to
#     the unpatched block's exact op sequence -- byte-for-byte identical
#     to the legacy `qwen_temporal_embedding=off` path. This lets us keep
#     the historical single-frame training/eval numerically stable while
#     opting into MEM only for K>1 head views.
#
#   * K>1 is a bounded perturbation of pretrain behavior: a fixed
#     sinusoidal time embedding e(t) with e(0)=0 is added to the ATTENTION
#     BRANCH input only (not the residual stream), a causal time-softmax
#     mixes V across frames at each (image-token position, head), and the
#     standard per-image spatial attention runs afterwards with the mixed
#     V. QKV weights are reused across both steps (one qkv pass total).
#
#   * Vision tower output stays 49/196 tokens per view. The driver
#     `spacetime_forward_visual` returns ONLY the current-frame tokens
#     from `last_hidden_state`, and only the current-frame rows from
#     `pooler_output` after `visual.merger`. Downstream packed sequence
#     length, `_pack_multimodal`, and `compute_3d_position_ids` are
#     therefore identical to the K=1 path. This is exactly the design
#     that lets Florence's DaViT MEM patch avoid `max_len_seq=512`
#     overflow (see `models/spacetime_davit.py` header).
#
# Frame-slot convention (matches the RoboDojo history handler and
# `hy_vla/space_time_attention.py`): within a `(batch, view)` group the
# frames are laid out with slot K-1 as the CURRENT frame and slot 0 the
# oldest. e(t) is indexed by slot, so at K=1 the single (current) frame
# receives e(0)=0 -> strict identity.
#
# Contract with the driver:
#   1. The Qwen3.5 `_encode_image_native` caller MUST arrange `pixel_values`
#      so that frames belonging to the same (batch, view) group are
#      contiguous and ordered from slot 0 (oldest) to slot K-1 (current).
#      The `Qwen35Backend._encode_valid_views` history path already
#      compacts `[B*V, num_frames, C, H, W]` in that order, so the
#      driver only needs to hand us a `time_group_sizes` tensor telling
#      us how many frames each group contains.
#   2. The driver calls `set_time_context(vision_model, time_group_sizes,
#      grid_thw)` before invoking the model's forward, and clears the
#      context immediately after. Any exception during forward also
#      clears the context so the patched module never leaks state across
#      calls.
# ------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Base of the sinusoidal e(t) table. Chosen to match the Hy-VLA and
# `spacetime_davit.py` defaults so the temporal frequency spectrum lines
# up across all backends.
QWEN_SPACETIME_TIME_EMBED_BASE = 100.0


# ---------------------------------------------------------------------------
# Time-context container and (un)installation helpers
# ---------------------------------------------------------------------------


class _TimeContext:
    """Per-forward context describing the frame layout inside the ViT batch.

    Attributes
    ----------
    group_sizes:
        1-D `torch.long` tensor giving the number of frames K_g in each
        `(batch, view)` group. Sum(group_sizes) must equal the number of
        images the driver hands to `visual(...)`. Groups appear in the
        exact order they are stacked in `pixel_values` / `image_grid_thw`.
    tokens_per_frame:
        Integer, the pre-merge patch-token count per image (`grid_h * grid_w`).
        All frames in every group are assumed to share this value; the
        driver validates this against `image_grid_thw` before installing
        the context.
    """

    __slots__ = ("group_sizes", "tokens_per_frame")

    def __init__(self, group_sizes: torch.Tensor, tokens_per_frame: int) -> None:
        if group_sizes.dim() != 1:
            raise ValueError(
                "SpaceTime time-context group_sizes must be 1-D, got "
                f"{tuple(group_sizes.shape)}."
            )
        if group_sizes.dtype != torch.long:
            raise TypeError(
                "SpaceTime time-context group_sizes must be torch.long, got "
                f"{group_sizes.dtype}."
            )
        if bool((group_sizes < 1).any()):
            raise ValueError(
                "SpaceTime time-context group_sizes must all be >= 1."
            )
        if tokens_per_frame <= 0:
            raise ValueError(
                f"SpaceTime tokens_per_frame must be positive, got {tokens_per_frame}."
            )
        self.group_sizes = group_sizes
        self.tokens_per_frame = int(tokens_per_frame)


@contextlib.contextmanager
def set_time_context(
    visual: nn.Module,
    group_sizes: torch.Tensor,
    tokens_per_frame: int,
):
    """Attach a `_TimeContext` to `visual` for the duration of a forward.

    The `SpaceTimeVisionBlock` reads the attribute at each block; the
    context manager guarantees the attribute is cleared even if the
    vision model raises.
    """
    context = _TimeContext(group_sizes=group_sizes, tokens_per_frame=tokens_per_frame)
    previous = getattr(visual, "_hy_spacetime_context", None)
    visual._hy_spacetime_context = context
    try:
        yield context
    finally:
        if previous is None:
            if hasattr(visual, "_hy_spacetime_context"):
                delattr(visual, "_hy_spacetime_context")
        else:
            visual._hy_spacetime_context = previous


def _resolve_time_context(module: nn.Module) -> _TimeContext | None:
    """Walk up the wrapped-block chain to fetch the current context."""
    parent = getattr(module, "_hy_visual_parent", None)
    if parent is None:
        return None
    return getattr(parent, "_hy_spacetime_context", None)


# ---------------------------------------------------------------------------
# Fixed sinusoidal e(t) with e(0) = 0
# ---------------------------------------------------------------------------


def _build_time_pe(
    num_frames: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = QWEN_SPACETIME_TIME_EMBED_BASE,
) -> torch.Tensor:
    """Return `(num_frames, dim)` sinusoidal e(t) with e(0)=0.

    Uses the same construction as `hy_vla/space_time_attention.py` and
    `models/spacetime_davit.py`: even dims sin(t*f), odd dims cos(t*f)-1.
    The `-1` pins e(0) to exactly zero so the K=1 branch stays a strict
    identity to the unpatched block.
    """
    if dim <= 0 or dim % 2:
        raise ValueError(
            f"Qwen3.5 SpaceTime time-embed dim must be positive/even, got {dim}."
        )
    t = torch.arange(num_frames, dtype=torch.float32, device=device).unsqueeze(1)
    inv_freq = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(base) / dim)
    )
    pe = torch.empty(num_frames, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(t * inv_freq)
    pe[:, 1::2] = torch.cos(t * inv_freq) - 1.0
    return pe.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Wrapped Qwen3.5 vision block
# ---------------------------------------------------------------------------


class SpaceTimeVisionBlock(nn.Module):
    """`Qwen3_5VisionBlock` wrapper with MEM causal time mixing.

    Adopts the base block's `norm1 / attn / norm2 / mlp` submodules by
    reference; no new parameters are introduced. When the driver has not
    installed a time context, or the context contains no group with K>1,
    the wrapper is numerically identical to the base block.

    The wrapped forward keeps the standard Qwen3.5 API surface exactly:
    the driver calls the block with the same `(hidden_states, cu_seqlens,
    position_embeddings, **kwargs)` triple that `Qwen3_5VisionModel.forward`
    already emits, and returns a `[seq_len, hidden_size]` tensor with the
    identical shape and dtype semantics.
    """

    def __init__(self, base_block: nn.Module, parent_visual: nn.Module) -> None:
        super().__init__()
        # Adopt submodules by reference. Keeping the attribute names
        # identical is what makes `state_dict()` on the containing
        # `nn.ModuleList` produce keys byte-for-byte identical to the
        # unpatched block. `nn.Module.__setattr__` will register these as
        # children, which is fine -- both the wrapper and the source
        # `Qwen3_5VisionBlock` point at the SAME `nn.Module` instances,
        # so parameters remain unique (each key resolves to one shared
        # tensor via reference equality) and DeepSpeed sees them exactly
        # once because the source block is replaced in `blocks[i]`.
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp

        # A back-pointer to the containing `Qwen3_5VisionModel` so the
        # forward can pick up the driver-installed `_TimeContext`.
        # Registered as a non-persistent, non-parameter attribute; not
        # stored via `add_module` so it does not appear in `state_dict()`
        # or `named_children()`.
        object.__setattr__(self, "_hy_visual_parent", parent_visual)

        # Cache the block hidden dim once so the fixed sinusoidal table
        # can be rebuilt on-device inside `forward`. The attribute lookup
        # is a Python int, not a parameter, and therefore stays out of
        # `state_dict()`.
        head_dim = int(self.attn.head_dim)
        num_heads = int(self.attn.num_heads)
        object.__setattr__(self, "_hy_hidden_dim", int(head_dim * num_heads))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Numerically identical to `Qwen3_5VisionBlock.forward`."""
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        context = _resolve_time_context(self)
        if context is None or bool((context.group_sizes > 1).any()) is False:
            # K=1 or no context -> strict identity.
            return self._base_forward(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Bounded-perturbation MEM path. Structure mirrors the Hy-VLA
        # SpaceTimeBlock closely:
        #   1. norm1(x) + e(t) as the attention-branch input;
        #   2. compute Q/K/V from the same qkv linear as the base attn;
        #   3. causal time-softmax mixes V across frames within each
        #      `(batch, view)` group at every (image-token position, head);
        #   4. run the standard per-image spatial attention on (q, k,
        #      v_mixed), reusing the base `attn.proj`;
        #   5. add residual, then the untouched MLP branch.

        group_sizes = context.group_sizes.to(device=hidden_states.device)
        tokens_per_frame = context.tokens_per_frame

        # Sanity: sum of frames * tokens_per_frame must equal seq_len.
        expected_seq_len = int(group_sizes.sum().item()) * tokens_per_frame
        if hidden_states.shape[0] != expected_seq_len:
            raise RuntimeError(
                "Qwen3.5 SpaceTime block expected seq_len "
                f"{expected_seq_len} = tokens_per_frame({tokens_per_frame}) * "
                f"sum(group_sizes)({int(group_sizes.sum().item())}), got "
                f"{hidden_states.shape[0]}."
            )

        # -- (1) attention-branch input with e(t) --
        # Each group indexes e(t) locally as 0..k-1, with slot k-1 being
        # the current frame. That preserves two invariants:
        #   * K=1 groups (e.g. wrist views under IMG_HISTORY_PER_VIEW=6,1,1)
        #     see e(0) = 0 on their single current frame, so they are
        #     byte-for-byte identical to a K=1-only encode on the
        #     attention-branch input.
        #   * K>1 groups get a monotone e(0)..e(K-1) tail with e(0)
        #     pinned to zero, matching the Hy-VLA/Florence MEM recipe.
        max_frames = int(group_sizes.max().item())
        pe_table = _build_time_pe(
            max_frames,
            self._hy_hidden_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )  # (max_K, D)

        frame_time_pe = pe_table.new_zeros(int(group_sizes.sum().item()), self._hy_hidden_dim)
        cursor = 0
        for k_val in group_sizes.tolist():
            k = int(k_val)
            frame_time_pe[cursor : cursor + k] = pe_table[:k]
            cursor += k

        # Broadcast to per-token e(t): every image contributes
        # tokens_per_frame consecutive positions in `hidden_states`.
        time_pe = frame_time_pe.repeat_interleave(tokens_per_frame, dim=0)  # (seq_len, D)
        # Hy parity: add e(t) BEFORE the LayerNorm, so norm1 sees
        # (x + pe). This keeps the K=1 identity strict when pe[0]=0
        # and matches both `hy_vla/space_time_attention.py` and
        # `models/spacetime_davit.py` layer semantics. The residual
        # stream still adds the un-embedded `hidden_states` further
        # down, exactly like the base block does.
        h_attn_input = self.norm1(hidden_states + time_pe)

        # -- (2) Q/K/V from base attn --
        attn = self.attn
        seq_len = hidden_states.shape[0]
        qkv = attn.qkv(h_attn_input).reshape(seq_len, 3, attn.num_heads, -1)
        query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
        # Import the vendored rotary util lazily so this module stays
        # importable in environments without the qwen3_5 site-package
        # (e.g. Florence-only .venv). Fatal at first use, never at
        # import time.
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_rotary_pos_emb_vision,
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )
        # (seq_len, H, d) each.

        # -- (3) causal time-softmax mixes V across frames --
        value_mixed = _time_softmax_on_value(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            group_sizes=group_sizes,
            tokens_per_frame=tokens_per_frame,
            scaling=attn.scaling,
        )

        # -- (4) standard per-image spatial attention with mixed V --
        # We reuse the base attention's implementation exactly, save for
        # substituting V. That gives us one W_O application and preserves
        # any flash / SDPA kernel selection the base attention already
        # decided on.
        attn_out = _spatial_attention(
            attn=attn,
            query_states=query_states,
            key_states=key_states,
            value_states=value_mixed,
            cu_seqlens=cu_seqlens,
            training=self.training,
            extra_kwargs=kwargs,
        )

        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Causal time softmax and spatial attention helpers
# ---------------------------------------------------------------------------


def _time_softmax_on_value(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    group_sizes: torch.Tensor,
    tokens_per_frame: int,
    scaling: float,
) -> torch.Tensor:
    """Return V mixed across frames within each (batch, view) group.

    Inputs are the post-RoPE q, k, v tensors of shape `(seq_len, H, d)`
    laid out as `[group0_frame0, group0_frame1, ..., group0_frame{K0-1},
    group1_frame0, ...]`, with each frame contributing `tokens_per_frame`
    consecutive rows.

    The output has the same shape as `value_states`; groups whose K==1
    are copied through unchanged (single-frame identity).
    """
    if query_states.shape != key_states.shape or query_states.shape != value_states.shape:
        raise RuntimeError(
            "Qwen3.5 SpaceTime time softmax requires matching q/k/v shapes; "
            f"got q={tuple(query_states.shape)}, k={tuple(key_states.shape)}, "
            f"v={tuple(value_states.shape)}."
        )

    num_heads = query_states.shape[1]
    head_dim = query_states.shape[2]
    device = value_states.device

    v_mixed = value_states.clone()

    cursor = 0
    group_sizes_list = group_sizes.tolist()
    for k_val in group_sizes_list:
        k = int(k_val)
        group_len = k * tokens_per_frame
        if k == 1:
            cursor += group_len
            continue

        # Slice the (K * N, H, d) rows for this group.
        q_g = query_states[cursor : cursor + group_len]
        k_g = key_states[cursor : cursor + group_len]
        v_g = value_states[cursor : cursor + group_len]

        # Fold (K, N) so time is the softmax axis at every (n, head).
        # Layout: (K, N, H, d) -> transpose to (N, H, K, d).
        q_kn = q_g.view(k, tokens_per_frame, num_heads, head_dim).permute(1, 2, 0, 3)
        k_kn = k_g.view(k, tokens_per_frame, num_heads, head_dim).permute(1, 2, 0, 3)
        v_kn = v_g.view(k, tokens_per_frame, num_heads, head_dim).permute(1, 2, 0, 3)

        # Score in fp32 to avoid bf16 accumulation error in the softmax.
        scores = torch.matmul(q_kn.float(), k_kn.float().transpose(-2, -1)) * scaling
        causal = torch.triu(
            torch.ones(k, k, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))
        weights = scores.softmax(dim=-1).to(v_kn.dtype)  # (N, H, K, K)
        v_mixed_kn = torch.matmul(weights, v_kn)  # (N, H, K, d)

        # Unpermute back to (K, N, H, d) then flatten to (K*N, H, d).
        v_mixed_slice = v_mixed_kn.permute(2, 0, 1, 3).contiguous().view(group_len, num_heads, head_dim)
        v_mixed[cursor : cursor + group_len] = v_mixed_slice

        cursor += group_len

    return v_mixed


def _spatial_attention(
    *,
    attn: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    training: bool,
    extra_kwargs: dict | None = None,
) -> torch.Tensor:
    """Reproduce `Qwen3_5VisionAttention.forward` from post-RoPE q/k/v.

    We avoid calling `attn.forward` directly because that would re-run
    the qkv linear and RoPE. Instead we replicate the tail of the base
    method verbatim (kernel selection, cu_seqlens dispatch, final proj),
    so any future kernel-selection change in the vendored file requires
    revisiting exactly one place: this helper.

    `extra_kwargs` mirrors the base attention's `**kwargs` pass-through:
    the containing `Qwen3_5VisionModel.forward` may add flash-specific
    hints (e.g. training-time position-id metadata inserted by the
    `@merge_with_config_defaults` decorator) that must reach the
    selected attention kernel unchanged.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils.generic import is_flash_attention_requested

    if extra_kwargs is None:
        extra_kwargs = {}

    seq_length = query_states.shape[0]
    # (seq_len, H, d) -> (1, H, seq_len, d), matching the base file.
    q = query_states.transpose(0, 1).unsqueeze(0)
    k = key_states.transpose(0, 1).unsqueeze(0)
    v = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )

    if is_flash_attention_requested(attn.config):
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            attn,
            q,
            k,
            v,
            attention_mask=None,
            scaling=attn.scaling,
            dropout=0.0 if not training else attn.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **extra_kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (q, k, v)
        ]
        attn_outputs = [
            attention_interface(
                attn,
                qi,
                ki,
                vi,
                attention_mask=None,
                scaling=attn.scaling,
                dropout=0.0 if not training else attn.attention_dropout,
                is_causal=False,
                **extra_kwargs,
            )[0]
            for qi, ki, vi in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return attn.proj(attn_output)


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------


def apply_spacetime_patch(
    visual: nn.Module,
    stride: int = 4,
) -> nn.Module:
    """Install `SpaceTimeVisionBlock` at every `stride`-th vision block.

    Idempotent: calling it a second time re-uses the wrappers already in
    place. Only `Qwen3_5VisionBlock` instances are wrapped; anything
    else (e.g. `SpaceTimeVisionBlock` from a previous call) is skipped.

    Layer selection follows the Florence port
    (`models/spacetime_davit.py:281`) and the Hy-VLA original:
    `blocks[stride-1::stride]`, so the last block is always wrapped.
    The last-block guarantee is important because it forces the mixed
    representation to reach the merger, i.e. the tokens the LLM
    ultimately consumes are affected by cross-frame mixing.

    Parameters
    ----------
    visual:
        The `Qwen3_5VisionModel` instance whose `.blocks` should be
        wrapped. The instance is modified in-place; the return value is
        the same object for chaining convenience.
    stride:
        Interval between wrapped blocks. Default 4 mirrors the Hy-VLA
        MEM recipe, which wraps every fourth block. Must be >= 1.
    """
    if stride < 1:
        raise ValueError(f"apply_spacetime_patch stride must be >= 1, got {stride}.")

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionBlock

    blocks: nn.ModuleList = visual.blocks
    for idx in range(stride - 1, len(blocks), stride):
        block = blocks[idx]
        if isinstance(block, SpaceTimeVisionBlock):
            continue
        if not isinstance(block, Qwen3_5VisionBlock):
            raise TypeError(
                "apply_spacetime_patch expected Qwen3_5VisionBlock at "
                f"blocks[{idx}], got {type(block).__name__}."
            )
        blocks[idx] = SpaceTimeVisionBlock(block, parent_visual=visual)

    # Record the stride on the model for later introspection; not
    # serialized because `Qwen3_5VisionModel.state_dict` walks children,
    # not arbitrary attributes.
    visual._hy_spacetime_stride = int(stride)
    return visual


# ---------------------------------------------------------------------------
# Driver for the Qwen backend
# ---------------------------------------------------------------------------


def encode_with_current_frame_only(
    *,
    visual: nn.Module,
    get_image_features_fn,
    pixel_values_patches: torch.Tensor,
    image_grid_thw: torch.Tensor,
    group_sizes: torch.Tensor,
    tokens_per_frame: int,
    group_sizes_list: tuple[int, ...] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run one Qwen3.5 vision forward with MEM-style time mixing.

    Semantics
    ---------
    * The caller supplies patches for *all* frames in group order (oldest
      to current within each group). This function attaches the required
      `_TimeContext` to `visual`, invokes `get_image_features_fn(patches,
      grids)`, then slices out **only the current-frame rows** of the
      resulting merged (49/view) and pre-merge (196/view) features. The
      returned lists therefore look identical in shape/count to a K=1
      encode of the current frames alone, but their values incorporate
      MEM time mixing inside the wrapped vision blocks.
    * `get_image_features_fn` must be `owner.vlm.model.get_image_features`
      (or an equivalent callable that forwards `pixel_values` and
      `image_grid_thw` to `visual`). This indirection keeps this module
      independent of the exact backend model class.

    Returns
    -------
    (merged_current_rows, detail_current_rows), each a list of length
    `len(group_sizes)`, entry `g` being the current-frame tokens for
    group `g`:
      * `merged_current_rows[g]` has shape `[49, D_out]` (post-merger),
      * `detail_current_rows[g]` has shape `[196, D_hidden]` (pre-merger).

    Rationale
    ---------
    The RoboDojo per-view history handler emits frames oldest-first
    (`hy_vla/data/robodojo_dataset.py:721-745`), and the caller in
    `Qwen35Backend._encode_valid_views` compacts wrist padding before
    calling us so `group_sizes` faithfully describes the number of real
    frames per (batch, view) group. Returning only the current frame's
    rows keeps the packed multimodal sequence length (49/view merged,
    196/view detail) and `compute_3d_position_ids` completely untouched.
    """
    if pixel_values_patches.dim() != 2:
        raise ValueError(
            "encode_with_current_frame_only expects patchified `pixel_values` "
            f"with shape (num_tokens, patch_dim), got {tuple(pixel_values_patches.shape)}."
        )
    total_images = int(image_grid_thw.shape[0])
    group_counts = tuple(int(k) for k in group_sizes.tolist())
    if group_sizes_list is not None and tuple(group_sizes_list) != group_counts:
        raise ValueError("group_sizes_list must match group_sizes in order and count")
    if int(group_sizes.sum().item()) != total_images:
        raise ValueError(
            "encode_with_current_frame_only: sum(group_sizes)="
            f"{int(group_sizes.sum().item())} must equal image_grid_thw rows "
            f"({total_images})."
        )

    # Every frame is one 224x224 image, so every row of image_grid_thw
    # must be (1, sqrt(tokens_per_frame), sqrt(tokens_per_frame)). The
    # QWEN_POLICY_IMAGE_SIZE contract is enforced by `_encode_image_native`
    # before we get here; we still guard against silently drifting grids.
    if bool((image_grid_thw[:, 1] * image_grid_thw[:, 2] != tokens_per_frame).any()):
        raise RuntimeError(
            "encode_with_current_frame_only: all frames must share "
            f"tokens_per_frame={tokens_per_frame}, got grids "
            f"{image_grid_thw.tolist()}."
        )

    with set_time_context(
        visual,
        group_sizes=group_sizes.to(dtype=torch.long),
        tokens_per_frame=tokens_per_frame,
    ):
        outputs = get_image_features_fn(
            pixel_values=pixel_values_patches,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    # Merged tokens are `spatial_merge_size**2`-times fewer per image
    # than pre-merger tokens. For the Qwen policy contract this is
    # 4x, giving 196 detail / 49 merged tokens per view.
    merge_ratio = tokens_per_frame // int(_infer_merged_tokens_per_frame(visual, tokens_per_frame))
    if merge_ratio <= 0:
        raise RuntimeError(
            "encode_with_current_frame_only failed to infer merger ratio; "
            f"tokens_per_frame={tokens_per_frame}."
        )
    merged_tokens_per_frame = tokens_per_frame // merge_ratio

    # `pooler_output` is a tuple/list of per-image tensors split by
    # `get_image_features` in the current qwen3_5 file; older forks
    # returned a single stacked tensor. Support both so this helper
    # stays robust across the site's Transformers pin. `last_hidden_state`
    # remains a single stacked tensor over all images.
    pooler_output = outputs.pooler_output
    if isinstance(pooler_output, torch.Tensor):
        merged_tokens_expected = tokens_per_frame // (
            int(getattr(visual, "spatial_merge_unit", 0) or (
                int(visual.spatial_merge_size) ** 2
            ))
        )
        pooler_output = list(
            torch.split(
                pooler_output,
                [merged_tokens_expected] * total_images,
                dim=0,
            )
        )
    elif isinstance(pooler_output, (list, tuple)):
        pooler_output = list(pooler_output)
    else:
        raise TypeError(
            "encode_with_current_frame_only expected pooler_output as "
            "Tensor or tuple/list of Tensors, got "
            f"{type(pooler_output).__name__}."
        )
    if len(pooler_output) != total_images:
        raise RuntimeError(
            "encode_with_current_frame_only: pooler_output length "
            f"{len(pooler_output)} does not match total_images={total_images}."
        )

    detail_all = outputs.last_hidden_state
    if detail_all.shape[0] != total_images * tokens_per_frame:
        raise RuntimeError(
            "encode_with_current_frame_only: last_hidden_state rows "
            f"{detail_all.shape[0]} != total_images*tokens_per_frame="
            f"{total_images * tokens_per_frame}."
        )

    # Assemble one (merged, detail) pair per group, taking the last
    # frame (current frame) only.
    merged_current: list[torch.Tensor] = []
    detail_current: list[torch.Tensor] = []
    image_cursor = 0
    detail_cursor = 0
    for k_val in group_counts:
        k = int(k_val)
        current_image_idx = image_cursor + k - 1
        merged_current.append(pooler_output[current_image_idx])
        detail_start = (current_image_idx) * tokens_per_frame
        detail_current.append(detail_all[detail_start : detail_start + tokens_per_frame])
        image_cursor += k
        detail_cursor += k * tokens_per_frame

    _ = merged_tokens_per_frame  # currently unused, kept for future validation
    return merged_current, detail_current


def _infer_merged_tokens_per_frame(visual: nn.Module, tokens_per_frame: int) -> int:
    """Return the post-merger token count per image.

    `Qwen3_5VisionPatchMerger` collapses `spatial_merge_size**2`
    pre-merger tokens into one post-merger token, so the ratio is
    `spatial_merge_size**2`. We read this from the model config rather
    than hard-coding it, so a future Qwen size that changes the merger
    ratio does not silently produce wrong slicing.
    """
    merge_size = int(getattr(visual, "spatial_merge_size", 0) or 0)
    if merge_size <= 0:
        raise RuntimeError(
            "Cannot infer Qwen3.5 spatial_merge_size from vision model."
        )
    merged = tokens_per_frame // (merge_size * merge_size)
    if merged * merge_size * merge_size != tokens_per_frame:
        raise RuntimeError(
            "Qwen3.5 tokens_per_frame is not divisible by "
            f"spatial_merge_size^2 ({merge_size ** 2}); "
            f"got tokens_per_frame={tokens_per_frame}."
        )
    return merged
