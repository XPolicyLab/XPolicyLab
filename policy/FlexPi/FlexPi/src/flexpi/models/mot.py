from __future__ import annotations

from typing import Callable, ClassVar, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .wan_video_dit import flash_attention, modulate, rope_apply
from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy-initialized compiled flex_attention — only imported/compiled when needed
_compiled_flex_attn: Callable | None = None


def _get_compiled_flex_attn() -> Callable:
    global _compiled_flex_attn
    if _compiled_flex_attn is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex_attn = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attn


class MoT(nn.Module):
    """Mixture-of-Transformers with configurable attention backend.

    Args:
        mixtures: Dict mapping expert names to expert modules.
        mot_checkpoint_mixed_attn: Use gradient checkpointing for mixed attention.
        attn_mode: ``'sdpa'`` (default) uses Flash Attention / SDPA.
            ``'flex'`` uses FlexAttention for training forward passes
            (reads ``BlockMask`` from ``MoT.attention_mask`` class variable)
            and falls back to SDPA for inference paths
            (``prefill_video_cache``, ``forward_action_with_video_cache``).
    """

    # Class-level mask — set before each forward pass by init_flex_mask()
    attention_mask: ClassVar[Optional["BlockMask"]] = None  # noqa: F821
    attention_mask_seq_len: ClassVar[Optional[int]] = None

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        attn_mode: str = "sdpa",
        hbridge_enabled: bool = False,
        hbridge_bottom_ratio: float = 0.25,
        hbridge_top_ratio: float = 0.25,
    ):
        super().__init__()
        if attn_mode not in ("sdpa", "flex"):
            raise ValueError(f"`attn_mode` must be 'sdpa' or 'flex', got {attn_mode!r}")
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")
        if not (0.0 <= hbridge_bottom_ratio <= 1.0):
            raise ValueError(f"`hbridge_bottom_ratio` must be in [0,1], got {hbridge_bottom_ratio}")
        if not (0.0 <= hbridge_top_ratio <= 1.0):
            raise ValueError(f"`hbridge_top_ratio` must be in [0,1], got {hbridge_top_ratio}")
        if hbridge_bottom_ratio + hbridge_top_ratio > 1.0:
            raise ValueError(
                f"`hbridge_bottom_ratio + hbridge_top_ratio` must be <= 1.0, "
                f"got {hbridge_bottom_ratio} + {hbridge_top_ratio}"
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.attn_mode = attn_mode
        self.hbridge_enabled = hbridge_enabled
        self.hbridge_bottom_ratio = hbridge_bottom_ratio
        self.hbridge_top_ratio = hbridge_top_ratio
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, "
            f"num_layers={self.num_layers}, attn_mode={self.attn_mode}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")
        if self.hbridge_enabled:
            n_bottom = int(self.num_layers * self.hbridge_bottom_ratio)
            n_top = int(self.num_layers * self.hbridge_top_ratio)
            n_middle = self.num_layers - n_bottom - n_top
            logger.info(
                f"HBridge enabled: bottom={n_bottom}, middle={n_middle}, top={n_top} layers "
                f"(ratios: bottom={self.hbridge_bottom_ratio}, top={self.hbridge_top_ratio})"
            )

    def _is_outer_layer(self, layer_idx: int) -> bool:
        """Return True if this layer is in the bottom or top HBridge band.

        Outer layers process each sub-stream independently (no cross-modal attention).
        Middle layers run full joint attention. Returns False if HBridge is disabled.
        """
        if not self.hbridge_enabled:
            return False
        n_bottom = int(self.num_layers * self.hbridge_bottom_ratio)
        n_top = int(self.num_layers * self.hbridge_top_ratio)
        return layer_idx < n_bottom or layer_idx >= self.num_layers - n_top

    def set_attn_mode(self, mode: str):
        """Switch attention backend at runtime (e.g. 'flex' for training, 'sdpa' for inference)."""
        if mode not in ("sdpa", "flex"):
            raise ValueError(f"`mode` must be 'sdpa' or 'flex', got {mode!r}")
        self.attn_mode = mode

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        """Mixed attention with configurable backend.

        When ``attn_mode='flex'`` and no explicit ``attention_mask`` is passed,
        uses FlexAttention via the class-level ``MoT.attention_mask`` BlockMask.
        Otherwise falls back to SDPA (used by inference paths and ``attn_mode='sdpa'``).
        """
        if self.attn_mode == "flex" and MoT.attention_mask is not None and attention_mask is None:
            # FlexAttention path (training).
            # The BlockMask is created at a 128-aligned padded size.
            # Pad Q/K/V to match, run attention, then strip.
            actual_seq = q_cat.shape[1]
            mask_total = MoT.attention_mask.shape[-1]  # padded size
            pad_len = mask_total - actual_seq

            def _forward_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                if pad_len > 0:
                    q = F.pad(q, (0, 0, 0, pad_len))  # pad seq dim
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                q4d = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
                k4d = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
                v4d = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
                half_dtypes = (torch.float16, torch.bfloat16)
                if q4d.dtype not in half_dtypes:
                    q4d, k4d, v4d = q4d.to(torch.bfloat16), k4d.to(torch.bfloat16), v4d.to(torch.bfloat16)
                out = _get_compiled_flex_attn()(
                    q4d, k4d, v4d,
                    block_mask=MoT.attention_mask,
                    kernel_options={
                        "BLOCK_M": 64, "BLOCK_N": 64,
                        "BLOCK_M1": 32, "BLOCK_N1": 64,
                        "BLOCK_M2": 64, "BLOCK_N2": 32,
                    },
                )
                out = rearrange(out, "b n s d -> b s (n d)")
                if pad_len > 0:
                    out = out[:, :actual_seq, :]  # strip padding
                return out

            if self.mot_checkpoint_mixed_attn and self.training:
                return torch.utils.checkpoint.checkpoint(
                    _forward_flex, q_cat, k_cat, v_cat, use_reentrant=False,
                )
            return _forward_flex(q_cat, k_cat, v_cat)

        # SDPA path (default, and fallback for inference in flex mode).
        # attention_mask=None runs maskless SDPA — attn_backend="auto" drops
        # provably all-True masks so flash/cuDNN can dispatch instead of the
        # masked memory-efficient kernel (same math, faster kernel).
        attn_mask = attention_mask.to(device=q_cat.device) if attention_mask is not None else None

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_sub_stream_lens: Optional[list[int]] = None,
        video_sub_stream_self_masks: Optional[list[torch.Tensor]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].
            video_sub_stream_lens: Optional list of sub-stream sizes within the video
                stream (e.g. ``[Sv_raw, Sd, Sp]`` for Latent/3D/Unified variants).
                Must sum to ``Sv``. Only used when HBridge is enabled.
            video_sub_stream_self_masks: Per-sub-stream self-attention masks. In
                outer HBridge layers, each video sub-stream attends only to itself.

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        # None = full visibility (attn_backend="auto" drops the all-True mask).
        if video_attention_mask is not None:
            if video_attention_mask.ndim != 2:
                raise ValueError(
                    f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
                )
            if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
                raise ValueError(
                    f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
                )
            if video_attention_mask.shape[0] != video_tokens.shape[1]:
                raise ValueError(
                    "`video_attention_mask` seq length mismatch: "
                    f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
                )

        hbridge_active = (
            self.hbridge_enabled
            and video_sub_stream_lens is not None
            and video_sub_stream_self_masks is not None
            and len(video_sub_stream_lens) > 1  # nothing to split if only one sub-stream
        )
        if hbridge_active:
            if len(video_sub_stream_lens) != len(video_sub_stream_self_masks):
                raise ValueError(
                    f"`video_sub_stream_lens` ({len(video_sub_stream_lens)}) and "
                    f"`video_sub_stream_self_masks` ({len(video_sub_stream_self_masks)}) must have same length."
                )
            if sum(video_sub_stream_lens) != video_tokens.shape[1]:
                raise ValueError(
                    "`video_sub_stream_lens` sum must equal video token length: "
                    f"sum={sum(video_sub_stream_lens)} vs tokens={video_tokens.shape[1]}"
                )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            if hbridge_active and self._is_outer_layer(layer_idx):
                # HBridge outer layer: per-sub-stream self-attention within video.
                mixed_chunks = []
                offset = 0
                for sub_len, sub_mask in zip(video_sub_stream_lens, video_sub_stream_self_masks):
                    q_sub = q[:, offset:offset + sub_len, :]
                    k_sub = k[:, offset:offset + sub_len, :]
                    v_sub = v[:, offset:offset + sub_len, :]
                    mixed_chunks.append(
                        self._mixed_attention(q_cat=q_sub, k_cat=k_sub, v_cat=v_sub, attention_mask=sub_mask)
                    )
                    offset += sub_len
                mixed = torch.cat(mixed_chunks, dim=1)
            else:
                # Video prefill uses only video self-attention mask.
                mixed = self._mixed_attention(
                    q_cat=q,
                    k_cat=k,
                    v_cat=v,
                    attention_mask=video_attention_mask,
                )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_only_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.
            action_only_attention_mask: Optional [Sa, Sa] mask for HBridge outer
                layers. When ``hbridge_enabled`` and provided, outer layers skip
                the cached video K/V entirely (action queries attend to action
                K/V only), reducing per-step attention cost from O(Sa*(Sv+Sa))
                to O(Sa^2).

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        hbridge_active = (
            self.hbridge_enabled and action_only_attention_mask is not None
        )
        if hbridge_active:
            if action_only_attention_mask.ndim != 2:
                raise ValueError(
                    f"`action_only_attention_mask` must be 2D, got shape {tuple(action_only_attention_mask.shape)}"
                )
            if (
                action_only_attention_mask.shape[0] != action_seq_len
                or action_only_attention_mask.shape[1] != action_seq_len
            ):
                raise ValueError(
                    f"`action_only_attention_mask` must be [{action_seq_len}, {action_seq_len}], "
                    f"got {tuple(action_only_attention_mask.shape)}"
                )

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            if hbridge_active and self._is_outer_layer(layer_idx):
                # HBridge outer layer: action attends only to action K/V (skip cached video).
                mixed = self._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_action,
                    v_cat=v_action,
                    attention_mask=action_only_attention_mask,
                )
            else:
                # Mixed attention: action queries attend to cached video K/V plus current action K/V.
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)
                mixed = self._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def _prefill_video_cache_inner(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_sub_stream_lens: Optional[list[int]] = None,
        video_sub_stream_self_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Compile-friendly core of `prefill_video_cache`.

        Skips validation, NVTX, gradient checkpointing, and dict outputs.
        Inference-only. Supports HBridge per-sub-stream attention in outer
        layers when ``hbridge_enabled`` and sub-stream args are provided.
        """
        hbridge_active = (
            self.hbridge_enabled
            and video_sub_stream_lens is not None
            and video_sub_stream_self_masks is not None
            and len(video_sub_stream_lens) > 1
        )
        expert = self.mixtures["video"]
        x = video_tokens
        cache_k_list: list[torch.Tensor] = []
        cache_v_list: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q, k, v,
                residual_x, gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                _use_ckpt,
            ) = self._build_expert_attention_io(
                expert=expert, block=block, x=x,
                freqs=video_freqs, t_mod=video_t_mod,
            )
            if hbridge_active and self._is_outer_layer(layer_idx):
                mixed_chunks = []
                offset = 0
                for sub_len, sub_mask in zip(video_sub_stream_lens, video_sub_stream_self_masks):
                    q_sub = q[:, offset:offset + sub_len, :]
                    k_sub = k[:, offset:offset + sub_len, :]
                    v_sub = v[:, offset:offset + sub_len, :]
                    mixed_chunks.append(
                        self._mixed_attention(q_cat=q_sub, k_cat=k_sub, v_cat=v_sub, attention_mask=sub_mask)
                    )
                    offset += sub_len
                mixed = torch.cat(mixed_chunks, dim=1)
            else:
                mixed = self._mixed_attention(
                    q_cat=q, k_cat=k, v_cat=v,
                    attention_mask=video_attention_mask,
                )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=video_context_payload,
            )
            cache_k_list.append(k)
            cache_v_list.append(v)
        return x, cache_k_list, cache_v_list

    def _forward_action_with_video_cache_inner(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        action_only_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compile-friendly core of `forward_action_with_video_cache`.

        Caller pre-slices the action attention mask and flattens the K/V cache
        into parallel lists. Skips validation, NVTX, and gradient checkpointing.
        Inference-only. When ``hbridge_enabled`` and ``action_only_attention_mask``
        is provided, outer layers skip the cached video K/V (action-only attention).
        """
        hbridge_active = (
            self.hbridge_enabled and action_only_attention_mask is not None
        )
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_ckpt,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            if hbridge_active and self._is_outer_layer(layer_idx):
                mixed = self._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_action,
                    v_cat=v_action,
                    attention_mask=action_only_attention_mask,
                )
            else:
                k_video = video_cache_k[layer_idx]
                v_video = video_cache_v[layer_idx]
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)
                mixed = self._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=action_context_payload,
            )
        return x

    def _forward_joint_inner(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        attention_mask: Optional[torch.Tensor],
        sub_stream_lens: Optional[list[int]] = None,
        sub_stream_self_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compile-friendly variant of `forward` for the 2-expert "video"+"action"
        # joint case (auxiliary streams like DINO/pointmap are merged into the
        # video stream by the caller). Skips dict-keyed iteration, validation,
        # and gradient checkpointing so inductor can capture a single CUDA Graph.
        # `attention_mask=None` routes to FlexAttention via `MoT.attention_mask`.
        # Must produce the same outputs as `forward({"video": ..., "action": ...}, ...)`
        # for valid inputs.
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        hbridge_active = (
            self.hbridge_enabled
            and sub_stream_lens is not None
            and sub_stream_self_masks is not None
        )
        Sv = video_tokens.shape[1]
        Sa = action_tokens.shape[1]
        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_layers):
            v_block = video_expert.blocks[layer_idx]
            a_block = action_expert.blocks[layer_idx]
            (
                qv, kv, vv, residual_v, gate_msa_v,
                shift_mlp_v, scale_mlp_v, gate_mlp_v, _ckpt_v,
            ) = self._build_expert_attention_io(
                expert=video_expert, block=v_block, x=x_video,
                freqs=video_freqs, t_mod=video_t_mod,
            )
            (
                qa, ka, va, residual_a, gate_msa_a,
                shift_mlp_a, scale_mlp_a, gate_mlp_a, _ckpt_a,
            ) = self._build_expert_attention_io(
                expert=action_expert, block=a_block, x=x_action,
                freqs=action_freqs, t_mod=action_t_mod,
            )

            q_cat = torch.cat([qv, qa], dim=1)
            k_cat = torch.cat([kv, ka], dim=1)
            v_cat = torch.cat([vv, va], dim=1)

            if hbridge_active and self._is_outer_layer(layer_idx):
                mixed_chunks = []
                offset = 0
                for sub_len, sub_mask in zip(sub_stream_lens, sub_stream_self_masks):
                    q_sub = q_cat[:, offset:offset + sub_len, :]
                    k_sub = k_cat[:, offset:offset + sub_len, :]
                    v_sub = v_cat[:, offset:offset + sub_len, :]
                    mixed_chunks.append(
                        self._mixed_attention(
                            q_cat=q_sub, k_cat=k_sub, v_cat=v_sub,
                            attention_mask=sub_mask,
                        )
                    )
                    offset += sub_len
                mixed = torch.cat(mixed_chunks, dim=1)
            else:
                mixed = self._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat,
                    attention_mask=attention_mask,
                )

            mixed_v = mixed[:, :Sv, :]
            mixed_a = mixed[:, Sv:Sv + Sa, :]

            x_video = self._apply_expert_post_block(
                block=v_block,
                residual_x=residual_v,
                mixed_attn_out=mixed_v,
                gate_msa=gate_msa_v,
                shift_mlp=shift_mlp_v,
                scale_mlp=scale_mlp_v,
                gate_mlp=gate_mlp_v,
                context_payload=video_context_payload,
            )
            x_action = self._apply_expert_post_block(
                block=a_block,
                residual_x=residual_a,
                mixed_attn_out=mixed_a,
                gate_msa=gate_msa_a,
                shift_mlp=shift_mlp_a,
                scale_mlp=scale_mlp_a,
                gate_mlp=gate_mlp_a,
                context_payload=action_context_payload,
            )
        return x_video, x_action

    # ------------------------------------------------------------------
    # Joint KV-split (prefill clean anchors once, decode noisy per step).
    # The first-frame anchors (ff_v+ff_d+ff_p) attend only to anchors and are
    # modulated at t=0, so their per-layer K/V is step-invariant — prefill once
    # and reuse across every denoise step. Decode recomputes only the noisy
    # tokens (rem_v/rem_d/rem_p + action), reading [cached anchor K/V | fresh
    # noisy K/V]. Math matches `_forward_joint_inner` at the noisy positions
    # (bf16 attention-kernel-shape drift only; the anchor OUTPUT is identical).
    # HBridge: outer layers use the block-diagonal sub-stream mask, inner the
    # full mask — passed in pre-restricted to the anchor / noisy query rows.
    # ------------------------------------------------------------------
    def prefill_joint_anchor_kv(
        self,
        anchor_tokens: torch.Tensor,
        anchor_freqs: torch.Tensor,
        anchor_t_mod: torch.Tensor,
        anchor_context_payload: Optional[dict],
        anchor_inner_mask: torch.Tensor,
        anchor_outer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the video-expert over the clean anchor tokens and stack per-layer
        K/V. Returns ``k_all, v_all`` of shape ``[num_layers, B, Sanc, H*Dh]``
        (same stacked contract as the action-path ``prefill_video_cache``)."""
        ve = self.mixtures["video"]
        x = anchor_tokens
        ks, vs = [], []
        for li in range(self.num_layers):
            blk = ve.blocks[li]
            q, k, v, res, g_msa, sh, sc, g_mlp, _ = self._build_expert_attention_io(
                expert=ve, block=blk, x=x, freqs=anchor_freqs, t_mod=anchor_t_mod,
            )
            ks.append(k)
            vs.append(v)
            m = anchor_outer_mask if (self.hbridge_enabled and self._is_outer_layer(li)) else anchor_inner_mask
            mixed = self._mixed_attention(q_cat=q, k_cat=k, v_cat=v, attention_mask=m)
            x = self._apply_expert_post_block(
                block=blk, residual_x=res, mixed_attn_out=mixed, gate_msa=g_msa,
                shift_mlp=sh, scale_mlp=sc, gate_mlp=g_mlp,
                context_payload=anchor_context_payload,
            )
        return torch.stack(ks, dim=0), torch.stack(vs, dim=0)

    def decode_joint_noisy(
        self,
        noisy_video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        noisy_video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        noisy_video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        noisy_video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        anchor_pos_idx: torch.Tensor,
        noisy_pos_idx: torch.Tensor,
        video_seq_len: int,
        decode_inner_mask: torch.Tensor,
        decode_outer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the noisy tokens (rem video + action) attending to
        ``[cached anchor K/V | fresh noisy K/V]``. ``k_all/v_all`` are the
        stacked anchor cache from :meth:`prefill_joint_anchor_kv`;
        ``anchor_pos_idx``/``noisy_pos_idx`` scatter them back into full video
        order. Returns ``(noisy_video_out, action_out)``."""
        ve = self.mixtures["video"]
        ae = self.mixtures["action"]
        xv = noisy_video_tokens
        xa = action_tokens
        n_noisy_v = noisy_video_tokens.shape[1]
        for li in range(self.num_layers):
            vb = ve.blocks[li]
            ab = ae.blocks[li]
            qv, kv, vv, res_v, gmv, shv, scv, gmlv, _ = self._build_expert_attention_io(
                expert=ve, block=vb, x=xv, freqs=noisy_video_freqs, t_mod=noisy_video_t_mod,
            )
            qa, ka, va, res_a, gma, sha, sca, gmla, _ = self._build_expert_attention_io(
                expert=ae, block=ab, x=xa, freqs=action_freqs, t_mod=action_t_mod,
            )
            B, HD = kv.shape[0], kv.shape[-1]
            # assemble full-length video K/V: anchors cached, noisy fresh
            k_vid = kv.new_zeros((B, video_seq_len, HD)).index_copy(
                1, anchor_pos_idx, k_all[li]).index_copy(1, noisy_pos_idx, kv)
            v_vid = vv.new_zeros((B, video_seq_len, HD)).index_copy(
                1, anchor_pos_idx, v_all[li]).index_copy(1, noisy_pos_idx, vv)
            k_full = torch.cat([k_vid, ka], dim=1)
            v_full = torch.cat([v_vid, va], dim=1)
            q_noisy = torch.cat([qv, qa], dim=1)
            m = decode_outer_mask if (self.hbridge_enabled and self._is_outer_layer(li)) else decode_inner_mask
            mixed = self._mixed_attention(q_cat=q_noisy, k_cat=k_full, v_cat=v_full, attention_mask=m)
            mv = mixed[:, :n_noisy_v, :]
            ma = mixed[:, n_noisy_v:, :]
            xv = self._apply_expert_post_block(
                block=vb, residual_x=res_v, mixed_attn_out=mv, gate_msa=gmv,
                shift_mlp=shv, scale_mlp=scv, gate_mlp=gmlv,
                context_payload=noisy_video_context_payload,
            )
            xa = self._apply_expert_post_block(
                block=ab, residual_x=res_a, mixed_attn_out=ma, gate_msa=gma,
                shift_mlp=sha, scale_mlp=sca, gate_mlp=gmla,
                context_payload=action_context_payload,
            )
        return xv, xa

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        sub_stream_lens: Optional[list[int]] = None,
        sub_stream_self_masks: Optional[list[torch.Tensor]] = None,
    ):
        """Forward through all layers with mixed attention.

        When ``attention_mask`` is ``None`` and ``attn_mode='flex'``, uses
        FlexAttention via the class-level ``MoT.attention_mask`` BlockMask.

        When ``hbridge_enabled`` and ``sub_stream_lens`` / ``sub_stream_self_masks``
        are both provided, outer (bottom/top) layers run per-sub-stream attention
        independently using the provided self-masks. Inner layers use the full
        ``attention_mask`` joint attention as before. Sub-streams are slices of
        the concatenated ``[expert_0_tokens || expert_1_tokens || ...]`` sequence;
        ``sub_stream_lens`` must sum to that total length.
        """
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask is not None:
            if attention_mask.ndim not in (2, 3):
                raise ValueError(
                    f"`attention_mask` must be 2D [S, S] or 3D [B, S, S], "
                    f"got shape {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape[-1] != attention_mask.shape[-2]:
                raise ValueError(f"`attention_mask` must be square in trailing dims, got shape {tuple(attention_mask.shape)}")
        elif self.attn_mode != "flex" or MoT.attention_mask is None:
            raise ValueError(
                "`attention_mask` is None but attn_mode is not 'flex' or "
                "MoT.attention_mask is not set. Call init_flex_mask() or pass a 2D bool mask."
            )

        hbridge_active = (
            self.hbridge_enabled
            and sub_stream_lens is not None
            and sub_stream_self_masks is not None
        )
        if hbridge_active:
            if len(sub_stream_lens) != len(sub_stream_self_masks):
                raise ValueError(
                    f"`sub_stream_lens` ({len(sub_stream_lens)}) and "
                    f"`sub_stream_self_masks` ({len(sub_stream_self_masks)}) must have same length."
                )
            for i, (slen, smask) in enumerate(zip(sub_stream_lens, sub_stream_self_masks)):
                if smask.ndim not in (2, 3) or smask.shape[-1] != slen or smask.shape[-2] != slen:
                    raise ValueError(
                        f"`sub_stream_self_masks[{i}]` must be 2D [S,S] or 3D [B,S,S] of size {slen}, "
                        f"got shape {tuple(smask.shape)}"
                    )

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask is not None and attention_mask.shape[-1] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[-1]} vs tokens={total_seq}"
                )
            if hbridge_active and sum(sub_stream_lens) != total_seq:
                raise ValueError(
                    "`sub_stream_lens` sum must equal total token length: "
                    f"sum={sum(sub_stream_lens)} vs total={total_seq}"
                )

            if hbridge_active and self._is_outer_layer(layer_idx):
                # HBridge outer layer: per-sub-stream self-attention only.
                mixed_chunks = []
                offset = 0
                for sub_len, sub_mask in zip(sub_stream_lens, sub_stream_self_masks):
                    q_sub = q_cat[:, offset:offset + sub_len, :]
                    k_sub = k_cat[:, offset:offset + sub_len, :]
                    v_sub = v_cat[:, offset:offset + sub_len, :]
                    mixed_chunks.append(
                        self._mixed_attention(q_cat=q_sub, k_cat=k_sub, v_cat=v_sub, attention_mask=sub_mask)
                    )
                    offset += sub_len
                mixed = torch.cat(mixed_chunks, dim=1)
            else:
                mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
