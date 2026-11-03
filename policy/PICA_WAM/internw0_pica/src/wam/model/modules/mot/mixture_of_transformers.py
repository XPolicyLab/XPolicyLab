from __future__ import annotations

import os
from contextlib import nullcontext

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbones.wan22.wan_video_dit import flash_attention, modulate, rope_apply
from wam.utils.logging_config import get_logger
from .kv_cache import MoTCacheMixin
from .compile import MoTCompileMixin

logger = get_logger(__name__)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - depends on torch build.
    create_block_mask = None
    flex_attention = None


def _round_up_to_multiple(value: int, multiple: int) -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def _wam_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    kernel_options: dict[str, int | bool],
    key_mask: Optional[torch.Tensor],
    empty_key_batch: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run FlexAttention with the same per-sample key semantics as SDPA."""

    score_mod = None
    if key_mask is not None:

        def score_mod(score, b, h, q_idx, kv_idx):
            valid = key_mask[b, kv_idx]
            if empty_key_batch is not None:
                valid = valid | (empty_key_batch[b] & (q_idx == kv_idx))
            return torch.where(valid, score, -torch.inf)

    return flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        kernel_options=kernel_options,
    )


def _profile_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


class MoT(MoTCacheMixin, MoTCompileMixin, nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        checkpoint_layer_stride: int = 1,
        compile_mode: str = "off",
        compile_gradient_checkpointing: bool = False,
        compile_action_context_pad_to: int = 0,
        attention_backend: str = "flex",
        flex_block_size: int = 64,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self._init_compile(compile_mode, compile_gradient_checkpointing, compile_action_context_pad_to)
        self.checkpoint_layer_stride = int(checkpoint_layer_stride)
        if self.checkpoint_layer_stride < 1:
            raise ValueError("checkpoint_layer_stride must be at least 1")
        self.checkpoint_extra_layers = frozenset()
        self.checkpoint_preserve_rng_state = True
        self.attention_backend = str(attention_backend or "flex").strip().lower()
        if self.attention_backend not in {"flex", "sdpa"}:
            raise ValueError(f"`attention_backend` must be 'flex' or 'sdpa', got {attention_backend!r}.")
        if self.attention_backend == "flex" and (flex_attention is None or create_block_mask is None):
            raise RuntimeError(
                "MoT attention_backend='flex' requires torch.nn.attention.flex_attention, "
                "but it is not available in this torch build."
            )
        self.flex_block_size = max(16, int(flex_block_size))
        if self.flex_block_size not in {16, 32, 64, 128}:
            raise ValueError(
                "`flex_block_size` must be one of {16, 32, 64, 128}; "
                f"got {self.flex_block_size}."
            )
        self.flex_mask_block_size = 128
        self._flex_block_mask_cache: dict[tuple[Any, ...], Any] = {}
        self._dynamic_flex_block_mask_key: Optional[tuple[Any, ...]] = None
        self._dynamic_flex_block_mask_result = None
        self._flex_attention_compiled = bool(
            self.attention_backend == "flex" and hasattr(torch, "compile") and torch.cuda.is_available()
        )
        self._flex_attention = (
            torch.compile(_wam_flex_attention, dynamic=True, fullgraph=True)
            if self._flex_attention_compiled
            else _wam_flex_attention
        )
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
        
        self._log_expert_summary("Initialized")

    @staticmethod
    def _num_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def _log_expert_summary(self, verb: str) -> None:
        logger.info(
            "%s MoT with experts: %s, num_layers=%d, attention_backend=%s",
            verb,
            self.expert_order,
            self.num_layers,
            self.attention_backend,
        )
        for name in self.expert_order:
            logger.debug(
                "  Expert '%s': num_params=%.2f B",
                name,
                self._num_params(self.mixtures[name]) / 1e9,
            )

    def _validate_new_expert(self, name: str, expert: nn.Module) -> None:
        if len(expert.blocks) != self.num_layers:
            raise ValueError(
                f"Expert {name!r} must have {self.num_layers} layers, got {len(expert.blocks)}."
            )
        if int(expert.num_heads) != int(self.num_heads):
            raise ValueError(
                f"Expert {name!r} num_heads mismatch: {expert.num_heads} vs {self.num_heads}."
            )
        if int(expert.attn_head_dim) != int(self.attn_head_dim):
            raise ValueError(
                f"Expert {name!r} attn_head_dim mismatch: {expert.attn_head_dim} vs {self.attn_head_dim}."
            )

    def add_expert(self, name: str, expert: nn.Module) -> None:
        name = str(name)
        if name in self.mixtures:
            raise ValueError(f"MoT already has expert {name!r}.")
        self._validate_new_expert(name, expert)
        self.mixtures[name] = expert
        self.expert_order.append(name)
        self._flex_block_mask_cache.clear()
        object.__setattr__(self, "_compiled_groups", {})
        self._log_expert_summary("Updated")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        # Enter the owning block through Module.__call__ so ZeRO-3 gathers the
        # block-level modulation parameter before it is read.
        return block(t_mod=t_mod, modulation_only=True)

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask=None,
        flex_key_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        allow_self_when_all_masked: bool = False,
        force_sdpa: bool = False,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            if self.attention_backend == "sdpa" or force_sdpa or (attn_mask.ndim == 4 and attn_mask.shape[0] > 1):
                effective_mask = self._apply_key_mask(
                    attn_mask,
                    flex_key_mask,
                    allow_self_when_all_masked=allow_self_when_all_masked,
                )
                effective_mask = self._apply_query_mask(
                    effective_mask,
                    query_mask,
                )
                return flash_attention(
                    q=q,
                    k=k,
                    v=v,
                    num_heads=self.num_heads,
                    ctx_mask=effective_mask,
                )

            block_mask = flex_block_mask
            if block_mask is None:
                block_mask = self._build_flex_block_mask(
                    attention_mask=attn_mask,
                    batch_size=int(q.shape[0]),
                    query_len=int(q.shape[1]),
                    key_len=int(k.shape[1]),
                    device=q.device,
                )
            batch_size, query_len, _ = q.shape
            key_len = int(k.shape[1])
            q_heads = q.reshape(
                batch_size, query_len, self.num_heads, self.attn_head_dim
            ).transpose(1, 2)
            k_heads = k.reshape(
                batch_size, key_len, self.num_heads, self.attn_head_dim
            ).transpose(1, 2)
            v_heads = v.reshape(
                batch_size, key_len, self.num_heads, self.attn_head_dim
            ).transpose(1, 2)
            out = self._flex_attention(
                q_heads,
                k_heads,
                v_heads,
                block_mask,
                {"BLOCK_M": self.flex_block_size, "BLOCK_N": self.flex_block_size}
                if not self.training else {},
                flex_key_mask,
                (
                    (~flex_key_mask.any(dim=1)).contiguous()
                    if flex_key_mask is not None and allow_self_when_all_masked
                    else None
                ),
            )
            return out.transpose(1, 2).reshape(
                batch_size,
                query_len,
                self.num_heads * self.attn_head_dim,
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
                preserve_rng_state=self.checkpoint_preserve_rng_state,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_query_mask(
        attention_mask: torch.Tensor,
        query_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if query_mask is None:
            return attention_mask
        if attention_mask.ndim not in (2, 4):
            raise ValueError(
                "`attention_mask` must be 2D or 4D when applying query_mask, "
                f"got {tuple(attention_mask.shape)}"
            )
        if query_mask.ndim != 2:
            raise ValueError(
                f"`query_mask` must be [B,S], got {tuple(query_mask.shape)}"
            )
        if int(attention_mask.shape[-2]) != int(query_mask.shape[1]):
            raise ValueError(
                "`query_mask` seq length mismatch: "
                f"mask={query_mask.shape[1]} vs attention queries={attention_mask.shape[-2]}"
            )
        if int(attention_mask.shape[-1]) != int(query_mask.shape[1]):
            raise ValueError("Query masking requires square self-attention.")
        effective = attention_mask.to(device=query_mask.device, dtype=torch.bool)
        if effective.ndim == 2:
            effective = effective.unsqueeze(0).unsqueeze(1)
        elif int(effective.shape[0]) not in (1, int(query_mask.shape[0])):
            raise ValueError(
                "`attention_mask` batch dimension must be 1 or match query_mask: "
                f"{effective.shape[0]} vs {query_mask.shape[0]}"
            )
        valid_queries = query_mask[:, None, :, None].to(dtype=torch.bool)
        self_only = torch.eye(
            int(query_mask.shape[1]),
            dtype=torch.bool,
            device=query_mask.device,
        )[None, None]
        return torch.where(valid_queries, effective, self_only)

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
        context_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            def _mask_rows(mask: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
                if mask is None:
                    return None
                if mask.dim() == 3:
                    return mask[:, int(start) : int(end), :]
                elif mask.dim() == 4:
                    return mask[:, :, int(start) : int(end), :]
                raise ValueError(f"Context mask must be [B,S,L] or [B,1,S,L], got {tuple(mask.shape)}")

            def _finalize_context_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if mask is None:
                    return None
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                elif mask.dim() != 4:
                    raise ValueError(f"Context mask must be [B,S,L] or [B,1,S,L], got {tuple(mask.shape)}")
                if mask.shape[-1] > 0:
                    row_has_key = mask.any(dim=-1, keepdim=True)
                    fallback = torch.zeros_like(mask)
                    fallback[..., 0:1] = True
                    mask = torch.where(row_has_key, mask, fallback)
                return mask

            if context is not None:
                raw_mask = _mask_rows(
                    context_payload.get("mask"), 0, int(x.shape[1])
                )
                context_row_enabled = None
                if raw_mask is not None:
                    context_row_enabled = raw_mask.any(dim=-1, keepdim=True)
                    if context_row_enabled.dim() == 4:
                        context_row_enabled = context_row_enabled.squeeze(1)
                context_mask = _finalize_context_mask(raw_mask)
                if context_kv is None:
                    cross_out = block.cross_attn(
                        block.norm3(x), context, ctx_mask=context_mask
                    )
                else:
                    cross_out = block.cross_attn.forward_with_kv_cache(
                        block.norm3(x),
                        context_kv[0],
                        context_kv[1],
                        ctx_mask=context_mask,
                    )
                if context_row_enabled is not None:
                    cross_out = cross_out * context_row_enabled.to(dtype=cross_out.dtype)

                x = x + cross_out
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
        context_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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
            _context_kv=context_kv,
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
                context_kv=_context_kv,
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
                preserve_rng_state=self.checkpoint_preserve_rng_state,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def _forward_eager(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        key_masks_all: Optional[Dict[str, torch.Tensor]] = None,
        query_masks_all: Optional[Dict[str, torch.Tensor]] = None,
        capture_layers: Optional[Dict[str, set[int]]] = None,
        capture_batch_masks: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
        profiler=None,
        max_layer: Optional[int] = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask is None:
            raise ValueError("MoT.forward requires `attention_mask`.")
        if attention_mask is not None:
            if attention_mask.ndim not in (2, 4):
                raise ValueError(
                    f"`attention_mask` must be 2D [S,S] or 4D [B,1,S,S], got shape {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape[-2] != attention_mask.shape[-1]:
                raise ValueError(
                    f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
                )
            if attention_mask.ndim == 4:
                if int(attention_mask.shape[1]) != 1:
                    raise ValueError(
                        f"`attention_mask` 4D shape must be [B,1,S,S], got {tuple(attention_mask.shape)}"
                    )
                batch_size = next(iter(embeds_all.values())).shape[0]
                if int(attention_mask.shape[0]) not in (1, int(batch_size)):
                    raise ValueError(
                        "`attention_mask` batch dimension must be 1 or match token batch: "
                        f"{attention_mask.shape[0]} vs {batch_size}"
                    )

        requested_captures = {
            name: frozenset(layers) for name, layers in (capture_layers or {}).items()
        }
        captured: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in requested_captures
        }

        query_masks, query_mask_all = self._build_query_masks_all(
            embeds_all, query_masks_all
        )
        tokens_all = {
            name: tokens.masked_fill(~query_masks[name].unsqueeze(-1), 0.0)
            if name in query_masks
            else tokens
            for name, tokens in embeds_all.items()
        }
        with _profile_section(profiler, "key_mask_build"):
            key_mask_all = self._build_key_mask_all(embeds_all, key_masks_all)

        flex_key_mask = key_mask_all
        force_sdpa = False
        if self.attention_backend == "flex" and key_mask_all is not None:
            # Per-sample Flex score modulation is slower than fused SDPA for
            # this sequence length. Keep exact key-mask semantics and select
            # the sparse kernel only for all-valid batches.
            force_sdpa = not bool(key_mask_all.all().item())
            if not force_sdpa:
                flex_key_mask = None
        if query_mask_all is not None and not bool(query_mask_all.all().item()):
            force_sdpa = True

        total_seq = sum(int(tokens.shape[1]) for tokens in embeds_all.values())
        if int(attention_mask.shape[-2]) != total_seq:
            raise ValueError(
                "Attention mask seq length mismatch: "
                f"mask={attention_mask.shape[-2]} vs tokens={total_seq}"
            )
        first_tokens = next(iter(embeds_all.values()))
        flex_block_mask = None
        if not force_sdpa:
            flex_block_mask = self._build_flex_block_mask(
                attention_mask=attention_mask,
                batch_size=int(first_tokens.shape[0]),
                query_len=total_seq,
                key_len=total_seq,
                device=first_tokens.device,
            )

        profile_mot_layers = profiler is not None and _env_flag(
            "WAM_PROFILE_MOT_LAYERS"
        )
        end_layer = self.num_layers if max_layer is None else int(max_layer) + 1
        if not 0 < end_layer <= self.num_layers:
            raise ValueError("max_layer must be within the MoT depth.")
        for layer_idx in range(end_layer):
            layer_prefix = f"layer_{layer_idx:02d}" if profile_mot_layers else "layer"
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

                with _profile_section(profiler, f"{layer_prefix}/prepare_{name}"):
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
            if attention_mask is not None and attention_mask.shape[-2] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[-2]} vs tokens={total_seq}"
                )
            with _profile_section(profiler, f"{layer_prefix}/mixed_attention"):
                mixed = self._mixed_attention(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=attention_mask,
                    flex_block_mask=flex_block_mask,
                    flex_key_mask=flex_key_mask,
                    query_mask=query_mask_all,
                    force_sdpa=force_sdpa,
                )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                with _profile_section(profiler, f"{layer_prefix}/post_{name}"):
                    updated_tokens = self._apply_post_with_optional_checkpoint(
                        block=block,
                        residual_x=cached_expert["residual_x"],
                        gate_msa=cached_expert["gate_msa"],
                        shift_mlp=cached_expert["shift_mlp"],
                        scale_mlp=cached_expert["scale_mlp"],
                        gate_mlp=cached_expert["gate_mlp"],
                        use_gradient_checkpointing=(
                            cached_expert["use_gradient_checkpointing"]
                            and (
                                layer_idx % self.checkpoint_layer_stride == 0
                                or layer_idx in self.checkpoint_extra_layers
                            )
                        ),
                        mixed_slice=mixed_slice,
                        context_payload=context_payload,
                    )

                if name in query_masks:
                    updated_tokens = updated_tokens.masked_fill(
                        ~query_masks[name].unsqueeze(-1), 0.0
                    )
                tokens_all[name] = updated_tokens
                start = end

            for name, layers in requested_captures.items():
                if layer_idx in layers:
                    mask = (capture_batch_masks or {}).get(name, {}).get(layer_idx)
                    captured[name][layer_idx] = (
                        tokens_all[name] if mask is None else tokens_all[name][mask]
                    )

        if requested_captures:
            return tokens_all, captured
        return tokens_all

    def _build_flex_block_mask(
        self,
        *,
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ):
        """Build one structural mask and reuse it across all eager MoT layers."""

        del batch_size
        if self.attention_backend != "flex":
            return None
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if tuple(mask.shape[-2:]) != (int(query_len), int(key_len)):
            raise ValueError(
                "Attention mask shape mismatch for FlexAttention: "
                f"mask={tuple(mask.shape[-2:])}, q={query_len}, k={key_len}."
            )

        if mask.ndim == 2:

            def mask_mod(b, h, q_idx, kv_idx):
                return mask[q_idx, kv_idx]

        elif (
            mask.ndim == 4
            and int(mask.shape[0]) == 1
            and int(mask.shape[1]) == 1
        ):

            def mask_mod(b, h, q_idx, kv_idx):
                return mask[0, 0, q_idx, kv_idx]

        else:
            raise ValueError(
                "FlexAttention structural mask must be [Q,K] or [1,1,Q,K]; "
                "pass per-sample key validity through `flex_key_mask`, "
                f"got {tuple(mask.shape)}."
            )

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=int(query_len),
            KV_LEN=int(key_len),
            device=device,
            BLOCK_SIZE=self.flex_mask_block_size,
        )

    def _build_key_mask_all(
        self,
        embeds_all: Dict[str, torch.Tensor],
        key_masks_all: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if key_masks_all is None:
            return None
        key_mask_chunks = []
        for name in self.expert_order:
            key_mask = key_masks_all.get(name)
            if key_mask is None:
                key_mask = torch.ones(
                    embeds_all[name].shape[:2],
                    dtype=torch.bool,
                    device=embeds_all[name].device,
                )
            if key_mask.ndim != 2 or tuple(key_mask.shape) != tuple(
                embeds_all[name].shape[:2]
            ):
                raise ValueError(
                    f"`key_masks_all['{name}']` must be [B,S], got {tuple(key_mask.shape)} "
                    f"for tokens {tuple(embeds_all[name].shape[:2])}."
                )
            key_mask_chunks.append(
                key_mask.to(device=embeds_all[name].device, dtype=torch.bool)
            )
        return torch.cat(key_mask_chunks, dim=1)

    def _build_query_masks_all(
        self,
        embeds_all: Dict[str, torch.Tensor],
        query_masks_all: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if query_masks_all is None:
            return {}, None
        normalized: dict[str, torch.Tensor] = {}
        chunks: list[torch.Tensor] = []
        for name in self.expert_order:
            query_mask = query_masks_all.get(name)
            if query_mask is None:
                query_mask = torch.ones(
                    embeds_all[name].shape[:2],
                    dtype=torch.bool,
                    device=embeds_all[name].device,
                )
            if query_mask.ndim != 2 or tuple(query_mask.shape) != tuple(
                embeds_all[name].shape[:2]
            ):
                raise ValueError(
                    f"`query_masks_all['{name}']` must be [B,S], got "
                    f"{tuple(query_mask.shape)} for tokens "
                    f"{tuple(embeds_all[name].shape[:2])}."
                )
            query_mask = query_mask.to(
                device=embeds_all[name].device, dtype=torch.bool
            )
            normalized[name] = query_mask
            chunks.append(query_mask)
        return normalized, torch.cat(chunks, dim=1)

    def forward(self, embeds_all, attention_mask, freqs_all, context_all, t_mod_all,
                key_masks_all=None, query_masks_all=None, capture_layers=None,
                capture_batch_masks=None, profiler=None):
        forward = self._forward_eager if self.compile_mode == "off" else self._compiled_forward_or_eager
        return forward(embeds_all, attention_mask, freqs_all, context_all, t_mod_all,
                       key_masks_all, query_masks_all, capture_layers,
                       capture_batch_masks, profiler)
