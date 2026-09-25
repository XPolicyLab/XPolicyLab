from __future__ import annotations

import os
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbones.wan22.wan_video_dit import flash_attention, modulate, rope_apply
from wam.utils.logging_config import get_logger
from .kv_cache import MoTCacheMixin

logger = get_logger(__name__)

_MOT_COMPILE_GROUP_SIZE = 1
_MOT_CHECKPOINT_GROUP_SIZE = 3

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - depends on torch build.
    create_block_mask = None
    flex_attention = None


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


class MoT(MoTCacheMixin, nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        checkpoint_layer_stride: int = 1,
        checkpoint_extra_layers: Optional[Sequence[int]] = None,
        checkpoint_preserve_rng_state: bool = True,
        attention_backend: str = "flex",
        flex_block_size: int = 64,
        compile_mode: str = "off",
        compile_gradient_checkpointing: bool = False,
        compile_action_context_pad_to: int = 0,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.checkpoint_layer_stride = int(checkpoint_layer_stride)
        if self.checkpoint_layer_stride < 1:
            raise ValueError(
                "`checkpoint_layer_stride` must be at least 1, got "
                f"{checkpoint_layer_stride}."
            )
        self.checkpoint_extra_layers = frozenset(
            int(layer_idx) for layer_idx in (checkpoint_extra_layers or ())
        )
        self.checkpoint_preserve_rng_state = bool(checkpoint_preserve_rng_state)
        self.compile_mode = str(compile_mode or "off").strip().lower()
        if self.compile_mode not in {"off", "default", "reduce-overhead"}:
            raise ValueError(
                "`compile_mode` must be one of "
                f"('off', 'default', 'reduce-overhead'), got {compile_mode!r}."
            )
        if self.compile_mode != "off" and self.mot_checkpoint_mixed_attn:
            raise ValueError(
                "MoT torch.compile validation requires mot_checkpoint_mixed_attn=false."
            )
        self.compile_gradient_checkpointing = bool(
            compile_gradient_checkpointing
        )
        if self.compile_gradient_checkpointing and self.compile_mode == "off":
            raise ValueError(
                "MoT compile gradient checkpointing requires compile_mode != 'off'."
            )
        self.compile_action_context_pad_to = int(compile_action_context_pad_to)
        if self.compile_action_context_pad_to < 0:
            raise ValueError("`compile_action_context_pad_to` must be non-negative.")
        self._compile_fallback_reasons: set[str] = set()
        self._compile_profiler_notice_logged = False
        self._compile_context_padding_logged = False
        self.attention_backend = str(attention_backend or "flex").strip().lower()
        if self.attention_backend not in {"flex", "sdpa"}:
            raise ValueError(
                "`attention_backend` must be one of {'flex', 'sdpa'}, "
                f"got {attention_backend!r}."
            )
        if (
            self.attention_backend == "flex"
            and (flex_attention is None or create_block_mask is None)
        ):
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
            self.attention_backend == "flex"
            and self.compile_mode == "off"
            and hasattr(torch, "compile")
            and torch.cuda.is_available()
        )
        if self.attention_backend == "flex":
            self._flex_attention = (
                torch.compile(_wam_flex_attention, dynamic=True, fullgraph=True)
                if self._flex_attention_compiled
                else _wam_flex_attention
            )
        else:
            self._flex_attention = None
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )
        if self.checkpoint_layer_stride > 1:
            logger.info(
                "Using selective expert checkpointing every %d layers.",
                self.checkpoint_layer_stride,
            )
        if self.compile_gradient_checkpointing:
            logger.info(
                "Using MoT compile gradient checkpointing: "
                "compile_granularity=layer checkpoint_group_size=%d "
                "use_reentrant=false placement=outside_compiled_graph.",
                _MOT_CHECKPOINT_GROUP_SIZE,
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        invalid_extra_layers = sorted(
            layer_idx
            for layer_idx in self.checkpoint_extra_layers
            if layer_idx < 0 or layer_idx >= self.num_layers
        )
        if invalid_extra_layers:
            raise ValueError(
                "`checkpoint_extra_layers` must contain valid zero-based layer "
                f"indices for {self.num_layers} layers, got {invalid_extra_layers}."
            )
        if self.checkpoint_extra_layers:
            logger.info(
                "Using additional expert checkpoint layers: %s.",
                sorted(self.checkpoint_extra_layers),
            )
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

        # Keep compiler wrappers outside nn.Module registration. Some torch
        # versions return OptimizedModule objects, which must not alter
        # state_dict keys or duplicate optimizer parameters. One wrapper per
        # MoT layer preserves the autograd boundaries needed for ZeRO-1
        # communication overlap. The callable factory below also gives every
        # range an independent Python code object so Dynamo does not count all
        # range specializations against one frame's recompile limit.
        object.__setattr__(self, "_compiled_groups", {})

        self._log_expert_summary("Initialized")

    @staticmethod
    def _num_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def _log_expert_summary(self, verb: str) -> None:
        logger.info(
            "%s MoT with experts: %s, num_layers=%d, attention_backend=%s, "
            "compile_mode=%s",
            verb,
            self.expert_order,
            self.num_layers,
            self.attention_backend,
            self.compile_mode,
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

    def _compile_group_ranges(
        self,
        requested_captures: Dict[str, tuple[int, ...]],
    ) -> tuple[tuple[int, int], ...]:
        """Return one independent compiler boundary for every MoT layer."""
        boundaries = set(range(0, self.num_layers, _MOT_COMPILE_GROUP_SIZE))
        boundaries.add(self.num_layers)
        ordered = sorted(boundaries)
        return tuple(zip(ordered, ordered[1:]))

    def _checkpoint_group_ranges(
        self,
        requested_captures: Dict[str, tuple[int, ...]],
    ) -> tuple[tuple[int, int], ...]:
        """Return three-layer checkpoint ranges split at capture boundaries."""
        boundaries = set(range(0, self.num_layers, _MOT_CHECKPOINT_GROUP_SIZE))
        boundaries.add(self.num_layers)
        for layers in requested_captures.values():
            boundaries.update(
                layer_idx + 1
                for layer_idx in layers
                if 0 <= layer_idx < self.num_layers
            )
        ordered = sorted(boundaries)
        return tuple(zip(ordered, ordered[1:]))

    def _get_compiled_group(self, start_layer: int, end_layer: int):
        start_layer = int(start_layer)
        end_layer = int(end_layer)
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise IndexError(
                "MoT compile group range out of bounds: "
                f"[{start_layer}, {end_layer}) for {self.num_layers} layers."
            )
        if end_layer != start_layer + _MOT_COMPILE_GROUP_SIZE:
            raise ValueError(
                "MoT layerwise compile requires exactly one layer per boundary; "
                f"got [{start_layer}, {end_layer})."
            )
        group_range = (start_layer, end_layer)
        compiled_groups = self._compiled_groups
        if group_range not in compiled_groups:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "MoT compile_mode requires a PyTorch build with torch.compile."
                )
            logger.info(
                "Compiling MoT training layer %d/%d lazily: mode=%s "
                "granularity=layer "
                "forward=inductor backward=inductor "
                "dynamic=false fullgraph=false cudagraphs=%s "
                "mix_order_reduction=default",
                start_layer + 1,
                self.num_layers,
                self.compile_mode,
                self.compile_mode == "reduce-overhead"
                and not self.compile_gradient_checkpointing,
            )
            compile_kwargs = {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": False,
            }
            if self.compile_gradient_checkpointing:
                # PyTorch 2.10 CUDA Graph Trees can recycle saved AOTAutograd
                # activations before the non-reentrant checkpoint backward
                # consumes them. Traditional CUDA Graphs avoid that exception
                # but produce incorrect gradients for this graph. Keep
                # Inductor compilation and default reduction scheduling, and
                # disable CUDA Graph capture only for checkpointed MoT.
                compile_kwargs["options"] = {
                    "triton.cudagraphs": False,
                }
            else:
                compile_kwargs["mode"] = self.compile_mode
            compiled_group = torch.compile(
                self._make_compilable_group_callable(start_layer, end_layer),
                **compile_kwargs,
            )
            compiled_groups[group_range] = compiled_group
        return compiled_groups[group_range]

    def _make_compilable_group_callable(
        self,
        start_layer: int,
        end_layer: int,
    ):
        """Create a Dynamo frame dedicated to one fixed MoT layer.

        ``functools.partial`` objects are wrapped by Dynamo with the shared
        ``external_utils.inner`` code object. With more than eight bound ranges
        that frame hits Dynamo's default recompile limit and the remaining
        ranges fall back to eager. Cloning this ordinary function's code object
        makes the intended one-graph-per-layer boundary explicit without
        changing process-global Dynamo limits. Compile checkpointing is applied
        by the eager orchestrator outside this callable so checkpoint operators
        and RNG-state helpers never enter the persistent compiler graph.
        """
        start_layer = int(start_layer)
        end_layer = int(end_layer)

        def group_forward(
            video_embed: torch.Tensor,
            action_embed: torch.Tensor,
            attention_mask: torch.Tensor,
            video_freqs: torch.Tensor,
            action_freqs: torch.Tensor,
            video_context: Optional[torch.Tensor],
            video_context_mask: Optional[torch.Tensor],
            action_context: Optional[torch.Tensor],
            action_context_mask: Optional[torch.Tensor],
            video_t_mod: torch.Tensor,
            action_t_mod: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._forward_compilable_group(
                start_layer,
                end_layer,
                video_embed,
                action_embed,
                attention_mask,
                video_freqs,
                action_freqs,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                video_t_mod,
                action_t_mod,
            )

        code_name = f"_forward_compilable_group_{start_layer}_{end_layer}"
        group_forward.__code__ = group_forward.__code__.replace(co_name=code_name)
        group_forward.__name__ = code_name
        group_forward.__qualname__ = f"{type(self).__qualname__}.{code_name}"
        group_forward._mot_layer_range = (start_layer, end_layer)
        return group_forward

    def _pad_compiled_action_context(
        self,
        context: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pad Action cross-attention inputs to one compiler-stable length."""
        target = self.compile_action_context_pad_to
        if context is None or target == 0:
            return context, mask
        if context.ndim != 3:
            raise ValueError(
                "Compiled Action context must be [B,L,D], got "
                f"shape={tuple(context.shape)}."
            )
        length = int(context.shape[1])
        if length > target:
            raise ValueError(
                "Compiled Action context exceeds "
                f"model.mot_compile_action_context_pad_to: {length} > {target}. "
                "Increase the configured limit; the context will not be truncated."
            )
        if mask is None:
            mask = torch.ones(
                (context.shape[0], length), dtype=torch.bool, device=context.device
            )
        elif (
            mask.ndim not in (2, 3)
            or int(mask.shape[0]) != int(context.shape[0])
            or int(mask.shape[-1]) != length
        ):
            raise ValueError(
                "Compiled Action context mask must be [B,L] or [B,Q,L], got "
                f"mask={tuple(mask.shape)} context={tuple(context.shape)}."
            )
        if length == target:
            return context, mask
        pad = target - length
        if not self._compile_context_padding_logged:
            logger.info(
                "Padding MoT Action context to %d tokens for a stable compiled graph.",
                target,
            )
            self._compile_context_padding_logged = True
        return F.pad(context, (0, 0, 0, pad)), F.pad(mask, (0, pad), value=False)

    def _compile_fallback(self, reason: str) -> None:
        if reason in self._compile_fallback_reasons:
            return
        self._compile_fallback_reasons.add(reason)
        logger.warning("MoT compile path disabled for this call: %s", reason)

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
        allow_self_when_all_masked: bool = False,
        force_sdpa: bool = False,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            if self.attention_backend == "sdpa" or force_sdpa:
                effective_mask = self._apply_key_mask(
                    attn_mask,
                    flex_key_mask,
                    allow_self_when_all_masked=allow_self_when_all_masked,
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
                {
                    # Keep the generated Triton kernel within the shared-memory
                    # limit of evaluation GPUs such as RTX 4090.  Leaving these
                    # unset lets FlexAttention autotune a 128-wide kernel that
                    # can require 114,688 bytes, above Ada's 101,376-byte limit.
                    "BLOCK_M": self.flex_block_size,
                    "BLOCK_N": self.flex_block_size,
                },
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

            def _mask_rows(
                mask: Optional[torch.Tensor], start: int, end: int
            ) -> Optional[torch.Tensor]:
                if mask is None:
                    return None
                if mask.dim() == 3:
                    return mask[:, int(start) : int(end), :]
                elif mask.dim() == 4:
                    return mask[:, :, int(start) : int(end), :]
                raise ValueError(
                    f"Context mask must be [B,S,L] or [B,1,S,L], got {tuple(mask.shape)}"
                )

            def _finalize_context_mask(
                mask: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if mask is None:
                    return None
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                elif mask.dim() != 4:
                    raise ValueError(
                        f"Context mask must be [B,S,L] or [B,1,S,L], got {tuple(mask.shape)}"
                    )
                if mask.shape[-1] > 0:
                    row_has_key = mask.any(dim=-1, keepdim=True)
                    fallback = torch.zeros_like(mask)
                    fallback[..., 0:1] = True
                    mask = torch.where(row_has_key, mask, fallback)
                return mask

            if context is not None:
                raw_mask = _mask_rows(context_payload.get("mask"), 0, int(x.shape[1]))
                context_row_enabled = None
                if raw_mask is not None:
                    context_row_enabled = raw_mask.any(dim=-1, keepdim=True)
                    if context_row_enabled.dim() == 4:
                        context_row_enabled = context_row_enabled.squeeze(1)
                context_mask = _finalize_context_mask(raw_mask)
                cross_out = block.cross_attn(
                    block.norm3(x), context, ctx_mask=context_mask
                )
                if context_row_enabled is not None:
                    cross_out = cross_out * context_row_enabled.to(
                        dtype=cross_out.dtype
                    )

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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
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

    def _forward_impl(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        key_masks_all: Optional[Dict[str, torch.Tensor]] = None,
        capture_layers: Optional[Dict[str, set[int]]] = None,
        profiler=None,
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

        tokens_all = {k: v for k, v in embeds_all.items()}
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
        for layer_idx in range(self.num_layers):
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

                tokens_all[name] = updated_tokens
                start = end

            for name, layers in requested_captures.items():
                if layer_idx in layers:
                    captured[name][layer_idx] = tokens_all[name]

        if requested_captures:
            return tokens_all, captured
        return tokens_all

    def _forward_compilable_layer(
        self,
        layer_idx: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fixed video/action MoT layer inside one compiler boundary."""
        tokens_all = {"video": video_embed, "action": action_embed}
        freqs_all = {"video": video_freqs, "action": action_freqs}
        t_mod_all = {"video": video_t_mod, "action": action_t_mod}
        context_all = {
            "video": {"context": video_context, "mask": video_context_mask},
            "action": {"context": action_context, "mask": action_context_mask},
        }
        q_chunks = []
        k_chunks = []
        v_chunks = []
        cached = {}
        seq_lens = []

        for name in ("video", "action"):
            expert = self.mixtures[name]
            block = expert.blocks[layer_idx]
            x = tokens_all[name]
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
                freqs=freqs_all[name],
                t_mod=t_mod_all[name],
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

        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)
        if attention_mask.shape[-2] != q_cat.shape[1]:
            raise ValueError(
                "Attention mask seq length mismatch: "
                f"mask={attention_mask.shape[-2]} vs tokens={q_cat.shape[1]}"
            )
        mixed = self._mixed_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
            # Avoid nesting the separately compiled Flex kernel inside the
            # per-layer AOTAutograd graph. This is the colleague-validated
            # Flash/SDPA compile path; eager mode still uses sparse Flex.
            force_sdpa=True,
        )

        start = 0
        for name, seq_len in zip(("video", "action"), seq_lens):
            end = start + seq_len
            cached_expert = cached[name]
            tokens_all[name] = self._apply_post_with_optional_checkpoint(
                block=cached_expert["block"],
                residual_x=cached_expert["residual_x"],
                gate_msa=cached_expert["gate_msa"],
                shift_mlp=cached_expert["shift_mlp"],
                scale_mlp=cached_expert["scale_mlp"],
                gate_mlp=cached_expert["gate_mlp"],
                use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                mixed_slice=mixed[:, start:end, :],
                context_payload=context_all[name],
            )
            start = end

        return tokens_all["video"], tokens_all["action"]

    def _forward_compilable_group(
        self,
        start_layer: int,
        end_layer: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MoT layer inside its own compiler/autograd boundary."""
        video_out = video_embed
        action_out = action_embed
        for layer_idx in range(start_layer, end_layer):
            video_out, action_out = self._forward_compilable_layer(
                layer_idx,
                video_out,
                action_out,
                attention_mask,
                video_freqs,
                action_freqs,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                video_t_mod,
                action_t_mod,
            )
        return video_out, action_out

    def _run_compiled_layer_group(
        self,
        compiled_layers: tuple,
        start_layer: int,
        video_embed: torch.Tensor,
        action_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        *,
        profiler=None,
        profile_mot_layers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one checkpoint group while retaining per-layer compile frames."""
        video_out = video_embed
        action_out = action_embed
        for offset, compiled_layer in enumerate(compiled_layers):
            layer_idx = int(start_layer) + offset
            layer_prefix = (
                f"layer_{layer_idx:02d}" if profile_mot_layers else "layer"
            )
            try:
                with _profile_section(profiler, f"{layer_prefix}/compiled"):
                    video_out, action_out = compiled_layer(
                        video_out,
                        action_out,
                        attention_mask,
                        video_freqs,
                        action_freqs,
                        video_context,
                        video_context_mask,
                        action_context,
                        action_context_mask,
                        video_t_mod,
                        action_t_mod,
                    )
            except Exception as exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "MoT layerwise torch.compile failed at "
                        f"layer={layer_idx}, mode={self.compile_mode}."
                    )
                raise
        return video_out, action_out

    def _forward_compiled_groups(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        key_masks_all: Dict[str, torch.Tensor],
        requested_captures: Dict[str, tuple[int, ...]],
        profiler=None,
    ):
        with _profile_section(profiler, "key_mask_build"):
            key_mask_all = self._build_key_mask_all(embeds_all, key_masks_all)
            effective_attention_mask = self._apply_key_mask(
                attention_mask, key_mask_all
            )

        video_context = context_all.get("video") or {}
        action_context = context_all.get("action") or {}
        video_out = embeds_all["video"]
        action_out = embeds_all["action"]
        captured: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in requested_captures
        }
        requested_capture_sets = {
            name: frozenset(layers) for name, layers in requested_captures.items()
        }
        profile_mot_layers = profiler is not None and _env_flag(
            "WAM_PROFILE_MOT_LAYERS"
        )

        for start_layer, end_layer in self._checkpoint_group_ranges(
            requested_captures
        ):
            compiled_layers = tuple(
                self._get_compiled_group(layer_idx, layer_idx + 1)
                for layer_idx in range(start_layer, end_layer)
            )
            group_forward = partial(
                self._run_compiled_layer_group,
                compiled_layers,
                start_layer,
                profiler=profiler,
                profile_mot_layers=profile_mot_layers,
            )
            group_args = (
                video_out,
                action_out,
                effective_attention_mask,
                freqs_all["video"],
                freqs_all["action"],
                video_context.get("context"),
                video_context.get("mask"),
                action_context.get("context"),
                action_context.get("mask"),
                t_mod_all["video"],
                t_mod_all["action"],
            )
            if self.compile_gradient_checkpointing and self.training:
                checkpoint_kwargs = {"use_reentrant": False}
                if not self.checkpoint_preserve_rng_state:
                    checkpoint_kwargs["preserve_rng_state"] = False
                video_out, action_out = torch.utils.checkpoint.checkpoint(
                    group_forward,
                    *group_args,
                    **checkpoint_kwargs,
                )
            else:
                video_out, action_out = group_forward(*group_args)

            last_layer = end_layer - 1
            group_outputs = {"video": video_out, "action": action_out}
            for name, layers in requested_capture_sets.items():
                if last_layer in layers:
                    captured[name][last_layer] = group_outputs[name]

        tokens = {"video": video_out, "action": action_out}
        if not requested_captures:
            return tokens
        return tokens, captured

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        key_masks_all: Optional[Dict[str, torch.Tensor]] = None,
        capture_layers: Optional[Dict[str, set[int]]] = None,
        profiler=None,
    ):
        if self.compile_mode == "off":
            return self._forward_impl(
                embeds_all,
                attention_mask,
                freqs_all,
                context_all,
                t_mod_all,
                key_masks_all,
                capture_layers,
                profiler,
            )

        fallback_reason = None
        if not self.training:
            fallback_reason = "only training forward is validated"
        elif self.expert_order != ["video", "action"]:
            fallback_reason = "the compiled core requires exactly video/action experts"
        elif attention_mask is None:
            fallback_reason = "attention_mask is required"
        elif any(name not in embeds_all for name in ("video", "action")):
            fallback_reason = "video/action embeddings are required"
        elif any(name not in freqs_all for name in ("video", "action")):
            fallback_reason = "video/action frequencies are required"
        elif any(name not in t_mod_all for name in ("video", "action")):
            fallback_reason = "video/action timestep modulation is required"
        elif embeds_all["video"].device.type != "cuda":
            fallback_reason = "CUDA tensors are required"

        if fallback_reason is not None:
            self._compile_fallback(fallback_reason)
            return self._forward_impl(
                embeds_all,
                attention_mask,
                freqs_all,
                context_all,
                t_mod_all,
                key_masks_all,
                capture_layers,
                profiler,
            )

        video_context = context_all.get("video") or {}
        action_context = context_all.get("action") or {}
        action_context_tensor, action_context_mask = self._pad_compiled_action_context(
            action_context.get("context"), action_context.get("mask")
        )
        requested_captures = {
            name: tuple(sorted(int(layer) for layer in layers))
            for name, layers in (capture_layers or {}).items()
        }
        unsupported_capture_experts = set(requested_captures) - {"video", "action"}
        if unsupported_capture_experts:
            reason = (
                "capture_layers contains unsupported experts: "
                f"{sorted(unsupported_capture_experts)}"
            )
            self._compile_fallback(reason)
            return self._forward_impl(
                embeds_all,
                attention_mask,
                freqs_all,
                context_all,
                t_mod_all,
                key_masks_all,
                capture_layers,
                profiler,
            )
        if profiler is not None and not self._compile_profiler_notice_logged:
            logger.info(
                "MoT compile mode keeps the outer mot profiler section but "
                "reports one compiled section per MoT layer; operator-level "
                "ranges remain compiler-managed."
            )
            self._compile_profiler_notice_logged = True
        compiled_context_all = {
            "video": video_context,
            "action": {
                "context": action_context_tensor,
                "mask": action_context_mask,
            },
        }
        return self._forward_compiled_groups(
            embeds_all,
            attention_mask,
            freqs_all,
            compiled_context_all,
            t_mod_all,
            key_masks_all or {},
            requested_captures,
            profiler,
        )
