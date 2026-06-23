from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

from .wan_video_dit import flash_attention, modulate, rope_apply
from ahawam.utils.logging_config import get_logger

logger = get_logger(__name__)


class LayerwiseChunkKVCacheEditor(nn.Module):
    """Build per-chunk first-frame K/V deltas from obs-conditioned queries."""

    def __init__(
        self,
        *,
        query_dim: int,
        num_layers: int,
        num_heads: int,
        attn_hidden_dim: int,
        gate_init: float = -4.0,
        use_delta_gate: bool = True,
    ) -> None:
        super().__init__()
        if attn_hidden_dim % num_heads != 0:
            raise ValueError(
                f"`attn_hidden_dim` ({attn_hidden_dim}) must be divisible by `num_heads` ({num_heads})."
            )
        self.query_dim = int(query_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_hidden_dim = int(attn_hidden_dim)
        self.use_delta_gate = bool(use_delta_gate)
        self.head_dim = self.attn_hidden_dim // self.num_heads
        self.layer_query_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.query_dim),
                    nn.Linear(self.query_dim, self.attn_hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_delta_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.attn_hidden_dim),
                    nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.attn_hidden_dim, 2 * self.attn_hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.delta_gate = nn.Parameter(
            torch.full((self.num_layers,), float(gate_init))
        )
        for proj in self.layer_delta_proj:
            last = cast(nn.Linear, proj[-1])
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def build_layer_updated_cache(
        self,
        *,
        layer_idx: int,
        chunk_queries: torch.Tensor,
        first_frame_keys: torch.Tensor,
        first_frame_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if chunk_queries.ndim != 4:
            raise ValueError(
                "`chunk_queries` must be [B, N, Q, D], "
                f"got shape {tuple(chunk_queries.shape)}"
            )
        if first_frame_keys.ndim != 3 or first_frame_values.ndim != 3:
            raise ValueError(
                "`first_frame_keys` and `first_frame_values` must be [B, S, H*Dh], "
                f"got {tuple(first_frame_keys.shape)} and {tuple(first_frame_values.shape)}"
            )
        query_proj_weight = cast(nn.LayerNorm, self.layer_query_proj[layer_idx][0]).weight
        chunk_queries = chunk_queries.to(
            device=query_proj_weight.device,
            dtype=query_proj_weight.dtype,
        )
        first_frame_keys = first_frame_keys.to(
            device=query_proj_weight.device,
            dtype=query_proj_weight.dtype,
        )
        first_frame_values = first_frame_values.to(
            device=query_proj_weight.device,
            dtype=query_proj_weight.dtype,
        )
        batch_size, num_chunks, num_queries, _ = chunk_queries.shape
        first_frame_tokens = int(first_frame_keys.shape[1])
        query = self.layer_query_proj[layer_idx](chunk_queries).view(
            batch_size, num_chunks, num_queries, self.num_heads, self.head_dim
        )
        k0 = first_frame_keys.view(
            batch_size, first_frame_tokens, self.num_heads, self.head_dim
        )
        v0 = first_frame_values.view(
            batch_size, first_frame_tokens, self.num_heads, self.head_dim
        )

        scores = torch.einsum("bnqhd,bshd->bhnqs", query, k0)
        scores = scores / (float(self.head_dim) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        routed = torch.einsum("bhnqs,bshd->bnqhd", weights, v0)

        token_scores = torch.einsum("bshd,bnqhd->bhnsq", k0, routed)
        token_scores = token_scores / (float(self.head_dim) ** 0.5)
        token_weights = torch.softmax(token_scores, dim=-1)
        decoded = torch.einsum("bhnsq,bnqhd->bnshd", token_weights, routed).reshape(
            batch_size,
            num_chunks,
            first_frame_tokens,
            self.attn_hidden_dim,
        )

        delta = self.layer_delta_proj[layer_idx](decoded)
        delta_k, delta_v = delta.chunk(2, dim=-1)
        base_k = first_frame_keys.unsqueeze(1).expand(-1, num_chunks, -1, -1)
        base_v = first_frame_values.unsqueeze(1).expand(-1, num_chunks, -1, -1)
        if self.use_delta_gate:
            gate = torch.sigmoid(self.delta_gate[layer_idx]).to(
                device=delta.device,
                dtype=delta.dtype,
            )
            updated_k = base_k + gate * delta_k
            updated_v = base_v + gate * delta_v
        else:
            updated_k = base_k + delta_k
            updated_v = base_v + delta_v
        return {
            "k": updated_k,
            "v": updated_v,
            "delta_k": delta_k,
            "delta_v": delta_v,
        }


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
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
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        first_expert = cast(Any, self.mixtures[self.expert_order[0]])
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = cast(Any, self.mixtures[name])
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
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )
        action_expert = cast(Any, self.mixtures["action"])
        action_expert_hidden = int(getattr(action_expert, "hidden_dim"))
        # HACK: This was introduced for an old experimental action-branch idea that
        # was not used in the final version. This meaningless parameter accidentally
        # ended up in released checkpoints, so it is kept only for compatibility;
        # remove it for training from scratch if you also remove the dependent code paths.
        self.action_branch_embedding = nn.Parameter(
            torch.zeros(2, action_expert_hidden)
        )
        self.chunk_kv_cache_editor: Optional[LayerwiseChunkKVCacheEditor] = None

    def configure_chunk_kv_cache_editor(
        self,
        *,
        query_dim: Optional[int] = None,
        use_delta_gate: bool = True,
    ) -> None:
        if query_dim is None:
            self.chunk_kv_cache_editor = None
            return
        self.chunk_kv_cache_editor = LayerwiseChunkKVCacheEditor(
            query_dim=int(query_dim),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            attn_hidden_dim=self.num_heads * self.attn_head_dim,
            use_delta_gate=use_delta_gate,
        ).to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
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
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)
        k_cat = k_cat.to(device=q_cat.device, dtype=q_cat.dtype)
        v_cat = v_cat.to(device=q_cat.device, dtype=q_cat.dtype)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return cast(
                torch.Tensor,
                torch_checkpoint.checkpoint(
                    _forward,
                    q_cat,
                    k_cat,
                    v_cat,
                    use_reentrant=False,
                ),
            )
        return _forward(q_cat, k_cat, v_cat)

    def _chunk_routed_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _forward(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            return flash_attention(
                q=q_in,
                k=k_in,
                v=v_in,
                num_heads=self.num_heads,
                ctx_mask=ctx_mask,
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return cast(
                torch.Tensor,
                torch_checkpoint.checkpoint(
                    _forward,
                    q,
                    k,
                    v,
                    use_reentrant=False,
                ),
            )
        return _forward(q, k, v)

    def _forward_chunk_routed_prior_only_attention(
        self,
        *,
        q_action: torch.Tensor,
        k_action: torch.Tensor,
        v_action: torch.Tensor,
        updated_keys: torch.Tensor,
        updated_values: torch.Tensor,
        action_seq_len: int,
        chunk_size: int,
    ) -> torch.Tensor:
        batch_size = int(q_action.shape[0])
        num_chunks = int(updated_keys.shape[1])

        def chunk_view(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size, num_chunks, int(chunk_size), x.shape[-1])

        q_prior = chunk_view(q_action)
        k_prior = chunk_view(k_action)
        v_prior = chunk_view(v_action)

        prior_k = torch.cat([updated_keys, k_prior], dim=2).reshape(
            batch_size * num_chunks,
            int(updated_keys.shape[2]) + int(chunk_size),
            -1,
        )
        prior_v = torch.cat([updated_values, v_prior], dim=2).reshape(
            batch_size * num_chunks,
            int(updated_values.shape[2]) + int(chunk_size),
            -1,
        )
        q_flat = q_prior.reshape(batch_size * num_chunks, int(chunk_size), -1)
        mixed_prior = self._chunk_routed_attention(
            q=q_flat,
            k=prior_k,
            v=prior_v,
        ).reshape(batch_size, action_seq_len, -1)
        return mixed_prior

    def build_chunk_updated_video_kv_cache(
        self,
        *,
        video_kv_cache: list[dict[str, torch.Tensor]],
        chunk_queries: torch.Tensor,
        video_tokens_per_frame: int,
        chunk_index: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        editor = self.chunk_kv_cache_editor
        if editor is None:
            raise ValueError("Chunk KV cache editor is not configured.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if chunk_queries.ndim != 4:
            raise ValueError(
                "`chunk_queries` must be [B, N, Q, D], "
                f"got shape {tuple(chunk_queries.shape)}"
            )
        if chunk_index < 0 or chunk_index >= int(chunk_queries.shape[1]):
            raise ValueError(
                f"`chunk_index` out of range: {chunk_index} for {chunk_queries.shape[1]} chunks."
            )
        one_chunk_queries = chunk_queries[:, chunk_index : chunk_index + 1]
        updated_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx, layer_cache in enumerate(video_kv_cache):
            first_k = layer_cache["k"][:, : int(video_tokens_per_frame)]
            first_v = layer_cache["v"][:, : int(video_tokens_per_frame)]
            updated = editor.build_layer_updated_cache(
                layer_idx=layer_idx,
                chunk_queries=one_chunk_queries,
                first_frame_keys=first_k,
                first_frame_values=first_v,
            )
            updated_cache.append({"k": updated["k"][:, 0], "v": updated["v"][:, 0]})
        return updated_cache

    def forward_prior_action_with_chunk_updated_kv(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict[str, torch.Tensor]],
        video_attention_mask: torch.Tensor,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict[str, torch.Tensor]],
        chunk_queries: torch.Tensor,
        video_tokens_per_frame: int,
        action_chunk_size: int,
        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> dict[str, torch.Tensor]:
        editor = self.chunk_kv_cache_editor
        if editor is None:
            raise ValueError("Chunk KV cache editor is not configured.")
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("MoT requires both video and action experts.")

        video_expert = cast(Any, self.mixtures["video"])
        action_expert = cast(Any, self.mixtures["action"])
        action_seq_len = int(action_tokens.shape[1])
        x_video = video_tokens
        prior_embed = self.action_branch_embedding[1].to(
            device=action_tokens.device, dtype=action_tokens.dtype
        )
        x_action = action_tokens + prior_embed.view(1, 1, -1)

        for layer_idx in range(self.num_layers):
            video_block = cast(Any, video_expert.blocks[layer_idx])
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_video,
                shift_video_mlp,
                scale_video_mlp,
                gate_video_mlp,
                video_checkpoint,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            if history_video_kv_cache is not None:
                hist_k = history_video_kv_cache[layer_idx]["k"].to(
                    device=k_video.device, dtype=k_video.dtype
                )
                hist_v = history_video_kv_cache[layer_idx]["v"].to(
                    device=v_video.device, dtype=v_video.dtype
                )
                k_video_attn = torch.cat([hist_k, k_video], dim=1)
                v_video_attn = torch.cat([hist_v, v_video], dim=1)
            else:
                k_video_attn = k_video
                v_video_attn = v_video
            mixed_video = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_video_attn,
                v_cat=v_video_attn,
                attention_mask=video_attention_mask,
            )
            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_video,
                shift_mlp=shift_video_mlp,
                scale_mlp=scale_video_mlp,
                gate_mlp=gate_video_mlp,
                use_gradient_checkpointing=video_checkpoint,
                mixed_slice=mixed_video,
                context_payload=video_context_payload,
            )

            first_k = k_video[:, : int(video_tokens_per_frame)]
            first_v = v_video[:, : int(video_tokens_per_frame)]
            updated = editor.build_layer_updated_cache(
                layer_idx=layer_idx,
                chunk_queries=chunk_queries,
                first_frame_keys=first_k,
                first_frame_values=first_v,
            )

            action_block = cast(Any, action_expert.blocks[layer_idx])
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_action,
                shift_action_mlp,
                scale_action_mlp,
                gate_action_mlp,
                action_checkpoint,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            mixed_action = self._forward_chunk_routed_prior_only_attention(
                q_action=q_action,
                k_action=k_action,
                v_action=v_action,
                updated_keys=updated["k"],
                updated_values=updated["v"],
                action_seq_len=action_seq_len,
                chunk_size=int(action_chunk_size),
            )
            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_action,
                shift_mlp=shift_action_mlp,
                scale_mlp=scale_action_mlp,
                gate_mlp=gate_action_mlp,
                use_gradient_checkpointing=action_checkpoint,
                mixed_slice=mixed_action,
                context_payload=action_context_payload,
            )

        return {"video": x_video, "action_prior": x_action}

    def compute_cross_attn_kv(
        self,
        context: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Pre-compute per-layer cross-attention K/V for action context."""
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `compute_cross_attn_kv`."
            )

        expert = cast(Any, self.mixtures["action"])
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
            kv_cache.append(
                {
                    "k": block.cross_attn.norm_k(block.cross_attn.k(context)),
                    "v": block.cross_attn.v(context),
                }
            )
        return kv_cache

    def prefill_video_and_editor_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict[str, torch.Tensor]],
        video_attention_mask: torch.Tensor,
        chunk_queries: torch.Tensor,
        video_tokens_per_frame: int,
        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        editor = self.chunk_kv_cache_editor
        if editor is None:
            raise ValueError("Chunk KV cache editor is not configured.")
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for editor cache prefill.")

        video_expert = cast(Any, self.mixtures["video"])
        x_video = video_tokens
        editor_cache: list[dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            video_block = cast(Any, video_expert.blocks[layer_idx])
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_video,
                shift_video_mlp,
                scale_video_mlp,
                gate_video_mlp,
                video_checkpoint,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            if history_video_kv_cache is not None:
                hist_k = history_video_kv_cache[layer_idx]["k"].to(
                    device=k_video.device, dtype=k_video.dtype
                )
                hist_v = history_video_kv_cache[layer_idx]["v"].to(
                    device=v_video.device, dtype=v_video.dtype
                )
                k_video_attn = torch.cat([hist_k, k_video], dim=1)
                v_video_attn = torch.cat([hist_v, v_video], dim=1)
            else:
                k_video_attn = k_video
                v_video_attn = v_video
            mixed_video = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_video_attn,
                v_cat=v_video_attn,
                attention_mask=video_attention_mask,
            )
            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_video,
                shift_mlp=shift_video_mlp,
                scale_mlp=scale_video_mlp,
                gate_mlp=gate_video_mlp,
                use_gradient_checkpointing=video_checkpoint,
                mixed_slice=mixed_video,
                context_payload=video_context_payload,
            )
            first_k = k_video[:, : int(video_tokens_per_frame)]
            first_v = v_video[:, : int(video_tokens_per_frame)]
            updated = editor.build_layer_updated_cache(
                layer_idx=layer_idx,
                chunk_queries=chunk_queries,
                first_frame_keys=first_k,
                first_frame_values=first_v,
            )
            editor_cache.append({"k": updated["k"], "v": updated["v"]})

        return editor_cache

    def forward_action_prior_only_with_editor_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict[str, torch.Tensor]],
        editor_cache: list[dict[str, torch.Tensor]],
        action_chunk_size: int,
    ) -> torch.Tensor:
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert.")

        action_expert = cast(Any, self.mixtures["action"])
        action_seq_len = int(action_tokens.shape[1])
        x_action = action_tokens

        for layer_idx in range(self.num_layers):
            action_block = cast(Any, action_expert.blocks[layer_idx])
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_action,
                shift_action_mlp,
                scale_action_mlp,
                gate_action_mlp,
                action_checkpoint,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            mixed_action = self._forward_chunk_routed_prior_only_attention(
                q_action=q_action,
                k_action=k_action,
                v_action=v_action,
                updated_keys=editor_cache[layer_idx]["k"],
                updated_values=editor_cache[layer_idx]["v"],
                action_seq_len=action_seq_len,
                chunk_size=int(action_chunk_size),
            )
            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_action,
                shift_mlp=shift_action_mlp,
                scale_mlp=scale_action_mlp,
                gate_mlp=gate_action_mlp,
                use_gradient_checkpointing=action_checkpoint,
                mixed_slice=mixed_action,
                context_payload=action_context_payload,
            )

        return x_action

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict[str, torch.Tensor]],
        cross_attn_kv: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            context_mask = context_payload.get("mask")
            if context_mask is not None and context_mask.dim() == 3:
                context_mask = context_mask.unsqueeze(1)
            if cross_attn_kv is not None:
                q = block.cross_attn.norm_q(block.cross_attn.q(block.norm3(x)))
                x = x + block.cross_attn.o(
                    flash_attention(
                        q=q,
                        k=cross_attn_kv["k"],
                        v=cross_attn_kv["v"],
                        num_heads=block.cross_attn.num_heads,
                        ctx_mask=context_mask,
                    )
                )
            elif context is not None:
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert: Any,
        block: Any,
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
        context_payload: Optional[dict[str, torch.Tensor]],
        cross_attn_kv: Optional[dict[str, torch.Tensor]] = None,
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
            _block: Any = block,
            _context_payload: Optional[dict[str, torch.Tensor]] = context_payload,
            _cross_attn_kv: Optional[dict[str, torch.Tensor]] = cross_attn_kv,
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
                cross_attn_kv=_cross_attn_kv,
            )

        if use_gradient_checkpointing and self.training:
            return cast(
                torch.Tensor,
                torch_checkpoint.checkpoint(
                    _post_fn,
                    mixed_slice,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_reentrant=False,
                ),
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache_with_prefix(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict[str, torch.Tensor]],
        prefix_video_kv_cache: list[dict[str, torch.Tensor]],
        prefix_video_seq_len: int,
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video cache with history prefix KV prepended per layer.

        Current video tokens attend to [prefix_kv, self_kv] at each layer.
        Only the current video K/V is stored in the returned cache.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for video cache prefill.")
        if len(prefix_video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_video_kv_cache` must contain {self.num_layers} layers, "
                f"got {len(prefix_video_kv_cache)}."
            )
        current_seq_len = int(video_tokens.shape[1])
        total_seq_len = int(prefix_video_seq_len) + current_seq_len
        if video_attention_mask.shape != (current_seq_len, total_seq_len):
            raise ValueError(
                "`video_attention_mask` shape mismatch: "
                f"got {tuple(video_attention_mask.shape)} vs "
                f"expected {(current_seq_len, total_seq_len)}."
            )
        expert = cast(Any, self.mixtures["video"])
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
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
            prefix_layer = prefix_video_kv_cache[layer_idx]
            k_prefix = prefix_layer["k"]
            v_prefix = prefix_layer["v"]
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=torch.cat([k_prefix, k], dim=1),
                v_cat=torch.cat([v_prefix, v], dim=1),
                attention_mask=video_attention_mask,
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
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict[str, torch.Tensor]],
        video_attention_mask: torch.Tensor,
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

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim not in (2, 3):
            raise ValueError(
                "`video_attention_mask` must be 2D [S,S] or 3D [B,S,S], "
                f"got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[-2] != video_attention_mask.shape[-1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[-1] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[-1]} vs tokens={video_tokens.shape[1]}"
            )

        expert = cast(Any, self.mixtures["video"])
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
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

    def _forward_action_with_video_cache_inner(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        video_kv_cache_k: list[torch.Tensor],
        video_kv_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compile-friendly action branch with cached video K/V."""
        expert = cast(Any, self.mixtures["action"])
        x = action_tokens
        context_payload = None
        if action_context is not None or action_context_mask is not None:
            context_payload = {
                "context": action_context,
                "mask": action_context_mask,
            }

        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self._split_modulation(block, action_t_mod)
            )
            attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

            q = block.self_attn.norm_q(block.self_attn.q(attn_input))
            k = block.self_attn.norm_k(block.self_attn.k(attn_input))
            v = block.self_attn.v(attn_input)
            q = rope_apply(q, action_freqs, block.num_heads)
            k = rope_apply(k, action_freqs, block.num_heads)

            k_cat = torch.cat([video_kv_cache_k[layer_idx], k], dim=1)
            v_cat = torch.cat([video_kv_cache_v[layer_idx], v], dim=1)
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=context_payload,
            )
        return x

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict[str, torch.Tensor]],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_history_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        action_history_seq_len: int = 0,
        cross_attn_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
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

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `forward_action_with_video_cache`."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = (
            int(video_seq_len) + int(action_history_seq_len) + action_seq_len
        )
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        if (
            cross_attn_kv_cache is not None
            and len(cross_attn_kv_cache) != self.num_layers
        ):
            raise ValueError(
                f"`cross_attn_kv_cache` must contain {self.num_layers} layers, got {len(cross_attn_kv_cache)}."
            )
        # Use the action query rows from the joint [video+action] mask.
        action_query_start = int(video_seq_len) + int(action_history_seq_len)
        action_attention_mask = attention_mask[
            action_query_start:total_seq_len, :total_seq_len
        ]

        expert = cast(Any, self.mixtures["action"])
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
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

            k_prefix = [k_video]
            v_prefix = [v_video]
            if action_history_kv_cache is not None:
                if len(action_history_kv_cache) != self.num_layers:
                    raise ValueError(
                        f"`action_history_kv_cache` must contain {self.num_layers} layers, got {len(action_history_kv_cache)}."
                    )
                history_cache = action_history_kv_cache[layer_idx]
                k_history = history_cache["k"]
                v_history = history_cache["v"]
                if (
                    k_history.shape[1] != action_history_seq_len
                    or v_history.shape[1] != action_history_seq_len
                ):
                    raise ValueError(
                        f"`action_history_kv_cache[{layer_idx}]` seq len mismatch, expected {action_history_seq_len}."
                    )
                k_prefix.append(k_history)
                v_prefix.append(v_history)

            # Mixed attention: action queries attend to cached video K/V plus cached clean action history and current action K/V.
            k_cat = torch.cat([*k_prefix, k_action], dim=1)
            v_cat = torch.cat([*v_prefix, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            layer_cross_attn_kv = (
                cross_attn_kv_cache[layer_idx]
                if cross_attn_kv_cache is not None
                else None
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
                cross_attn_kv=layer_cross_attn_kv,
            )
        return x

    def prefill_action_history_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict[str, torch.Tensor]],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_history_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        action_history_seq_len: int = 0,
        cross_attn_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        prefill the clean action chunk, produce the kv cache of the clean chunk
        """

        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `prefill_action_history_with_video_cache`."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if (
            attention_mask.ndim != 2
            or attention_mask.shape[0] != attention_mask.shape[1]
        ):
            raise ValueError(
                f"`attention_mask` must be square 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = (
            int(video_seq_len) + int(action_history_seq_len) + action_seq_len
        )
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        if (
            cross_attn_kv_cache is not None
            and len(cross_attn_kv_cache) != self.num_layers
        ):
            raise ValueError(
                f"`cross_attn_kv_cache` must contain {self.num_layers} layers, got {len(cross_attn_kv_cache)}."
            )
        action_query_start = int(video_seq_len) + int(action_history_seq_len)
        action_attention_mask = attention_mask[
            action_query_start:total_seq_len, :total_seq_len
        ]

        expert = cast(Any, self.mixtures["action"])
        x = action_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = cast(Any, expert.blocks[layer_idx])
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
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )
            k_prefix = [k_video]
            v_prefix = [v_video]
            if action_history_kv_cache is not None:
                if len(action_history_kv_cache) != self.num_layers:
                    raise ValueError(
                        f"`action_history_kv_cache` must contain {self.num_layers} layers, got {len(action_history_kv_cache)}."
                    )
                history_cache = action_history_kv_cache[layer_idx]
                k_history = history_cache["k"]
                v_history = history_cache["v"]
                if (
                    k_history.shape[1] != action_history_seq_len
                    or v_history.shape[1] != action_history_seq_len
                ):
                    raise ValueError(
                        f"`action_history_kv_cache[{layer_idx}]` seq len mismatch, expected {action_history_seq_len}."
                    )
                k_prefix.append(k_history)
                v_prefix.append(v_history)
            k_cat = torch.cat([*k_prefix, k_action], dim=1)
            v_cat = torch.cat([*v_prefix, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            layer_cross_attn_kv = (
                cross_attn_kv_cache[layer_idx]
                if cross_attn_kv_cache is not None
                else None
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
                cross_attn_kv=layer_cross_attn_kv,
            )
            kv_cache.append({"k": k_action, "v": v_action})
        return kv_cache

    @staticmethod
    def append_kv_cache(
        base_kv_cache: Optional[list[dict[str, torch.Tensor]]],
        delta_kv_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        if base_kv_cache is None:
            return [{"k": layer["k"], "v": layer["v"]} for layer in delta_kv_cache]
        if len(base_kv_cache) != len(delta_kv_cache):
            raise ValueError(
                f"`base_kv_cache` and `delta_kv_cache` layer mismatch: {len(base_kv_cache)} vs {len(delta_kv_cache)}."
            )
        merged: list[dict[str, torch.Tensor]] = []
        for base_layer, delta_layer in zip(base_kv_cache, delta_kv_cache):
            merged.append(
                {
                    "k": torch.cat([base_layer["k"], delta_layer["k"]], dim=1),
                    "v": torch.cat([base_layer["v"], delta_layer["v"]], dim=1),
                }
            )
        return merged

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict[str, torch.Tensor]]],
        t_mod_all: Dict[str, torch.Tensor],
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

        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = cast(Any, self.mixtures[name])
                block = cast(Any, expert.blocks[layer_idx])
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
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

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
                    use_gradient_checkpointing=cached_expert[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
