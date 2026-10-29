"""Mixture of transformers: one expert per stream, one joint attention per layer.

Every expert keeps its own transformer weights (or shares them, as RGB and depth
do). In each layer the experts compute their queries, keys and values
separately, attend jointly under a single ``[query, key]`` mask, and then apply
their own output projection, cross-attention and MLP.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from prismwam.utils.logging_config import get_logger

from .wan_video_dit import flash_attention, modulate, rope_apply

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures:
            raise ValueError("`mixtures` must include a 'video' expert.")
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.attn_head_dim = first_expert.attn_head_dim
        for name in self.expert_order:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    f"All experts must have same attn_head_dim; got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
            if int(expert.num_heads) != self.num_heads:
                raise ValueError(
                    f"Expert {name!r} has {expert.num_heads} heads but joint attention uses {self.num_heads}."
                )
        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum((p.numel() for p in expert.parameters())) / 1000000000.0:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward, q_cat, k_cat, v_cat, use_reentrant=False
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
            cross_attn_input = block.norm3(x)
            branches = context_payload["cross_attention"]
            for modality, branch_payload in branches.items():
                if modality != "text":
                    raise ValueError(
                        f"Only text cross-attention is supported, got `{modality}`."
                    )
                context = branch_payload.get("context")
                if context is None:
                    continue
                context_mask = branch_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                scale = float(branch_payload.get("scale", 1.0))
                delta = block.cross_attn(
                    cross_attn_input, context, ctx_mask=context_mask
                )
                x = x + scale * delta
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor
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
            context_payload: Optional ``{"cross_attention": {"text": {...}}}``
                dict with the text ``context`` [B, L, D] and its ``mask``.

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
            mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        )

    def prefill_prefix_expert_cache(
        self,
        *,
        expert_name: str,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context_payload: Optional[dict],
        attention_mask: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Run the leading expert once and cache each layer's K/V.

        Only the first expert can be cached: removing a middle or trailing
        expert would change sequence ordering and mask semantics. The cached
        expert must also be independent of every later expert in
        ``attention_mask``; callers enforce that topology.
        """
        if not self.expert_order or expert_name != self.expert_order[0]:
            raise ValueError(
                f"`expert_name` must be the leading MoT expert {self.expert_order[0]!r}, got {expert_name!r}."
            )
        if attention_mask.ndim not in (2, 3):
            raise ValueError(
                f"`attention_mask` must be 2D [S,S] or 3D [B,S,S], got {tuple(attention_mask.shape)}."
            )
        prefix_seq_len = int(tokens.shape[1])
        if tuple(attention_mask.shape[-2:]) != (prefix_seq_len, prefix_seq_len):
            raise ValueError(
                f"Prefix attention mask must match the prefix token count: mask={tuple(attention_mask.shape[-2:])}, tokens={prefix_seq_len}."
            )
        expert = self.mixtures[expert_name]
        x = tokens
        kv_cache: list[dict[str, Any]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
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
                expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod
            )
            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v, attention_mask=attention_mask
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
                context_payload=context_payload,
            )
            kv_cache.append(
                {
                    "k": k,
                    "v": v,
                    "prefix_expert_name": expert_name,
                    "prefix_seq_len": prefix_seq_len,
                }
            )
        return kv_cache

    def forward_with_prefix_expert_cache(
        self,
        *,
        prefix_expert_name: str,
        prefix_kv_cache: list[dict[str, Any]],
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run all later experts against a cached leading-expert prefix.

        Prefix query outputs are unnecessary: attention is row-independent,
        and a directed topology guarantees that the prefix never reads later
        K/V. The later experts still share one mixed-attention call per layer,
        so their mutual information flow is identical to :meth:`forward`.
        """
        if not self.expert_order or prefix_expert_name != self.expert_order[0]:
            raise ValueError(
                f"`prefix_expert_name` must be the leading MoT expert {self.expert_order[0]!r}, got {prefix_expert_name!r}."
            )
        suffix_names = self.expert_order[1:]
        if not suffix_names:
            raise ValueError("A cached-prefix forward requires suffix experts.")
        for payload_name, payload in (
            ("embeds_all", embeds_all),
            ("freqs_all", freqs_all),
            ("context_all", context_all),
            ("t_mod_all", t_mod_all),
        ):
            missing = [name for name in suffix_names if name not in payload]
            extras = sorted(set(payload) - set(suffix_names))
            if missing or extras:
                raise ValueError(
                    f"`{payload_name}` must contain exactly suffix experts {suffix_names}; missing={missing}, extras={extras}."
                )
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_kv_cache` must contain one entry per layer: expected {self.num_layers}, got {len(prefix_kv_cache)}."
            )
        if attention_mask.ndim not in (2, 3):
            raise ValueError(
                f"`attention_mask` must be 2D or 3D, got {tuple(attention_mask.shape)}."
            )
        first_cache = prefix_kv_cache[0]
        prefix_seq_len = int(first_cache.get("prefix_seq_len", -1))
        suffix_seq_len = sum((int(embeds_all[name].shape[1]) for name in suffix_names))
        expected_total = prefix_seq_len + suffix_seq_len
        mask_query_len = int(attention_mask.shape[-2])
        mask_key_len = int(attention_mask.shape[-1])
        if prefix_seq_len <= 0 or mask_key_len != expected_total:
            raise ValueError(
                f"Cached-prefix attention layout mismatch: prefix={prefix_seq_len}, suffix={suffix_seq_len}, mask={tuple(attention_mask.shape[-2:])}."
            )
        if mask_query_len not in (expected_total, suffix_seq_len):
            raise ValueError(
                f"Cached-prefix attention mask must be full square or a suffix-query rectangle: expected query rows {expected_total} or {suffix_seq_len}, got {mask_query_len}."
            )
        tokens_all = {name: embeds_all[name] for name in suffix_names}
        batch_size = int(tokens_all[suffix_names[0]].shape[0])
        suffix_attention_mask = (
            attention_mask[..., prefix_seq_len:, :]
            if mask_query_len == expected_total
            else attention_mask
        )
        for layer_idx in range(self.num_layers):
            layer_cache = prefix_kv_cache[layer_idx]
            if layer_cache.get("prefix_expert_name") != prefix_expert_name:
                raise ValueError(
                    f"Prefix cache layer {layer_idx} belongs to {layer_cache.get('prefix_expert_name')!r}, not {prefix_expert_name!r}."
                )
            if int(layer_cache.get("prefix_seq_len", -1)) != prefix_seq_len:
                raise ValueError(
                    f"Prefix cache layer {layer_idx} has inconsistent sequence length."
                )
            k_prefix = layer_cache.get("k")
            v_prefix = layer_cache.get("v")
            if not isinstance(k_prefix, torch.Tensor) or not isinstance(
                v_prefix, torch.Tensor
            ):
                raise TypeError(
                    f"Prefix cache layer {layer_idx} must contain tensor K/V."
                )
            if (
                int(k_prefix.shape[0]) != batch_size
                or int(v_prefix.shape[0]) != batch_size
                or int(k_prefix.shape[1]) != prefix_seq_len
                or (int(v_prefix.shape[1]) != prefix_seq_len)
            ):
                raise ValueError(f"Prefix cache layer {layer_idx} K/V shape mismatch.")
            q_chunks = []
            k_chunks = [k_prefix]
            v_chunks = [v_prefix]
            seq_lens = []
            cached = {}
            for name in suffix_names:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
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
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(int(tokens_all[name].shape[1]))
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }
            mixed = self._mixed_attention(
                q_cat=torch.cat(q_chunks, dim=1),
                k_cat=torch.cat(k_chunks, dim=1),
                v_cat=torch.cat(v_chunks, dim=1),
                attention_mask=suffix_attention_mask,
            )
            start = 0
            for name, seq_len in zip(suffix_names, seq_lens):
                end = start + seq_len
                cached_expert = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=cached_expert["block"],
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed[:, start:end],
                    context_payload=context_all[name],
                )
                start = end
        return tokens_all

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")
        if attention_mask.ndim not in (2, 3):
            raise ValueError(
                f"`attention_mask` must be 2D [S,S] or 3D [B,S,S]; got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[-2] != attention_mask.shape[-1]:
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
            if attention_mask.shape[-1] != q_cat.shape[1]:
                raise ValueError(
                    f"Attention mask seq length mismatch: mask={attention_mask.shape[-1]} vs tokens={q_cat.shape[1]}"
                )
            mixed = self._mixed_attention(
                q_cat=q_cat,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=attention_mask,
            )
            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                cached_expert = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=cached_expert["block"],
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed[:, start:end, :],
                    context_payload=context_all.get(name),
                )
                start = end
        return tokens_all
