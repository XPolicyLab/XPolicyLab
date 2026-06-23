from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LayoutSegment:
    expert: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return int(self.end) - int(self.start)


def _expert_num_heads(expert: nn.Module) -> int:
    if hasattr(expert, "num_heads"):
        return int(expert.num_heads)
    return int(expert.config.num_attention_heads)


def _expert_attn_head_dim(expert: nn.Module) -> int:
    if hasattr(expert, "attn_head_dim"):
        return int(expert.attn_head_dim)
    return int(expert.config.attention_head_dim)


def _project_self_attn_qkv(attn, hidden_states: torch.Tensor):
    if getattr(attn, "fused_projections", False) and hasattr(attn, "to_qkv"):
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    return query, key, value


def _apply_rotary(hidden_states: torch.Tensor, rotary_emb, heads: int) -> torch.Tensor:
    freqs_cos, freqs_sin = rotary_emb
    hidden_states = hidden_states.unflatten(2, (heads, -1))
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.flatten(2, 3).type_as(hidden_states)


def _slice_rotary(rotary_emb, start: int, end: int):
    return rotary_emb[0][:, start:end], rotary_emb[1][:, start:end]


class MoT(nn.Module):
    """Mixture-of-Transformers layer mixer for GWP-style Wan blocks.

    The module owns its experts so checkpoint keys include ``mot.*``. Hidden
    sizes may differ across experts, but every expert must expose the same
    number of layers, attention heads, and attention head dim so mixed Q/K/V
    tensors have a common width.
    """

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("MoT requires both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)

        first = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first.blocks)
        self.num_heads = _expert_num_heads(first)
        self.attn_head_dim = _expert_attn_head_dim(first)
        self.inner_dim = self.num_heads * self.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same num_layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if _expert_num_heads(expert) != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {_expert_num_heads(expert)}"
                )
            if _expert_attn_head_dim(expert) != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {_expert_attn_head_dim(expert)}"
                )

    @staticmethod
    def _split_block_temb(block, temb: torch.Tensor):
        if temb.ndim == 4:
            chunks = (block.scale_shift_table.unsqueeze(0).to(temb.device) + temb.float()).chunk(6, dim=2)
            return tuple(x.squeeze(2) for x in chunks)
        chunks = (block.scale_shift_table.to(temb.device) + temb.float()).chunk(6, dim=1)
        return chunks

    def _build_attention_io(
        self,
        block,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb,
    ):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._split_block_temb(block, temb)
        norm_hidden_states = (
            block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)

        query, key, value = _project_self_attn_qkv(block.attn1, norm_hidden_states)
        query = block.attn1.norm_q(query)
        key = block.attn1.norm_k(key)
        query = _apply_rotary(query, rotary_emb, self.num_heads)
        key = _apply_rotary(key, rotary_emb, self.num_heads)

        return {
            "q": query,
            "k": key,
            "v": value,
            "residual": hidden_states,
            "gate_msa": gate_msa,
            "c_shift_msa": c_shift_msa,
            "c_scale_msa": c_scale_msa,
            "c_gate_msa": c_gate_msa,
        }

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device, dtype=q_cat.dtype)
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q = q.unflatten(2, (self.num_heads, self.attn_head_dim)).transpose(1, 2)
            k = k.unflatten(2, (self.num_heads, self.attn_head_dim)).transpose(1, 2)
            v = v.unflatten(2, (self.num_heads, self.attn_head_dim)).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            return out.transpose(1, 2).flatten(2, 3)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(_forward, q_cat, k_cat, v_cat, use_reentrant=False)
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_post_block(block, mixed_slice: torch.Tensor, cached: dict, context: Optional[torch.Tensor]):
        hidden_states = cached["residual"]
        attn_output = block.attn1.to_out[0](mixed_slice)
        attn_output = block.attn1.to_out[1](attn_output)
        hidden_states = (hidden_states.float() + attn_output * cached["gate_msa"]).type_as(hidden_states)

        if context is not None:
            norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
            hidden_states = hidden_states + block.attn2(norm_hidden_states, context, None, None)

        norm_hidden_states = (
            block.norm3(hidden_states.float()) * (1 + cached["c_scale_msa"]) + cached["c_shift_msa"]
        ).type_as(hidden_states)
        ff_output = block.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * cached["c_gate_msa"]).type_as(hidden_states)
        return hidden_states

    @staticmethod
    def _normalize_layout(layout: Iterable[LayoutSegment | dict]) -> list[LayoutSegment]:
        out = []
        for seg in layout:
            if isinstance(seg, LayoutSegment):
                out.append(seg)
            else:
                out.append(LayoutSegment(seg["expert"], int(seg["start"]), int(seg["end"])))
        return out

    def _assemble_from_layout(self, tensors_by_expert: dict, layout: list[LayoutSegment]) -> torch.Tensor:
        return torch.cat([tensors_by_expert[seg.expert][:, seg.start:seg.end] for seg in layout], dim=1)

    def _scatter_to_experts(
        self,
        mixed: torch.Tensor,
        layout: list[LayoutSegment],
        reference_by_expert: dict,
    ) -> dict[str, torch.Tensor]:
        scattered = {
            name: torch.empty_like(reference_by_expert[name])
            for name in reference_by_expert
        }
        cursor = 0
        for seg in layout:
            next_cursor = cursor + seg.length
            scattered[seg.expert][:, seg.start:seg.end] = mixed[:, cursor:next_cursor]
            cursor = next_cursor
        return scattered

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        rotary_all: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        context_all: Dict[str, Optional[torch.Tensor]],
        t_mod_all: Dict[str, torch.Tensor],
        layout: Iterable[LayoutSegment | dict],
    ) -> Dict[str, torch.Tensor]:
        layout = self._normalize_layout(layout)
        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_all, k_all, v_all, cached_all = {}, {}, {}, {}
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                io = self._build_attention_io(
                    block=block,
                    hidden_states=tokens_all[name],
                    temb=t_mod_all[name],
                    rotary_emb=rotary_all[name],
                )
                q_all[name], k_all[name], v_all[name] = io["q"], io["k"], io["v"]
                cached_all[name] = io

            q_cat = self._assemble_from_layout(q_all, layout)
            k_cat = self._assemble_from_layout(k_all, layout)
            v_cat = self._assemble_from_layout(v_all, layout)
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
            mixed_by_expert = self._scatter_to_experts(mixed, layout, q_all)

            for name in self.expert_order:
                block = self.mixtures[name].blocks[layer_idx]
                tokens_all[name] = self._apply_post_block(
                    block=block,
                    mixed_slice=mixed_by_expert[name],
                    cached=cached_all[name],
                    context=context_all.get(name),
                )

        return tokens_all

    def prefill_prefix_cache(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        rotary_all: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        context_all: Dict[str, Optional[torch.Tensor]],
        t_mod_all: Dict[str, torch.Tensor],
        layout: Iterable[LayoutSegment | dict],
    ) -> list[dict[str, torch.Tensor]]:
        layout = self._normalize_layout(layout)
        tokens_all = {k: v for k, v in embeds_all.items()}
        kv_cache = []

        for layer_idx in range(self.num_layers):
            q_all, k_all, v_all, cached_all = {}, {}, {}, {}
            for name in self.expert_order:
                if name not in tokens_all:
                    continue
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                io = self._build_attention_io(block, tokens_all[name], t_mod_all[name], rotary_all[name])
                q_all[name], k_all[name], v_all[name] = io["q"], io["k"], io["v"]
                cached_all[name] = io

            q_cat = self._assemble_from_layout(q_all, layout)
            k_cat = self._assemble_from_layout(k_all, layout)
            v_cat = self._assemble_from_layout(v_all, layout)
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
            mixed_by_expert = self._scatter_to_experts(mixed, layout, q_all)
            kv_cache.append({"k": k_cat, "v": v_cat})

            for name in tokens_all:
                block = self.mixtures[name].blocks[layer_idx]
                tokens_all[name] = self._apply_post_block(
                    block=block,
                    mixed_slice=mixed_by_expert[name],
                    cached=cached_all[name],
                    context=context_all.get(name),
                )

        return kv_cache

    def forward_action_with_prefix_cache(
        self,
        action_tokens: torch.Tensor,
        action_rotary,
        action_t_mod: torch.Tensor,
        action_context: Optional[torch.Tensor],
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} cache layers, got {len(prefix_kv_cache)}")

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            io = self._build_attention_io(block, x, action_t_mod, action_rotary)
            cache = prefix_kv_cache[layer_idx]
            k_cat = torch.cat([cache["k"], io["k"]], dim=1)
            v_cat = torch.cat([cache["v"], io["v"]], dim=1)
            mixed = self._mixed_attention(io["q"], k_cat, v_cat, attention_mask)
            x = self._apply_post_block(block, mixed, io, action_context)
        return x


__all__ = ["LayoutSegment", "MoT", "_slice_rotary"]
