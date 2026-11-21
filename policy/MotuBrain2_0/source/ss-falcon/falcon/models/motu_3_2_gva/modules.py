import torch
import torch.nn as nn
from torch.nn import RMSNorm

from falcon.distributed import all_to_all_2_or_3D, get_cp_group
from falcon.ops import all_to_all_flat
from soma.inference.models.operators import attend
from soma.inference.models.operators import chunk_qkv as multi_qkv_chunk_func
from soma.inference.models.operators import (
    rmsnorm_cached_rope as fused_rmsnorm_cache_rope_func,
)
from soma.inference.models.operators import split_vac as split_vac_padded
from soma.inference.models.operators import unite_vac as unite_vac_padded

from .model_config import GVAConfig


def _get_unite_split_vac_kwargs(packed_seq_params: dict):
    return dict(
        cuVACs=packed_seq_params["cuVACs"],
        totalVAC=packed_seq_params["totalVAC"],
        maxVAC=packed_seq_params["maxVAC"],
        cuVs=packed_seq_params["cuVs"],
        totalV=packed_seq_params["totalV"],
        maxV=packed_seq_params["maxV"],
        cuVs_padded=packed_seq_params["cuVs"],
        totalV_padded=packed_seq_params["totalV"],
        maxV_padded=packed_seq_params["maxV"],
        hasV=packed_seq_params.get("hasV", True),
        cuAs=packed_seq_params["cuAs"],
        totalA=packed_seq_params["totalA"],
        maxA=packed_seq_params["maxA"],
        cuAs_padded=packed_seq_params["cuAs_padded"],
        totalA_padded=packed_seq_params["totalA_padded"],
        maxA_padded=packed_seq_params["maxA_padded"],
        hasA=packed_seq_params.get("hasA", True),
        cuCs=packed_seq_params["cuCs"],
        totalC=packed_seq_params["totalC"],
        maxC=packed_seq_params["maxC"],
        cuCs_padded=packed_seq_params["cuCs_padded"],
        totalC_padded=packed_seq_params["totalC_padded"],
        maxC_padded=packed_seq_params["maxC_padded"],
        hasC=packed_seq_params.get("hasC", True),
    )


class FullJointAttention(nn.Module):
    """Audio Version Full Joint Attention

    Use QKARoPE for audio version full joint attention.

    Args:
        motu_cfg (GAConfig): GVA configuration
        layer_number (int): Layer number
    """

    def __init__(self, motu_cfg: GVAConfig, layer_number: int):
        super().__init__()
        self.motu_cfg = motu_cfg
        hidden_size = motu_cfg.hidden_size
        hidden_size_per_attention_head = motu_cfg.hidden_size_per_attention_head
        self.layer_number = layer_number
        self.is_last_layer = layer_number == motu_cfg.num_layers

        self.x_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.a_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.c_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)
        self.k_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)
        self.a_q_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)
        self.a_k_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)
        self.c_q_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)
        self.c_k_layernorm = RMSNorm(hidden_size_per_attention_head, eps=1e-6)

        # projection layers
        self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if not self.is_last_layer:
            self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # RoPE is now fused into RMSNorm (fused_rmsnorm_cache_rope_func) using
        # per-forward caches built from motu_cfg.video_rope_axes / audio_rope_axes.
        # No standalone RoPE module is held anymore (was self.rope3d / self.rope1d).

    def get_vac_q_k_v(self, hidden_states: torch.Tensor, audio_hidden_states: torch.Tensor, context: torch.Tensor):
        x_qkv = self.x_qkv(hidden_states)
        a_qkv = self.a_qkv(audio_hidden_states)
        c_qkv = self.c_qkv(context)
        return x_qkv, a_qkv, c_qkv

    def get_query_key_value_tensors_from_vac_qkv(
        self,
        x_qkv: torch.Tensor,
        a_qkv: torch.Tensor,
        c_qkv: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [T, 3 * D] ---> [T, H, 3 * d]
        num_attention_heads = self.motu_cfg.num_attention_heads
        hidden_size_per_attention_head = self.motu_cfg.hidden_size_per_attention_head

        x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v = multi_qkv_chunk_func(
            [x_qkv, a_qkv, c_qkv],
            num_heads=num_attention_heads,
            head_dim=hidden_size_per_attention_head,
        )

        # Context: RMSNorm only (no RoPE), then all-to-all (flat).
        # [C_local, H, d] ---> [totalC, H // cp_size, d]
        c_q, c_k = self.c_q_layernorm(c_q), self.c_k_layernorm(c_k)
        c_q, c_k, c_v = all_to_all_flat([c_q, c_k, c_v])

        # Video: fused RMSNorm + RoPE on LOCAL tokens (BEFORE a2a — the cache is
        # aligned to this rank's flat local window). Pass raw .weight (fused
        # kernel does the RMSNorm internally). Then all-to-all.
        # [V_local, H, d] --fused--> [V_local, H, d] --a2a--> [totalV, H // cp_size, d]
        v_cos, v_sin, v_partner = packed_seq_params["video_rope_cache"]
        x_q, x_k = fused_rmsnorm_cache_rope_func(
            x_q,
            x_k,
            v_cos,
            v_sin,
            v_partner,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            1e-6,
        )
        x_q, x_k, x_v = all_to_all_flat([x_q, x_k, x_v])

        # Audio: fused RMSNorm + RoPE on LOCAL tokens, then all-to-all.
        # [A_local, H, d] --fused--> [A_local, H, d] --a2a--> [totalA, H // cp_size, d]
        a_cos, a_sin, a_partner = packed_seq_params["audio_rope_cache"]
        a_q, a_k = fused_rmsnorm_cache_rope_func(
            a_q,
            a_k,
            a_cos,
            a_sin,
            a_partner,
            self.a_q_layernorm.weight,
            self.a_k_layernorm.weight,
            1e-6,
        )
        a_q, a_k, a_v = all_to_all_flat([a_q, a_k, a_v])

        # unite video, audio and context tokens
        unite_vac_kwargs = _get_unite_split_vac_kwargs(packed_seq_params)
        query = unite_vac_padded(x_q, a_q, c_q, **unite_vac_kwargs)
        key = unite_vac_padded(x_k, a_k, c_k, **unite_vac_kwargs)
        value = unite_vac_padded(x_v, a_v, c_v, **unite_vac_kwargs)

        return query, key, value  # type: ignore

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get query, key, value tensors from hidden states and context."""
        x_qkv, a_qkv, c_qkv = self.get_vac_q_k_v(hidden_states, audio_hidden_states, context)
        query, key, value = self.get_query_key_value_tensors_from_vac_qkv(x_qkv, a_qkv, c_qkv, packed_seq_params)
        return query, key, value

    def core_attention(self, query, key, value, packed_seq_params) -> torch.Tensor:
        cu_seqlens = packed_seq_params["cuVACs"]
        max_seqlen = packed_seq_params["maxVAC"]
        attn_out = attend(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )
        return attn_out  # type: ignore

    def proj_out(
        self, attn_out: torch.Tensor, packed_seq_params: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # split attn_out into a and c(padded)
        split_vac_kwargs = _get_unite_split_vac_kwargs(packed_seq_params)
        attn_out = attn_out.reshape(attn_out.shape[0], -1)
        # x: [totalV, H, d], a: [totalA, H, d], c: [totalC, H, d]
        x, a, c = split_vac_padded(attn_out, **split_vac_kwargs)  # type: ignore

        # scatter along head dim (flat CP: no interleaving needed, seq already global)
        x = all_to_all_2_or_3D(x, scatter_idx=0, gather_idx=1, group=get_cp_group())
        x = self.x_proj(x)

        a = all_to_all_2_or_3D(a, scatter_idx=0, gather_idx=1, group=get_cp_group())
        a = self.a_proj(a)

        if not self.is_last_layer:
            c = all_to_all_2_or_3D(c, scatter_idx=0, gather_idx=1, group=get_cp_group())
            c = self.c_proj(c)

        return x, a, c

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply full joint attention to hidden states and context.
        Args:
            hidden_states (torch.Tensor): [T, D]
            audio_hidden_states (torch.Tensor): [T, D]
            context (torch.Tensor): [T, D]
            packed_seq_params (dict): dict

        Returns:
            x (torch.Tensor): [T, D]
            a (torch.Tensor): [T, D]
            c (torch.Tensor): [T, D]
        """
        # get query, key, value tensors [L, H, d], L = totalAC = totalA + totalC
        query, key, value = self.get_query_key_value_tensors(
            hidden_states, audio_hidden_states, context, packed_seq_params
        )
        # attention output [L, H, d]
        attn_out = self.core_attention(query, key, value, packed_seq_params)
        return self.proj_out(attn_out, packed_seq_params)

    def init_weights(self):
        nn.init.xavier_normal_(self.x_qkv.weight)
        nn.init.xavier_normal_(self.a_qkv.weight)
        nn.init.xavier_normal_(self.c_qkv.weight)
        self.q_layernorm.reset_parameters()
        self.k_layernorm.reset_parameters()
        self.a_q_layernorm.reset_parameters()
        self.a_k_layernorm.reset_parameters()
        self.c_q_layernorm.reset_parameters()
        self.c_k_layernorm.reset_parameters()
        nn.init.xavier_normal_(self.x_proj.weight)
        nn.init.xavier_normal_(self.a_proj.weight)
        if not self.is_last_layer:
            nn.init.xavier_normal_(self.c_proj.weight)
