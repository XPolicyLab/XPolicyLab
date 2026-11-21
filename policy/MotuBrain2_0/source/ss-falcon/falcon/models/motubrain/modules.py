"""Attention modules for Motubrain."""

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from falcon.distributed import all_to_all_2_or_3D, get_cp_group
from falcon.models.motubrain.model_config import MotubrainConfig
from falcon.models.motu_3_2_gva.modules import FullJointAttention
from falcon.ops import all_to_all_flat
from soma.inference.models.operators import attend
from soma.inference.models.operators import chunk_qkv as multi_qkv_chunk_func
from soma.inference.models.operators import (
    rmsnorm_cached_rope as fused_rmsnorm_cache_rope_func,
)
from soma.inference.models.operators import split_vac as split_vac_padded
from soma.inference.models.operators import split_vc as split_vc_padded
from soma.inference.models.operators import unite_vac as unite_vac_padded
from soma.inference.models.operators import unite_vc as unite_vc_padded


def _get_unite_split_vac_kwargs(packed_seq_params: dict) -> dict:
    return {
        "cuVACs": packed_seq_params["cuVACs"],
        "totalVAC": packed_seq_params["totalVAC"],
        "maxVAC": packed_seq_params["maxVAC"],
        "cuVs": packed_seq_params["cuVs"],
        "totalV": packed_seq_params["totalV"],
        "maxV": packed_seq_params["maxV"],
        "cuVs_padded": packed_seq_params["cuVs"],
        "totalV_padded": packed_seq_params["totalV"],
        "maxV_padded": packed_seq_params["maxV"],
        "hasV": packed_seq_params.get("hasV", True),
        "cuAs": packed_seq_params["cuAs"],
        "totalA": packed_seq_params["totalA"],
        "maxA": packed_seq_params["maxA"],
        "cuAs_padded": packed_seq_params["cuAs_padded"],
        "totalA_padded": packed_seq_params["totalA_padded"],
        "maxA_padded": packed_seq_params["maxA_padded"],
        "hasA": packed_seq_params.get("hasA", True),
        "cuCs": packed_seq_params["cuCs"],
        "totalC": packed_seq_params["totalC"],
        "maxC": packed_seq_params["maxC"],
        "cuCs_padded": packed_seq_params["cuCs_padded"],
        "totalC_padded": packed_seq_params["totalC_padded"],
        "maxC_padded": packed_seq_params["maxC_padded"],
        "hasC": packed_seq_params.get("hasC", True),
    }


def _get_unite_split_vc_kwargs(packed_seq_params: dict) -> dict:
    return {
        "cuVCs": packed_seq_params["cuVCs"],
        "totalVC": packed_seq_params["totalVC"],
        "maxVC": packed_seq_params["maxVC"],
        "cuVs": packed_seq_params["cuVs"],
        "totalV": packed_seq_params["totalV"],
        "maxV": packed_seq_params["maxV"],
        "cuVs_padded": packed_seq_params["cuVs"],
        "totalV_padded": packed_seq_params["totalV"],
        "maxV_padded": packed_seq_params["maxV"],
        "cuCs": packed_seq_params["cuCs"],
        "totalC": packed_seq_params["totalC"],
        "maxC": packed_seq_params["maxC"],
        "cuCs_padded": packed_seq_params["cuCs_padded"],
        "totalC_padded": packed_seq_params["totalC_padded"],
        "maxC_padded": packed_seq_params["maxC_padded"],
    }


def _filter_video_keys(
    key: torch.Tensor,
    value: torch.Tensor,
    packed: dict,
    base: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove invalid video keys while retaining the complete query layout."""
    indices = packed.get(f"{base}_key_indices")
    if indices is None:
        return (
            key,
            value,
            packed[f"cu{base}s"],
            packed[f"max{base}"],
        )
    return (
        key.index_select(0, indices),
        value.index_select(0, indices),
        packed[f"cu{base}s_key"],
        packed[f"max{base}_key"],
    )


class ActionConditionProxy(nn.Module):
    """Project external condition tokens into action-attention keys and values."""

    def __init__(self, motu_cfg: MotubrainConfig, feature_dim: int):
        super().__init__()
        self.num_heads = motu_cfg.num_attention_heads
        self.head_dim = motu_cfg.hidden_size_per_attention_head
        self.embed = nn.Linear(feature_dim, motu_cfg.hidden_size, bias=False)
        self.kv = nn.Linear(motu_cfg.hidden_size, 2 * motu_cfg.hidden_size, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self) -> None:
        """Initialize trainable projections in their original operation order."""
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.kv.weight)
        self.k_norm.reset_parameters()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project CP-local features and exchange sequence/head partitions."""
        projected = self.kv(self.embed(features))
        projected = projected.reshape(projected.shape[0], self.num_heads, 2 * self.head_dim)
        key, value = projected.chunk(2, dim=-1)
        key, value = all_to_all_flat([self.k_norm(key), value.contiguous()])
        return key, value


class _MotubrainAttention(FullJointAttention):
    """Reuse GVA projections, RoPE, CP exchange, and output projection."""

    def __init__(self, motu_cfg: MotubrainConfig, layer_number: int):
        super().__init__(motu_cfg, layer_number)
        self.condition_proxies = nn.ModuleDict({
            name: ActionConditionProxy(motu_cfg, condition["feature_dim"])
            for name, condition in motu_cfg.action_conditions.items()
            if layer_number in condition["layers"]
        })

    def init_weights(self) -> None:
        super().init_weights()
        for proxy in self.condition_proxies.values():
            proxy.init_weights()

    def _action_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed: dict,
        base: str,
    ) -> torch.Tensor:
        if base == "VAC":
            key, value, cumulative, maximum = _filter_video_keys(
                key,
                value,
                packed,
                base,
            )
        else:
            cumulative = packed["cuAs_padded"]
            maximum = packed["maxA_padded"]
        for name, proxy in self.condition_proxies.items():
            condition = packed["action_conditions"][name]
            condition_key, condition_value = proxy(condition["features"][self.layer_number])
            indices, cumulative, maximum = packed["condition_key_layouts"][self.layer_number][name][base]
            key = torch.cat((key, condition_key), dim=0).index_select(0, indices)
            value = torch.cat((value, condition_value), dim=0).index_select(0, indices)
        return attend(
            query,
            key,
            value,
            cu_seqlens_q=packed["cuAs_padded"],
            cu_seqlens_k=cumulative,
            max_seqlen_q=packed["maxA_padded"],
            max_seqlen_k=maximum,
        ).reshape(query.shape[0], -1)

    def _project_modal_qkv(
        self,
        hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, ...]:
        """Project action through GVA's audio path using Falcon kernels."""
        x_qkv, a_qkv, c_qkv = self.get_vac_q_k_v(
            hidden_states,
            action_hidden_states,
            context,
        )
        modal_qkv = multi_qkv_chunk_func(
            [x_qkv, a_qkv, c_qkv],
            num_heads=self.motu_cfg.num_attention_heads,
            head_dim=self.motu_cfg.hidden_size_per_attention_head,
        )
        x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v = modal_qkv

        c_q, c_k = self.c_q_layernorm(c_q), self.c_k_layernorm(c_k)
        c_q, c_k, c_v = all_to_all_flat([c_q, c_k, c_v])

        video_cos, video_sin, video_partner = packed_seq_params["video_rope_cache"]
        x_q, x_k = fused_rmsnorm_cache_rope_func(
            x_q,
            x_k,
            video_cos,
            video_sin,
            video_partner,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            1e-6,
        )
        x_q, x_k, x_v = all_to_all_flat([x_q, x_k, x_v])

        action_cos, action_sin, action_partner = packed_seq_params["audio_rope_cache"]
        a_q, a_k = fused_rmsnorm_cache_rope_func(
            a_q,
            a_k,
            action_cos,
            action_sin,
            action_partner,
            self.a_q_layernorm.weight,
            self.a_k_layernorm.weight,
            1e-6,
        )
        a_q, a_k, a_v = all_to_all_flat([a_q, a_k, a_v])
        return x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v

    def _project_modal_outputs(
        self,
        x_out: torch.Tensor,
        action_out: torch.Tensor,
        context_out: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return global attention outputs to Falcon's flat CP layout."""
        x_out = all_to_all_2_or_3D(
            x_out,
            scatter_idx=0,
            gather_idx=1,
            group=get_cp_group(),
        )
        action_out = all_to_all_2_or_3D(
            action_out,
            scatter_idx=0,
            gather_idx=1,
            group=get_cp_group(),
        )
        x_out = self.x_proj(x_out)
        action_out = self.a_proj(action_out)
        if not self.is_last_layer:
            context_out = all_to_all_2_or_3D(
                context_out,
                scatter_idx=0,
                gather_idx=1,
                group=get_cp_group(),
            )
            context_out = self.c_proj(context_out)
        return x_out, action_out, context_out


class MotubrainJointAttention(_MotubrainAttention):
    """GVA joint video/action/caption attention with Motubrain RoPE metadata."""

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modal_qkv = self._project_modal_qkv(
            hidden_states,
            action_hidden_states,
            context,
            packed_seq_params,
        )
        x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v = modal_qkv
        kwargs = _get_unite_split_vac_kwargs(packed_seq_params)
        return (
            unite_vac_padded(x_q, a_q, c_q, **kwargs),
            unite_vac_padded(x_k, a_k, c_k, **kwargs),
            unite_vac_padded(x_v, a_v, c_v, **kwargs),
        )

    def core_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_seq_params: dict,
    ) -> torch.Tensor:
        key, value, cumulative, maximum = _filter_video_keys(
            key,
            value,
            packed_seq_params,
            "VAC",
        )
        return attend(
            query,
            key,
            value,
            cu_seqlens_q=packed_seq_params["cuVACs"],
            cu_seqlens_k=cumulative,
            max_seqlen_q=packed_seq_params["maxVAC"],
            max_seqlen_k=maximum,
        )

    def proj_out(
        self,
        attn_out: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out = attn_out.reshape(attn_out.shape[0], -1)
        x_out, action_out, context_out = split_vac_padded(
            attn_out,
            **_get_unite_split_vac_kwargs(packed_seq_params),
        )
        return self._project_modal_outputs(
            x_out,
            action_out,
            context_out,
            packed_seq_params,
        )


class MotubrainDecoupledAttention(_MotubrainAttention):
    """V+C joint attention with independent action self-attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modal_qkv = self._project_modal_qkv(
            hidden_states,
            action_hidden_states,
            context,
            packed_seq_params,
        )
        x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v = modal_qkv

        vc_kwargs = _get_unite_split_vc_kwargs(packed_seq_params)
        vc_q = unite_vc_padded(x_q, c_q, **vc_kwargs)
        vc_k = unite_vc_padded(x_k, c_k, **vc_kwargs)
        vc_v = unite_vc_padded(x_v, c_v, **vc_kwargs)
        vc_k, vc_v, vc_cumulative, vc_maximum = _filter_video_keys(
            vc_k,
            vc_v,
            packed_seq_params,
            "VC",
        )
        vc_out = attend(
            vc_q,
            vc_k,
            vc_v,
            cu_seqlens_q=packed_seq_params["cuVCs"],
            cu_seqlens_k=vc_cumulative,
            max_seqlen_q=packed_seq_params["maxVC"],
            max_seqlen_k=vc_maximum,
        ).reshape(vc_q.shape[0], -1)
        x_out, context_out = split_vc_padded(vc_out, **vc_kwargs)

        action_out = self._action_attention(a_q, a_k, a_v, packed_seq_params, "A_padded")
        return self._project_modal_outputs(
            x_out,
            action_out,
            context_out,
            packed_seq_params,
        )


class MotubrainV2AAttention(_MotubrainAttention):
    """V+C self-attention plus action queries over V+A+C."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        context: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modal_qkv = self._project_modal_qkv(
            hidden_states,
            action_hidden_states,
            context,
            packed_seq_params,
        )
        x_q, x_k, x_v, a_q, a_k, a_v, c_q, c_k, c_v = modal_qkv

        vc_kwargs = _get_unite_split_vc_kwargs(packed_seq_params)
        vc_q = unite_vc_padded(x_q, c_q, **vc_kwargs)
        vc_k = unite_vc_padded(x_k, c_k, **vc_kwargs)
        vc_v = unite_vc_padded(x_v, c_v, **vc_kwargs)
        vc_k, vc_v, vc_cumulative, vc_maximum = _filter_video_keys(
            vc_k,
            vc_v,
            packed_seq_params,
            "VC",
        )
        vc_out = attend(
            vc_q,
            vc_k,
            vc_v,
            cu_seqlens_q=packed_seq_params["cuVCs"],
            cu_seqlens_k=vc_cumulative,
            max_seqlen_q=packed_seq_params["maxVC"],
            max_seqlen_k=vc_maximum,
        ).reshape(vc_q.shape[0], -1)
        x_out, context_out = split_vc_padded(vc_out, **vc_kwargs)

        vac_kwargs = _get_unite_split_vac_kwargs(packed_seq_params)
        vac_k = unite_vac_padded(x_k, a_k, c_k, **vac_kwargs)
        vac_v = unite_vac_padded(x_v, a_v, c_v, **vac_kwargs)
        action_out = self._action_attention(a_q, vac_k, vac_v, packed_seq_params, "VAC")
        return self._project_modal_outputs(
            x_out,
            action_out,
            context_out,
            packed_seq_params,
        )
