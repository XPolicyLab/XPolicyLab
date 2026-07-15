# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half

ENABLE_ATTENTION_BIAS=os.environ.get("ENABLE_ATTENTION_BIAS", "true").lower() in ["true", "1"]

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return hidden_states.repeat_interleave(n_rep, dim=1)


def repeat_batch(
    hidden_states: torch.Tensor,
    target_batch_size: int,
) -> torch.Tensor:
    """
    Repeats the batch dimension of `hidden_states` to reach `target_batch_size`.

    Args:
        hidden_states (torch.Tensor): Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        target_batch_size (int): Target batch size (must be a multiple of current batch size)

    Returns:
        torch.Tensor: Tensor with updated batch size
    """
    batch = hidden_states.size(0)
    if batch == target_batch_size:
        return hidden_states

    n_rep = target_batch_size // batch
    return hidden_states.repeat_interleave(n_rep, dim=0)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb[:, None]


class LinearWithNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = Qwen2RMSNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return self.norm(hidden_states)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        head_dim=64,
        kv_heads=2,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=ENABLE_ATTENTION_BIAS)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.q_norm = Qwen2RMSNorm(self.head_dim)
        self.k_norm = Qwen2RMSNorm(self.head_dim)

    def forward(self, hidden_state, past_key_values, position_embeds, attn_mask=None):
        bsz, q_len, _ = hidden_state.size()

        qkv = self.qkv_proj(hidden_state)

        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeds

        if cos.ndim == 4:
            cos = cos[0]
            sin = sin[0]

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        k_cache, v_cache = past_key_values
        k_cache = repeat_batch(k_cache, bsz)
        v_cache = repeat_batch(v_cache, bsz)

        k_cache = repeat_kv(k_cache, self.kv_group)
        v_cache = repeat_kv(v_cache, self.kv_group)

        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size=768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        head_dim=64,
        kv_heads=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn = Attention(
            hidden_size=self.hidden_size,
            head_dim=head_dim,
            kv_heads=kv_heads,
        )

        self.mlp = MLP(hidden_size=self.hidden_size)

        self.input_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.post_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)

        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states,
        past_key_values,
        position_embeds,
        t_embeds,
        attn_mask=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(
            6, dim=1
        )

        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.attn(
            hidden_states,
            past_key_values,
            position_embeds,
            attn_mask=attn_mask,
        )
        hidden_states = residual + gate_msa * hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states

        return hidden_states

    def forward_perf(
        self,
        hidden_states,
        past_key_values,
        position_embeds,
        t_embeds,
        attn_mask,
        repeat,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(
            6, dim=1
        )

        # LN + AdaLN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)

        # Flatten hidden states
        repeated_bsz, q_len, _ = hidden_states.shape
        assert repeated_bsz % repeat == 0
        original_bsz = repeated_bsz // repeat
        hidden_states = hidden_states.view(original_bsz, repeat * q_len, -1)

        # Attention
        assert attn_mask.shape[0] == original_bsz
        hidden_states = self.attn(
            hidden_states,
            past_key_values,
            position_embeds,
            attn_mask=attn_mask,
        )

        # Reshape back
        hidden_states = hidden_states.view(repeated_bsz, q_len, -1)

        # AdaLN
        hidden_states = residual + gate_msa * hidden_states

        # LN + AdaLN + FFN
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states

        return hidden_states

class DiT(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        layer_num=8,
        is_causal=True,
        head_dim=128,
        kv_heads=2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.is_causal = is_causal

        # layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    kv_heads=kv_heads,
                )
                for _ in range(layer_num)
            ]
        )
        self.layer_num = layer_num

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, t_embeds):
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start_idx + i],
                position_embeds,
                t_embeds,
                attn_mask=attn_mask,
            )
        return hidden_states
    
    def forward_perf(self, hidden_states, past_key_values, attn_mask, position_embeds, t_embeds, repeat):
        repeated_bsz, q_len, _ = hidden_states.shape
        assert repeated_bsz % repeat == 0
        original_bsz = repeated_bsz // repeat
        position_embeds = tuple(
            x.reshape(original_bsz, repeat * q_len, -1)
            for x in position_embeds
        )
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward_perf(
                hidden_states,
                past_key_values[start_idx + i],
                position_embeds,
                t_embeds,
                attn_mask=attn_mask,
                repeat=repeat,
            )
        return hidden_states


