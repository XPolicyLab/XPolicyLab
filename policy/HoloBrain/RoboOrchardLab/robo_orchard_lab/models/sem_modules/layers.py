# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import math

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from robo_orchard_lab.utils.build import build


def linear_act_ln(
    embed_dims,
    in_loops,
    out_loops,
    input_dims=None,
    act_cfg=None,
):
    if act_cfg is None:
        act_cfg = dict(type=nn.ReLU, inplace=True)

    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(build(act_cfg))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class ScalarEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        max_period=10000,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, t):
        t_freq = self.freqs * t[:, None]
        t_freq = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64)
                .float()
                .to(device)
                / self.dim
            )
        )
        freqs = (
            torch.arange(max_position_embeddings)[:, None].to(self.inv_freq)
            @ self.inv_freq[None]
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def forward(self, x, position_ids):
        # x: [bs,h,n,c] or [bs,n,c]
        # position_ids: [bs,n]
        position_ids = position_ids.to(torch.int32)
        cos = self.cos.to(x)[position_ids]
        sin = self.sin.to(x)[position_ids]
        if x.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)  # b 1 n c
        x = (x * cos) + (rotate_half(x) * sin)
        return x


class RotaryAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        max_position_embeddings=128,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, all_head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.proj = nn.Linear(all_head_dim, embed_dims)
        self.position_encoder = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings
        )
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        pre_scale=True,
        key_padding_mask=None,
        identity=None,
        **kwargs,
    ):
        if identity is None:
            identity = query

        B, N, C = query.shape  # noqa: N806
        M = key.shape[1]  # noqa: N806
        q = self.q_proj(query)
        k = self.k_proj(key)
        if value is None:
            value = key
        v = self.v_proj(value)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,n,c
        k = k.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,m,c

        q, k = self.apply_position_encode(q, query_pos, k, key_pos)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        # Equivalent Implementation
        # attn = torch.einsum("bhnc,bhmc->bhnm", q, k) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.shape[0] == B:
                attn_mask = attn_mask.unsqueeze(1)
            attn = torch.where(attn_mask, float("-inf"), attn)

        if key_padding_mask is not None:
            attn = torch.where(
                key_padding_mask[:, None, None], float("-inf"), attn
            )

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = x + identity
        return x

    def apply_position_encode(self, q, query_pos, k, key_pos):
        if query_pos is not None:
            q = self.position_encoder(q, query_pos)
        if key_pos is not None:
            k = self.position_encoder(k, key_pos)
        return q, k


class JointGraphAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, all_head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)

        self.proj = nn.Linear(all_head_dim, embed_dims)
        self.position_encoder = ScalarEmbedder(embed_dims)
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        identity=None,
        **kwargs,
    ):
        if identity is None:
            identity = query
        B, N, C = query.shape  # noqa: N806
        M = key.shape[1]  # noqa: N806
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,n,c
        k = k.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,m,c

        query_pos = self.position_encoder(query_pos.flatten()).reshape(
            -1, N, M, C
        )  # bs, n, m, c
        query_pos = query_pos.unflatten(-1, (self.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )
        if B != query_pos.shape[0]:
            query_pos = query_pos.tile(B // query_pos.shape[0], 1, 1, 1, 1)

        q = q[:, :, :, None] * query_pos

        attn = (q * k.unsqueeze(2)).sum(-1) * self.scale
        # Equivalent Implementation
        # attn = torch.einsum("bhnmc,bhmc->bhnm", q, k) * self.scale

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = x + identity
        return x


class AdaRMSNorm(nn.RMSNorm):
    def __init__(
        self,
        normalized_shape,
        condition_dims,
        num_condition_mlp_layers=2,
        elementwise_affine=False,
        eps=1e-6,
        zero=False,
        **kwargs,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            **kwargs,
        )
        self.zero = zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                condition_dims, normalized_shape * (2 if not zero else 6)
            ),
        )

    def forward(self, x, c):
        x = super().forward(x)
        return self.apply_ada_func(x, c)

    def apply_ada_func(self, x, c):
        dims = x.shape[-1]
        ada_scale_shift = self.adaLN_modulation(c).unflatten(-1, (dims, -1))
        if ada_scale_shift.dim() != 4:
            ada_scale_shift = ada_scale_shift[:, None]
        x = x * (1 + ada_scale_shift[..., 0]) + ada_scale_shift[..., 1]
        if self.zero:
            gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                x.squeeze(dim=-1)
                for x in ada_scale_shift[..., 2:].chunk(4, dim=-1)
            ]
        else:
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class UpsampleHead(nn.Module):
    def __init__(
        self, upsample_sizes, input_dim, dims, norm, act, norm_act_idx
    ):
        super().__init__()
        self.norm_act_idx = norm_act_idx
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.act_and_norm = nn.ModuleList()
        dims = [input_dim] + dims
        for i, size in enumerate(upsample_sizes):
            if i in norm_act_idx:
                norm["normalized_shape"] = dims[i]
                self.act_and_norm.append(
                    nn.Sequential(build(act), build(norm))
                )
            else:
                self.act_and_norm.append(None)

            self.upsamples.append(
                nn.Upsample(
                    size=(size, 1), mode="bilinear", align_corners=True
                ),
            )
            self.convs.append(nn.Conv1d(dims[i], dims[i + 1], 3, padding=1))

    def forward(self, x):
        bs, num_joint, num_chunk, state_dims = x.shape
        x = x.flatten(0, 1)
        for i, layer in enumerate(self.upsamples):
            if i in self.norm_act_idx:
                x = self.act_and_norm[i](x)
            x = x.permute(0, 2, 1)
            x = self.convs[i](layer(x.unsqueeze(-1)).squeeze(-1))
            x = x.permute(0, 2, 1)
        x = x.unflatten(0, (bs, num_joint))
        return x
