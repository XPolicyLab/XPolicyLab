from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .transformer_wa_casual import (
    FeedForward,
    FP32LayerNorm,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed1D,
)


class ActionStateBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_head_dim: int,
        eps: float = 1e-6,
        cross_attn_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=attn_head_dim,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )
        self.attn2 = WanAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=attn_head_dim,
            eps=eps,
            cross_attention_dim_head=attn_head_dim,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(hidden_dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)


class ActionStateDiT(nn.Module):
    """Compact state/action expert for MoT.

    Hidden size is independent of the Wan video expert. Self-attention projects
    to ``num_heads * attn_head_dim`` so mixed attention can share K/V/Q width
    with the video expert.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        text_dim: int = 4096,
        freq_dim: int = 256,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.action_dim),
        )

        self.timesteps_proj = Timesteps(num_channels=self.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=self.freq_dim, time_embed_dim=self.hidden_dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        self.text_embedder = PixArtAlphaTextProjection(self.text_dim, self.hidden_dim, act_fn="gelu_tanh")
        self.action_rope = WanRotaryPosEmbed1D(self.attn_head_dim, rope_max_seq_len)

        self.blocks = nn.ModuleList(
            [
                ActionStateBlock(
                    hidden_dim=self.hidden_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    eps=eps,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _embed_token_timesteps(
        self,
        timestep: torch.Tensor,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim == 1:
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            if timestep.shape[0] != batch_size:
                raise ValueError(f"Expected timestep length {batch_size}, got {tuple(timestep.shape)}")
            timestep = timestep[:, None].expand(batch_size, seq_len)
        elif timestep.ndim == 2:
            if tuple(timestep.shape) != (batch_size, seq_len):
                raise ValueError(f"Expected timestep shape {(batch_size, seq_len)}, got {tuple(timestep.shape)}")
        else:
            raise ValueError(f"Expected timestep ndim 1 or 2, got {timestep.ndim}")

        timestep_emb = self.timesteps_proj(timestep.flatten())
        time_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep_emb.dtype != time_dtype and time_dtype != torch.int8:
            timestep_emb = timestep_emb.to(time_dtype)
        temb = self.time_embedder(timestep_emb).reshape(batch_size, seq_len, self.hidden_dim).to(dtype=dtype)
        t_mod = self.time_proj(self.act_fn(temb)).unflatten(2, (6, self.hidden_dim))
        return temb, t_mod

    def pre_dit(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> dict:
        if state.ndim != 3:
            raise ValueError(f"state must be [B,S,D], got {tuple(state.shape)}")
        if action.ndim != 3:
            raise ValueError(f"action must be [B,T,D], got {tuple(action.shape)}")
        if state.shape[0] != action.shape[0]:
            raise ValueError(f"state/action batch mismatch: {state.shape[0]} vs {action.shape[0]}")

        batch_size = state.shape[0]
        state_tokens = self.state_encoder(state)
        action_tokens = self.action_encoder(action)
        tokens = torch.cat([state_tokens, action_tokens], dim=1)

        timesteps = torch.cat([state_timestep, action_timestep], dim=1)
        temb, t_mod = self._embed_token_timesteps(
            timesteps,
            seq_len=tokens.shape[1],
            batch_size=batch_size,
            dtype=tokens.dtype,
        )
        context = self.text_embedder(encoder_hidden_states).to(dtype=tokens.dtype)
        rotary_emb = self.action_rope(tokens)
        num_state_tokens = state_tokens.shape[1]
        num_action_tokens = action_tokens.shape[1]

        return {
            "tokens": tokens,
            "state_tokens": state_tokens,
            "action_tokens": action_tokens,
            "rotary_emb": rotary_emb,
            "t_mod": t_mod,
            "temb": temb,
            "context": context,
            "meta": {
                "num_state_tokens": num_state_tokens,
                "num_action_tokens": num_action_tokens,
            },
        }

    def post_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        return self.action_decoder(action_tokens)


__all__ = ["ActionStateBlock", "ActionStateDiT"]
