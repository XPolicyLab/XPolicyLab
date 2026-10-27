"""Wan2.1 image-to-video experts of PrismWAM*.

Each expert reads 36 input channels, as in Wan2.1 I2V: 16 latent channels to
denoise, 4 mask channels and 16 channels holding the observed latents in the
leading frames. It also cross-attends to CLIP features of the current RGB frame.
The RGB and depth experts share one block stack; patch embedding, embeddings and
head belong to each expert. The MoT applies the blocks layer by layer, so the
experts provide only their input (``pre_dit``) and output (``post_dit``) stages.

Adapted from Wan2.1 (https://github.com/Wan-Video/Wan2.1), Apache-2.0.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..wan22.wan_video_dit import (
    CrossAttention,
    GateModule,
    RMSNorm,
    SelfAttention,
    flash_attention,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

# CLIP ViT-H/14 image features: a class token and 16 x 16 patches.
CLIP_TOKENS = 257
CLIP_DIM = 1280
MASK_CHANNELS = 4


class FP32LayerNorm(nn.LayerNorm):
    """Layer norm evaluated in float32, returned in the input dtype."""

    def forward(self, x):
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps
        ).type_as(x)


class ImageTextCrossAttention(CrossAttention):
    """Cross-attention to the text and, separately, to the CLIP image tokens.

    The context is an ``(image, text)`` pair; ``ctx_mask`` covers the text only.
    """

    def __init__(
        self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6
    ):
        super().__init__(hidden_dim, attn_head_dim, num_heads, eps)
        self.k_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.norm_k_img = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, ctx, ctx_mask=None):
        image, text = ctx
        q = self.norm_q(self.q(x))
        out = flash_attention(
            q, self.norm_k(self.k(text)), self.v(text), self.num_heads, ctx_mask
        )
        out = out + flash_attention(
            q, self.norm_k_img(self.k_img(image)), self.v_img(image), self.num_heads
        )
        return self.o(out)


class I2VBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = ImageTextCrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps
        )
        self.norm1 = FP32LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = FP32LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = FP32LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Sequence[int], eps: float = 1e-6
    ):
        super().__init__()
        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t):
        modulation = self.modulation.to(dtype=t.dtype, device=t.device)
        shift, scale = (modulation + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class ImageProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class I2VExpert(nn.Module):
    """One Wan2.1 I2V expert; ``num_condition_frames`` latents are observed.

    The RGB expert observes two history latents and denoises nothing at
    inference: its two input positions hold pure noise at t=1000. The depth
    expert observes the current depth latent and denoises three latents.
    Self-attention within an expert is bidirectional.
    """

    def __init__(
        self,
        *,
        num_condition_frames: int,
        hidden_dim: int = 5120,
        ffn_dim: int = 13824,
        num_heads: int = 40,
        attn_head_dim: int = 128,
        num_layers: int = 40,
        in_dim: int = 36,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        patch_size: Sequence[int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        if in_dim != 2 * out_dim + MASK_CHANNELS:
            raise ValueError("I2V input must hold latents, mask and observed latents.")
        self.num_condition_frames = int(num_condition_frames)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                I2VBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(hidden_dim, out_dim, self.patch_size, eps)
        self.img_emb = ImageProjection(CLIP_DIM, hidden_dim)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        self.use_gradient_checkpointing = False

    def pre_dit(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        clip_fea: torch.Tensor,
        condition_latents: torch.Tensor,
    ) -> dict[str, Any]:
        if x.ndim != 5 or x.shape[1] != self.out_dim:
            raise ValueError(
                f"`x` must be [B,{self.out_dim},T,H,W], got {tuple(x.shape)}."
            )
        b, _, t, h, w = x.shape
        observed = self.num_condition_frames
        if tuple(condition_latents.shape) != (b, self.out_dim, observed, h, w):
            raise ValueError(
                f"`condition_latents` must be [B,{self.out_dim},{observed},H,W], got {tuple(condition_latents.shape)}."
            )
        if timestep.ndim != 1 or timestep.shape[0] not in (1, b):
            raise ValueError(
                f"`timestep` must be [B] or [1], got {tuple(timestep.shape)}."
            )
        if tuple(clip_fea.shape) != (b, CLIP_TOKENS, CLIP_DIM):
            raise ValueError(
                f"`clip_fea` must be [B,{CLIP_TOKENS},{CLIP_DIM}], got {tuple(clip_fea.shape)}."
            )
        mask = torch.zeros((b, MASK_CHANNELS, t, h, w), dtype=x.dtype, device=x.device)
        mask[:, :, :observed] = 1
        known = torch.zeros((b, self.out_dim, t, h, w), dtype=x.dtype, device=x.device)
        known[:, :, :observed] = condition_latents.to(device=x.device, dtype=x.dtype)
        x = self.patch_embedding(torch.cat([x, mask, known], dim=1))
        f, h, w = x.shape[2:]
        seq_len = f * h * w

        time_weight = self.time_embedding[0].weight
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(
                device=time_weight.device, dtype=time_weight.dtype
            )
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        text = self.text_embedding(context)
        image = self.img_emb(clip_fea.to(device=text.device, dtype=text.dtype))
        # Image tokens are always attended to; only the text is masked.
        text_mask = (
            context_mask.to(dtype=torch.bool).unsqueeze(1).expand(-1, seq_len, -1)
        )

        tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(seq_len, 1, -1)
            .to(tokens.device)
        )
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": (image, text),
            "context_mask": text_mask,
            "meta": {"grid_size": (f, h, w), "tokens_per_frame": h * w},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(tokens, pre_state["t"])
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
