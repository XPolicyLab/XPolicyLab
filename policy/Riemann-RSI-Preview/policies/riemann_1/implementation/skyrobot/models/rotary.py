# WanRotaryPosEmbed contains code from Robbyant/Lingbot-VA (Apache-2.0).
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# See ../SOURCE_NOTICE.md for exact scope, source revision and other helpers.
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64) / half)
    sinusoid = position.unsqueeze(-1) * freqs
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)

@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis
