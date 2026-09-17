"""SIPAI memory policy forward pass, with no training framework dependencies."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .gemma import AdaptiveGemmaRMSNorm, GemmaConfig, GemmaDecoderLayer, GemmaRMSNorm
from .siglip import SigLIPVisionEncoder


class Prefix(nn.Module):
    def __init__(self, history_pool_size):
        super().__init__()
        config = GemmaConfig(2048, 18, 16384, 8, 1, 256)
        self.vision_encoder = SigLIPVisionEncoder()
        self.token_embedding = nn.Embedding(257152, config.width)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(config.depth)])
        self.final_norms = nn.ModuleList([GemmaRMSNorm(config.width)])
        self.history_pool_size = history_pool_size

    def forward(self, images, history, history_mask, token_ids, token_mask):
        parts = [self.vision_encoder(image) for image in images]
        masks = [torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) for x in parts]
        batch, frames = history.shape[:2]
        indices = torch.nonzero(history_mask.reshape(-1), as_tuple=False).squeeze(-1)
        pool = self.history_pool_size
        if indices.numel():
            encoded = self.vision_encoder(history.flatten(0, 1).index_select(0, indices))
            spatial = math.isqrt(encoded.shape[1])
            grid = encoded.reshape(-1, spatial, spatial, 2048).permute(0, 3, 1, 2)
            encoded = F.adaptive_avg_pool2d(grid, (pool, pool))
            encoded = encoded.permute(0, 2, 3, 1).reshape(-1, pool * pool, 2048)
            scattered = encoded.new_zeros(batch * frames, pool * pool, 2048)
            scattered = scattered.index_copy(0, indices, encoded)
        else:
            scattered = parts[0].new_zeros(batch * frames, pool * pool, 2048)
        parts.insert(0, scattered.reshape(batch, frames * pool * pool, 2048))
        masks.insert(0, history_mask[:, :, None].expand(-1, -1, pool * pool).reshape(batch, -1))
        parts.append(self.token_embedding(token_ids) * math.sqrt(2048))
        masks.append(token_mask)
        hidden = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        attention = mask[:, None, :] & mask[:, :, None]
        positions = torch.cumsum(mask.int(), dim=1) - 1
        caches = []
        for layer in self.layers:
            hidden, cache = layer(hidden, positions, attention)
            caches.append(cache)
        return mask, caches


class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        config = GemmaConfig(1024, 18, 4096, 8, 1, 256, adaptive_norm=True)
        self.action_in_proj = nn.Linear(32, 1024)
        self.time_mlp_in = nn.Linear(1024, 1024)
        self.time_mlp_out = nn.Linear(1024, 1024)
        self.action_out_proj = nn.Linear(1024, 32)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(config.depth)])
        self.final_norms = nn.ModuleList([AdaptiveGemmaRMSNorm(1024)])

    def forward(self, mask, caches, noise, steps):
        batch, horizon = noise.shape[:2]
        device = noise.device
        attention = torch.cat(
            [
                mask[:, None, :].expand(-1, horizon, -1),
                torch.ones(batch, horizon, horizon, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )
        positions = mask.sum(-1)[:, None] + torch.arange(horizon, device=device)[None, :]
        fractions = torch.linspace(0.0, 1.0, 512, device=device, dtype=torch.float32)
        periods = 4e-3 * (4.0 / 4e-3) ** fractions
        dt, time_value = -1.0 / steps, 1.0
        actions = noise
        while time_value >= -dt / 2:
            time = torch.full((batch,), time_value, dtype=torch.float32, device=device)
            radians = torch.einsum("i,j->ij", time, 1.0 / periods * 2 * torch.pi)
            condition = torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)
            condition = F.silu(self.time_mlp_in(condition))
            condition = F.silu(self.time_mlp_out(condition))
            hidden = self.action_in_proj(actions)
            for layer, cache in zip(self.layers, caches, strict=True):
                hidden, _ = layer(hidden, positions, attention, cache, condition)
            hidden, _ = self.final_norms[0](hidden, condition)
            actions = actions + dt * self.action_out_proj(hidden).float()
            time_value += dt
        return actions


class SIPAINetwork(nn.Module):
    def __init__(self, history_pool_size=4):
        super().__init__()
        self.backbone = Prefix(history_pool_size)
        self.action_head = ActionHead()

    @torch.inference_mode()
    def forward(self, prepared, noise=None, steps=10):
        mask, caches = self.backbone(**prepared)
        if noise is None:
            noise = torch.randn(mask.shape[0], 50, 32, dtype=torch.float32, device=mask.device)
        return self.action_head(mask, caches, noise, steps)
