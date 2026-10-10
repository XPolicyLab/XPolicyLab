from __future__ import annotations

import torch
from torch import nn

from .base_layers import FP32LayerNorm
from .config import RobotwinTactileAEV41Config
from .layers import PlainRelationBlock
from .manifeel import ManiFeelForceDecoderV41, ManiFeelTactileAEV41Config
from .surface_encoder import FullResolutionForceEncoderV41
from .layout import SLOT_NAMES, pooling_regions


class SharedMaskedPooling(nn.Module):
    """One learned query and the same K/V/projection for every pooling range."""

    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.empty(128))
        nn.init.normal_(self.query, std=0.02)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.project = nn.Linear(128, 48)
        self.norm = FP32LayerNorm(48)

    def forward(self, features, ranges, support):
        x = features.flatten(2).transpose(1, 2)
        valid = (ranges & support).flatten(2)
        score = (self.key(x) * self.query).sum(-1) / (128 ** 0.5)
        score = score[:, None].expand(-1, 4, -1).masked_fill(~valid, -10000)
        weights = score.float().softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = weights.to(x.dtype) @ self.value(x)
        token_valid = valid.any(-1)
        return self.norm(self.project(pooled)) * token_valid[..., None], token_valid


class UnifiedTactileAE(nn.Module):
    """Packed physical surfaces -> [B,T,34,48] -> the same packed force fields.

    Packed input uses N observed surfaces, not B*32 padded CNN evaluations.
    No dataset identity or normalization parameters enter the learned model.
    """

    def __init__(self, *, variational=False):
        super().__init__()
        self.variational = bool(variational)
        if self.variational:
            self.posterior_mean = nn.Linear(48, 48)
            self.posterior_logvar = nn.Linear(48, 48)
            nn.init.eye_(self.posterior_mean.weight)
            nn.init.zeros_(self.posterior_mean.bias)
            nn.init.zeros_(self.posterior_logvar.weight)
            nn.init.constant_(self.posterior_logvar.bias, -6.0)
        self.surface_encoder = FullResolutionForceEncoderV41(RobotwinTactileAEV41Config())
        self.pool = SharedMaskedPooling()
        self.slot_embedding = nn.Embedding(34, 48)
        # Distinguish a quadrant token from a whole-pad token without a domain ID.
        self.range_embedding = nn.Embedding(2, 48)
        self.relation = nn.ModuleList([PlainRelationBlock(48, 4, 192, 0.1) for _ in range(2)])
        self.norm = FP32LayerNorm(48)
        self.decoder = ManiFeelForceDecoderV41(ManiFeelTactileAEV41Config())

    def forward(self, force, support, owner, slots, batch_size):
        # force [N,T,3,10,14], static support [N,1,10,14], owner [N].
        n, time = force.shape[:2]
        if force.shape[2:] != (3, 10, 14) or support.shape != (n, 1, 10, 14):
            raise ValueError("expected packed 10x14 surfaces and static support masks")
        ranges = pooling_regions(slots)
        expanded_support = support[:, None].expand(-1, time, -1, -1, -1).flatten(0, 1)
        expanded_ranges = ranges[:, None].expand(-1, time, -1, -1, -1).flatten(0, 1)
        features = self.surface_encoder(force.flatten(0, 1), expanded_support)
        local, valid = self.pool(features, expanded_ranges, expanded_support)
        with torch.no_grad():
            baseline, _ = self.pool(
                self.surface_encoder(torch.zeros_like(force[:, 0]), support), ranges, support
            )
        local = local.reshape(n, time, 4, 48)
        valid = valid.reshape(n, time, 4)
        # Only declared local tokens are scattered. Missing slots never overwrite
        # an observed slot, and gradients flow through the scatter operation.
        declared = slots.ge(0)
        surface_id, local_id = declared.nonzero(as_tuple=True)
        flat_destination = owner[surface_id] * 34 + slots[surface_id, local_id]
        token_values = local[surface_id, :, local_id].transpose(0, 1)
        token_valid = valid[surface_id, :, local_id].transpose(0, 1)
        packed = local.new_zeros(time, batch_size * 34, 48)
        packed = packed.index_copy(1, flat_destination, token_values)
        observed = torch.zeros(time, batch_size * 34, dtype=torch.bool, device=force.device)
        observed = observed.index_copy(1, flat_destination, token_valid)
        kind = slots[:, 1].ge(0).long()
        range_values = self.range_embedding(kind[surface_id])
        position = self.slot_embedding(slots[surface_id, local_id]) + range_values
        identity = local.new_zeros(batch_size * 34, 48).index_copy(0, flat_destination, position.to(local.dtype))
        tokens = (packed + identity[None]) * observed[..., None]
        tokens = tokens.reshape(time, batch_size, 34, 48).transpose(0, 1).flatten(0, 1)
        mask = observed.reshape(time, batch_size, 34).transpose(0, 1).flatten(0, 1)
        # Empty samples (e.g. masked sensors) must not produce all-masked softmax NaNs.
        safe = mask.clone()
        safe[:, 0] |= ~mask.any(-1)
        for layer in self.relation:
            tokens = layer(tokens, safe) * mask[..., None]
        tokens = self.norm(tokens) * mask[..., None]
        tokens = tokens.reshape(batch_size, time, 34, 48)
        mask = mask.reshape(batch_size, time, 34)
        posterior = {}
        if self.variational:
            mu = self.posterior_mean(tokens).float() * mask[..., None]
            logvar = self.posterior_logvar(tokens).float().clamp(-12, 8) * mask[..., None]
            posterior = {'posterior_mean': mu, 'posterior_logvar': logvar}
            tokens = (mu + torch.randn_like(mu) * torch.exp(.5 * logvar)) if self.training else mu
            tokens = tokens * mask[..., None]
        selected = tokens[owner[:, None, None], torch.arange(time, device=force.device)[None, :, None],
                          slots.clamp_min(0)[:, None]]
        selected_valid = mask[owner[:, None, None], torch.arange(time, device=force.device)[None, :, None],
                              slots.clamp_min(0)[:, None]] & declared[:, None]
        decoded = self.decoder(selected.flatten(0, 1), selected_valid.flatten(0, 1),
                               expanded_support[:, None])
        return {
            **posterior,
            **{k: v[:, 0].reshape(n, time, *v.shape[2:]) for k, v in decoded.items()},
            "frame_tokens": tokens, "frame_token_valid": mask,
            "local_tokens": local, "local_valid": valid,
            "zero_local_tokens": baseline.detach(), "pooling_regions": ranges,
        }

    def schema(self):
        return {"architecture": "unified_tactile_vae" if self.variational else "unified_tactile_ae", "latent": "[B,T,34,48]",
                "slots": list(SLOT_NAMES), "shared_surface_encoder": True,
                "shared_pooling_query": True, "decoder_count": 1,
                "time_compression": False}
