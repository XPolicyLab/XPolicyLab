"""MoPA manipulation policy with one query bank and a flow action head."""

from __future__ import annotations

import math

import torch
from torch import nn

from .action_model import ArmQueryActionHead
from .backbone import QwenArmQueryBackbone


class ArmQueryPolicy(nn.Module):
    """End-to-end policy with a conditioning backbone and an arm-query action head.

    The conditioning backbone must expose ``hidden_size`` and
    ``encode(images, instructions, query_tokens) -> [B,Q,H]``. The returned
    tensor must remain connected to query_tokens for training the query bank.
    """

    def __init__(self, config: dict, backbone=None):
        super().__init__()
        self.config = dict(config)
        self.backbone = backbone if backbone is not None else QwenArmQueryBackbone(config)
        self.num_query_tokens = int(config.get("num_query_tokens", 8))
        self.repeated_diffusion_steps = int(config.get("repeated_diffusion_steps", 8))
        query_init_std = float(config.get("query_init_std", 0.02))
        if self.num_query_tokens <= 0 or self.repeated_diffusion_steps <= 0:
            raise ValueError("Query count and diffusion repeat count must be positive")
        if not math.isfinite(query_init_std) or query_init_std <= 0:
            raise ValueError("query_init_std must be finite and positive")
        self.arm_query_tokens = nn.Parameter(torch.empty(self.num_query_tokens, self.backbone.hidden_size))
        nn.init.normal_(self.arm_query_tokens, std=query_init_std)
        self.action_model = ArmQueryActionHead(config, self.backbone.hidden_size)
        if bool(config.get("freeze_backbone", False)):
            self.backbone.requires_grad_(False)

    def encode(self, images, instructions) -> torch.Tensor:
        queries = self.backbone.encode(images, instructions, self.arm_query_tokens)
        expected = (len(images), self.num_query_tokens, self.backbone.hidden_size)
        if tuple(queries.shape) != expected:
            raise ValueError(f"Backbone query shape must be {expected}, got {tuple(queries.shape)}")
        parameter = next(self.action_model.parameters())
        return queries.to(device=parameter.device, dtype=parameter.dtype)

    @staticmethod
    def _tensor(value, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def forward(self, images, instructions, state, actions) -> dict[str, torch.Tensor]:
        query = self.encode(images, instructions)
        state = self._tensor(state, query)
        actions = self._tensor(actions, query)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        # Validate before repeating so malformed ranks cannot broadcast silently.
        self.action_model._validate(query, state, actions)
        repeats = self.repeated_diffusion_steps
        loss = self.action_model(query.repeat(repeats, 1, 1),
                                 actions.repeat(repeats, 1, 1), state.repeat(repeats, 1, 1))
        return {"action_loss": loss, "action_dit_loss_manipulation": loss.detach()}

    @torch.no_grad()
    def predict_action(self, images, instructions, state) -> torch.Tensor:
        query = self.encode(images, instructions)
        return self.action_model.predict_action(query, self._tensor(state, query))
