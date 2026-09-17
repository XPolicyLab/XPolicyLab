"""MoPA's arm-query flow action head.

Arm-query and manipulation streams use independent attention/FFN experts
and attend to one another without a causal restriction. The model has no
mobility tokens, parameters, losses or sampler state.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """GR00T action/time MLP, with sin-then-cos time features."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")
        half = self.hidden_size // 2
        exponent = -torch.arange(half, device=actions.device, dtype=torch.float32)
        exponent = exponent * (math.log(10000.0) / half)
        phase = timesteps[:, None, None].float() * exponent.exp()[None, None, :]
        time_features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        action_features = self.layer1(actions)
        time_features = time_features.expand(-1, horizon, -1).to(action_features.dtype)
        hidden = self.layer2(torch.cat((action_features, time_features), dim=-1))
        return self.layer3(hidden * hidden.sigmoid())


class TimestepEncoder(nn.Module):
    """diffusers Timesteps(256, flip=True, shift=1) + TimestepEmbedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        exponent = -math.log(10000.0) * torch.arange(
            128, device=timesteps.device, dtype=torch.float32
        ) / 127.0
        phase = timesteps[:, None].float() * exponent.exp()[None, :]
        features = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.linear_2(F.silu(self.linear_1(features.to(self.linear_1.weight.dtype))))


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(time)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class QueryProjector(nn.Module):
    def __init__(self, qwen_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(qwen_hidden_dim)
        self.input_projection = nn.Linear(qwen_hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.input_projection(self.input_norm(x)))


class ExpertAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_bias: bool, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=attention_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)


class ExpertFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float, dropout: float):
        super().__init__()
        intermediate = round(hidden_dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(intermediate, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArmQueryJointAttentionBlock(nn.Module):
    """Joint attention between the arm-query and manipulation experts."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float,
                 attention_bias: bool = True, expansion: float = 4.0):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.arm_query_attention_norm = nn.LayerNorm(hidden_dim)
        self.manipulation_attention_norm = AdaLayerNorm(hidden_dim)
        self.arm_query_attention = ExpertAttention(hidden_dim, attention_bias, dropout)
        self.manipulation_attention = ExpertAttention(hidden_dim, attention_bias, dropout)
        self.arm_query_ff_norm = nn.LayerNorm(hidden_dim)
        self.manipulation_ff_norm = AdaLayerNorm(hidden_dim)
        self.arm_query_ff = ExpertFeedForward(hidden_dim, expansion, dropout)
        self.manipulation_ff = ExpertFeedForward(hidden_dim, expansion, dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, action: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = (self.arm_query_attention_norm(query),
                   self.manipulation_attention_norm(action, time))
        experts = (self.arm_query_attention, self.manipulation_attention)
        projections = []
        for projection in ("q_proj", "k_proj", "v_proj"):
            joined = torch.cat(tuple(getattr(expert, projection)(stream)
                                     for stream, expert in zip(streams, experts)), dim=1)
            projections.append(self._split_heads(joined))
        # Arm-query and manipulation tokens are mutually visible.
        attended = F.scaled_dot_product_attention(
            *projections, dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(query.shape[0], -1, query.shape[-1])
        query_output, action_output = attended.split((query.shape[1], action.shape[1]), dim=1)
        query = query + self.arm_query_attention.dropout(
            self.arm_query_attention.out_proj(query_output))
        action = action + self.manipulation_attention.dropout(
            self.manipulation_attention.out_proj(action_output))
        query = query + self.arm_query_ff(self.arm_query_ff_norm(query))
        action = action + self.manipulation_ff(self.manipulation_ff_norm(action, time))
        return query, action


class ArmQueryActionHead(nn.Module):
    """Uniform conditional flow matching over the complete Dojo joint vector."""

    def __init__(self, config: dict, qwen_hidden_dim: int):
        super().__init__()
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config["state_dim"])
        self.action_horizon = int(config.get("action_horizon", 4))
        self.num_query_tokens = int(config.get("num_query_tokens", 8))
        self.num_timestep_buckets = int(config.get("num_timestep_buckets", 1000))
        self.num_inference_timesteps = int(config.get("num_inference_timesteps", 4))
        self.qwen_hidden_dim = int(qwen_hidden_dim)
        # Native DiT-B uses 12*64 attention width; hidden_size=1024 is MLP width.
        width = int(config.get("input_embedding_dim", 768))
        mlp_width = int(config.get("hidden_size", 1024))
        heads = int(config.get("num_attention_heads", 12))
        layers = int(config.get("num_layers", 16))
        dropout = float(config.get("dropout", 0.2))
        expansion = float(config.get("ff_expansion_factor", 4.0))
        max_seq_len = int(config.get("max_seq_len", 1024))
        dimensions = (self.action_dim, self.state_dim, self.action_horizon,
                      self.num_query_tokens, self.num_timestep_buckets,
                      self.num_inference_timesteps, self.qwen_hidden_dim,
                      width, mlp_width, heads, layers)
        if any(value <= 0 for value in dimensions):
            raise ValueError("All model dimensions, head/layer counts and sampling counts must be positive")
        if width % heads or width % 2:
            raise ValueError("input_embedding_dim must be even and divisible by num_attention_heads")
        if not 0 <= dropout < 1 or expansion <= 0 or round(width * expansion) <= 0:
            raise ValueError("Invalid dropout or feed-forward expansion")
        if max_seq_len < self.action_horizon:
            raise ValueError("max_seq_len must cover action_horizon")
        if config.get("time_sampling", "uniform") != "uniform":
            raise ValueError("This Query-DMoT variant requires uniform flow-time sampling")
        if int(config.get("base_num_query_tokens", 0)) != 0:
            raise ValueError("MoPA Dojo uses no base query tokens")
        if int(config.get("in_context_condition_dim", 0)) != 0:
            raise ValueError("Fetch scene context is unavailable in Dojo; context dimension must be zero")

        self.arm_query_projector = QueryProjector(qwen_hidden_dim, width)
        self.manipulation_action_encoder = ActionEncoder(self.action_dim, width)
        self.manipulation_state_encoder = MLP(self.state_dim, mlp_width, width)
        self.manipulation_state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.add_pos_embed = bool(config.get("add_pos_embed", True))
        if self.add_pos_embed:
            self.manipulation_position_embedding = nn.Embedding(max_seq_len, width)
            nn.init.normal_(self.manipulation_position_embedding.weight, std=0.02)
        self.manipulation_timestep_encoder = TimestepEncoder(width)
        self.blocks = nn.ModuleList([
            ArmQueryJointAttentionBlock(width, heads, dropout,
                                        bool(config.get("attention_bias", True)), expansion)
            for _ in range(layers)
        ])
        self.manipulation_output_norm = AdaLayerNorm(width)
        self.manipulation_action_decoder = MLP(width, mlp_width, self.action_dim)

    def _validate(self, query: torch.Tensor, state: torch.Tensor,
                  actions: torch.Tensor | None = None) -> torch.Tensor:
        if query.ndim != 3 or tuple(query.shape[1:]) != (self.num_query_tokens, self.qwen_hidden_dim):
            raise ValueError("query must have shape [B, num_query_tokens, qwen_hidden_dim]")
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if tuple(state.shape) != (query.shape[0], 1, self.state_dim):
            raise ValueError(f"state must have shape [B, {self.state_dim}] or [B, 1, {self.state_dim}]")
        if actions is not None and tuple(actions.shape) != (
            query.shape[0], self.action_horizon, self.action_dim
        ):
            raise ValueError(f"actions must have shape [B, {self.action_horizon}, {self.action_dim}]")
        return state

    def _condition(self, query: torch.Tensor, state: torch.Tensor):
        return (self.arm_query_projector(query), self.manipulation_state_encoder(state)
                + self.manipulation_state_type_embedding)

    def _velocity(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                  query: torch.Tensor, state_token: torch.Tensor) -> torch.Tensor:
        action = self.manipulation_action_encoder(noisy, timesteps)
        if self.add_pos_embed:
            positions = torch.arange(self.action_horizon, device=noisy.device)
            action = action + self.manipulation_position_embedding(positions).unsqueeze(0)
        action = torch.cat((state_token, action), dim=1)
        time = self.manipulation_timestep_encoder(timesteps)
        for block in self.blocks:
            query, action = block(query, action, time)
        action = self.manipulation_output_norm(action, time)
        return self.manipulation_action_decoder(action[:, -self.action_horizon:])

    def forward(self, query: torch.Tensor, actions: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        state = self._validate(query, state, actions)
        noise = torch.randn_like(actions)
        flow_time = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        interpolation = flow_time[:, None, None]
        noisy = interpolation * actions + (1 - interpolation) * noise
        timesteps = (flow_time * self.num_timestep_buckets).long().clamp_(
            max=self.num_timestep_buckets - 1)
        query_tokens, state_token = self._condition(query, state)
        prediction = self._velocity(noisy, timesteps, query_tokens, state_token)
        return F.mse_loss(prediction, actions - noise)

    @torch.no_grad()
    def predict_action(self, query: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state = self._validate(query, state)
        actions = torch.randn(query.shape[0], self.action_horizon, self.action_dim,
                              device=query.device, dtype=query.dtype)
        query_tokens, state_token = self._condition(query, state)
        for step in range(self.num_inference_timesteps):
            timestep = min(int(step / self.num_inference_timesteps * self.num_timestep_buckets),
                           self.num_timestep_buckets - 1)
            timesteps = torch.full((query.shape[0],), timestep, device=query.device, dtype=torch.long)
            velocity = self._velocity(actions, timesteps, query_tokens, state_token)
            actions = actions + velocity / self.num_inference_timesteps
        return actions
