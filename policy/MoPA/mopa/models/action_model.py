"""Source Query-DMoT action head used by the MoPA adapter.

The head uses two learned Qwen query
banks, independent mobility/manipulation experts, a structured four-stream
attention mask, shared flow time with independent noises, and synchronous Euler
sampling.  The action vector is kept in dataset order; only the configured
slices determine which channels belong to the base and arm experts.
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

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, actions, timesteps):
        b, h, _ = actions.shape
        half = self.hidden_size // 2
        if 2 * half != self.hidden_size:
            raise ValueError("action encoder hidden_size must be even")
        exponent = -torch.arange(half, device=actions.device, dtype=torch.float32)
        exponent = exponent * (math.log(10000.0) / max(half, 1))
        phase = timesteps[:, None, None].float() * exponent.exp()[None, None, :]
        time = torch.cat((phase.sin(), phase.cos()), dim=-1).expand(-1, h, -1)
        hidden = self.layer2(torch.cat((self.layer1(actions), time.to(actions.dtype)), dim=-1))
        return self.layer3(hidden * hidden.sigmoid())


class TimestepEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, timesteps):
        exponent = -math.log(10000.0) * torch.arange(128, device=timesteps.device, dtype=torch.float32) / 127.0
        phase = timesteps[:, None].float() * exponent.exp()[None, :]
        features = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.linear_2(F.silu(self.linear_1(features.to(self.linear_1.weight.dtype))))


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x, time):
        scale, shift = self.linear(F.silu(time)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class QueryProjector(nn.Module):
    def __init__(self, qwen_hidden_dim: int, hidden_dim: int, count: int):
        super().__init__()
        self.qwen_hidden_dim, self.count = qwen_hidden_dim, count
        self.input_norm = nn.LayerNorm(qwen_hidden_dim)
        self.input_projection = nn.Linear(qwen_hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.count, self.qwen_hidden_dim):
            raise ValueError(f"query bank must have shape [B,{self.count},{self.qwen_hidden_dim}], got {tuple(x.shape)}")
        return self.output_norm(self.input_projection(self.input_norm(x)))


class ExpertAttention(nn.Module):
    def __init__(self, width: int, bias: bool, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=bias)
        self.k_proj = nn.Linear(width, width, bias=bias)
        self.v_proj = nn.Linear(width, width, bias=bias)
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)


class ExpertFFN(nn.Module):
    def __init__(self, width: int, expansion: float, dropout: float):
        super().__init__()
        inner = round(width * expansion)
        self.net = nn.Sequential(nn.Linear(width, inner), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(inner, width), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def build_query_dmot_action_attention_mask(arm_mask, base_mask, mobility_len, manipulation_len):
    if arm_mask.ndim != 2 or base_mask.ndim != 2 or arm_mask.shape[0] != base_mask.shape[0]:
        raise ValueError("arm/base query masks must have matching shape [B,Q]")
    b, aq, bq = arm_mask.shape[0], arm_mask.shape[1], base_mask.shape[1]
    if aq <= 0 or bq <= 0 or mobility_len <= 0 or manipulation_len <= 0:
        raise ValueError("all Query-DMoT streams must be nonempty")
    device = arm_mask.device
    ids = torch.cat((torch.zeros(aq, device=device, dtype=torch.long),
                     torch.ones(bq, device=device, dtype=torch.long),
                     torch.full((mobility_len,), 2, device=device, dtype=torch.long),
                     torch.full((manipulation_len,), 3, device=device, dtype=torch.long)))
    # Source connectivity. True means query/key visibility.
    connectivity = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0],
                                 [0, 1, 1, 1], [1, 0, 1, 1]], device=device, dtype=torch.bool)
    structural = connectivity[ids[:, None], ids[None, :]]
    valid = torch.cat((arm_mask.bool(), base_mask.bool(),
                       torch.ones(b, mobility_len + manipulation_len, device=device, dtype=torch.bool)), dim=1)
    allowed = structural[None, None] & valid[:, None, :, None] & valid[:, None, None, :]
    if not allowed[:, 0][valid].any(dim=-1).all():
        raise RuntimeError("every valid Query-DMoT token must attend to a key")
    return allowed


class QueryDMoTBlock(nn.Module):
    def __init__(self, width, heads, dropout, bias, expansion):
        super().__init__()
        if width % heads:
            raise ValueError("action width must be divisible by attention heads")
        self.heads, self.head_dim, self.dropout = heads, width // heads, dropout
        self.arm_q_norm = nn.LayerNorm(width)
        self.base_q_norm = nn.LayerNorm(width)
        self.mob_norm = AdaLayerNorm(width)
        self.manip_norm = AdaLayerNorm(width)
        self.arm_q_attn = ExpertAttention(width, bias, dropout)
        self.base_q_attn = ExpertAttention(width, bias, dropout)
        self.mob_attn = ExpertAttention(width, bias, dropout)
        self.manip_attn = ExpertAttention(width, bias, dropout)
        self.arm_q_ff_norm = nn.LayerNorm(width)
        self.base_q_ff_norm = nn.LayerNorm(width)
        self.mob_ff_norm = AdaLayerNorm(width)
        self.manip_ff_norm = AdaLayerNorm(width)
        self.arm_q_ff = ExpertFFN(width, expansion, dropout)
        self.base_q_ff = ExpertFFN(width, expansion, dropout)
        self.mob_ff = ExpertFFN(width, expansion, dropout)
        self.manip_ff = ExpertFFN(width, expansion, dropout)

    def _split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    def forward(self, arm_q, base_q, mob, manip, mob_time, manip_time, arm_mask, base_mask):
        streams = (self.arm_q_norm(arm_q), self.base_q_norm(base_q), self.mob_norm(mob, mob_time), self.manip_norm(manip, manip_time))
        experts = (self.arm_q_attn, self.base_q_attn, self.mob_attn, self.manip_attn)
        q = torch.cat([e.q_proj(x) for e, x in zip(experts, streams)], dim=1)
        k = torch.cat([e.k_proj(x) for e, x in zip(experts, streams)], dim=1)
        v = torch.cat([e.v_proj(x) for e, x in zip(experts, streams)], dim=1)
        mask = build_query_dmot_action_attention_mask(arm_mask, base_mask, mob.shape[1], manip.shape[1])
        y = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v),
                                            attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        y = self._merge(y)
        lengths = [x.shape[1] for x in streams]
        yo = torch.split(y, lengths, dim=1)
        arm_q = arm_q + self.arm_q_attn.dropout(self.arm_q_attn.out_proj(yo[0]))
        base_q = base_q + self.base_q_attn.dropout(self.base_q_attn.out_proj(yo[1]))
        mob = mob + self.mob_attn.dropout(self.mob_attn.out_proj(yo[2]))
        manip = manip + self.manip_attn.dropout(self.manip_attn.out_proj(yo[3]))
        arm_q = arm_q + self.arm_q_ff(self.arm_q_ff_norm(arm_q))
        base_q = base_q + self.base_q_ff(self.base_q_ff_norm(base_q))
        mob = mob + self.mob_ff(self.mob_ff_norm(mob, mob_time))
        manip = manip + self.manip_ff(self.manip_ff_norm(manip, manip_time))
        return arm_q, base_q, mob, manip


class DualQueryMoT(nn.Module):
    def __init__(self, width, layers, heads, dropout, bias, expansion):
        super().__init__()
        self.mob_time = TimestepEncoder(width)
        self.manip_time = TimestepEncoder(width)
        self.blocks = nn.ModuleList([QueryDMoTBlock(width, heads, dropout, bias, expansion) for _ in range(layers)])
        self.mob_out = AdaLayerNorm(width)
        self.manip_out = AdaLayerNorm(width)

    def forward(self, arm_q, base_q, mob, manip, timesteps):
        mt, at = self.mob_time(timesteps), self.manip_time(timesteps)
        arm_mask = torch.ones(arm_q.shape[:2], device=arm_q.device, dtype=torch.bool)
        base_mask = torch.ones(base_q.shape[:2], device=base_q.device, dtype=torch.bool)
        for block in self.blocks:
            arm_q, base_q, mob, manip = block(arm_q, base_q, mob, manip, mt, at, arm_mask, base_mask)
        return self.mob_out(mob, mt), self.manip_out(manip, at)


class ArmQueryActionHead(nn.Module):
    """Dual-query flow head; name retained for checkpoint/API compatibility."""
    def __init__(self, config: dict, qwen_hidden_dim: int):
        super().__init__()
        self.config = dict(config)
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config["state_dim"])
        self.action_horizon = int(config.get("action_horizon", 32))
        self.arm_num_query_tokens = int(config.get("arm_num_query_tokens", 8))
        self.base_num_query_tokens = int(config.get("base_num_query_tokens", 8))
        self.num_query_tokens = self.arm_num_query_tokens + self.base_num_query_tokens
        if int(config.get("num_query_tokens", self.num_query_tokens)) != self.num_query_tokens:
            raise ValueError("num_query_tokens does not match the arm/base query counts")
        self.num_timestep_buckets = int(config.get("num_timestep_buckets", 1000))
        self.num_inference_timesteps = int(config.get("num_inference_timesteps", 4))
        self.manipulation_action_range = tuple(config.get("manipulation_action_range", [0, self.action_dim // 2]))
        self.mobility_action_range = tuple(config.get("mobility_action_range", [self.action_dim // 2, self.action_dim]))
        if self.manipulation_action_range[0] != 0 or self.mobility_action_range[1] != self.action_dim or self.manipulation_action_range[1] != self.mobility_action_range[0]:
            raise ValueError("action ranges must partition the canonical vector")
        self.manipulation_dim = self.manipulation_action_range[1] - self.manipulation_action_range[0]
        self.mobility_dim = self.mobility_action_range[1] - self.mobility_action_range[0]
        if self.action_horizon <= 0 or self.num_timestep_buckets <= 0 or self.num_inference_timesteps <= 0:
            raise ValueError("horizon and sampling counts must be positive")
        if config.get("time_sampling", "uniform") != "uniform":
            raise ValueError("Query-DMoT requires uniform time sampling")
        self.mobility_loss_weight = float(config.get("mobility_loss_weight", 1.0))
        self.manipulation_loss_weight = float(config.get("manipulation_loss_weight", 1.0))
        if self.mobility_loss_weight < 0 or self.manipulation_loss_weight < 0 or self.mobility_loss_weight + self.manipulation_loss_weight <= 0:
            raise ValueError("loss weights must be nonnegative and not both zero")
        width, mlp_width = int(config.get("input_embedding_dim", 768)), int(config.get("hidden_size", 1024))
        heads, layers = int(config.get("num_attention_heads", 12)), int(config.get("num_layers", 16))
        self.width = width
        self.arm_query_projector = QueryProjector(qwen_hidden_dim, width, self.arm_num_query_tokens)
        self.base_query_projector = QueryProjector(qwen_hidden_dim, width, self.base_num_query_tokens)
        self.mobility_action_encoder = ActionEncoder(self.mobility_dim, width)
        self.manipulation_action_encoder = ActionEncoder(self.manipulation_dim, width)
        self.mobility_state_encoder = MLP(self.state_dim, mlp_width, width)
        self.manipulation_state_encoder = MLP(self.state_dim, mlp_width, width)
        self.mobility_state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.manipulation_state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        context_dim = int(config.get("in_context_condition_dim", 0) or 0)
        self.context_dim = context_dim
        if context_dim:
            self.mobility_context_encoder = MLP(context_dim, mlp_width, width)
            self.manipulation_context_encoder = MLP(context_dim, mlp_width, width)
            self.mobility_context_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.manipulation_context_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        max_seq_len = int(config.get("max_seq_len", 1024))
        self.add_pos_embed = bool(config.get("add_pos_embed", True))
        if self.add_pos_embed:
            self.mobility_position_embedding = nn.Embedding(max_seq_len, width)
            self.manipulation_position_embedding = nn.Embedding(max_seq_len, width)
            nn.init.normal_(self.mobility_position_embedding.weight, std=0.02)
            nn.init.normal_(self.manipulation_position_embedding.weight, std=0.02)
        self.model = DualQueryMoT(width, layers, heads, float(config.get("dropout", 0.2)), bool(config.get("attention_bias", True)), float(config.get("ff_expansion_factor", 4.0)))
        self.mobility_action_decoder = MLP(width, mlp_width, self.mobility_dim)
        self.manipulation_action_decoder = MLP(width, mlp_width, self.manipulation_dim)

    def _validate(self, query, state, actions=None):
        if query.ndim != 3 or tuple(query.shape[1:]) != (self.num_query_tokens, query.shape[-1]):
            raise ValueError(f"query must have shape [B,{self.num_query_tokens},H]")
        if state.ndim == 2:
            state = state[:, None]
        if state.ndim != 3 or tuple(state.shape) != (query.shape[0], 1, self.state_dim):
            raise ValueError(f"state must have shape [B,{self.state_dim}] or [B,1,{self.state_dim}]")
        if actions is not None and tuple(actions.shape) != (query.shape[0], self.action_horizon, self.action_dim):
            raise ValueError(f"actions must have shape [B,{self.action_horizon},{self.action_dim}]")
        return state

    def _condition(self, query, state, context=None):
        arm_q, base_q = torch.split(query, (self.arm_num_query_tokens, self.base_num_query_tokens), dim=1)
        arm_q, base_q = self.arm_query_projector(arm_q), self.base_query_projector(base_q)
        mob_state = self.mobility_state_encoder(state) + self.mobility_state_type_embedding
        manip_state = self.manipulation_state_encoder(state) + self.manipulation_state_type_embedding
        if self.context_dim:
            if context is None:
                raise ValueError("in_context_condition_dim is nonzero but no context was provided")
            if context.ndim == 2:
                context = context[:, None]
            if tuple(context.shape) != (query.shape[0], 1, self.context_dim):
                raise ValueError("context shape does not match in_context_condition_dim")
            mob_context = self.mobility_context_encoder(context) + self.mobility_context_type_embedding
            manip_context = self.manipulation_context_encoder(context) + self.manipulation_context_type_embedding
            mob_state, manip_state = torch.cat((mob_context, mob_state), 1), torch.cat((manip_context, manip_state), 1)
        return (arm_q, base_q), mob_state, manip_state

    def _velocity(self, noisy_mob, noisy_manip, timestep, cond):
        (arm_q, base_q), mob_state, manip_state = cond
        mob_action = self.mobility_action_encoder(noisy_mob, timestep)
        manip_action = self.manipulation_action_encoder(noisy_manip, timestep)
        if self.add_pos_embed:
            pos = torch.arange(self.action_horizon, device=noisy_mob.device)
            mob_action = mob_action + self.mobility_position_embedding(pos)[None]
            manip_action = manip_action + self.manipulation_position_embedding(pos)[None]
        mob = torch.cat((mob_state, mob_action), 1)
        manip = torch.cat((manip_state, manip_action), 1)
        mob, manip = self.model(arm_q, base_q, mob, manip, timestep)
        return self.mobility_action_decoder(mob[:, -self.action_horizon:]), self.manipulation_action_decoder(manip[:, -self.action_horizon:])

    def forward(self, query, actions, state, context=None):
        state = self._validate(query, state, actions)
        clean_manip = actions[..., self.manipulation_action_range[0]:self.manipulation_action_range[1]]
        clean_mob = actions[..., self.mobility_action_range[0]:self.mobility_action_range[1]]
        manip_noise, mob_noise = torch.randn_like(clean_manip), torch.randn_like(clean_mob)
        flow_time = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = flow_time[:, None, None]
        noisy_manip, noisy_mob = t * clean_manip + (1 - t) * manip_noise, t * clean_mob + (1 - t) * mob_noise
        buckets = (flow_time.float() * self.num_timestep_buckets).long().clamp(max=self.num_timestep_buckets - 1)
        cond = self._condition(query, state, context)
        mob_pred, manip_pred = self._velocity(noisy_mob, noisy_manip, buckets, cond)
        mob_loss, manip_loss = F.mse_loss(mob_pred, clean_mob - mob_noise), F.mse_loss(manip_pred, clean_manip - manip_noise)
        total = self.mobility_loss_weight * mob_loss + self.manipulation_loss_weight * manip_loss
        return {"action_loss": total, "action_dit_loss_mobility": mob_loss.detach(), "action_dit_loss_manipulation": manip_loss.detach()}

    @torch.no_grad()
    def predict_action(self, query, state, context=None):
        state = self._validate(query, state)
        dtype, device, b = query.dtype, query.device, query.shape[0]
        mob = torch.randn(b, self.action_horizon, self.mobility_dim, device=device, dtype=dtype)
        manip = torch.randn(b, self.action_horizon, self.manipulation_dim, device=device, dtype=dtype)
        cond = self._condition(query, state, context)
        step = 1.0 / self.num_inference_timesteps
        for i in range(self.num_inference_timesteps):
            flow = i / float(self.num_inference_timesteps)
            t = torch.full((b,), min(int(flow * self.num_timestep_buckets), self.num_timestep_buckets - 1), device=device, dtype=torch.long)
            mob_v, manip_v = self._velocity(mob, manip, t, cond)
            mob, manip = mob + step * mob_v, manip + step * manip_v
        return torch.cat((manip, mob), dim=-1)
