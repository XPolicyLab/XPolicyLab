"""
MM-DiT Action Header for Flow Matching (Ψ₀-style)

Aligned with: https://github.com/physical-superintelligence-lab/Psi0

Key differences from GR00T's cross-attention DiT:
  - Joint global attention between action tokens and VL condition tokens
  - Separate AdaLN-Zero modulation for each stream
  - Bidirectional information flow (not just action ← condition)
  - ActionProjectionIn: pure MLP + learned positional embedding (no timestep mixing)
  - ActionProjectionOut: x * scale + shift (no norm, no 1+scale)
  - ObservationProjection: VL proj + state token + sinusoidal pos encoding + dropout
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from transformers import PretrainedConfig


# ─── Config ──────────────────────────────────────────────────────────────────

MMDiTConfig = {
    "MMDiT-Psi0": {"hidden_dim": 1536, "num_heads": 24, "num_blocks": 6},
    "MMDiT-B": {"hidden_dim": 768, "num_heads": 12, "num_blocks": 6},
    "MMDiT-L": {"hidden_dim": 1024, "num_heads": 16, "num_blocks": 12},
    "MMDiT-H": {"hidden_dim": 1536, "num_heads": 24, "num_blocks": 12},
    "MMDiT-XL": {"hidden_dim": 1536, "num_heads": 24, "num_blocks": 18},
}


def masked_velocity_mse(
    pred_velocity: torch.Tensor,
    velocity_target: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute velocity MSE with optional per-dimension or per-timestep masking."""
    squared_diff = (pred_velocity - velocity_target) ** 2
    if action_mask is None:
        return squared_diff.mean()

    action_mask = action_mask.to(device=pred_velocity.device, dtype=torch.bool)
    if action_mask.ndim == 2:
        action_mask_expanded = action_mask[:, None, :].expand_as(squared_diff)
    elif action_mask.ndim == 3:
        if action_mask.shape != squared_diff.shape:
            raise ValueError(
                f"Per-timestep action mask must match {tuple(squared_diff.shape)}, got {tuple(action_mask.shape)}"
            )
        action_mask_expanded = action_mask
    else:
        raise ValueError(f"action_mask must have 2 or 3 dimensions, got {action_mask.ndim}")
    return (squared_diff * action_mask_expanded.float()).sum() / (
        action_mask_expanded.sum().clamp_min(1).float()
    )


# ─── Building Blocks ─────────────────────────────────────────────────────────


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with zero-init gating (6 modulation params). Matches Psi0."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if len(emb.shape) == 2:
            emb = emb.unsqueeze(1)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm for context_pre_only block (scale+shift only). Matches Psi0."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if len(emb.shape) == 2:
            emb = emb.unsqueeze(1)
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class VLAJointTransformerBlock(nn.Module):
    """
    MM-DiT block with joint attention. Matches Psi0's VLATransformerBlock.

    Two streams:
      - Action (A): AdaLN-Zero → Joint Attn → FFN
      - Observation (O): AdaLN-Zero → Joint Attn → FFN  (last block: context_pre_only)
    """

    def __init__(self, dim: int, num_heads: int, context_pre_only: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_pre_only = context_pre_only

        # Action stream norm
        self.norm1_act = AdaLayerNormZero(dim)

        # Observation stream norm
        if context_pre_only:
            self.norm1_obs = AdaLayerNormContinuous(dim)
        else:
            self.norm1_obs = AdaLayerNormZero(dim)

        # Joint attention projections — action stream (hidden_states in Psi0)
        self.to_q_act = nn.Linear(dim, dim, bias=True)
        self.to_k_act = nn.Linear(dim, dim, bias=True)
        self.to_v_act = nn.Linear(dim, dim, bias=True)
        self.to_out_act = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.Dropout(0.0),
        )

        # Joint attention projections — observation stream (encoder_hidden_states in Psi0)
        self.to_q_obs = nn.Linear(dim, dim, bias=True)
        self.to_k_obs = nn.Linear(dim, dim, bias=True)
        self.to_v_obs = nn.Linear(dim, dim, bias=True)
        if not context_pre_only:
            self.to_out_obs = nn.Linear(dim, dim, bias=True)

        # QK norm (RMSNorm per head, matches Psi0's attn.norm_q/norm_k/norm_added_q/norm_added_k)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Action FFN (GELU approximate, matches Psi0's FeedForward)
        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

        # Observation FFN (not needed for last block)
        if not context_pre_only:
            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_obs = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim * 4, dim),
            )

    def forward(
        self,
        action_hidden_states: torch.Tensor,  # (B, Ta, D)
        obs_hidden_states: torch.Tensor,     # (B, To, D)
        temb: torch.Tensor,                  # (B, D) or (B, T, D)
        obs_attention_mask: Optional[torch.Tensor] = None,
    ):
        B = action_hidden_states.shape[0]
        Ta = action_hidden_states.shape[1]
        To = obs_hidden_states.shape[1]

        # ─── Norm + Modulation ───
        norm_act, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(
            action_hidden_states, temb
        )

        # Psi0: obs uses temb[:,-1] if 3D, else temb
        obs_temb = temb[:, -1] if len(temb.shape) > 2 else temb

        if self.context_pre_only:
            norm_obs = self.norm1_obs(obs_hidden_states, obs_temb)
            gate_msa_obs = shift_mlp_obs = scale_mlp_obs = gate_mlp_obs = None
        else:
            norm_obs, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(
                obs_hidden_states, obs_temb
            )

        # ─── Joint Attention ───
        q_act = self.to_q_act(norm_act)
        k_act = self.to_k_act(norm_act)
        v_act = self.to_v_act(norm_act)

        q_obs = self.to_q_obs(norm_obs)
        k_obs = self.to_k_obs(norm_obs)
        v_obs = self.to_v_obs(norm_obs)

        def reshape_heads(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_act = reshape_heads(q_act, Ta)
        k_act = reshape_heads(k_act, Ta)
        v_act = reshape_heads(v_act, Ta)
        q_obs = reshape_heads(q_obs, To)
        k_obs = reshape_heads(k_obs, To)
        v_obs = reshape_heads(v_obs, To)

        # QK norm (separate norms for action and obs, matching Psi0)
        q_act = self.norm_q(q_act)
        k_act = self.norm_k(k_act)
        q_obs = self.norm_added_q(q_obs)
        k_obs = self.norm_added_k(k_obs)

        # Concat for joint attention: [action; obs]
        q = torch.cat([q_act, q_obs], dim=2)  # (B, H, Ta+To, head_dim)
        k = torch.cat([k_act, k_obs], dim=2)
        v = torch.cat([v_act, v_obs], dim=2)

        # Build attention mask if needed
        attn_mask = None
        if obs_attention_mask is not None:
            act_mask = torch.ones(
                B, 1, 1, Ta, device=action_hidden_states.device, dtype=torch.bool
            )
            obs_mask = (obs_attention_mask == 1)[:, None, None, :]  # (B, 1, 1, To)
            attn_mask = torch.cat([act_mask, obs_mask], dim=-1)  # (B, 1, 1, Ta+To)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        # Split back
        out = out.transpose(1, 2).reshape(B, Ta + To, self.dim)
        act_attn_out = out[:, :Ta]
        obs_attn_out = out[:, Ta:]

        # Output projections
        act_attn_out = self.to_out_act(act_attn_out)

        # ─── Action stream residual + FFN ───
        action_hidden_states = action_hidden_states + gate_msa_act * act_attn_out

        norm_act_ff = self.norm2_act(action_hidden_states)
        norm_act_ff = norm_act_ff * (1 + scale_mlp_act) + shift_mlp_act
        action_hidden_states = action_hidden_states + gate_mlp_act * self.ff_act(norm_act_ff)

        # ─── Observation stream residual + FFN ───
        if self.context_pre_only:
            obs_hidden_states = None
        else:
            obs_attn_out = self.to_out_obs(obs_attn_out)
            obs_hidden_states = obs_hidden_states + gate_msa_obs * obs_attn_out

            norm_obs_ff = self.norm2_obs(obs_hidden_states)
            norm_obs_ff = norm_obs_ff * (1 + scale_mlp_obs) + shift_mlp_obs
            obs_hidden_states = obs_hidden_states + gate_mlp_obs * self.ff_obs(norm_obs_ff)

        return action_hidden_states, obs_hidden_states


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Matches Psi0's _PositionalEncoding exactly."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """x: (seq_len, batch_size, d_model) → returns (seq_len, batch_size, d_model)"""
        pe = self.pe[:x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    """Timestep embedding. Matches Psi0's _TimeNetwork exactly."""

    def __init__(self, time_dim: int = 256, out_dim: int = 1536, learnable_w: bool = False):
        super().__init__()
        assert time_dim % 2 == 0
        half_dim = time_dim // 2

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.w = nn.Parameter(w, requires_grad=learnable_w)

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B,) or (B, T)
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)


class ActionProjectionIn(nn.Module):
    """Action token encoder. Matches Psi0's ActionProjectionIn exactly.

    Pure MLP projection + learned positional embedding. No timestep mixing.
    """

    def __init__(self, action_pred_horizon: int, action_dim: int, output_dim: int):
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim

        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),
        )
        self.dec_pos = nn.Parameter(
            torch.empty(action_pred_horizon, output_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        B = noisy_actions.shape[0]
        noise_acs = noisy_actions.reshape(B, -1, self.action_dim)
        ac_tokens = self.ac_proj(noise_acs)
        return ac_tokens + self.dec_pos.unsqueeze(0)


class ActionProjectionOut(nn.Module):
    """Final projection layer. Matches Psi0's ActionProjectionOut exactly.

    x * scale + shift → linear (no norm, no 1+scale).
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = x * scale + shift
        return self.linear(x)


class ObservationProjection(nn.Module):
    """Observation token builder. Matches Psi0's ObservationProjection.

    VL features → Linear proj → concat state token → sinusoidal pos encoding → dropout.
    """

    def __init__(self, vl_feature_dim: int, output_dim: int, state_dim: int = 0):
        super().__init__()
        self.views_proj = nn.Linear(vl_feature_dim, output_dim, bias=True)

        # State token: Dropout(0.2) → Linear (matches Psi0's _obs_proc)
        self.has_state = state_dim > 0
        if self.has_state:
            self._obs_proc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(state_dim, output_dim),
            )

        # Sinusoidal positional encoding (matches Psi0's enc_pos)
        self.enc_pos = _PositionalEncoding(d_model=output_dim)

        # Post-processing dropout (matches Psi0's post_proc with dropout=0.1)
        self.post_proc = nn.Dropout(0.1)

    def forward(self, vl_features: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        vl_features: (B, S, vl_feature_dim)
        state: (B, 1, state_dim) or None
        Returns: (B, S_total, output_dim)
        """
        tokens = self.views_proj(vl_features)  # (B, S, D)

        if self.has_state and state is not None:
            state_token = self._obs_proc(state)  # (B, 1, D)
            tokens = torch.cat([tokens, state_token], dim=1)  # (B, S+1, D)

        # Psi0 order: tokenize_obs applies post_proc BEFORE enc_pos
        tokens = self.post_proc(tokens)

        # Sinusoidal positional encoding (Psi0: s_t.T(0,1) → enc_pos → add → T(0,1))
        s_t = tokens.transpose(0, 1)  # (S, B, D)
        pos = self.enc_pos(s_t)       # (S, B, D)
        tokens = (s_t + pos).transpose(0, 1)  # (B, S, D)

        return tokens


# ─── Main Module ─────────────────────────────────────────────────────────────


@dataclass
class MMDiTActionHeadConfig(PretrainedConfig):
    hidden_dim: int = field(default=1536, metadata={"help": "MM-DiT hidden dimension"})
    num_heads: int = field(default=24, metadata={"help": "Number of attention heads"})
    num_blocks: int = field(default=12, metadata={"help": "Number of MM-DiT blocks"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension"})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon (chunk size)"})
    state_dim: int = field(default=None, metadata={"help": "State/proprioception dim"})
    vl_feature_dim: int = field(default=2048, metadata={"help": "VLM hidden dim (input)"})
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Unused, kept for config compat"})
    add_pos_embed: bool = field(default=True, metadata={"help": "Unused, kept for config compat"})
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=10, metadata={"help": "Euler steps at inference"})
    state_placement: str = field(default="condition", metadata={"help": "Where to place state: 'condition' or 'noise'"})
    state_dropout_ratio: float = field(default=0.0)
    prediction_type: str = field(default="velocity", metadata={"help": "'velocity' (flow matching) or 'sample' (JiT-style x0 prediction with v-space loss)"})
    t_eps: float = field(default=5e-2, metadata={"help": "Clamp min for (1-t) in sample mode to avoid division explosion"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class MMDiTFlowmatchingActionHead(nn.Module):
    """
    MM-DiT based flow-matching action head (Ψ₀-style).

    Architecture aligned with Psi0's ActionTransformerModel:
      - _TimeNetwork(time_dim=256, out_dim=hidden_dim)
      - ActionProjectionIn(action_dim → hidden_dim, learned pos embed)
      - ObservationProjection(VL proj + state token + sinusoidal pos + dropout)
      - N × VLATransformerBlock (joint attention, last one context_pre_only)
      - ActionProjectionOut(x * scale + shift → linear)
    """

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        action_model_type = config.action_model_type
        mmdit_cfg = MMDiTConfig[action_model_type]
        hidden_dim = mmdit_cfg["hidden_dim"]
        num_heads = mmdit_cfg["num_heads"]
        num_blocks = mmdit_cfg["num_blocks"]

        self.hidden_dim = hidden_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.config = config

        # Time embedding (Psi0: _TimeNetwork(time_dim=256, out_dim=action_hidden_dim))
        self.time_embed = _TimeNetwork(time_dim=256, out_dim=hidden_dim)

        # Action projection in (Psi0: ActionProjectionIn)
        self.action_proj_in = ActionProjectionIn(
            action_pred_horizon=config.action_horizon,
            action_dim=config.action_dim,
            output_dim=hidden_dim,
        )

        # Observation projection (Psi0: ObservationProjection)
        vl_feature_dim = getattr(config, "vl_feature_dim", 2048)
        state_dim = getattr(config, "state_dim", 0) or 0
        self.state_placement = getattr(config, "state_placement", "condition")
        if self.state_placement not in {"condition", "noise"}:
            raise ValueError(
                f"state_placement must be 'condition' or 'noise', got '{self.state_placement}'"
            )

        # For condition placement, ObservationProjection handles state internally
        # For noise placement, state goes to action side via separate projection
        obs_state_dim = state_dim if self.state_placement == "condition" else 0
        self.obs_proj = ObservationProjection(
            vl_feature_dim=vl_feature_dim,
            output_dim=hidden_dim,
            state_dim=obs_state_dim,
        )

        if self.state_placement == "noise" and state_dim > 0:
            self.state_noise_proj = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(state_dim, hidden_dim),
            )

        # MM-DiT transformer blocks
        self.transformer_blocks = nn.ModuleList([
            VLAJointTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                context_pre_only=(i == num_blocks - 1),
            )
            for i in range(num_blocks)
        ])

        # Action output projection (Psi0: ActionProjectionOut)
        self.action_proj_out = ActionProjectionOut(hidden_dim, config.action_dim)

        # Noise schedule (flow matching)
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        self.state_dropout_ratio = getattr(config, "state_dropout_ratio", 0.0)
        self.prediction_type = getattr(config, "prediction_type", "velocity")
        if self.prediction_type not in {"velocity", "sample"}:
            raise ValueError(
                f"prediction_type must be 'velocity' or 'sample', got '{self.prediction_type}'"
            )
        self.t_eps = getattr(config, "t_eps", 5e-2)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total MMDiT Action Head parameters: {total_params:,}")

    def _build_obs_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        obs_hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is None:
            return None

        obs_seq_len = obs_hidden_states.shape[1]
        encoder_attention_mask = encoder_attention_mask.to(device=obs_hidden_states.device)

        if encoder_attention_mask.shape[1] == obs_seq_len:
            return encoder_attention_mask

        if (
            state is not None
            and encoder_attention_mask.shape[1] + 1 == obs_seq_len
        ):
            state_mask = torch.ones(
                encoder_attention_mask.shape[0],
                1,
                device=obs_hidden_states.device,
                dtype=encoder_attention_mask.dtype,
            )
            return torch.cat([encoder_attention_mask, state_mask], dim=1)

        raise ValueError(
            "encoder_attention_mask length must match observation tokens "
            f"({encoder_attention_mask.shape[1]} vs {obs_seq_len})"
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None,
        encoder_attention_mask=None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """
        Training forward: compute flow-matching loss.

        vl_embs: (B, S, vl_feature_dim) — VLM hidden states
        actions: (B, T, action_dim) — ground truth action chunk
        state: (B, 1, state_dim) — proprioception (optional)
        action_mask: (B, action_dim) — True for real action dims, False for padded dims.
        """
        device = vl_embs.device
        B = actions.shape[0]

        # Sample noise and time
        noise = torch.randn_like(actions)
        t = self.sample_time(B, device=device, dtype=actions.dtype)
        t_broadcast = t[:, None, None]

        noisy_actions = (1 - t_broadcast) * noise + t_broadcast * actions
        velocity_target = actions - noise        # Time embedding (scale to [0, 1000] for proper sinusoidal frequency activation)
        temb = self.time_embed(t * self.num_timestep_buckets)  # (B, D)

        # Action tokens (no timestep mixed in — matches Psi0)
        action_hidden_states = self.action_proj_in(noisy_actions)  # (B, T, D)

        # State dropout (applied regardless of placement)
        if self.training and self.state_dropout_ratio > 0 and state is not None:
            drop_mask = torch.rand(B, 1, 1, device=device) < self.state_dropout_ratio
            state = state.masked_fill(drop_mask, 0.0)

        # Place state on noise side or condition side
        if self.state_placement == "noise" and state is not None:
            obs_hidden_states = self.obs_proj(vl_embs, None)
            state_token = self.state_noise_proj(state)  # (B, 1, D)
            action_hidden_states = torch.cat([state_token, action_hidden_states], dim=1)
        else:
            obs_hidden_states = self.obs_proj(vl_embs, state)

        obs_attention_mask = self._build_obs_attention_mask(
            encoder_attention_mask, obs_hidden_states,
            state if self.state_placement == "condition" else None,
        )

        # Run MM-DiT blocks
        for block in self.transformer_blocks:
            action_hidden_states, obs_hidden_states = block(
                action_hidden_states=action_hidden_states,
                obs_hidden_states=obs_hidden_states,
                temb=temb,
                obs_attention_mask=obs_attention_mask,
            )

        # Strip prepended state token if noise placement
        if self.state_placement == "noise" and state is not None:
            action_hidden_states = action_hidden_states[:, 1:]

        # Output projection
        model_out = self.action_proj_out(action_hidden_states, temb)  # (B, T, action_dim)

        if self.prediction_type == "sample":
            # JiT-style: model outputs predicted x0, then convert to velocity.
            # Loss is still in v-space (AML A1): ‖v̂ - v‖² with v̂ = (x̂0 - z_t)/(1-t).
            # Apply the same clamped denominator to BOTH pred and target so that
            # a perfect x̂0 == actions yields exactly zero loss even when 1-t < t_eps.
            one_minus_t = (1 - t_broadcast).clamp_min(self.t_eps)
            pred_velocity = (model_out - noisy_actions) / one_minus_t
            velocity_target = (actions - noisy_actions) / one_minus_t
        else:
            pred_velocity = model_out

        # MSE loss on velocity. When mixing embodiments with different action
        # widths, padded dimensions must not train the model toward zeros.
        loss = masked_velocity_mse(pred_velocity, velocity_target, action_mask)
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference: denoise actions via Euler integration from optional explicit noise."""
        B = vl_embs.shape[0]
        device = vl_embs.device
        expected_shape = (B, self.action_horizon, self.action_dim)
        if noise is None:
            actions = torch.randn(expected_shape, device=device, dtype=vl_embs.dtype)
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
            if noise.device != device:
                raise ValueError(f"noise must be on {device}, got {noise.device}")
            if noise.dtype != vl_embs.dtype:
                raise ValueError(f"noise must have dtype {vl_embs.dtype}, got {noise.dtype}")
            actions = noise.clone()
        # Integrate in fp32 no matter what dtype the blocks run in, matching openpi
        # (pi0_pytorch casts the transformer output to fp32 before action_out_proj and
        # keeps dt/time/x_t fp32). Until now this held only by accident: the velocity came
        # back fp32 from a hard-coded fp32 autocast and promoted the accumulator on the
        # first Euler step. It is also what lets the caller do .numpy(), which has no bf16.
        actions = actions.to(torch.float32)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Pre-compute obs tokens (constant across denoising steps)
        if self.state_placement == "noise" and state is not None:
            obs_hidden_states_base = self.obs_proj(vl_embs, None)
            state_token = self.state_noise_proj(state)  # (B, 1, D)
        else:
            obs_hidden_states_base = self.obs_proj(vl_embs, state)
            state_token = None

        obs_attention_mask = self._build_obs_attention_mask(
            encoder_attention_mask, obs_hidden_states_base,
            state if self.state_placement == "condition" else None,
        )

        for step in range(num_steps):
            t_val = step / float(num_steps)
            t_tensor = torch.full((B,), t_val, device=device, dtype=vl_embs.dtype)

            # Time embedding (scale to [0, 1000])
            temb = self.time_embed(t_tensor * self.num_timestep_buckets)  # (B, D)

            # Action tokens
            action_hidden_states = self.action_proj_in(actions)

            # Prepend state token to action side if noise placement
            if state_token is not None:
                action_hidden_states = torch.cat([state_token, action_hidden_states], dim=1)

            # Run through transformer blocks
            obs_hidden_states = obs_hidden_states_base.clone()
            for block in self.transformer_blocks:
                action_hidden_states, obs_hidden_states = block(
                    action_hidden_states=action_hidden_states,
                    obs_hidden_states=obs_hidden_states,
                    temb=temb,
                    obs_attention_mask=obs_attention_mask,
                )

            # Strip state token before output
            if state_token is not None:
                action_hidden_states = action_hidden_states[:, 1:]

            # Predict velocity and Euler step. fp32 from here on: the blocks may be bf16,
            # but the velocity and the accumulator are not (see the cast above).
            model_out = self.action_proj_out(action_hidden_states, temb).to(torch.float32)
            if self.prediction_type == "sample":
                # JiT-style: model_out is x̂0; convert to velocity.
                # (1 - t_val) is a scalar in (0, 1] for t_val in [0, (N-1)/N], no clamp needed.
                pred_velocity = (model_out - actions) / (1.0 - t_val)
            else:
                pred_velocity = model_out
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_mmdit_action_model(config=None):
    """Factory: build MMDiTFlowmatchingActionHead from global framework config."""
    return MMDiTFlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    B = 2
    vl_embs = torch.randn(B, 80, 2048)
    actions = torch.randn(B, 50, 14)
    state = torch.randn(B, 1, 14)

    for placement in ["condition", "noise"]:
        for pred_type in ["velocity", "sample"]:
            print(f"\n{'='*60}")
            print(f"Testing state_placement = '{placement}', prediction_type = '{pred_type}'")
            print(f"{'='*60}")

            cfg = OmegaConf.create({
                "framework": {
                    "action_model": {
                        "action_model_type": "MMDiT-B",
                        "action_dim": 14,
                        "state_dim": 14,
                        "action_horizon": 50,
                        "num_inference_timesteps": 10,
                        "vl_feature_dim": 2048,
                        "num_target_vision_tokens": 32,
                        "add_pos_embed": True,
                        "noise_beta_alpha": 1.5,
                        "noise_beta_beta": 1.0,
                        "noise_s": 0.999,
                        "num_timestep_buckets": 1000,
                        "state_placement": placement,
                        "state_dropout_ratio": 0.2,
                        "prediction_type": pred_type,
                    }
                }
            })

            model = MMDiTFlowmatchingActionHead(full_config=cfg)
            print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

            loss = model(vl_embs, actions, state)
            print(f"Training loss: {loss.item():.4f}")

            model.eval()
            pred = model.predict_action(vl_embs, state)
            print(f"Predicted actions shape: {pred.shape}")
            assert pred.shape == (B, 50, 14), f"Shape mismatch: {pred.shape}"
            print("PASSED")
