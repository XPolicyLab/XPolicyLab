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

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config ──────────────────────────────────────────────────────────────────

MMDiTConfig = {
    "MMDiT-Psi0": {"hidden_dim": 1536, "num_heads": 24, "num_blocks": 6},
}


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

    def forward(self, x, position_ids=None):
        """Preserve checkpoint PE shape; extend the sinusoid for history inputs."""
        if position_ids is None:
            pe = self.pe[:x.shape[0]]
            return pe.repeat((1, x.shape[1], 1)).detach().clone()
        table = self.pe[:, 0]
        if x.shape[0] > table.shape[0]:
            position = torch.arange(table.shape[0], x.shape[0], device=x.device, dtype=torch.float32)[:, None]
            div_term = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
                                 * -(np.log(10000.0) / x.shape[-1]))
            extra = torch.empty(x.shape[0] - table.shape[0], x.shape[-1], device=x.device, dtype=torch.float32)
            extra[:, 0::2] = torch.sin(position * div_term)
            extra[:, 1::2] = torch.cos(position * div_term)
            table = torch.cat([table, extra.to(table.dtype)], dim=0)
        return table[position_ids].detach()


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

    def forward(self, vl_features: torch.Tensor, state: Optional[torch.Tensor] = None, attention_mask=None):
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
        position_ids = None
        if attention_mask is not None:
            mask = attention_mask.to(device=tokens.device, dtype=torch.long)
            if self.has_state and state is not None:
                mask = torch.cat([mask, torch.ones(mask.shape[0], 1, device=mask.device, dtype=mask.dtype)], dim=1)
            position_ids = (mask.cumsum(-1) - 1).clamp_min(0).transpose(0, 1)
        pos = self.enc_pos(s_t, position_ids=position_ids)  # (S, B, D)
        tokens = (s_t + pos).transpose(0, 1)  # (B, S, D)

        return tokens


# ─── Main Module ─────────────────────────────────────────────────────────────


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
        self.compute_dtype = torch.float32

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
        self.obs_proj = ObservationProjection(
            vl_feature_dim=vl_feature_dim,
            output_dim=hidden_dim,
            state_dim=state_dim,
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

        self.num_timestep_buckets = config.num_timestep_buckets

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

    def compute_autocast(self, device):
        """Keep the native FP32 action-head boundary inside outer BF16 inference."""
        return torch.autocast(
            device_type=torch.device(device).type,
            dtype=self.compute_dtype,
            enabled=True,
        )


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
        integration_dtype = vl_embs.dtype
        expected_shape = (B, self.action_horizon, self.action_dim)
        if noise is None:
            actions = torch.randn(expected_shape, device=device, dtype=integration_dtype)
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
            if noise.device != device:
                raise ValueError(f"noise must be on {device}, got {noise.device}")
            if noise.dtype != vl_embs.dtype:
                raise ValueError(f"noise must have dtype {vl_embs.dtype}, got {noise.dtype}")
            actions = noise.clone()
        # Preserve native noise precision, then integrate in FP32.
        actions = actions.to(torch.float32)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Pre-compute obs tokens (constant across denoising steps)
        obs_hidden_states_base = self.obs_proj(
            vl_embs, state,
            attention_mask=encoder_attention_mask if getattr(self, "history_position_ids", False) else None,
        )
        obs_attention_mask = self._build_obs_attention_mask(
            encoder_attention_mask, obs_hidden_states_base, state,
        )

        for step in range(num_steps):
            t_val = step / float(num_steps)
            t_tensor = torch.full((B,), t_val, device=device, dtype=integration_dtype)

            # Time embedding (scale to [0, 1000])
            temb = self.time_embed(t_tensor * self.num_timestep_buckets)  # (B, D)

            # Action tokens
            action_hidden_states = self.action_proj_in(actions)

            # Run through transformer blocks
            obs_hidden_states = obs_hidden_states_base.clone()
            for block in self.transformer_blocks:
                action_hidden_states, obs_hidden_states = block(
                    action_hidden_states=action_hidden_states,
                    obs_hidden_states=obs_hidden_states,
                    temb=temb,
                    obs_attention_mask=obs_attention_mask,
                )

            # The checkpoint predicts the sample; convert it to Euler velocity.
            model_out = self.action_proj_out(action_hidden_states, temb).to(torch.float32)
            pred_velocity = (model_out - actions) / (1.0 - t_val)
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
