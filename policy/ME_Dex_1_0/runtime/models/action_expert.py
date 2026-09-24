"""ME-Dex-1.0 action expert used by the inference runtime."""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from wan.modules.model import WanRMSNorm, WanLayerNorm

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.

    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | tensor): a list of positions to be encoded, size (M,).
    Returns:
        out (tensor): resulting positional embedding, size (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return torch.from_numpy(emb).float()


@dataclass
class ActionExpertConfig:
    """State-conditioned action expert settings."""

    dim: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 30
    state_dim: int = 14
    action_dim: int = 14
    chunk_size: int = 17
    num_registers: int = 4
    eps: float = 1e-6


class StateActionEncoder(nn.Module):
    """Encoder for robot states and actions."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.state_encoder = self.build_mlp(
            in_features=config.state_dim,
            out_features=config.dim
        )

        self.action_encoder = self.build_mlp(
            in_features=config.action_dim,
            out_features=config.dim
        )

        # Create fixed sinusoidal positional embeddings (chunk_size + 1 + num_registers)
        max_seq_len = config.chunk_size + 1 + config.num_registers
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            config.dim,
            np.arange(max_seq_len)
        )
        # Register as buffer (non-trainable)
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))  # [1, chunk_size+1+num_registers, dim]

    @staticmethod
    def build_mlp(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor, registers: torch.Tensor = None) -> torch.Tensor:
        """
        Encode state and action tokens separately then concatenate, optionally with registers.

        Args:
            state_tokens: [B, 1, state_dim] - initial state
            action_tokens: [B, action_chunk_size, action_dim] - action sequence
            registers: [B, num_registers, dim] - optional register tokens

        Returns:
            Encoded sequence [B, chunk_size + num_registers, dim] if registers provided
            Encoded sequence [B, chunk_size, dim] if no registers
        """
        B = state_tokens.shape[0]
        chunk_size = state_tokens.shape[1] + action_tokens.shape[1]

        # Encode state tokens: direct encoding without squeeze/unsqueeze
        state_encoded = self.state_encoder(state_tokens)  # [B, 1, dim]

        # Encode action tokens: direct encoding
        action_encoded = self.action_encoder(action_tokens)  # [B, action_chunk_size, dim]

        # Concatenate state and action encodings
        encoded = torch.cat([state_encoded, action_encoded], dim=1)  # [B, chunk_size, dim]

        # Optionally concatenate registers
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)  # [B, chunk_size + num_registers, dim]

        # Add positional embeddings to all tokens (including registers)
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]

        return encoded


class ActionExpertBlock(nn.Module):
    """
    Action Expert Block.

    This block owns action-side parameters only (Q/K/V/O and norms) that map
    action tokens to the head space of external backbones (WAN / VLM).
    The actual attention is executed by the backbone self-attention modules
    via a MoT (mixture-of-tokens) interface; this block provides projections
    and FFN, while higher-level modules orchestrate call order.
    """

    def __init__(self, config: ActionExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config

        # Layer norms (WAN style) - only need one for joint attention and one for FFN
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)  # For trimodal joint attention
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)  # For FFN

        # WAN-side action projections and norms (MoT: action -> WAN head space for trimodal joint attention)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.wan_action_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        # normalize Q/K in WAN unified dim
        self.wan_action_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_action_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)

        # FFN (Action Expert's own)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.ffn_dim, config.dim)
        )

        # Timestep modulation (AdaLN style, 6 parameters)
        # 3 params each for: self-attn residual (WAN-action), FFN (alpha/beta/gamma)
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)


class ActionDecoder(nn.Module):
    """Final layer to decode action predictions."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.norm = WanLayerNorm(config.dim, eps=config.eps)

        self.action_head = nn.Sequential(nn.Linear(config.dim, config.action_dim))

        # Timestep modulation for head input (WAN Head style: 2-way modulation)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim**0.5)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode action predictions.

        Args:
            x: Features [B, chunk_size, dim]
            time_emb: Time embeddings [B, chunk_size, dim] for head modulation

        Returns:
            Action predictions [B, chunk_size, action_dim]
        """
        # WAN Head-style modulation using time_emb
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0, e1 = (self.modulation.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
        return self.action_head(z)

class ActionExpert(nn.Module):
    """State-conditioned action expert for joint WAN attention."""

    def __init__(self, config: ActionExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        self.freq_dim = 256  # Sinusoidal embedding dimension (same as WAN)

        self.input_encoder = StateActionEncoder(config)

        # Timestep embedding (same structure as WAN)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6)  # 6 parameters: 3 for WAN-Action joint attn + 3 for FFN
        )

        self.blocks = nn.ModuleList([
            ActionExpertBlock(config, wan_config) for _ in range(config.num_layers)
        ])

        # Register tokens for global attention (optional)
        # When num_registers == 0, do not create parameter to avoid shape issues
        if config.num_registers > 0:
            self.registers = nn.Parameter(
                torch.empty(1, config.num_registers, config.dim).normal_(std=0.02)
            )
        else:
            self.registers = None

        # Output decoder
        self.decoder = ActionDecoder(config)

        # Initialize weights and set dtype
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-initialize output layer
        nn.init.zeros_(self.decoder.action_head[-1].weight)
        nn.init.zeros_(self.decoder.action_head[-1].bias)

        # Initialize time embedding layers
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
