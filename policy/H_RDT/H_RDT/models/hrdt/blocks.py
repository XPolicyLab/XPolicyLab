import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from models.hrdt.norm import RMSNorm
from models.hrdt.attention import Attention, CrossAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Source:
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    """
    A feed-forward network with SiLU activation.

    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Apply custom dimension factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HRDTBlock(nn.Module):
    """
    H-RDT block with self-attention, two cross-attention layers and feed-forward network
    Training mode controls which cross-attention layers to use:
    - 'lang': image + language cross-attention
    """
    def __init__(self, layer_idx: int, config: dict, training_mode: str = 'lang'):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.training_mode = training_mode
        
        # Validate training mode
        if training_mode not in ['lang']:
            raise ValueError(f"training_mode must be 'lang', got {training_mode}")
        
        # Self-attention layer
        self.attn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.attn = Attention(config)
        
        # Image cross-attention layer (always present)
        self.img_cross_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.img_cond_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.img_cross_attn = CrossAttention(config)
        
        # Language cross-attention layer
        self.lang_cross_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.lang_cond_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.lang_cross_attn = CrossAttention(config)
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = FeedForward(
            dim=self.hidden_size,
            hidden_dim=4*self.hidden_size,
            multiple_of=config["multiple_of"],
            ffn_dim_multiplier=config["ffn_dim_multiplier"],
        )
        
        # AdaLN modulation - keep original 9 parameters structure
        # self_attn(3) + cross_attn(3) + mlp(3) = 9 total
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 9*self.hidden_size, bias=True)
        )
        
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cross_contexts: dict = None,
        ):
        """
        Forward pass with two cross-attention layers based on training mode
        
        Args:
            x: Input state-action sequence
            t: Timestep embedding (no sentence token anymore)
            cross_contexts: Dictionary containing cross-attention contexts
                - 'img_c': Image features for cross-attention (always used)
                - 'lang_c': Language tokens for cross-attention (if training_mode='lang')
                - 'lang_attn_mask': Attention mask for language
        """
        if cross_contexts is None:
            cross_contexts = {}
            
        # Adaptive Layer Normalization - split into shifts, scales and gates
        shift_attn, scale_attn, gate_attn, \
        shift_cross, scale_cross, gate_cross, \
        shift_mlp, scale_mlp, gate_mlp \
            = self.adaLN_modulation(t).chunk(9, dim=1)
            
        # Self-attention
        h = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.attn_norm(x), shift_attn, scale_attn))
        
        # Image cross-attention (always present)
        img_c = cross_contexts.get('img_c')
        if img_c is not None:
            h = h + gate_cross.unsqueeze(1) * self.img_cross_attn(
                modulate(self.img_cross_norm(h), shift_cross, scale_cross),
                self.img_cond_norm(img_c), None)
        
        # Language cross-attention
        lang_c = cross_contexts.get('lang_c')
        lang_attn_mask = cross_contexts.get('lang_attn_mask')
        if lang_c is not None:
            # Apply additional cross-attention for language using same modulation parameters
            h = h + self.lang_cross_attn(
                self.lang_cross_norm(h),
                self.lang_cond_norm(lang_c), lang_attn_mask)
        
        # Feedforward network
        out = h + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        
        return out


class ActionDecoder(nn.Module):
    """
    The action decoder layer of H-RDT (previously called FinalLayer).
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.output_size = config["output_size"]

        self.ffn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = Mlp(
            in_features=self.hidden_size,
            hidden_features=self.hidden_size*4,
            out_features=self.output_size,
            act_layer=nn.SiLU, drop=0.0
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2*self.hidden_size, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.ffn_norm(x), shift, scale)
        x = self.ffn(x)
        return x


# Keep FinalLayer for backward compatibility
FinalLayer = ActionDecoder
