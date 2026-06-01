import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        if dim % 2 != 0:
            raise ValueError(f"dimension ({dim}) must be divisible by 2")
        self.time_mlp_in = nn.Linear(dim, dim)
        self.time_mlp_out = nn.Linear(dim, dim)
    def create_sinusoidal_pos_embedding(self,
        time: torch.tensor, 
        min_period: float=4e-3, 
        max_period: float=4.0, 
    ) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
        dtype = get_safe_dtype(torch.float64, time.device.type)
        fraction = torch.linspace(0.0, 1.0, self.dim // 2, dtype=dtype, device=time.device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    def forward(self, time: torch.tensor) -> torch.Tensor:
        time_emb = self.create_sinusoidal_pos_embedding(time).type(dtype=time.dtype)
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
    ) -> torch.FloatTensor:
        emb = self.linear_1(action)
        return emb
    
class ActionDecoder(nn.Module):
    def __init__(self, action_hidden_size: int, action_dim: int, num_layers: int=2):
        super().__init__()

        proj = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(action_hidden_size, action_hidden_size),
                nn.SiLU()
            ) for _ in range(num_layers - 1)]
        )
        proj.append(nn.Linear(action_hidden_size, action_dim))
        self.proj = nn.Sequential(*proj)

    def forward(
        self,
        action_embed: torch.FloatTensor,
    ) -> torch.FloatTensor:
        action = self.proj(action_embed)
        return action