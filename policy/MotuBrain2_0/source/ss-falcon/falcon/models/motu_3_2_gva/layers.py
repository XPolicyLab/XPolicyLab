import logging
import os
from functools import partial

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from falcon.distributed import get_cp_group
from falcon.modules import MLP, AdalnModulation
from falcon.ops.fused_norm_ops import (
    fused_varlen_batch_a_broadcast_mul,
    fused_varlen_batch_a_modulate,
    fused_varlen_batch_v_broadcast_mul,
    fused_varlen_batch_v_modulate,
    fused_varlen_c_broadcast_mul,
    fused_varlen_c_modulate,
)
from falcon.utils.common import nvtx_profiled_function
from soma.inference.models.operators import chunk_contiguous_tanh

from .model_config import GVAConfig
from .modules import FullJointAttention

logger = logging.getLogger(__name__)


def _skip_attn_use_reentrant() -> bool:
    return os.environ.get("AC_SKIP_ATTN_REENTRANT") == "1"


class MotuLayer(nn.Module):
    """Layer for the GVA model with audio modality

    Args:
        layer_number (int): Layer number
        motu_cfg (GVAConfig): GVA configuration
    """

    def __init__(self, layer_number: int, motu_cfg: GVAConfig):
        super().__init__()
        self.motu_cfg = motu_cfg
        self.layer_number = layer_number
        self.is_last_layer = layer_number == motu_cfg.num_layers

        self.x_adamod = AdalnModulation(motu_cfg)
        self.a_adamod = AdalnModulation(motu_cfg)
        self.c_adamod = AdalnModulation(motu_cfg)
        self.norm = RMSNorm(motu_cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attention = FullJointAttention(motu_cfg, layer_number)
        self.x_mlp = MLP(motu_cfg)
        self.a_mlp = MLP(motu_cfg)
        if not self.is_last_layer:
            self.c_mlp = MLP(motu_cfg)

        self.cp_group = get_cp_group()

    @nvtx_profiled_function
    def forward(
        self,
        hidden_states: torch.Tensor,
        a_hidden_states: torch.Tensor | None,
        context: torch.Tensor,
        adaln: torch.Tensor,
        packed_seq_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): Hidden states
            a_hidden_states (torch.Tensor | None): Audio hidden states
            context (torch.Tensor): Context
            adaln (torch.Tensor): Adaln
            packed_seq_params (dict): Packed sequence parameters

        Returns:
            output_hidden_states (torch.Tensor): Output hidden states
            a_mod (torch.Tensor | None): Output audio hidden states
            c_mod (torch.Tensor): Output context
        """
        # pre-process video modality
        x_residual = hidden_states
        x_scale_mha, x_gate_mha, x_scale_mlp, x_gate_mlp = chunk_contiguous_tanh(
            self.x_adamod(adaln), tanh_mask=0b1010
        )
        x_mod = fused_varlen_batch_v_modulate(hidden_states, x_scale_mha, packed_seq_params)

        # pre-process context modality
        c_residual = context
        c_scale_mha, c_gate_mha, c_scale_mlp, c_gate_mlp = chunk_contiguous_tanh(
            self.c_adamod(adaln), tanh_mask=0b1010
        )
        c_mod = fused_varlen_c_modulate(context, c_scale_mha, packed_seq_params)

        a_residual = a_hidden_states
        a_scale_mha, a_gate_mha, a_scale_mlp, a_gate_mlp = chunk_contiguous_tanh(
            self.a_adamod(adaln), tanh_mask=0b1010
        )
        a_mod = fused_varlen_batch_a_modulate(a_hidden_states, a_scale_mha, packed_seq_params)

        # joint attention between video, audio, and context modalities
        x_mod, a_mod, c_mod = self.self_attention(x_mod, a_mod, c_mod, packed_seq_params)

        # post-process video modality
        x_mod = x_residual + fused_varlen_batch_v_broadcast_mul(x_mod, x_gate_mha, packed_seq_params)
        x_residual = x_mod
        x_mod = fused_varlen_batch_v_modulate(x_mod, x_scale_mlp, packed_seq_params)
        x_mod = self.x_mlp(x_mod)
        x_mod = x_residual + fused_varlen_batch_v_broadcast_mul(x_mod, x_gate_mlp, packed_seq_params)

        a_mod = a_residual + fused_varlen_batch_a_broadcast_mul(a_mod, a_gate_mha, packed_seq_params)
        a_residual = a_mod
        a_mod = fused_varlen_batch_a_modulate(a_mod, a_scale_mlp, packed_seq_params)
        a_mod = self.a_mlp(a_mod)
        a_mod = a_residual + fused_varlen_batch_a_broadcast_mul(a_mod, a_gate_mlp, packed_seq_params)

        # post-process context modality if not the last layer
        if not self.is_last_layer:
            c_mod = c_residual + fused_varlen_c_broadcast_mul(c_mod, c_gate_mha, packed_seq_params)
            c_residual = c_mod
            c_mod = fused_varlen_c_modulate(c_mod, c_scale_mlp, packed_seq_params)
            c_mod = self.c_mlp(c_mod)
            c_mod = c_residual + fused_varlen_c_broadcast_mul(c_mod, c_gate_mlp, packed_seq_params)

        return x_mod, a_mod, c_mod

    def init_weights(self):
        self.x_adamod.init_weights()
        self.a_adamod.init_weights()
        self.c_adamod.init_weights()
        self.norm.reset_parameters()
        self.self_attention.init_weights()
        self.x_mlp.init_weights()
        self.a_mlp.init_weights()
        if not self.is_last_layer:
            self.c_mlp.init_weights()


class FinalLayer(nn.Module):
    """Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py#L125-L142

    Args:
        hidden_size (int): Hidden size
        patch_size (int): Patch size
        out_channels (int): Output channels
        audio_in_channels (int): Audio input channels
        cuVs (torch.Tensor): Cumulative sequence lengths for video
        maxV (int): Maximum sequence length for video
        cuAs (torch.Tensor | None): Cumulative sequence lengths for audio
        maxA (int | None): Maximum sequence length for audio
    """

    def __init__(self, hidden_size: int, out_channels: int, audio_in_channels: int):
        super().__init__()
        # video output layers
        self.norm = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        # audio output layers
        self.a_norm = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.a_adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.a_linear = nn.Linear(hidden_size, audio_in_channels, bias=True)
        self.cp_group = get_cp_group()

    def forward(self, x: torch.Tensor, a: torch.Tensor | None, adaln: torch.Tensor, mb_meta: dict):
        """

        Args:
            x (torch.Tensor): Video hidden states
            a (torch.Tensor | None): Audio hidden states
            adaln (torch.Tensor): timestep embedding
            mb_meta (dict): Metadata for micro batch

        Returns:
            x (torch.Tensor): Video output
            a (torch.Tensor | None): Audio output
        """
        scale = self.adaln_modulation(adaln).contiguous()
        x = fused_varlen_batch_v_modulate(x, scale, mb_meta)
        x = self.linear(x)

        if a is not None:
            scale = self.a_adaln_modulation(adaln).contiguous()
            a = fused_varlen_batch_a_modulate(a, scale, mb_meta)
            a = self.a_linear(a)

        return x, a

    def init_weights(self):
        self.norm.reset_parameters()
        self.a_norm.reset_parameters()

        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.a_linear.weight, 0)
        nn.init.constant_(self.adaln_modulation[1].weight, 0)
        nn.init.constant_(self.a_adaln_modulation[1].weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.a_linear.bias, 0)
        nn.init.constant_(self.adaln_modulation[1].bias, 0)
        nn.init.constant_(self.a_adaln_modulation[1].bias, 0)
