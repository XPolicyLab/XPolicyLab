"""Transformer layers for Motubrain."""

import torch
import torch.nn as nn

from falcon.models.motubrain.model_config import MotubrainConfig
from falcon.models.motu_3_2_gva.layers import (
    FinalLayer,
    MotuLayer,
)
from falcon.modules import AdalnModulation
from falcon.ops.fused_norm_ops import (
    fused_varlen_batch_a_modulate,
    fused_varlen_batch_v_modulate,
)


class _StreamAdalnModulation(AdalnModulation):
    """Select one stream from paired video/action timestep embeddings."""

    def __init__(self, cfg: MotubrainConfig, stream_index: int):
        super().__init__(cfg)
        self.stream_index = stream_index

    def forward(self, adaln: torch.Tensor) -> torch.Tensor:
        if adaln.ndim != 3 or adaln.shape[1] != 2:
            raise ValueError("Motubrain AdaLN expects [batch, video/action, hidden]")
        return super().forward(adaln[:, self.stream_index])


def _install_stream_adaln(
    layer: MotuLayer,
    motu_cfg: MotubrainConfig,
) -> None:
    layer.x_adamod = _StreamAdalnModulation(motu_cfg, stream_index=0)
    layer.a_adamod = _StreamAdalnModulation(motu_cfg, stream_index=1)
    layer.c_adamod = _StreamAdalnModulation(motu_cfg, stream_index=0)


class MotubrainLayer(MotuLayer):
    """GVA layer with independent video/action conditioning."""

    def __init__(
        self,
        layer_number: int,
        motu_cfg: MotubrainConfig,
        attention_class: type[nn.Module],
    ):
        super().__init__(layer_number, motu_cfg)
        _install_stream_adaln(self, motu_cfg)
        self.self_attention = attention_class(motu_cfg, layer_number)


class MotubrainFinalLayer(FinalLayer):
    """GVA output layer conditioned by separate video/action timesteps."""

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        adaln: torch.Tensor,
        mb_meta: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adaln.ndim != 3 or adaln.shape[1] != 2:
            raise ValueError("Motubrain final layer expects paired video/action AdaLN")
        video_adaln = adaln[:, 0]
        action_adaln = adaln[:, 1]

        video_scale = self.adaln_modulation(video_adaln).contiguous()
        x = fused_varlen_batch_v_modulate(x, video_scale, mb_meta)
        x = self.linear(x)

        action_scale = self.a_adaln_modulation(action_adaln).contiguous()
        action = fused_varlen_batch_a_modulate(
            action,
            action_scale,
            mb_meta,
        )
        return x, self.a_linear(action)
