from dataclasses import dataclass, field
from typing import List, Tuple
from rsi.deployment import locations
import torch

from skyrobot.data.data_config import (
    EmbodimentId,
    Embodiment_DIM,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
)


@dataclass
class SkyrobotWAMConfig:
    """Model config for Skyrobot_WAM."""

    text_len: int = 512
    in_dim: int = 48
    dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 4096
    action_dim: int = MAX_ACTION_DIM
    action_per_latent: int = 16
    chunk_size: int = 1
    state_dim: int = MAX_STATE_DIM
    out_dim: int = 48
    num_heads: int = 16
    num_layers: int = 30
    rope_max_seq_len: int = 1024
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

    patch_size: Tuple[int, int, int] = (1, 2, 2)
    vae_stride: Tuple[int, int, int] = (4, 16, 16)
    vae_checkpoint: str = field(default_factory=lambda: locations()["encoder_path"])

    t5_dtype: torch.dtype = torch.bfloat16
    t5_checkpoint: str = field(default_factory=lambda: locations()["encoder_path"])

    supported_embodiments: List[EmbodimentId] = field(
        default_factory=lambda: [
            EmbodimentId.Latent_action,
        ]
    )
    num_embodiments: int = 1  # 由 __post_init__ 派生

    def __post_init__(self):
        assert len(self.supported_embodiments) > 0, \
            "supported_embodiments must be non-empty"
        self.num_embodiments = len(self.supported_embodiments)
        # 每个 supported embodiment 的实际维度必须 ≤ pad 目标
        for e in self.supported_embodiments:
            ad = Embodiment_DIM.action_dims[EmbodimentId(int(e))]
            sd = Embodiment_DIM.state_dims[EmbodimentId(int(e))]
            assert ad <= self.action_dim, (
                f"{EmbodimentId(int(e)).name}: action_dim={ad} exceeds "
                f"model pad target {self.action_dim} (MAX_ACTION_DIM)"
            )
            assert sd <= self.state_dim, (
                f"{EmbodimentId(int(e)).name}: state_dim={sd} exceeds "
                f"model pad target {self.state_dim} (MAX_STATE_DIM)"
            )
