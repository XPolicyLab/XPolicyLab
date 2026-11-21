import torch
import torch.nn as nn

from falcon.config import LazyConfig
from falcon.ops.fused_kernel import bias_swiglu_impl
from falcon.utils.common import nvtx_profiled_function


class AdalnModulation(nn.Module):
    def __init__(self, cfg: LazyConfig):
        super().__init__()
        self.cfg = cfg
        hidden_size = cfg.hidden_size
        adaln_hidden_size = cfg.adaln_hidden_size if cfg.adaln_hidden_size is not None else hidden_size

        self.act = nn.SiLU()
        self.modulate_proj = nn.Linear(adaln_hidden_size, 4 * hidden_size, bias=True)

    @nvtx_profiled_function
    def forward(self, adaln: torch.Tensor) -> torch.Tensor:
        return self.modulate_proj(self.act(adaln))

    def init_weights(self):
        nn.init.constant_(self.modulate_proj.weight, 0)
        nn.init.constant_(self.modulate_proj.bias, 0)


class MLP(nn.Module):
    def __init__(self, cfg: LazyConfig):
        super().__init__()
        self.cfg = cfg
        hidden_size = cfg.hidden_size
        ffn_hidden_size = cfg.ffn_hidden_size

        self.linear_fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.act = nn.SiLU()
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = self.linear_fc1(x)
        intermediate = bias_swiglu_impl(intermediate)
        return self.linear_fc2(intermediate)

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_fc1.weight)
        nn.init.xavier_normal_(self.linear_fc2.weight)
