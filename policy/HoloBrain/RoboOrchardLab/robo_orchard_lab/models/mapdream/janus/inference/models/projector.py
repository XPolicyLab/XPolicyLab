# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from typing import Tuple, Union

import torch
import torch.nn as nn
from easydict import EasyDict


class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(
        self,
        x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """Project visual features into the language space.

        Args:
            x_or_tuple
                (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
                if it is a tuple of torch.Tensor, then it comes from the
                hybrid vision encoder, and x = high_res_x, low_res_x;
                otherwise it is the feature from the single vision
                encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        return self.layers(x)


if __name__ == "__main__":
    cfg = EasyDict(
        input_dim=1024,
        n_embed=2048,
        depth=2,
        projector_type="low_high_hybrid_split_mlp_gelu",
    )
    inputs = (torch.rand(4, 576, 1024), torch.rand(4, 576, 1024))

    m = MlpProjector(cfg)
    out = m(inputs)
    print(out.shape)
