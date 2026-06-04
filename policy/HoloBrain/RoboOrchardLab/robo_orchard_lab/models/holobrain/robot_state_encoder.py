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

from typing import List, Optional, Union

import torch
from robo_orchard_core.utils.config import Config
from torch import nn

from robo_orchard_lab.models.holobrain.layers import linear_act_ln
from robo_orchard_lab.models.holobrain.utils import apply_joint_mask
from robo_orchard_lab.models.mixin import TorchModuleCfgType_co
from robo_orchard_lab.utils.build import DelayInitDictType, build

MODULE_TYPE = TorchModuleCfgType_co | DelayInitDictType


class HoloBrainEncoderTransformerConfig(Config):
    """Transformer config of HoloBrain robot state encoder.

    Args:
        joint_self_attn (Optional[MODULE_TYPE]): Joint dimension self-attention
            module.
        norm_layer (MODULE_TYPE): Normalization layer.
        ffn (MODULE_TYPE): Feed-forward network.
        operation_order (Optional[List[str]]): Sequence of operations
            (attn/FFN/norm) in transformer decoder.
        temp_cross_attn (Optional[MODULE_TYPE]): Self attention module across
            time steps.
        pre_norm (bool): Use pre-normalization or post-normalization.
    """

    joint_self_attn: Optional[MODULE_TYPE]
    norm_layer: MODULE_TYPE
    ffn: MODULE_TYPE
    operation_order: List[Union[str, None]]
    temp_self_attn: Optional[MODULE_TYPE] = None
    pre_norm: bool = True

    @property
    def op_config_map(self) -> dict:
        return {
            "norm": self.norm_layer,
            "ffn": self.ffn,
            "joint_self_attn": self.joint_self_attn,
            "temp_cross_attn": self.temp_self_attn,
        }


class HoloBrainEncoderBaseConfig(Config):
    """Base config of HoloBrain robot state encoder.

    Args:
        embed_dims (int): Embedding dimension size.
        state_dims (int): Dimension size of robot state. Defaults to 8, refer
            to [a, x, y, z, qw, qx, qy, qz].
        act_cfg (Optional[MODULE_TYPE]): Activation function configuration.
        chunk_size (int): Step size for chunking prediction horizon.
    """

    embed_dims: int = 256
    state_dims: int = 8
    act_cfg: Optional[MODULE_TYPE] = None
    chunk_size: int = 1


class HoloBrainRobotStateEncoder(nn.Module):
    """Spatial Enhanced Manipulation (HoloBrain) Robot State Encoder.

    Robot state encoder implementation from the paper
       'https://arxiv.org/abs/2505.16196'.
    """

    def __init__(
        self,
        transformer_cfg: HoloBrainEncoderTransformerConfig,
        base_cfg: HoloBrainEncoderBaseConfig,
    ):
        super().__init__()
        transformer_cfg = HoloBrainEncoderTransformerConfig.model_validate(
            transformer_cfg
        )
        base_cfg = HoloBrainEncoderBaseConfig.model_validate(base_cfg)
        self.embed_dims = base_cfg.embed_dims
        self.chunk_size = base_cfg.chunk_size
        self.state_dims = base_cfg.state_dims
        self.act_cfg = base_cfg.act_cfg
        self.pre_norm = transformer_cfg.pre_norm

        self.input_fc = nn.Sequential(
            *linear_act_ln(
                self.embed_dims,
                2,
                2,
                self.state_dims * self.chunk_size,
                act_cfg=self.act_cfg,
            ),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.layers = []
        self.operation_order = transformer_cfg.operation_order
        self.layers = nn.ModuleList(
            [
                build(transformer_cfg.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

    def forward(self, robot_state, joint_distance=None, joint_mask=None):
        bs, num_step, num_link = robot_state.shape[:3]
        robot_state = robot_state.permute(0, 2, 1, 3)

        if joint_mask is not None:
            robot_state = apply_joint_mask(robot_state, joint_mask)

        num_chunk = num_step // self.chunk_size
        robot_state = robot_state.reshape(bs, num_link, num_chunk, -1)
        x = self.input_fc(robot_state)
        joint_distance = joint_distance.tile(num_chunk, 1, 1)
        temp_pos = torch.arange(num_chunk)[None].tile(bs * num_link, 1).to(x)
        if self.pre_norm:
            identity = x
        else:
            identity = None
        for op, layer in zip(self.operation_order, self.layers, strict=False):
            if layer is None:
                continue
            elif op == "joint_self_attn":
                x = x.permute(0, 2, 1, 3).flatten(0, 1)
                if identity is not None:
                    _identity = identity.permute(0, 2, 1, 3).flatten(0, 1)
                else:
                    _identity = None
                x = layer(
                    query=x,
                    key=x,
                    value=x,
                    query_pos=joint_distance,
                    identity=_identity,
                )
                x = x.unflatten(0, (bs, num_chunk)).permute(0, 2, 1, 3)
            if op == "temp_self_attn":
                x = x.flatten(0, 1)
                if identity is not None:
                    _identity = identity.flatten(0, 1)
                else:
                    _identity = None
                x = layer(
                    query=x,
                    key=x,
                    value=x,
                    query_pos=temp_pos,
                    key_pos=temp_pos,
                    identity=_identity,
                )
                x = x.unflatten(0, (bs, num_link))
            elif op == "ffn":
                x = layer(x, identity=identity)
            elif op == "norm":
                if self.pre_norm:
                    identity = x
                x = layer(x)
        return x  # bs, num_link, num_chunk, c
