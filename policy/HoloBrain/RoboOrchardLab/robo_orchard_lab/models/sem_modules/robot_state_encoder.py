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

import torch
from torch import nn

from robo_orchard_lab.models.sem_modules.layers import linear_act_ln
from robo_orchard_lab.utils.build import build


class SEMRobotStateEncoder(nn.Module):
    """Spatial Enhanced Manipulation (SEM) Robot State Encoder.

    Robot state encoder implementation from the paper
       'https://arxiv.org/abs/2505.16196'.
    """

    def __init__(
        self,
        embed_dims,
        joint_self_attn,
        norm_layer,
        ffn,
        temp_self_attn=None,
        state_dims=8,
        num_encoder=4,
        act_cfg=None,
        operation_order=None,
        chunk_size=5,
        pre_norm=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.chunk_size = chunk_size
        self.pre_norm = pre_norm

        self.input_fc = nn.Sequential(
            *linear_act_ln(
                embed_dims, 2, 2, state_dims * chunk_size, act_cfg=act_cfg
            ),
            nn.Linear(embed_dims, embed_dims),
        )

        self.layers = []
        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = [
                "joint_self_attn",
                "norm",
                "ffn",
                "norm",
            ] * num_encoder
        self.op_config_map = {
            "joint_self_attn": joint_self_attn,
            "temp_self_attn": temp_self_attn,
            "norm": norm_layer,
            "ffn": ffn,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

    def forward(self, robot_state, joint_distance=None):
        bs, num_step, num_link = robot_state.shape[:3]
        robot_state = robot_state.permute(0, 2, 1, 3)
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
