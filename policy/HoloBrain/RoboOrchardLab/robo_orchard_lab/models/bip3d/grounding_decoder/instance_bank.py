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

import os

import numpy as np
import torch
from torch import nn

__all__ = ["InstanceBank"]


class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_grad=True,
        feat_grad=True,
        anchor_in_camera=True,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims

        self.anchor_name = "anchor.npy"
        if isinstance(anchor, str):
            if os.path.isabs(anchor):
                self.anchor_name = os.path.split(anchor)[-1]
            else:
                self.anchor_name = anchor
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3:  # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        self.anchor = anchor[:num_anchor]
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([1, self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.anchor_in_camera = anchor_in_camera

    def save_anchor(self, directory):
        output_file = os.path.join(directory, self.anchor_name)
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(output_file, self.anchor)

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def bbox_transform(self, bbox, matrix):
        # bbox: bs, n, 9
        # matrix: bs, cam, 4, 4
        # output: bs, n*cam, 9
        bbox = bbox.unsqueeze(dim=2)
        matrix = matrix.unsqueeze(dim=1)
        points = bbox[..., :3]
        points_extend = torch.concat(
            [points, torch.ones_like(points[..., :1])], dim=-1
        )
        points_trans = torch.matmul(matrix, points_extend[..., None])[
            ..., :3, 0
        ]

        size = bbox[..., 3:6].tile(1, 1, points_trans.shape[2], 1)
        angle = bbox[..., 6:].tile(1, 1, points_trans.shape[2], 1)

        bbox = torch.cat([points_trans, size, angle], dim=-1)
        bbox = bbox.flatten(1, 2)
        return bbox

    def get(self, batch_size, inputs=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, self.anchor.shape[0], 1)
        )
        anchor = torch.tile(
            instance_feature.new_tensor(self.anchor)[None], (batch_size, 1, 1)
        )

        if self.anchor_in_camera:
            cam2global = np.linalg.inv(inputs["extrinsic"].cpu().numpy())
            cam2global = torch.from_numpy(cam2global).to(anchor)
            anchor = self.bbox_transform(anchor, cam2global)
            instance_feature = instance_feature.tile(1, cam2global.shape[1], 1)

        return (
            instance_feature,
            anchor,
        )
