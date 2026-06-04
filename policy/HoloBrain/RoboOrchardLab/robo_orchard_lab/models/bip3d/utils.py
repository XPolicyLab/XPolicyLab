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


def deformable_format(
    feature_maps,
    spatial_shapes=None,
    level_start_index=None,
    flat_batch=False,
    batch_size=None,
):
    if spatial_shapes is None:
        if flat_batch and feature_maps[0].dim() > 4:
            feature_maps = [x.flatten(end_dim=-4) for x in feature_maps]
        feat_flatten = []
        spatial_shapes = []
        for _, feat in enumerate(feature_maps):
            spatial_shape = torch._shape_as_tensor(feat)[-2:].to(feat.device)
            feat = feat.flatten(start_dim=-2).transpose(-1, -2)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, -2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return feat_flatten, spatial_shapes, level_start_index
    else:
        split_size = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        feature_maps = feature_maps.transpose(-1, -2)
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
        for i, _ in enumerate(feature_maps):
            feature_maps[i] = feature_maps[i].unflatten(
                -1, (spatial_shapes[i, 0], spatial_shapes[i, 1])
            )
            if batch_size is not None:
                if isinstance(batch_size, int):
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, (batch_size, -1)
                    )
                else:
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, batch_size + (-1,)
                    )
        return feature_maps
