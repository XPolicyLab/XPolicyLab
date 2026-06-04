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

from typing import List, Optional

import torch
import torch.nn as nn

try:
    from robo_orchard_lab.ops.deformable_aggregation import (
        deformable_aggregation_func,
    )
except ImportError:
    deformable_aggregation_func = None
from robo_orchard_lab.models.bip3d.grounding_decoder.utils import linear_act_ln
from robo_orchard_lab.models.layers.transformer_layers import FFN
from robo_orchard_lab.utils.build import build

__all__ = [
    "DeformableFeatureAggregation",
]


class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: Optional[dict] = None,
        use_camera_embed=False,
        residual_mode="add",
        batch_first=True,
        ffn_cfg=None,
        with_value_proj=False,
        filter_outlier=False,
        with_depth=False,
        min_depth=None,
        max_depth=None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        assert batch_first, "only support batch_first=True"
        assert deformable_aggregation_func is not None, (
            "please install the deformable_aggregation module"
        )
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        if kps_generator is not None:
            kps_generator["embed_dims"] = embed_dims
            self.kps_generator = build(kps_generator)
            self.num_pts = self.kps_generator.num_pts
        else:
            self.kps_generator = None
            self.num_pts = 1
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_act_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )
        if ffn_cfg is not None:
            ffn_cfg["embed_dims"] = embed_dims
            self.ffn = FFN(**ffn_cfg)
            self.norms = nn.ModuleList(
                build(dict(type=nn.LayerNorm), embed_dims) for _ in range(2)
            )
        else:
            self.ffn = None
        self.with_value_proj = with_value_proj
        self.filter_outlier = filter_outlier
        if self.with_value_proj:
            self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.with_depth = with_depth
        if self.with_depth:
            assert min_depth is not None and max_depth is not None
            self.min_depth = min_depth
            self.max_depth = max_depth

    def init_weight(self):
        nn.init.constant(self.weights_fc.weight.data, 0.0)
        nn.init.constant(self.weights_fc.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data, 0.0)
        nn.init.constant(self.output_proj.bias.data, 0.0)

    def get_spatial_shape_3D(self, spatial_shape, depth_dim):  # noqa: N802
        spatial_shape_depth = (
            spatial_shape.new_ones(*spatial_shape.shape[:-1], 1) * depth_dim
        )
        spatial_shape_3D = torch.cat(  # noqa: N806
            [spatial_shape, spatial_shape_depth], dim=-1
        )
        return spatial_shape_3D.contiguous()

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        depth_prob: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if self.kps_generator is not None:
            key_points = self.kps_generator(anchor, instance_feature)
        else:
            key_points = anchor[:, :, None]

        points_2d, depth, mask = self.project_points(
            key_points,
            metas["projection_mat"],
            metas.get("image_wh"),
        )
        weights = self._get_weights(
            instance_feature, anchor_embed, metas, mask
        )

        if self.with_value_proj:
            feature_maps[0] = self.value_proj(feature_maps[0])

        points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
            bs, num_anchor * self.num_pts, -1, 2
        )
        weights = (
            weights.permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .reshape(
                bs,
                num_anchor * self.num_pts,
                -1,
                self.num_levels,
                self.num_groups,
            )
        )

        if self.with_depth:
            assert depth_prob is not None, (
                "depth_prob is required when with_depth is True"
            )
            depth = depth.permute(0, 2, 3, 1).reshape(
                bs, num_anchor * self.num_pts, -1, 1
            )
            # normalize depth to [0, depth_prob.shape[-1]-1]
            depth = (depth - self.min_depth) / (
                self.max_depth - self.min_depth
            )
            depth = depth * (depth_prob.shape[-1] - 1)
            features = deformable_aggregation_func(
                *feature_maps, points_2d, weights, depth_prob, depth
            )
        else:
            features = deformable_aggregation_func(
                *feature_maps, points_2d, weights
            )
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        features = features.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        if self.ffn is not None:
            output = self.norms[0](output)
            output = self.ffn(output)
            output = self.norms[1](output)
        return output

    def _get_weights(
        self, instance_feature, anchor_embed, metas=None, mask=None
    ):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            assert metas is not None, (
                "metas is required when camera_encoder is used"
            )
            num_cams = metas["projection_mat"].shape[1]
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = self.weights_fc(feature)
        if mask is not None and self.filter_outlier:
            num_cams = weights.shape[2]
            mask = mask.permute(0, 2, 1, 3)[..., None, :, None]
            weights = weights.reshape(
                bs,
                num_anchor,
                num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
            weights = weights.masked_fill(
                torch.logical_and(~mask, mask.sum(dim=2, keepdim=True) != 0),
                float("-inf"),
            )
        weights = (
            weights.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                -1,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, -1, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        mask = depth > 1e-5
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        mask = mask & (points_2d[..., 0] > 0) & (points_2d[..., 1] > 0)
        if image_wh is not None:
            image_wh = image_wh.reshape(-1, 2)[0]
            points_2d = points_2d / image_wh
            mask = mask & (points_2d[..., 0] < 1) & (points_2d[..., 1] < 1)
        return points_2d, depth, mask
