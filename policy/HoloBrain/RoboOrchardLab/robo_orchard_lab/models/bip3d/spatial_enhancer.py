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

from robo_orchard_lab.models.bip3d.utils import deformable_format
from robo_orchard_lab.models.layers.transformer_layers import FFN, MLP


class BatchDepthProbGTGenerator(nn.Module):
    """Generate depth prob gt for spatial enhancer.

    To generate the ground truth required for thespatial enhancer,
    the GT should have a shape of [bs, num_view, num_feature, num_depth],
    where num_feature represents the total number of feature vectors in the
    multi-stride feature maps. Each image feature corresponds to a patch
    (composed of stride x stride pixels), and the GT indicates the depth
    distribution of all pixels within that patch.

    Compute the depth distribution for a single pixel using linear
    interpolation, example:

    .. code-block: text

        min_depth, max_depth, num_depth = 0, 3, 4
        depth_anchor: [0, 1, 2, 3]
        pixel_depth: 1.2
        Since 1.2 lies between depth anchors 1 and 2, we interpolate:
            Weight for 1: (2 - 1.2) / (2 - 1) = 0.8
            Weight for 2: (1.2 - 1) / (2 - 1) = 0.2
        depth_prob_gt: [0, 0.8, 0.2, 0]

    Invalid Depth Handling:
        If a pixel's depth is ≤0 or >=valid_max_depth, it is marked as
        invalid. The valid pixel ratio in a patch is used as the loss
        weight for that patch. If more than valid_threshold% of pixels in
        a patch are valid, the entire patch is deemed invalid, and the
        corresponding depth_prob_gt weights are set to 0.
    """

    def __init__(
        self,
        stride,
        min_depth=0.25,
        max_depth=10,
        num_depth=64,
        origin_stride=1,
        input_key="depths",
        output_key="depth_prob_gt",
        max_valid_depth=None,
        valid_threshold=-1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.stride = [x // origin_stride for x in stride]
        self.origin_stride = origin_stride
        self.input_key = input_key
        self.output_key = output_key
        self.max_valid_depth = max_valid_depth
        self.valid_threshold = valid_threshold

    def __call__(self, data):
        if not self.training:
            return data
        bs, num_view, _, h, w = data[self.input_key].shape
        depth = data[self.input_key].flatten(0, 1)

        if self.origin_stride != 1:
            depth = torch.nn.functional.interpolate(
                depth, scale_factor=1 / self.origin_stride
            )
            h = h // self.origin_stride
            w = w // self.origin_stride

        if self.valid_threshold >= 0:
            valid_mask = depth > 0
            if self.max_valid_depth is not None:
                valid_mask = torch.logical_and(
                    valid_mask, depth < self.max_depth
                )

        depth = torch.clip(depth, min=self.min_depth, max=self.max_depth)
        depth_anchor = torch.linspace(
            self.min_depth, self.max_depth, self.num_depth
        ).to(depth)[None, :, None, None]
        distance = depth - depth_anchor
        depth_anchor = depth_anchor.expand_as(distance)

        min_dis_index_a = torch.abs(distance).min(dim=1, keepdims=True)[1]
        depth_a = torch.gather(depth_anchor, 1, min_dis_index_a)

        min_dis_index_b = torch.where(
            depth_a < depth, min_dis_index_a + 1, min_dis_index_a - 1
        )
        min_dis_index_b = torch.clip(min_dis_index_b, 0, self.num_depth - 1)
        depth_b = torch.gather(depth_anchor, 1, min_dis_index_b)

        sparse_depth = torch.zeros_like(distance)
        sparse_depth = sparse_depth.scatter_(
            1, min_dis_index_a, torch.abs(depth_a)
        )
        sparse_depth = sparse_depth.scatter_(
            1, min_dis_index_b, torch.abs(depth_b)
        )
        sparse_depth_inv = (
            sparse_depth.sum(dim=1, keepdims=True) - sparse_depth
        )

        depth_prob = torch.where(
            sparse_depth != 0,
            (depth - sparse_depth_inv) / (sparse_depth - sparse_depth_inv),
            0,
        )  # bs*num_view, num_depth, h, w

        if self.valid_threshold >= 0:
            depth_prob = torch.where(valid_mask, depth_prob, 0)
            valid_mask = valid_mask.to(depth_prob)

        output = []
        if self.valid_threshold >= 0:
            valid_ratio = []
        for s in self.stride:
            tmp = torch.nn.functional.avg_pool2d(
                depth_prob, kernel_size=s, stride=s
            )
            if self.valid_threshold >= 0:
                _valid = torch.nn.functional.avg_pool2d(
                    valid_mask, kernel_size=s, stride=s
                )
                _valid = torch.where(_valid < self.valid_threshold, 0, _valid)
                tmp = torch.where(
                    _valid == 0,
                    0,
                    tmp / _valid,
                )
                valid_ratio.append(_valid.reshape(bs * num_view, -1))
            tmp = tmp.reshape(bs * num_view, self.num_depth, -1)
            output.append(tmp)

        output = torch.cat(output, dim=2).permute(0, 2, 1).clip(min=0, max=1)
        data[self.output_key] = output.unflatten(0, (bs, num_view))
        if self.valid_threshold >= 0:
            valid_ratio = torch.cat(valid_ratio, dim=1).unflatten(
                0, (bs, num_view)
            )
            data[self.output_key] = [data[self.output_key], valid_ratio]
        return data


class DepthFusionSpatialEnhancer(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        feature_3d_dim=32,
        num_depth_layers=2,
        min_depth=0.25,
        max_depth=10,
        num_depth=64,
        with_feature_3d=True,
        loss_depth_weight=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.feature_3d_dim = feature_3d_dim
        self.num_depth_layers = num_depth_layers
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_depth = num_depth
        self.with_feature_3d = with_feature_3d
        self.loss_depth_weight = loss_depth_weight

        fusion_dim = self.embed_dims + self.feature_3d_dim
        if self.with_feature_3d:
            self.pts_prob_pre_fc = nn.Linear(
                self.embed_dims, self.feature_3d_dim
            )
            dim = self.feature_3d_dim * 2
            fusion_dim += self.feature_3d_dim
        else:
            dim = self.embed_dims
        self.pts_prob_fc = MLP(
            dim,
            dim,
            self.num_depth,
            self.num_depth_layers,
        )
        self.pts_fc = nn.Linear(3, self.feature_3d_dim)
        self.fusion_fc = nn.Sequential(
            FFN(embed_dims=fusion_dim, feedforward_channels=1024),
            nn.Linear(fusion_dim, self.embed_dims),
        )
        self.fusion_norm = nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        feature_maps,
        inputs: dict,
        feature_3d=None,
        **kwargs,
    ):
        feature_2d, spatial_shapes, _ = deformable_format(feature_maps)
        pts = self.get_pts(
            spatial_shapes,
            inputs["image_wh"],
            inputs["projection_mat"],
            feature_2d.device,
            feature_2d.dtype,
        )

        if self.with_feature_3d:
            feature_3d = deformable_format(feature_3d)[0]
            depth_prob_feat = self.pts_prob_pre_fc(feature_2d)
            depth_prob_feat = torch.cat([depth_prob_feat, feature_3d], dim=-1)
            depth_prob = self.pts_prob_fc(depth_prob_feat).softmax(dim=-1)
            feature_fused = [feature_2d, feature_3d]
        else:
            depth_prob = self.pts_prob_fc(feature_2d).softmax(dim=-1)
            feature_fused = [feature_2d]

        pts_feature = self.pts_fc(pts)
        pts_feature = (depth_prob.unsqueeze(dim=-1) * pts_feature).sum(dim=-2)
        feature_fused.append(pts_feature)
        feature_fused = torch.cat(feature_fused, dim=-1)
        feature_fused = self.fusion_fc(feature_fused) + feature_2d
        feature_fused = self.fusion_norm(feature_fused)
        feature_fused = deformable_format(feature_fused, spatial_shapes)
        if self.loss_depth_weight > 0 and self.training:
            loss_depth = self.depth_prob_loss(depth_prob, inputs)
        else:
            loss_depth = None
        return feature_fused, depth_prob, loss_depth

    def get_pts(self, spatial_shapes, image_wh, projection_mat, device, dtype):
        image_wh = image_wh.reshape(-1, 2)[0]
        pixels = []
        for _, shape in enumerate(spatial_shapes):
            stride = image_wh[0] / shape[1]
            u = torch.linspace(
                0, image_wh[0] - stride, shape[1], device=device, dtype=dtype
            )
            v = torch.linspace(
                0, image_wh[1] - stride, shape[0], device=device, dtype=dtype
            )
            u = u[None].tile(shape[0], 1)
            v = v[:, None].tile(1, shape[1])
            uv = torch.stack([u, v], dim=-1).flatten(0, 1)
            pixels.append(uv)
        pixels = torch.cat(pixels, dim=0)[:, None]
        depths = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.num_depth,
            device=device,
            dtype=dtype,
        )
        depths = depths[None, :, None]
        pts = pixels * depths
        depths = depths.tile(pixels.shape[0], 1, 1)
        pts = torch.cat([pts, depths, torch.ones_like(depths)], dim=-1)

        pts = torch.linalg.solve(
            projection_mat.mT.unsqueeze(dim=2), pts, left=False
        )[..., :3]  # b,cam,N,3
        return pts

    def depth_prob_loss(self, depth_prob, inputs):
        gt = inputs["depth_prob_gt"]
        if isinstance(gt, list):
            gt, weight = gt
        else:
            weight = None

        loss_depth = torch.nn.functional.binary_cross_entropy(
            depth_prob, gt, reduction="none"
        ).mean(dim=-1)
        if weight is not None:
            loss_depth = loss_depth * weight
        loss_depth = loss_depth.mean() * self.loss_depth_weight

        # loss_depth = (
        #     torch.nn.functional.binary_cross_entropy(
        #         depth_prob[mask], inputs["depth_prob_gt"][mask]
        #     )
        #     * self.loss_depth_weight
        # )
        return loss_depth
