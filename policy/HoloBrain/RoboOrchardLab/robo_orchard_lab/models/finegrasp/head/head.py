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

import math

import pointnet2.pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.pointnet2_utils import CylinderQueryAndGroup

from robo_orchard_lab.models.finegrasp.utils import (
    batch_viewpoint_params_to_matrix,
    generate_grasp_views,
)


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points, is_training):
        B, _, num_seed = seed_features.size()  # noqa: N806
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous()  # [B, 1024, 300]
        end_points["view_score"] = view_score

        if is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(
                -1, -1, self.num_view
            )
            view_score_min = view_score_min.unsqueeze(-1).expand(
                -1, -1, self.num_view
            )
            view_score_ = (view_score_ - view_score_min) / (
                view_score_max - view_score_min + 1e-8
            )

            top_view_inds = []
            for i in range(B):
                # assert sum(view_score_[i]) > 0
                if min(torch.sum(view_score_[i], dim=1)) > 0:
                    top_view_inds_batch = torch.multinomial(
                        view_score_[i], 1, replacement=False
                    )
                else:
                    print("min(torch.sum(view_score_[i], dim=1)) <= 0")
                    top_view_inds_batch = torch.max(
                        view_score_[i], dim=1, keepdim=True
                    )[1]
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(
                -1
            )  # [B, 1024]
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # [B, 1024]

            top_view_inds_ = (
                top_view_inds.view(B, num_seed, 1, 1)
                .expand(-1, -1, -1, 3)
                .contiguous()
            )
            template_views = generate_grasp_views(self.num_view).to(
                features.device
            )  # (num_view, 3)
            template_views = (
                template_views.view(1, 1, self.num_view, 3)
                .expand(B, num_seed, -1, -1)
                .contiguous()
            )
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(
                2
            )  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(
                vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device
            )
            vp_rot = batch_viewpoint_params_to_matrix(
                -vp_xyz_, batch_angle
            ).view(B, num_seed, 3, 3)
            end_points["grasp_top_view_xyz"] = vp_xyz
            end_points["grasp_top_view_rot"] = vp_rot

        end_points["grasp_top_view_inds"] = top_view_inds
        return end_points, res_features


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(
            seed_features
        )  # (B, 3, num_seed)
        end_points["objectness_score"] = graspable_score[:, :2]
        end_points["graspness_score"] = graspable_score[:, 2]
        return end_points


class CylinderGroup(nn.Module):
    def __init__(
        self,
        nsample,
        seed_feature_dim,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax=0.04,
    ):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]
        mlps2 = [3 + 256, 256, 256]

        self.grouper = CylinderQueryAndGroup(
            radius=cylinder_radius,
            hmin=hmin,
            hmax=hmax,
            nsample=nsample,
            use_xyz=True,
            normalize_xyz=True,
        )
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        # local interaction module
        self.local_interaction_module = AttentionModule(
            dim=3 + 256, n_head=1, msa_dropout=0.05
        )
        self.mlps2 = pt_utils.SharedMLP(mlps2, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        coords = (
            seed_xyz_graspable.transpose(-1, -2)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.nsample)
        )
        grouped_feature = self.grouper(
            seed_xyz_graspable.float(),
            seed_xyz_graspable.float(),
            vp_rot.float(),
            seed_features_graspable.float(),
        )
        new_features = self.mlps(grouped_feature)
        new_features = (
            torch.cat([new_features, coords], dim=1)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.nsample, 256 + 3)
        )
        new_features = self.local_interaction_module(
            new_features, new_features, new_features, mask=None
        )
        new_features = (
            new_features.view(
                seed_xyz_graspable.shape[0],
                seed_xyz_graspable.shape[1],
                self.nsample,
                3 + 256,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        new_features = self.mlps2(new_features)
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )
        new_features = new_features.squeeze(-1)
        return new_features


class LocalInteraction(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv_angle_feature = nn.Conv1d(256, 64, 1)
        self.conv_depth_feature = nn.Conv1d(256, 64, 1)
        self.conv_width_feature = nn.Conv1d(256, 64, 1)
        self.conv_score_feature = nn.Conv1d(256, 64, 1)

        # global interaction module
        self.global_interaction_module = AttentionModule(
            dim=64, n_head=1, msa_dropout=0.05
        )

        self.conv_angle = nn.Conv1d(64, num_angle + 1, 1)
        self.conv_depth = nn.Conv1d(64, num_depth + 1, 1)
        self.conv_width = nn.Conv1d(64, 1, 1)
        self.conv_score = nn.Conv1d(
            64, 6, 1
        )  # use classification for score learning

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()  # noqa: N806

        angle_features = self.conv_angle_feature(vp_features)
        depth_features = self.conv_depth_feature(vp_features)
        width_features = self.conv_width_feature(vp_features)
        score_features = self.conv_score_feature(vp_features)

        angle_features = (
            angle_features.permute(0, 2, 1)
            .contiguous()
            .view(-1, 64)
            .unsqueeze(1)
        )
        depth_features = (
            depth_features.permute(0, 2, 1)
            .contiguous()
            .view(-1, 64)
            .unsqueeze(1)
        )
        width_features = (
            width_features.permute(0, 2, 1)
            .contiguous()
            .view(-1, 64)
            .unsqueeze(1)
        )
        score_features = (
            score_features.permute(0, 2, 1)
            .contiguous()
            .view(-1, 64)
            .unsqueeze(1)
        )

        interaction_feature = torch.cat(
            [angle_features, depth_features, width_features, score_features],
            dim=1,
        )
        interaction_feature = self.global_interaction_module(
            interaction_feature,
            interaction_feature,
            interaction_feature,
            mask=None,
        )

        angle_features = (
            interaction_feature[:, 0, :].view(B, -1, 64).permute(0, 2, 1)
        )
        depth_features = (
            interaction_feature[:, 1, :].view(B, -1, 64).permute(0, 2, 1)
        )
        width_features = (
            interaction_feature[:, 2, :].view(B, -1, 64).permute(0, 2, 1)
        )
        score_features = (
            interaction_feature[:, 3, :].view(B, -1, 64).permute(0, 2, 1)
        )

        angle_features = self.conv_angle(angle_features)
        depth_features = self.conv_depth(depth_features)
        width_features = self.conv_width(width_features)
        score_features = self.conv_score(score_features)

        # split prediction
        end_points["grasp_angle_pred"] = (
            angle_features  # [B, 12, num_seed_points]
        )
        end_points["grasp_depth_pred"] = (
            depth_features  # [B, 4, num_seed_points]
        )
        end_points["grasp_score_pred"] = (
            score_features  # [B, 1, num_seed_points]
        )
        end_points["grasp_width_pred"] = (
            width_features  # [B, 6, num_seed_points]
        )
        return end_points


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries, keys, values, mask=None, dropout=None):
        """Scaled dot-product attention.

        Args:
            queries : torch.Tensor
                A tensor of shape (B, H, S, headdim), representing the queries.
            keys : torch.Tensor
                A tensor of shape (B, H, L, headdim), representing the keys.
            values : torch.Tensor
                A tensor of shape (B, H, L, headdim), representing the values.
            mask : torch.Tensor, optional
                A tensor of shape (B, 1, S, L) used for masking the attention
                scores. Default is None.
            dropout : torch.nn.Module, optional
                Dropout layer applied to attention weights. Default is None.

        Returns:
            torch.Tensor
                The output of the attention mechanism,
                typically of shape (B, H, S, headdim).
        """
        head_dim = queries.size(-1)
        scores = (
            queries @ keys.transpose(-1, -2) / math.sqrt(head_dim)
        )  # B x H x S x L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values  # B x H x S x head_dim

    def forward(self, query, key, value, mask=None):
        """Computes the attention scores between query and key.

        Args:
            query: torch.Tensor
                A tensor of shape (B, S, D), where B is the batch size, S is
                the sequence length of the query, D is the feature dimension.
            key: torch.Tensor
                A tensor of shape (B, L, D), where B is the batch size, L is
                the sequence length of the key, D is the feature dimension.
            value: torch.Tensor
                A tensor of shape (B, L, D), where B is the batch size, L is
                the sequence length of the value, D is the feature dimension.
            mask: torch.Tensor, optional
                A tensor of shape (B, S, L) used to mask out certain positions
                in the attention computation (default is None).

        Returns:
            torch.Tensor
                The output tensor after applying attention to the value,
                of shape (B, S, D).
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x S x L, 1 for heads
        queries, keys, values = [
            layer(x)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
            for layer, x in zip(
                self.linears[:3], (query, key, value), strict=False
            )
        ]  # (bs, nhead, max_len, head_dim) for word feat
        result = self.attention(
            queries, keys, values, mask, self.dropout
        )  # (bs, nhead, max_len, head_dim)
        result = (
            result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        )  # (bs, max_len, dim)

        return self.linears[-1](result)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head, msa_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiHeadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask):
        msa = self.msa(q, k, v, mask)
        x = self.norm1(v + msa)

        return x
