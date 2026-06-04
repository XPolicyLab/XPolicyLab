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

import MinkowskiEngine as ME  # noqa: N817
import numpy as np
import torch
import torch.nn as nn
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

from robo_orchard_lab.models.finegrasp.backbone import MinkUNet
from robo_orchard_lab.models.finegrasp.head import (
    ApproachNet,
    CylinderGroup,
    GraspableNet,
    LocalInteraction,
)
from robo_orchard_lab.models.finegrasp.utils import process_grasp_labels
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
)


class GroupTransformerFusion(nn.Module):
    def __init__(self, feature_dim, num_groups):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: [B, G, D, N]
        B, G, D, N = x.shape  # noqa: N806

        x = x.permute(1, 3, 0, 2)  # [G, N, B, D]
        x = x.reshape(G, N * B, D)  # [G, N*B, D]

        out = self.transformer(x)  # [G, N*B, D]

        attn_weights = torch.softmax(self.attn(out), dim=0)  # [G, N*B, 1]
        out = (out * attn_weights).sum(dim=0)  # [N*B, D]
        out = out.reshape(N, B, D).permute(1, 2, 0)  # [B, D, N]

        return out


class FineGrasp(ModelMixin):  # noqa: N801
    cfg: "FineGraspConfig"  # for type hint

    def __init__(self, cfg: "FineGraspConfig"):
        super().__init__(cfg)
        self.cfg = cfg
        self.seed_feature_dim = cfg.seed_feat_dim
        self.num_depth = cfg.num_depth
        self.num_angle = cfg.num_angle
        self.num_seed_points = cfg.num_seed_points
        self.num_view = cfg.num_view
        self.voxel_size = cfg.voxel_size
        self.cylinder_radius = cfg.cylinder_radius
        self.cylinder_groups = cfg.cylinder_groups
        self.use_normal = cfg.use_normal
        self.loss = cfg.loss

        if self.use_normal is True:
            in_channels = 6  # xyz + normal
        else:
            in_channels = 3

        # Backbone
        self.backbone = MinkUNet(
            in_channels=in_channels, out_channels=self.seed_feature_dim, D=3
        )

        # Objectness and graspness
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)

        # View Selection
        self.view = ApproachNet(
            self.num_view,
            seed_feature_dim=self.seed_feature_dim,
        )

        # Cylinder
        self.cy_groups = torch.nn.ModuleList(
            [
                CylinderGroup(
                    nsample=16,
                    cylinder_radius=self.cylinder_radius * cylinder_group,
                    seed_feature_dim=self.seed_feature_dim,
                )
                for cylinder_group in self.cylinder_groups
            ]
        )

        self.fuse_multi_scale = GroupTransformerFusion(
            256, len(cfg.cylinder_groups)
        )

        # Depth and Score searching
        self.grasp_head = LocalInteraction(
            num_angle=self.num_angle, num_depth=self.num_depth
        )

        self.graspness_threshold = cfg.graspness_threshold

    def forward(self, end_points):
        is_training = self.training
        seed_xyz = end_points[
            "point_clouds"
        ]  # use all sampled point cloud, [B, point_num (20000)， 3]
        if self.use_normal is True:
            seed_normal = end_points[
                "cloud_normal"
            ]  # use all sampled point cloud, [B, point_num (20000)， 3]
            seed_point_input = torch.cat([seed_xyz, seed_normal], dim=-1)
            seed_init_feature = seed_point_input.cpu()
        else:
            seed_point_input = seed_xyz
            seed_init_feature = np.ones_like(seed_point_input.cpu()).astype(
                np.float32
            )

        B, point_num, _ = seed_point_input.shape  # noqa: N806

        # Generate input to meet the Minkowski Engine
        coordinates_batch, features_batch = ME.utils.sparse_collate(
            [coord for coord in end_points["coordinates_for_voxel"]],
            [feat for feat in seed_init_feature],
        )

        (
            coordinates_batch,
            features_batch,
            _,
            end_points["quantize2original"],
        ) = ME.utils.sparse_quantize(
            coordinates_batch,
            features_batch,
            return_index=True,
            return_inverse=True,
        )

        coordinates_batch = coordinates_batch.cuda()
        features_batch = features_batch.cuda()

        # [points of the whole scenes after quantize, 3(coors) + 1(index)]
        end_points["coors"] = coordinates_batch

        # [points of the whole scenes after quantize, 3 (input feature dim)]
        end_points["feats"] = features_batch
        mink_input = ME.SparseTensor(
            features_batch, coordinates=coordinates_batch
        )

        # Minkowski Backbone
        seed_features = self.backbone(mink_input).F
        seed_features = (
            seed_features[end_points["quantize2original"]]
            .view(B, point_num, -1)
            .transpose(1, 2)
        )
        # [B (batch size), 512 (feature dim), 20000 (points in a scene)]

        # Generate the masks of the objectness and the graspness
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points["objectness_score"]
        # [B (batch size), 2 (object classification), 20000 (points per scene)]
        graspness_score = end_points["graspness_score"].squeeze(
            1
        )  # [B (batch size), 20000 (points in a scene)]
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = objectness_pred == 1
        graspness_mask = graspness_score > self.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # Generate the downsample point (1024 per scene)
        # using the furthest point sampling
        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_normal_graspable = []
        graspable_num_batch = 0.0
        for i in range(B):
            cur_mask = graspable_mask[i]
            if cur_mask.sum() < 1:
                print("No graspable points in the scene")
                continue

            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            fps_idxs = furthest_point_sample(
                cur_seed_xyz, self.num_seed_points
            )
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            cur_seed_xyz = (
                gather_operation(cur_seed_xyz_flipped, fps_idxs)
                .transpose(1, 2)
                .squeeze(0)
                .contiguous()
            )
            if self.use_normal is True:
                cur_seed_normal = seed_normal[i][cur_mask]
                cur_seed_normal_flipped = (
                    cur_seed_normal.unsqueeze(0).transpose(1, 2).contiguous()
                )
                cur_seed_normal = (
                    gather_operation(cur_seed_normal_flipped.float(), fps_idxs)
                    .squeeze(0)
                    .contiguous()
                )
            cur_feat_flipped = (
                cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            )
            cur_feat = (
                gather_operation(cur_feat_flipped.float(), fps_idxs)
                .squeeze(0)
                .contiguous()
            )

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            if self.use_normal is True:
                seed_normal_graspable.append(cur_seed_normal)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        if self.use_normal is True:
            seed_normal_graspable = torch.stack(seed_normal_graspable, 0)
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = torch.stack(seed_features_graspable)
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["normal_graspable"] = seed_normal_graspable
        end_points["D: Graspable Points"] = graspable_num_batch / B

        # Select the view for each point
        end_points, res_feat = self.view(
            seed_features_graspable, end_points, is_training
        )
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = seed_features_graspable + res_feat
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]

        # Generate the labels
        if is_training:
            # generate the scene-level grasp labels from the object-level
            # grasp label and the object poses
            # map the scene sampled points to the labeled object points
            # (note that the labeled object points and the sampled points
            # may not 100% match due to the sampling and argumentation)
            grasp_top_views_rot, end_points = process_grasp_labels(
                end_points, self.cfg
            )
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # Cylinder grouping
        group_features = []
        for cy_group in self.cy_groups:
            group_features.append(
                cy_group(
                    seed_xyz_graspable.contiguous(),
                    seed_features_graspable.contiguous(),
                    grasp_top_views_rot,
                )
            )

        group_features = torch.stack(group_features, dim=1)
        group_features = self.fuse_multi_scale(group_features)

        # Width and score predicting
        end_points = self.grasp_head(group_features, end_points)

        if is_training:
            for loss_func in self.loss:
                end_points = loss_func(end_points)

        return end_points


MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class FineGraspConfig(TorchModuleCfg[FineGrasp]):
    class_type: ClassType_co[FineGrasp] = FineGrasp

    # Model hyperparameters
    model_name: str = "FineGrasp"
    seed_feat_dim: int = 512
    graspness_threshold: float = 0.1
    grasp_max_width: float = 0.1
    num_depth: int = 4
    num_view: int = 300
    num_angle: int = 12
    num_seed_points: int = 1024
    voxel_size: float = 0.005
    cylinder_radius: float = 0.05
    cylinder_groups: list = [0.25, 0.5, 0.75, 1.0]
    use_normal: bool = False

    loss: MODULE_TPYE | None = None
