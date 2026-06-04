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

import types

import torch
from torch.amp import custom_bwd, custom_fwd  # type: ignore
from torch.autograd.function import Function, once_differentiable

try:
    from robo_orchard_lab.ops.deformable_aggregation import (
        deformable_aggregation_ext as da,  # type: ignore
    )
except ImportError:
    da = None

try:
    from robo_orchard_lab.ops.deformable_aggregation import (
        deformable_aggregation_with_depth_ext as dad,  # type: ignore
    )
except ImportError:
    dad = None


__all__ = ["deformable_aggregation_func", "feature_maps_format"]


def load_deformable_aggregation(with_depth=False) -> types.ModuleType:
    from pathlib import Path

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    if not with_depth:
        src_files = [
            root / "src/deformable_aggregation.cpp",
            root / "src/deformable_aggregation_cuda.cu",
        ]
    else:
        src_files = [
            root / "src/deformable_aggregation_with_depth.cpp",
            root / "src/deformable_aggregation_with_depth_cuda.cu",
        ]

    func: types.ModuleType = load(
        f"defordeformable_aggregation_{'with_depth_' if with_depth else ''}ext",  # noqa: E501
        src_files,  # type: ignore
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )  # type: ignore
    return func


if da is None:  # type: ignore
    da: types.ModuleType = load_deformable_aggregation()

if dad is None:  # type: ignore
    dad: types.ModuleType = load_deformable_aggregation(True)


class DeformableAggregationFunction(Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = da.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        da.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


class DeformableAggregationWithDepthFunction(Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        num_depths,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = dad.deformable_aggregation_with_depth_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx._num_depths = num_depths
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        num_depths = ctx._num_depths
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        dad.deformable_aggregation_with_depth_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
            None,
        )


def deformable_aggregation_func(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
    depth_prob=None,
    depth=None,
):
    if depth_prob is not None and depth is not None:
        mc_ms_feat = torch.cat([mc_ms_feat, depth_prob], dim=-1)
        sampling_location = torch.cat([sampling_location, depth], dim=-1)
        return DeformableAggregationWithDepthFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            depth_prob.shape[-1],
        )
    else:
        return DeformableAggregationFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )


def feature_maps_format(feature_maps, inverse=False):
    bs, num_cams = feature_maps[0].shape[:2]
    if not inverse:
        spatial_shape = []
        scale_start_index = [0]

        col_feats = []
        for _, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            scale_start_index.append(
                feat.shape[-1] * feat.shape[-2] + scale_start_index[-1]
            )
            col_feats.append(
                torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
            )
        scale_start_index.pop()
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
        feature_maps = [
            col_feats,
            torch.tensor(
                spatial_shape,
                dtype=torch.int64,
                device=col_feats.device,
            ),
            torch.tensor(
                scale_start_index,
                dtype=torch.int64,
                device=col_feats.device,
            ),
        ]
    else:
        spatial_shape = feature_maps[1].int()
        split_size = (spatial_shape[:, 0] * spatial_shape[:, 1]).tolist()
        feature_maps = feature_maps[0].permute(0, 1, 3, 2)
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = feat.reshape(
                feat.shape[:3] + (spatial_shape[i, 0], spatial_shape[i, 1])
            )
    return feature_maps
