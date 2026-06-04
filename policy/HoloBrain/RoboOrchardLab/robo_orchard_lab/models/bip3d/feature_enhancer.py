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
from typing import List, Union

import torch
from torch import Tensor, nn

from robo_orchard_lab.models.bip3d.layers import (
    DeformableDetrTransformerEncoderLayer,
    DetrTransformerEncoderLayer,
    SinePositionalEncoding,
    SingleScaleBiAttentionBlock,
)
from robo_orchard_lab.models.bip3d.utils import deformable_format


def get_text_sine_pos_embed(
    pos_tensor: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """Generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. For example,
            input tensor is [x,y], the results will be [pos(y), pos(x)].
            Defaults to True.

    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack(
            (sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        return sin_x

    pos_res = [
        sine_func(x)
        for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    ]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def get_encoder_reference_points(
    spatial_shapes: Tensor,
    valid_ratios: Tensor,
    device: Union[torch.device, str],
) -> Tensor:
    """Get the reference points used in encoder.

    Args:
        spatial_shapes (Tensor): Spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).
        valid_ratios (Tensor): The ratios of the valid width and the valid
            height relative to the width and the height of features in all
            levels, has shape (bs, num_levels, 2).
        device (obj:`device` or str): The device acquired by the
            `reference_points`.

    Returns:
        Tensor: Reference points used in decoder, has shape (bs, length,
        num_levels, 2).
    """

    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):  # noqa: N806
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=torch.float32, device=device
            ),
            torch.linspace(
                0.5, W - 0.5, W, dtype=torch.float32, device=device
            ),
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # [bs, sum(hw), num_level, 2]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


class TextImageDeformable2DEnhancer(nn.Module):
    def __init__(
        self,
        num_layers,
        text_img_attn_block: dict,
        img_attn_block: dict,
        text_attn_block: dict,
        positional_encoding: dict,
        embed_dims=256,
        num_feature_levels=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.embed_dims = embed_dims
        self.positional_encoding = positional_encoding
        self.text_img_attn_blocks = nn.ModuleList()
        self.img_attn_blocks = nn.ModuleList()
        self.text_attn_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.text_img_attn_blocks.append(
                SingleScaleBiAttentionBlock(**text_img_attn_block)
            )
            self.img_attn_blocks.append(
                DeformableDetrTransformerEncoderLayer(**img_attn_block)
            )
            self.text_attn_blocks.append(
                DetrTransformerEncoderLayer(**text_attn_block)
            )
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding
        )
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )

    def forward(
        self,
        feature_maps: List[Tensor],
        text_dict: dict,
        **kwargs,
    ):
        with_cams = feature_maps[0].dim() == 5
        if with_cams:
            bs, num_cams = feature_maps[0].shape[:2]
            feature_maps = [x.flatten(0, 1) for x in feature_maps]
        else:
            bs = feature_maps[0].shape[0]
            num_cams = 1
        pos_2d = self.get_2d_position_embed(feature_maps)
        feature_2d, spatial_shapes, level_start_index = deformable_format(
            feature_maps
        )

        reference_points = get_encoder_reference_points(
            spatial_shapes,
            valid_ratios=feature_2d.new_ones(
                [bs * num_cams, self.num_feature_levels, 2]
            ),
            device=feature_2d.device,
        )

        text_feature = text_dict["embedded"]
        pos_text = get_text_sine_pos_embed(
            text_dict["position_ids"][..., None],
            num_pos_feats=self.embed_dims,
            exchange_xy=False,
        )

        for layer_id in range(self.num_layers):
            feature_2d_fused = feature_2d[:, level_start_index[-1] :]
            if with_cams:
                feature_2d_fused = feature_2d_fused.unflatten(
                    0, (bs, num_cams)
                )
                feature_2d_fused = feature_2d_fused.flatten(1, 2)
            feature_2d_fused, text_feature = self.text_img_attn_blocks[
                layer_id
            ](
                feature_2d_fused,
                text_feature,
                attention_mask_l=text_dict["text_token_mask"],
            )
            if with_cams:
                feature_2d_fused = feature_2d_fused.unflatten(
                    1, (num_cams, -1)
                )
                feature_2d_fused = feature_2d_fused.flatten(0, 1)
            feature_2d = torch.cat(
                [feature_2d[:, : level_start_index[-1]], feature_2d_fused],
                dim=1,
            )

            feature_2d = self.img_attn_blocks[layer_id](
                query=feature_2d,
                query_pos=pos_2d,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=None,
            )

            text_attn_mask = text_dict.get("masks")
            if text_attn_mask is not None:
                text_num_heads = self.text_attn_blocks[layer_id].self_attn_cfg[
                    "num_heads"
                ]
                text_attn_mask = ~text_attn_mask.repeat(text_num_heads, 1, 1)
            text_feature = self.text_attn_blocks[layer_id](
                query=text_feature,
                query_pos=pos_text,
                attn_mask=text_attn_mask,
                key_padding_mask=None,
            )
        feature_2d = deformable_format(
            feature_2d, spatial_shapes, batch_size=bs if with_cams else None
        )
        return feature_2d, text_feature

    def get_2d_position_embed(self, feature_maps):
        pos_2d = []
        for lvl, feat in enumerate(feature_maps):
            batch_size, c, h, w = feat.shape
            pos = self.positional_encoding(None, feat)
            pos = pos.view(batch_size, c, h * w).permute(0, 2, 1)
            pos = pos + self.level_embed[lvl]
            pos_2d.append(pos)
        pos_2d = torch.cat(pos_2d, 1)
        return pos_2d
