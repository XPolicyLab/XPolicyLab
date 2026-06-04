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

# This file was originally copied from the [mmdetection] repository:
# https://github.com/open-mmlab/mmdetection
# Modifications have been made to fit the needs of this project.

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from transformers.models.deformable_detr.modeling_deformable_detr import (
        MultiScaleDeformableAttentionFunction,
        load_cuda_kernels,
    )
except ImportError:
    from transformers.models.deformable_detr.modeling_deformable_detr import (
        MultiScaleDeformableAttention as MultiScaleDeformableAttentionFunction,
    )

    load_cuda_kernels = None

from robo_orchard_lab.models.layers.transformer_layers import (
    FFN,
    MultiheadAttention,
)
from robo_orchard_lab.utils.build import build

__all__ = [
    "MultiScaleDeformableAttention",
    "SinePositionalEncoding",
    "DetrTransformerEncoderLayer",
    "DeformableDetrTransformerEncoderLayer",
    "BiMultiHeadAttention",
    "SingleScaleBiAttentionBlock",
]

logger = logging.getLogger(__name__)


class MultiScaleDeformableAttention(nn.Module):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim).
            Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg: Optional[dict] = None,
        value_proj_ratio: float = 1.0,
    ):
        super().__init__()
        if load_cuda_kernels is not None:
            logger.warning("Deprecated, please use `transformers` >= 4.57.1.")
            load_cuda_kernels()
        else:
            self.func = MultiScaleDeformableAttentionFunction()

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(
                        n, type(n)
                    )
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            logger.warning(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        device = next(self.parameters()).device
        default_dtype = torch.get_default_dtype()
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32, device=device
        ).to(default_dtype) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        # xavier_init(self.value_proj, distribution='uniform', bias=0.)
        # xavier_init(self.output_proj, distribution='uniform', bias=0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
        self._is_init = True

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        if load_cuda_kernels is not None:
            output = MultiScaleDeformableAttentionFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = self.func(
                value,
                spatial_shapes,
                spatial_shapes.tolist(),
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()  # noqa: N806
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape  # noqa: N806
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device
            )
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device
            )
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, -1:, :] + self.eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, -1:] + self.eps)
                * self.scale
            )
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DetrTransformerEncoderLayer(nn.Module):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
    """

    def __init__(
        self,
        self_attn_cfg: Optional[dict] = None,
        ffn_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if self_attn_cfg is None:
            self_attn_cfg = dict(embed_dims=256, num_heads=8, dropout=0.0)
        if ffn_cfg is None:
            ffn_cfg = dict(
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=None,
            )

        self.self_attn_cfg = self_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert self.self_attn_cfg["batch_first"] is True

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        if self.norm_cfg is None:
            self.norm_cfg = dict(
                type=nn.LayerNorm, normalized_shape=self.embed_dims
            )
        norms_list = [build(self.norm_cfg) for _ in range(2)]
        self.norms = nn.ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        **kwargs,
    ):
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        if self.norm_cfg is None:
            self.norm_cfg = dict(
                type=nn.LayerNorm, normalized_shape=self.embed_dims
            )
        norms_list = [build(self.norm_cfg) for _ in range(2)]
        self.norms = nn.ModuleList(norms_list)


class BiMultiHeadAttention(nn.Module):
    """Bidirectional fusion Multi-Head Attention layer.

    Args:
        v_dim (int): The dimension of the vision input.
        l_dim (int): The dimension of the language input.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    MAX_CLAMP_VALUE = 50000

    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert self.head_dim * self.num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads "
            f"(got `embed_dim`: {self.embed_dim} "
            f"and `num_heads`: {self.num_heads})."
        )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(
        self,
        vision: Tensor,
        lang: Tensor,
        attention_mask_v: Optional[Tensor] = None,
        attention_mask_l: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, tgt_len, _ = vision.size()

        query_states = self.v_proj(vision) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(vision), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(
            *proj_shape
        )
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of "
                f"size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            # Do not increase -50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, min=-self.MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            # Do not increase 50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, max=self.MAX_CLAMP_VALUE)

        attn_weights_T = attn_weights.transpose(1, 2)  # noqa: N806
        attn_weights_l = (
            attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        )
        if self.clamp_min_for_underflow:
            # Do not increase -50000, data type half has quite limited range
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-self.MAX_CLAMP_VALUE
            )
        if self.clamp_max_for_overflow:
            # Do not increase 50000, data type half has quite limited range
            attn_weights_l = torch.clamp(
                attn_weights_l, max=self.MAX_CLAMP_VALUE
            )

        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :]
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert attention_mask_l.dim() == 2
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, -9e15
            )

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    "Attention mask should be of "
                    f"size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(
            attn_weights_v, p=self.dropout, training=self.training
        )
        attn_probs_l = F.dropout(
            attn_weights_l, p=self.dropout, training=self.training
        )

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output_v` should be of "
                f"size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
                f"but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (
            bsz * self.num_heads,
            src_len,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output_l` should be of size "
                f"{(bsz, self.num_heads, src_len, self.head_dim)}, "
                f"but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(
            bsz, self.num_heads, tgt_len, self.head_dim
        )
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(
            bsz, self.num_heads, src_len, self.head_dim
        )
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class SingleScaleBiAttentionBlock(nn.Module):
    """BiAttentionBlock Module.

    First, multi-level visual features are concat; Then the concat visual
    feature and lang feature are fused by attention; Finally the newly visual
    feature are split into multi levels.

    Args:
        v_dim (int): The dimension of the visual features.
        l_dim (int): The dimension of the language feature.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        init_values (float, optional):
            The initial value for the scaling parameter.
            Defaults to 1e-4.
    """

    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        init_values: float = 1e-4,
    ):
        super().__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # add layer scale for training stability
        self.gamma_v = nn.Parameter(
            init_values * torch.ones(v_dim), requires_grad=True
        )
        self.gamma_l = nn.Parameter(
            init_values * torch.ones(l_dim), requires_grad=True
        )

    def forward(
        self,
        visual: Tensor,
        lang: Tensor,
        attention_mask_v: Optional[Tensor] = None,
        attention_mask_l: Optional[Tensor] = None,
    ):
        """NN forward.

        Perform a single attention call between the visual and language
        inputs.

        Args:
            visual (Tensor): The visual input tensor.
            lang (Tensor): The language input tensor.
            attention_mask_v (Optional[Tensor]):
                An optional attention mask tensor for the visual input.
            attention_mask_l (Optional[Tensor]):
                An optional attention mask tensor for the language input.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the updated
                visual and language tensors after the attention call.
        """
        visual = self.layer_norm_v(visual)
        lang = self.layer_norm_l(lang)
        delta_v, delta_l = self.attn(
            visual,
            lang,
            attention_mask_v=attention_mask_v,
            attention_mask_l=attention_mask_l,
        )
        # visual, lang = visual + delta_v, l + delta_l
        visual = visual + self.gamma_v * delta_v
        lang = lang + self.gamma_l * delta_l
        return visual, lang
