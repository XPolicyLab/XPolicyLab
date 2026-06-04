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
#
# This file was originally copied from the [mmcv] repository:
# https://github.com/open-mmlab/mmcv
# Modifications have been made to fit the needs of this project.

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from robo_orchard_lab.utils.build import build

__all__ = ["LayerScale", "MultiheadAttention", "FFN", "MLP"]


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        inplace (bool): Whether performs operation in-place.
            Default: `False`.
        data_format (Literal["channels_first", "channels_last"]): The input data format,
            could be "channels_last" or "channels_first", representing (B, C, H, W) and
            (B, N, C) format data respectively. Default: "channels_last".
        scale (float): Initial value of scale factor. Default: 1.0
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: Literal[
            "channels_first", "channels_last"
        ] = "channels_last",  # noqa: E501
        scale: float = 1e-5,
    ):
        super().__init__()
        assert data_format in (
            "channels_last",
            "channels_first",
        ), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> Tensor:
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim). Default to False.
    """  # noqa: E501

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: dict | None = None,
        batch_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        if dropout_layer is None:
            dropout_layer = dict(type="Dropout", drop_prob=0.0)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, attn_drop, **kwargs
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        identity: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`.

        kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor | None): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor | None): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor | None): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor | None): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor | None): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor | None): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor | None): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims]
                if self.batch_first is False, else [bs, num_queries embed_dims].
        """  # noqa: E501

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=None,
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        assert num_fcs >= 2, (
            f"num_fcs should be no less than 2. got {num_fcs}."
        )
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        if act_cfg is None:
            act_cfg = dict(type=nn.ReLU, inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    build(act_cfg),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class MLP(nn.Module):
    """MLP layer.

    Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim], strict=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).

        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
