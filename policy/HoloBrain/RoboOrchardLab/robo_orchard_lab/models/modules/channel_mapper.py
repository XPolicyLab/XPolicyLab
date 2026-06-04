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

from typing import List, Optional, Tuple, Union

from torch import Tensor, nn

from robo_orchard_lab.utils.build import build


class ChannelMapper(nn.Module):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        num_outs (int | None): Number of output feature maps. There would
            be extra_convs when num_outs larger than the length of in_channels.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_type: str = "Conv2d",
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        bias: Union[bool, str] = "auto",
        num_outs: int | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                nn.Sequential(
                    getattr(nn, conv_type)(
                        in_channel,
                        out_channels,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    (
                        build(norm_cfg, out_channels)
                        if norm_cfg is not None
                        else nn.Identity()
                    ),
                    build(act_cfg) if act_cfg is not None else nn.Identity(),
                )
            )
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    nn.Sequential(
                        getattr(nn, conv_type)(
                            in_channel,
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                            bias=bias,
                        ),
                        (
                            build(norm_cfg, out_channels)
                            if norm_cfg is not None
                            else nn.Identity()
                        ),
                        (
                            build(act_cfg)
                            if act_cfg is not None
                            else nn.Identity()
                        ),
                    )
                )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
