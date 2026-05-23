# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from policy_heads.util.misc import is_main_process, NestedTensor
from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    This implementation is a copy-paste from torchvision.misc.ops with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101] produce NaNs.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # remove num_batches_tracked from state_dict if present
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """
        Forward pass for the frozen batch normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Base class for backbone networks.

    Args:
        backbone: The backbone model.
        train_backbone: Whether to train the backbone.
        num_channels: Number of output channels.
        return_interm_layers: Whether to return intermediate layers.
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # determine which layers to return
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        """
        Forward pass for the backbone.

        Args:
            tensor: Input tensor.

        Returns:
            Dictionary of feature maps from the specified layers.
        """
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.

    Args:
        name: Name of the ResNet model.
        train_backbone: Whether to train the backbone.
        return_interm_layers: Whether to return intermediate layers.
        dilation: Whether to use dilation in the last block.
    """
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)  # use pretrained model
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """
    Combines a backbone and a position encoding module.

    Args:
        backbone: The backbone model.
        position_embedding: The position encoding module.
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        Forward pass for the joiner.

        Args:
            tensor_list: NestedTensor containing input data and mask.

        Returns:
            Tuple of feature maps and position encodings.
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    """
    Builds the backbone model with position encoding.

    Args:
        args: Arguments containing configuration for the backbone.

    Returns:
        A model combining the backbone and position encoding.
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
