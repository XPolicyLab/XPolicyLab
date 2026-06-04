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

import copy
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from robo_orchard_core.datatypes.dataclass import TensorToMixin
from torch import nn

from robo_orchard_lab.utils import as_sequence, build


class BaseDataPreprocessor(nn.Module):
    """BaseDataPreprocessor.

    A base class for preprocessing data, particularly images
    This class handles image normalization, image channel manipulation,
    and optional grid masking augmentation.

    Args:
        mean (Optional[Sequence[float]]): The mean values for normalization.
        std (Optional[Sequence[float]]): The standard deviation values for
            normalization.
        channel_dim (int): The dimension along which the channels are located.
        channel_flip (bool): Whether to flip the channel order.
        channel_order (Optional[Sequence[int]]): The desired order of channels.
        hwc_to_chw (bool): Whether to convert the data format from HWC to CHW.
        image_keys (Union[List[str], str]): The keys corresponding to the
            images.
        depth_keys (Union[List[str], str]): The keys corresponding to the
            depths.
        grid_mask_config (Optional[Dict]): Configuration for the grid mask.
        batch_transforms: transforms for batch data.
    """

    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        channel_dim: int = -1,
        channel_flip: bool = False,
        channel_order: Optional[Sequence[int]] = None,
        hwc_to_chw: bool = True,
        image_keys: Union[Sequence[str], str] = "imgs",
        depth_keys: Union[Sequence[str], str] = "depths",
        unsqueeze_depth_channel: bool = False,
        batch_transforms: Optional[Sequence[Union[Dict, Callable]]] = None,
    ):
        super().__init__()
        self.mean = torch.Tensor(mean) if mean is not None else None
        self.std = torch.Tensor(std) if std is not None else None
        self.channel_dim = channel_dim
        self.channel_flip = channel_flip
        self.channel_order = channel_order
        self.hwc_to_chw = hwc_to_chw
        self.image_keys = as_sequence(image_keys)
        self.depth_keys = as_sequence(depth_keys)
        self.unsqueeze_depth_channel = unsqueeze_depth_channel
        self.batch_transforms = nn.ModuleList(
            [build(x) for x in as_sequence(batch_transforms)]
        )

    def cast_data(self, data: Any, device: str | torch.device = "cuda") -> Any:
        """Recursively move supported payloads onto ``device``.

        Mappings and sequences are traversed recursively while preserving their
        container type. :class:`torch.Tensor` values are moved with
        :meth:`torch.Tensor.to`, and values implementing
        :class:`TensorToMixin` are delegated to ``to(device=...)`` so custom
        dataclass-like containers can manage their own nested tensor transfer.
        Strings, bytes, ``None``, and unsupported objects are returned
        unchanged.

        Args:
            data (Any): Arbitrary nested payload containing tensors,
                ``TensorToMixin`` values, mappings, sequences, or pass-through
                scalars.
            device (str | torch.device, optional): Target device for supported
                tensor-bearing values. Default is ``"cuda"``.

        Returns:
            Any: Payload mirrored to ``device`` where supported, with the input
                container structure preserved.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key], device) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, TensorToMixin):
            return data.to(device=device)
        elif isinstance(data, Sequence):
            return type(data)(
                self.cast_data(sample, device) for sample in data
            )
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def process_img(self, imgs):
        idx = [slice(None)] * imgs.dim()
        if self.channel_order is not None:
            idx[self.channel_dim] = self.channel_order
        elif self.channel_flip:
            idx[self.channel_dim] = [2, 1, 0]
        imgs = imgs[idx]

        if self.mean is not None:
            idx = [None] * imgs.dim()
            idx[self.channel_dim] = slice(None)
            imgs = imgs - self.mean.to(imgs)[idx]
        if self.std is not None:
            idx = [None] * imgs.dim()
            idx[self.channel_dim] = slice(None)
            imgs = imgs / self.std.to(imgs)[idx]

        # XHWC to XCWH
        if self.hwc_to_chw:
            dims = list(range(imgs.dim()))
            dims = dims[:-3] + dims[-1:] + dims[-3:-1]
            imgs = torch.permute(imgs, dims)
        return imgs.contiguous()

    def forward(self, data, device="cuda"):
        data = self.cast_data(data, device)

        for key in self.image_keys:
            if key not in data:
                continue
            data[f"origin_{key}"] = data[key]
            data[key] = self.process_img(data[key])

        if self.hwc_to_chw:
            for key in self.depth_keys:
                if key not in data:
                    continue
                if self.unsqueeze_depth_channel:
                    data[key] = data[key][..., None]
                dims = list(range(data[key].dim()))
                dims = dims[:-3] + dims[-1:] + dims[-3:-1]
                data[key] = torch.permute(data[key], dims)

        data["img_mean"] = copy.deepcopy(self.mean)
        data["img_std"] = copy.deepcopy(self.std)

        for transform in self.batch_transforms:
            if transform is not None:
                data = transform(data)
        return data


class GridMask(nn.Module):
    def __init__(
        self,
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.7,
        apply_grid_mask_keys=("imgs", "depths"),
        image_keys="imgs",
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.apply_grid_mask_keys = apply_grid_mask_keys
        self.image_keys = image_keys

    def forward(self, data):
        if not self.training:
            return data
        for key in self.apply_grid_mask_keys:
            if key not in data:
                continue
            value = data[key]
            if value.dim() > 4:
                shape = value.shape
                value = value.flatten(end_dim=-4)
            else:
                shape = None
            if key in self.image_keys:
                mean, std = data.get("img_mean"), data.get("img_std")
                if mean is not None:
                    offset = -mean
                else:
                    offset = 0
                if std is not None:
                    offset = offset / std
                if isinstance(offset, torch.Tensor):
                    offset = (-mean / std)[:, None, None].to(value)
                else:
                    offset = None
            else:
                offset = None

            value = self._apply_grid_mask(value, offset=offset)
            if shape is not None:
                value = value.reshape(shape)
            data[key] = value
        return data

    def _apply_grid_mask(self, x, offset=None):
        if np.random.rand() > self.prob:
            return x
        n, c, h, w = x.size()
        x = x.reshape(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        ll = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + ll, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + ll, ww)
                mask[:, s:t] *= 0

        r = np.random.uniform(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.tensor(mask).to(x)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if offset is not None:
            x = x.view(n, c, h, w)
            mask = mask.view(n, c, h, w)
            x = x * mask + offset * (1 - mask)
            return x
        elif self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .cuda()
            )
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)
