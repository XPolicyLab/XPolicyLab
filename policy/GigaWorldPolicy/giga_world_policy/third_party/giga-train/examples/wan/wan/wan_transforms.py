from typing import Any, Dict

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from giga_train import TRANSFORMS


@TRANSFORMS.register
class WanTransform:
    """Video-to-tensor preprocessing for WAN.

    Selects `num_frames` from the input video, resizes to `(height, width)`,
    converts to CHW tensors, and scales to [-1, 1].
    """

    def __init__(self, num_frames: int, height: int, width: int):
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing to a single sample.

        Args:
            data_dict (dict): Contains `video` (Decord-like) and `prompt` (str).

        Returns:
            dict: New dict with `images` (Tensor[T, C, H, W]) and `prompt`.
        """
        video = data_dict['video']
        prompt = data_dict['prompt']
        indexes = np.linspace(0, len(video) - 1, self.num_frames, dtype=int)
        images = video.get_batch(indexes)
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images.asnumpy())
        images = images.permute(0, 3, 1, 2).contiguous()
        images = F.resize(images, (self.height, self.width), InterpolationMode.BILINEAR)
        images = images / 255.0 * 2.0 - 1.0
        new_data_dict: Dict[str, Any] = dict(images=images, prompt=prompt)
        return new_data_dict
