
from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image


class Normalize:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor:
        r"""Convert to tensor if needed and normalize by mean/std.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor: Normalized image or video tensor.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = transforms_F.to_tensor(video)  # division by 255 is applied in to_tensor()

        return transforms_F.normalize(tensor=data, mean=mean, std=std)
