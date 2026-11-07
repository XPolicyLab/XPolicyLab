import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as TF_F


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class CenterCropAspectResize(nn.Module):
    """Center-crop a video tensor to an aspect ratio, then resize it.

    The LeRobot loader supplies ``[T,C,H,W]`` tensors.  RoboColiseum's head
    and wrist cameras have different native geometries, so a direct resize
    would make training and benchmark inference disagree geometrically.  This
    transform establishes a common 4:3 camera contract before the views are
    composed into the model's T-shaped canvas.
    """

    def __init__(self, size, aspect_ratio: float = 4.0 / 3.0):
        super().__init__()
        if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
            raise ValueError(f"size must be [height,width], got {size!r}")
        if float(aspect_ratio) <= 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")
        self.size = [int(size[0]), int(size[1])]
        self.aspect_ratio = float(aspect_ratio)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"CenterCropAspectResize expects [T,C,H,W], got {tuple(x.shape)}")
        height, width = (int(value) for value in x.shape[-2:])
        current_ratio = width / height
        if current_ratio > self.aspect_ratio:
            crop_height = height
            crop_width = max(1, int(round(height * self.aspect_ratio)))
        else:
            crop_width = width
            crop_height = max(1, int(round(width / self.aspect_ratio)))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        x = TF_F.crop(x, top, left, crop_height, crop_width)
        return TF_F.resize(
            x,
            size=self.size,
            interpolation=TF_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
