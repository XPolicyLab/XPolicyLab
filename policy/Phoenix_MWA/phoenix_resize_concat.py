"""Direct-resize RGB transport for the isolated three-view deployment variant."""

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor


CAMERAS = ('head_camera', 'left_camera', 'right_camera')
INPUT_GEOMETRY = 'direct_bicubic_resize_256_per_view_horizontal_concat'


def resize_concat_rgb(observation):
    """Return [B=1,C=3,H=256,W=768], in head/left/right order."""
    views = []
    for camera in CAMERAS:
        rgb = np.asarray(observation['observation'][camera]['rgb'])
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[-1] != 3
            or min(rgb.shape[:2]) <= 0
        ):
            raise ValueError('Expected nonempty uint8 HWC RGB: ' + camera)
        image = resize(
            Image.fromarray(rgb),
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        views.append(to_tensor(image).mul(2).sub(1))
    panorama = torch.cat(views, dim=-1).unsqueeze(0)
    if panorama.shape != (1, 3, 256, 768) or not torch.isfinite(panorama).all():
        raise ValueError('Invalid direct-resize three-view panorama')
    return panorama


def current_rgb(observation):
    """Preserve camera boundaries required by the released Joint96 encoder.

    Horizontal concatenation is a transport layout, not whole-image encoding.
    The existing WAN and DINO/STVAE encoders retain their per-view semantics.
    """
    panorama = resize_concat_rgb(observation)
    return torch.stack(panorama.split(256, dim=-1), dim=1).contiguous()
