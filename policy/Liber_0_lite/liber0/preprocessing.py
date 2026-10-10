"""Image packing and action normalization."""
import numpy as np
import torch
from torchvision.transforms import Resize, functional as TF


def pack_images(images, resize, crop_scale):
    if len(images) != 4 or not 0 < crop_scale <= 1:
        raise ValueError('Expected four RGB images and a crop scale in (0, 1].')
    cameras = []
    for image in images:
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        tensor = Resize(resize)(tensor.to(torch.float32) / 255.0)
        height, width = tensor.shape[-2:]
        if crop_scale != 1.0:
            crop_h = max(1, min(height, int(round(height * crop_scale))))
            crop_w = max(1, min(width, int(round(width * crop_scale))))
            tensor = TF.resized_crop(tensor, (height - crop_h) // 2, (width - crop_w) // 2,
                                     crop_h, crop_w, [height, width],
                                     interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        cameras.append(TF.resize(tensor, [144, 192], interpolation=TF.InterpolationMode.BILINEAR, antialias=True))
    head, left, right, cue = cameras
    return (torch.cat([torch.cat([head, cue], -1), torch.cat([left, right], -1)], -2) - 0.5) / 0.5


def build_norm(stats):
    result = {}
    for name in ('state', 'action'):
        entry = stats[name]['default']
        mean = torch.as_tensor(np.asarray(entry['global_mean'], dtype=np.float32).reshape(-1))
        std = torch.as_tensor(np.asarray(entry['global_std'], dtype=np.float32).reshape(-1))
        result[name + '_scale'] = 1.0 / (std + 1e-8)
        result[name + '_offset'] = -mean / (std + 1e-8)
    return result


def denormalize(action, scale, offset):
    safe = scale.abs() > 1e-12
    inv_scale = torch.where(safe, 1.0 / scale, torch.zeros_like(scale))
    out = (action - offset) * inv_scale
    return torch.where(safe.expand_as(out), out, (-offset).expand_as(out))
