import json
import pickle
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from giga_datasets import image_utils


NormMode = Literal["minmax", "zscore"]
RefMode = Literal["cat_width", "three_views"]


def pad_t5_embedding(t5_embedding: torch.Tensor, target_len: int) -> torch.Tensor:
    if t5_embedding.ndim != 2:
        raise ValueError(f"t5_embedding must be 2D [seq, dim], got {t5_embedding.shape}")
    if t5_embedding.shape[0] >= target_len:
        return t5_embedding[:target_len]
    return torch.nn.functional.pad(t5_embedding, (0, 0, 0, target_len - t5_embedding.shape[0]), value=0)


def load_t5_embedding_from_pkl(pkl_path: str, target_len: int = 32) -> torch.Tensor:
    with open(pkl_path, "rb") as f:
        t5_embedding = torch.load(f)
    if not isinstance(t5_embedding, torch.Tensor):
        t5_embedding = torch.as_tensor(t5_embedding)
    return pad_t5_embedding(t5_embedding, target_len=target_len)


def load_stats(stats_dict_path: str) -> Dict:
    with open(stats_dict_path, "r") as f:
        return json.load(f)


def tensor_chw01_to_pil_rgb(image_chw: torch.Tensor) -> Image.Image:
    if image_chw.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {image_chw.shape}")
    image = image_chw.detach().cpu()
    if image.dtype != torch.float32:
        image = image.float()
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return PIL.Image.fromarray(image)


def center_crop_resize_to_multiple(image: Image.Image, dst_size: Tuple[int, int], multiple: int = 64) -> Image.Image:
    dst_width, dst_height = image_utils.get_image_size(image.size, dst_size, mode="area", multiple=multiple)
    width, height = image.size
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
    image = F.crop(image, y1, x1, dst_height, dst_width)
    return image


def random_crop_resize_to_multiple(
    image: Image.Image,
    dst_size: Tuple[int, int],
    multiple: int = 64,
    generator: Optional[np.random.Generator] = None,
) -> Image.Image:
    if generator is None:
        generator = np.random.default_rng()
    dst_width, dst_height = image_utils.get_image_size(image.size, dst_size, mode="area", multiple=multiple)
    width, height = image.size
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    x1 = int(generator.integers(0, max(1, new_width - dst_width + 1)))
    y1 = int(generator.integers(0, max(1, new_height - dst_height + 1)))
    image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
    image = F.crop(image, y1, x1, dst_height, dst_width)
    return image


def build_ref_image(
    images: Dict[str, torch.Tensor],
    dst_size: Tuple[int, int],
    crop_mode: Literal["center", "random"] = "center",
) -> Image.Image:

    high = tensor_chw01_to_pil_rgb(images["observation.images.cam_high"])
    left = tensor_chw01_to_pil_rgb(images["observation.images.cam_left_wrist"])
    right = tensor_chw01_to_pil_rgb(images["observation.images.cam_right_wrist"])
    dst_width, dst_height = dst_size
    single_w = dst_width // 3
    single_dst = (single_w, dst_height)
    if crop_mode == "random":
        proc = lambda im: random_crop_resize_to_multiple(im, single_dst, multiple=64)
    else:
        proc = lambda im: center_crop_resize_to_multiple(im, single_dst, multiple=64)
    imgs = [proc(im) for im in [high, left, right]]
    out = Image.new("RGB", (single_w * len(imgs), dst_height))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += single_w
    return out


@dataclass(frozen=True)
class NormalizationTensors:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    state_min: torch.Tensor
    state_max: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor


def extract_normalization_tensors(stats: Dict, device: torch.device, state_dim: int, action_dim: int) -> NormalizationTensors:
    state_mean = torch.tensor(stats["norm_stats"]["observation.state"]["mean"])[..., :state_dim].to(device=device)
    state_std = torch.tensor(stats["norm_stats"]["observation.state"]["std"])[..., :state_dim].to(device=device)
    state_min = torch.tensor(stats["norm_stats"]["observation.state"]["min"])[..., :state_dim].to(device=device)
    state_max = torch.tensor(stats["norm_stats"]["observation.state"]["max"])[..., :state_dim].to(device=device)

    action_mean = torch.tensor(stats["norm_stats"]["action"]["mean"][:action_dim])[..., :action_dim].to(device=device)
    action_std = torch.tensor(stats["norm_stats"]["action"]["std"][:action_dim])[..., :action_dim].to(device=device)
    action_min = torch.tensor(stats["norm_stats"]["action"]["min"][:action_dim])[..., :action_dim].to(device=device)
    action_max = torch.tensor(stats["norm_stats"]["action"]["max"][:action_dim])[..., :action_dim].to(device=device)

    return NormalizationTensors(
        state_mean=state_mean,
        state_std=state_std,
        state_min=state_min,
        state_max=state_max,
        action_mean=action_mean,
        action_std=action_std,
        action_min=action_min,
        action_max=action_max,
    )


def normalize_state(state: torch.Tensor, stats: NormalizationTensors, mode: NormMode) -> torch.Tensor:
    eps = 1e-8
    state = state[..., : stats.state_mean.shape[-1]]
    if mode == "minmax":
        state_range = stats.state_max - stats.state_min + eps
        norm = ((state - stats.state_min) / state_range) * 2 - 1
        return norm.clamp(-1, 1)
    if mode == "zscore":
        return (state - stats.state_mean) / stats.state_std.clamp_min(eps)
    raise ValueError(f"unknown mode: {mode}")


def denormalize_action(action: torch.Tensor, stats: NormalizationTensors, mode: NormMode) -> torch.Tensor:
    eps = 1e-8
    action = action[..., : stats.action_mean.shape[-1]]
    if mode == "minmax":
        action_range = stats.action_max - stats.action_min + eps
        return ((action + 1) / 2) * action_range + stats.action_min
    if mode == "zscore":
        return action * stats.action_std.clamp_min(eps) + stats.action_mean
    raise ValueError(f"unknown mode: {mode}")


def add_state_to_action(action: torch.Tensor, state: torch.Tensor, action_chunk: int, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=action.device)
    if mask.numel() != action.shape[-1]:
        raise ValueError(f"mask length ({mask.numel()}) must match action_dim ({action.shape[-1]})")
    state = state[..., : action.shape[-1]]
    state_rep = state.repeat(action_chunk, 1)
    return action + state_rep * mask
