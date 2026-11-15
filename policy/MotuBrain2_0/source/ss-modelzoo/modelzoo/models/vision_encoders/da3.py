"""Frozen DA3 global patch features and shared RGB preparation."""

import json
from collections.abc import Sequence

import torch
import torch.nn as nn

from morph.convs import image as image_convs


class FrozenDA3Backbone(nn.Module):
    """Load only DA3's backbone and return normalized global patch features.

    Inputs are ImageNet-normalized RGB [B, views, 3, H, W]. Output keys are
    one-based transformer layer numbers; values have shape [B, views * patches, D].
    DA3 restores camera order through its intermediate-layer output API.
    """

    def __init__(self, config_path: str, checkpoint_path: str, layers: Sequence[int]):
        super().__init__()
        # Optional dependency: ordinary Motubrain training does not import DA3.
        from depth_anything_3.model.dinov2 import dinov2
        from safetensors import safe_open

        with open(config_path, encoding="utf-8") as config_file:
            net_config = json.load(config_file)["config"]["net"]
        net_config.pop("__object__")
        self.layers = tuple(layers)
        indices = [layer - 1 for layer in self.layers]
        alt_start = net_config["alt_start"]
        if not indices or len(set(indices)) != len(indices) or indices != sorted(indices):
            raise ValueError("DA3 layers must be a nonempty, sorted list of unique layer numbers")
        if alt_start < 0 or any(index < alt_start or index % 2 != 1 for index in indices):
            raise ValueError("DA3 feature layers must select global attention blocks")
        net_config.update(out_layers=indices, cat_token=False)
        # Loading a frozen encoder must not advance the trainer's RNG.
        with torch.random.fork_rng(devices=[]):
            self.backbone = dinov2.DinoV2(**net_config)
        if max(indices) >= len(self.backbone.pretrained.blocks):
            raise ValueError("DA3 feature layer exceeds backbone depth")
        self.feature_dim = self.backbone.pretrained.embed_dim
        prefix = "model.backbone."
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            state = {
                key.removeprefix(prefix): checkpoint.get_tensor(key)
                for key in checkpoint.keys()
                if key.startswith(prefix)
            }
        self.backbone.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenDA3Backbone":
        """Keep the pretrained feature extractor frozen and in evaluation mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        """Encode all cameras jointly, without running depth or camera heads."""
        if images.ndim != 5 or images.shape[2] != 3 or any(size % 14 for size in images.shape[-2:]):
            raise ValueError("DA3 requires [B, views, 3, H, W] RGB with H/W divisible by 14")
        features, _ = self.backbone(images)
        return {layer: patches.flatten(1, 2) for layer, (patches, _) in zip(self.layers, features, strict=True)}


def prepare_images(
    frames: Sequence[torch.Tensor],
    image_size: tuple[int, int],
    device: str | torch.device | int,
) -> torch.Tensor:
    """Prepare ordered RGB camera crops in [-1, 1] for DA3.

    Each input is CHW. Resize in FP32, then apply ImageNet normalization and
    convert to BF16. Camera selection/cropping belongs to the caller. The
    operation order preserves the training encoder's rounding behavior.
    """
    height, width = image_size
    images = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("DA3 camera crops must be RGB [3, H, W]")
        frame = frame.to(device=device, dtype=torch.float32, non_blocking=True)
        images.append(
            image_convs.resize(
                frame,
                height=height,
                width=width,
                mode="direct",
                interpolation="bilinear",
                spatial_axes=(-2, -1),
            )
        )
    rgb = (torch.stack(images) + 1) * 0.5
    mean = rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return ((rgb - mean) / std).to(dtype=torch.bfloat16)
