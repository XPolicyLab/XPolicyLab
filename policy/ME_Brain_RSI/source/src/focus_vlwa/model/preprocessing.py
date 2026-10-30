import dataclasses
import logging
from collections.abc import Sequence

import torch

from focus_vlwa.data import image_tools

logger = logging.getLogger("focus_vlwa")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def observation_image_keys(images, image_keys: Sequence[str] = IMAGE_KEYS) -> list[str]:
    """Cameras first, then any extra views (instruction sheets, …)."""
    keys = [k for k in image_keys]
    for k in images:
        if k not in keys:
            keys.append(k)
    return keys


def _augment_images(image, key):
    """Apply one shared spatial/color transform to a group of related frames."""
    # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
    image = image / 2.0 + 0.5

    # Apply PyTorch-based augmentations
    if "wrist" not in key:
        # Geometric augmentations for non-wrist cameras
        height, width = image.shape[1:3]

        # Random crop and resize
        crop_height = int(height * 0.95)
        crop_width = int(width * 0.95)

        # Random crop
        max_h = height - crop_height
        max_w = width - crop_width
        if max_h > 0 and max_w > 0:
            # Use tensor operations instead of .item() for torch.compile compatibility
            start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
            start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
            image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

        # Resize back to original size
        image = torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

        # Random rotation (small angles)
        # Use tensor operations instead of .item() for torch.compile compatibility
        angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
        if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
            # Convert to radians
            angle_rad = angle * torch.pi / 180.0

            # Create rotation matrix
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

            # Apply rotation using grid_sample
            grid_x = torch.linspace(-1, 1, width, device=image.device)
            grid_y = torch.linspace(-1, 1, height, device=image.device)

            # Create meshgrid
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

            # Expand to batch dimension
            grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

            # Apply rotation transformation
            grid_x_rot = grid_x * cos_a - grid_y * sin_a
            grid_y_rot = grid_x * sin_a + grid_y * cos_a

            # Stack and reshape for grid_sample
            grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

            image = torch.nn.functional.grid_sample(
                image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    # Color augmentations for all cameras
    # Random brightness
    # Use tensor operations instead of .item() for torch.compile compatibility
    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
    image = image * brightness_factor

    # Random contrast
    # Use tensor operations instead of .item() for torch.compile compatibility
    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast_factor + mean

    # Random saturation (convert to HSV, modify S, convert back)
    # For simplicity, we'll just apply a random scaling to the color channels
    # Use tensor operations instead of .item() for torch.compile compatibility
    saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
    gray = image.mean(dim=-1, keepdim=True)
    image = gray + (image - gray) * saturation_factor

    # Clamp values to [0, 1]
    image = torch.clamp(image, 0, 1)

    # Back to [-1, 1]
    image = image * 2.0 - 1.0
    return image


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    history = getattr(observation, "history_images", None)
    for key in observation_image_keys(observation.images, image_keys):
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            if key == "base_0_rgb" and history is not None and history.ndim == 5:
                # One random transform per sample, shared by the current head and all history slots.
                groups = [_augment_images(torch.cat((image[b:b + 1], history[b]), dim=0), key)
                          for b in range(image.shape[0])]
                image = torch.stack([group[0] for group in groups])
                history = torch.stack([group[1:] for group in groups])
            else:
                image = _augment_images(image, key)

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    return dataclasses.replace(observation, images=out_images, image_masks=out_masks, history_images=history)
