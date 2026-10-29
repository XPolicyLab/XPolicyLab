"""Image normalization for policy adapters."""

import torch


def rgb_from_uint8(image: torch.Tensor) -> torch.Tensor:
    """Convert an HWC uint8 RGB image to float32 CHW in [-1, 1]."""
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "Expected HWC uint8 RGB; BGR conversion is the caller's responsibility."
        )
    return image.permute(2, 0, 1).float().div(127.5).sub(1.0)


def normalize_depth(
    image: torch.Tensor, *, minimum: float, maximum: float
) -> torch.Tensor:
    """Normalize HW metric depth using the checkpoint's range and units.

    No automatic mm/m conversion, invalid-pixel filling or depth refinement is
    performed. Those choices belong to the dataset and robot calibration.
    """
    if image.ndim != 2 or not torch.isfinite(image).all() or not maximum > minimum:
        raise ValueError("Expected finite HW depth and an increasing depth range.")
    return (
        (image.float().clamp(minimum, maximum) - minimum) / (maximum - minimum) * 2 - 1
    )[None]
