import functools

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F  # noqa: N812

import openpi.shared.array_typing as at


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
@at.typecheck
def resize_with_pad(
    images: at.UInt8[at.Array, "*b h w c"] | at.Float[at.Array, "*b h w c"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.UInt8[at.Array, "*b {height} {width} c"] | at.Float[at.Array, "*b {height} {width} c"]:
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), method=method
    )
    if images.dtype == jnp.uint8:
        # round from float back to uint8
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0 if images.dtype == jnp.uint8 else -1.0,
    )

    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images


def replace_padding_0to1_torch(image: torch.Tensor,) -> torch.Tensor:
    """PyTorch version of replace_padding_0to1. 
    OpenPI requires images with 0 value paddings, while VGGT series requires 1 value paddings.
    Here it achieves this bounding-box based padding replacement.
    Args:
        image: Tensor of shape [*b, h, w, c]
    Returns:
        Padding-replaced tensor with same shape as input
    """
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True

    b, h, w, c = image.shape
    device = image.device

    nonzero_any = (image != 0).any(dim=-1)

    row_any = nonzero_any.any(dim=2)
    col_any = nonzero_any.any(dim=1)

    top = row_any.to(torch.float32).argmax(dim=1)
    bottom = h - 1 - row_any.flip(dims=[1]).to(torch.float32).argmax(dim=1)
    left = col_any.to(torch.float32).argmax(dim=1)
    right = w - 1 - col_any.flip(dims=[1]).to(torch.float32).argmax(dim=1)

    has_any = row_any.any(dim=1)
    top = torch.where(has_any, top, torch.zeros_like(top))
    bottom = torch.where(has_any, bottom, torch.full_like(bottom, h - 1))
    left = torch.where(has_any, left, torch.zeros_like(left))
    right = torch.where(has_any, right, torch.full_like(right, w - 1))

    rows = torch.arange(h, device=device).view(1, h, 1)
    cols = torch.arange(w, device=device).view(1, 1, w)
    top_v = top.view(b, 1, 1)
    bottom_v = bottom.view(b, 1, 1)
    left_v = left.view(b, 1, 1)
    right_v = right.view(b, 1, 1)

    row_mask = (rows >= top_v) & (rows <= bottom_v)
    col_mask = (cols >= left_v) & (cols <= right_v)
    inside_mask = row_mask & col_mask

    padding_mask = ~inside_mask

    pixel_zero = (image == 0).all(dim=-1)

    final_mask = padding_mask & pixel_zero

    if final_mask.any():
        mask_exp = final_mask.unsqueeze(-1).expand_as(image)
        one_t = torch.tensor(1, dtype=image.dtype, device=device)
        image = torch.where(mask_exp, one_t, image)

    # Handle all-zero value images
    if (image == 0).all():
        image = torch.ones_like(image)

    if single:
        image = image.squeeze(0)
    return image