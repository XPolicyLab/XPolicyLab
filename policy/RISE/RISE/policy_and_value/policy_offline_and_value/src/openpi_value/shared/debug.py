import time
import jax
import jax.numpy as jnp
import jax.image
import dataclasses
import functools
from typing import Dict

# Original Version of Resize Function (without optimizations)
@dataclasses.dataclass(frozen=True)
class ResizeImagesOriginal:
    height: int
    width: int

    def __call__(self, data: Dict) -> Dict:
        # Original method that resizes images in a loop
        data["image"] = {
            k: resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()
        }
        return data

# Optimized Version of Resize Function (with batch processing)
@dataclasses.dataclass(frozen=True)
class ResizeImagesOptimized:
    height: int
    width: int

    def __call__(self, data: Dict) -> Dict:
        # Optimized method that resizes images in a batch
        data["image"] = {
            k: batch_resize(v, self.height, self.width) for k, v in data["image"].items()
        }
        return data

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def resize_with_pad(
    images: jax.Array,
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> jax.Array:
    """Resizes an image with padding, similar to tf.image.resize_with_pad."""
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension if not present
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = jax.image.resize(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), method=method
    )

    if images.dtype == jnp.uint8:
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    padded_images = jnp.zeros((images.shape[0], height, width, images.shape[3]), dtype=images.dtype)
    padded_images = padded_images.at[:, pad_h0:pad_h0+resized_height, pad_w0:pad_w0+resized_width, :].set(resized_images)

    if not has_batch_dim:
        padded_images = padded_images[0]

    return padded_images

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def batch_resize(images: jax.Array, height: int, width: int, method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR) -> jax.Array:
    """Resize a batch of images in parallel."""
    return jax.vmap(resize_with_pad, in_axes=(0, None, None, None))(images, height, width, method)

# Timing comparison function
def compare_resize_methods():
    # Test data: 10 images of size 256x256 with 3 color channels
    data = {
        "image": {
            f"img{i}": jnp.ones((1, 256, 256, 3), dtype=jnp.uint8) for i in range(10)
        }
    }

    # Measure time for the original method
    original_resize_fn = ResizeImagesOriginal(height=224, width=224)
    start_time = time.time()
    original_resized_data = original_resize_fn(data)
    original_duration = time.time() - start_time
    print(f"Original resizing time: {original_duration:.4f} seconds")

    # Measure time for the optimized method
    optimized_resize_fn = ResizeImagesOptimized(height=224, width=224)
    start_time = time.time()
    optimized_resized_data = optimized_resize_fn(data)
    optimized_duration = time.time() - start_time

    import pdb; pdb.set_trace()
    print(f"Optimized resizing time: {optimized_duration:.4f} seconds")

# Run the comparison
compare_resize_methods()
