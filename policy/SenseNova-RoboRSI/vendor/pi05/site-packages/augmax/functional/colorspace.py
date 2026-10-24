import jax
import jax.numpy as jnp

def identity(value):
    return value

def to_grayscale(pixel):
    pixel = jnp.broadcast_to(pixel.mean(axis=-1, keepdims=True), pixel.shape)
    return pixel

def adjust_brightness(value, brightness, invert=False):
    if not invert:
        return jnp.where(brightness < 0.0,
            value * (1.0 + brightness),
            value * (1.0 - brightness) + brightness
        )
    else:
        return jnp.where(brightness < 0.0,
            value / (1.0 + brightness),
            (value - brightness) / (1.0 - brightness)
        )

def adjust_contrast(value, contrast, invert=False):
    if invert:
        contrast = -contrast
    slant = jnp.tan((contrast + 1.0) * (jnp.pi / 4))
    # Invertible contrast transform
    # See https://www.desmos.com/calculator/yxnm5siet4
    p1 = (slant - jnp.square(slant)) / (2 * (1 - jnp.square(slant)))
    p2 = 1 - p1

    value = jnp.piecewise(value, [value < p1, value > p2], [
        lambda x: x / slant,
        lambda x: (x / slant) + 1 - 1 / slant,
        lambda x: slant * (x - 0.5) + 0.5
    ])
    return value
