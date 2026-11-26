# Copyright 2024 Konrad Heidler
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

def apply_perspective(xy: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    xyz = jnp.concatenate([xy, jnp.ones([1, *xy.shape[1:]])])
    xyz = jnp.tensordot(M, xyz, axes=1)
    yx, z = jnp.split(xyz, [2])
    return yx / z


def resample_image(image: jnp.ndarray, coordinates: jnp.ndarray, order: int=1, mode: str='nearest', cval: Any=0):
    H, W, *C = image.shape
    D, *S_out = coordinates.shape
    assert D == 2, f'Expected first dimension of coordinates array to have size 2, got {coordinates.shape}'
    coordinates = coordinates.reshape(2, -1)

    def resample_channel(channel: jnp.ndarray):
        return jnd.map_coordinates(channel, coordinates, order=order, mode=mode, cval=cval)

    if image.ndim == 2:
        resampled = resample_channel(image)
    elif image.ndim == 3:
        resampled = jax.vmap(resample_channel, in_axes=-1, out_axes=-1)(image)
    else:
        raise ValueError(f"Cannot resample image with {image.ndim} dimensions")

    resampled = resampled.reshape(*S_out, *C)

    return resampled


def log_uniform(key, shape=(), dtype=jnp.float32, minval=0.5, maxval=2.0):
    logmin = jnp.log(minval)
    logmax = jnp.log(maxval)

    sample = jax.random.uniform(key, shape=shape, minval=logmin, maxval=logmax)

    return  jnp.exp(sample)


def rgb_to_hsv(pixel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cf. https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae
    """
    value = jnp.max(pixel)
    range = value - jnp.min(pixel)
    argmax = jnp.argmax(pixel)
    second = jnp.mod(argmax + 1, 3)
    third  = jnp.mod(argmax + 2, 3)
    hue = jnp.where(range == 0.0,
        0.0,
        (2 * argmax + (pixel[second] - pixel[third]) / range) / 6
    )
    saturation = jnp.where(value == 0,
        0.0,
        range / value
    )

    return hue, saturation, value


def hsv_to_rgb(hue: jnp.ndarray, saturation: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """
    cf. https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae
    """
    n = jnp.array([5, 3, 1])
    k = jnp.mod(n + hue * 6, 6)
    f = value - value * saturation * jnp.maximum(0, jnp.minimum(jnp.minimum(k, 4-k), 1))
    return f
