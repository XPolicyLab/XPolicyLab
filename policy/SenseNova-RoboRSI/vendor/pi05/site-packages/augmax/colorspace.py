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
from abc import abstractmethod
from typing import List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import warnings

from .base import Transformation, BaseChain, InputType, same_type, PyTree, RNGKey
from .utils import log_uniform, rgb_to_hsv, hsv_to_rgb
from .functional import colorspace as F


class ColorspaceTransformation(Transformation):
    @abstractmethod
    def pixelwise(self, rng: RNGKey, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        return pixel

    def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree, invert=False) -> PyTree:
        op = partial(self.pixelwise, invert=invert)
        full_op = jax.jit(jax.vmap(jax.vmap(op, [None, 0], 0), [None, 1], 1))

        def transform_single(input, input_type):
            if same_type(input_type, InputType.IMAGE):
                return full_op(rng, input)
            else:
                return input

        return jax.tree_util.tree_map(transform_single, inputs, input_types)


class ColorspaceChain(ColorspaceTransformation, BaseChain):
    def __init__(self, *transforms: ColorspaceTransformation, input_types=None):
        super().__init__(input_types)
        self.transforms = transforms

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        N = len(self.transforms)
        subkeys = [None]*N if rng is None else jax.random.split(rng, N)

        transforms = self.transforms
        if invert:
            transforms = reversed(transforms)
            subkeys = reversed(subkeys)

        for transform, subkey in zip(transforms, subkeys):
            pixel = transform.pixelwise(subkey, pixel, invert=invert)
        return pixel


class ByteToFloat(ColorspaceTransformation):
    """Transforms images from uint8 representation (values 0-255)
    to normalized float representation (values 0.0-1.0)
    """
    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if invert:
            return jnp.clip(255.0 * pixel, 0, 255).astype(jnp.uint8)
        else:
            return pixel.astype(jnp.float32) / 255.0


class Normalize(ColorspaceTransformation):
    """Normalizes images using given coefficients using the mapping

    .. math::
        p_k \\longmapsto \\frac{p_k - \\mathtt{mean}_k}{\\mathtt{std}_k}

    Args:
        mean (jnp.ndarray): Mean values for each channel
        std (jnp.ndarray): Standard deviation for each channel
    """
    def __init__(self,
            mean: jnp.ndarray = np.array([0.485, 0.456, 0.406]),
            std: jnp.ndarray = np.array([0.229, 0.224, 0.225]),
            input_types=None
    ):
        super().__init__(input_types)
        self.mean = jnp.asarray(mean)
        self.std = jnp.asarray(std)

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if not invert:
            return (pixel - self.mean) / self.std
        else:
            return (pixel * self.std) + self.mean


class ChannelShuffle(ColorspaceTransformation):
    """Randomly shuffles an images channels.

    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        k1, k2 = jax.random.split(rng)
        do_apply = jax.random.bernoulli(k2, self.probability)
        if not invert:
            return jnp.where(do_apply,
                jax.random.permutation(k1, pixel),
                pixel
            )
        else:
            inv_permutation = jnp.argsort(jax.random.permutation(k1, pixel.shape[0]))
            return jnp.where(do_apply,
                pixel[inv_permutation],
                pixel
            )


class RandomGamma(ColorspaceTransformation):
    """Randomly adjusts the image gamma.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: Tuple[float, float]=(0.25, 4.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.range = range
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomGamma can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_gamma = log_uniform(k1, minval=self.range[0], maxval=self.range[1])
        gamma = jnp.where(jax.random.bernoulli(k2, self.probability), random_gamma, 1.0)

        if not invert:
            return jnp.power(pixel, gamma)
        else:
            return jnp.power(pixel, 1/gamma)


class RandomBrightness(ColorspaceTransformation):
    """Randomly adjusts the image brightness.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: Tuple[float, float] = (-1.0, 1.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.minval = range[0] / 2.0
        self.maxval = range[1] / 2.0
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomContrast can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_brightness = jax.random.uniform(k1, minval=self.minval, maxval=self.maxval)
        brightness = jnp.where(jax.random.bernoulli(k2, self.probability), random_brightness, 0.0)
        # cf. https://gitlab.gnome.org/GNOME/gimp/-/blob/master/app/operations/gimpoperationbrightnesscontrast.c

        return F.adjust_brightness(pixel, brightness, invert=invert)


class RandomContrast(ColorspaceTransformation):
    """Randomly adjusts the image contrast.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: Tuple[float, float] = (-1.0, 1.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.minval = range[0] / 2.0
        self.maxval = range[1] / 2.0
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomContrast can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_contrast = jax.random.uniform(k1, minval=self.minval, maxval=self.maxval)
        contrast = jnp.where(jax.random.bernoulli(k2, self.probability), random_contrast, 0.0)
        return F.adjust_contrast(pixel, contrast, invert=invert)


class ColorJitter(ColorspaceTransformation):
    """Randomly jitter the image colors.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            brightness: float = 0.1,
            contrast: float = 0.1,
            saturation: float = 0.1,
            hue: float = 0.1,
            p: float=0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = p

        self.keys_needed = sum(1 if val > 0 else 0
                for val in [brightness, contrast, saturation, hue])
        if p < 1:
            self.keys_needed += 1

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.shape != (3, ):
            raise ValueError(f"ColorJitter only supports RGB imagery for now, got {pixel.shape}")
        if pixel.dtype != jnp.float32:
            raise ValueError(f"ColorJitter can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")
        keys = jax.random.split(rng, self.keys_needed)
        hue, saturation, value = rgb_to_hsv(pixel)

        ops = ['brightness', 'contrast', 'hue', 'saturation']
        if invert:
            ops = reversed(ops)
            keys = reversed(keys)

        for op, key in zip(ops, keys):
            strength = getattr(self, op)
            if strength <= 0:
                continue
            amount = jax.random.uniform(key, minval=-strength, maxval=strength)
            if op == 'brightness':
                value = F.adjust_brightness(value, amount, invert=invert)
            elif op == 'contrast':
                value = F.adjust_contrast(value, amount, invert=invert)
            elif op == 'hue':
                if invert:
                    amount = -amount
                hue = hue + amount
            elif op == 'saturation':
                F.adjust_brightness(saturation, amount, invert=invert)

        transformed = hsv_to_rgb(hue, saturation, value)

        if self.probability < 1:
            do_apply = jax.random.bernoulli(rng, self.probability)
            transformed = jnp.where(do_apply, transformed, pixel)

        return transformed


class RandomGrayscale(ColorspaceTransformation):
    """Randomly converts the image to grayscale.
   
    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomGrayscale can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        if invert:
            warnings.warn("Trying to invert a Grayscale Filter, which is not invertible.")
            return pixel

        do_apply = jax.random.bernoulli(rng, self.probability)
        return jnp.where(do_apply,
                F.to_grayscale(pixel),
                pixel
        )


class RandomChannelGamma(ColorspaceTransformation):
    """Randomly adjusts each channel's gamma.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: Tuple[float, float]=(0.25, 4.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.range = range
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomGamma can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_gamma = log_uniform(k1, shape=pixel.shape, minval=self.range[0], maxval=self.range[1])
        gamma = jnp.where(jax.random.bernoulli(k2, self.probability), random_gamma, 1.0)

        if not invert:
            return jnp.power(pixel, gamma)
        else:
            return jnp.power(pixel, 1/gamma)


class Solarization(ColorspaceTransformation):
    """Randomly solarizes the image.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            threshold: float = 0.5,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.range = range
        self.threshold = threshold
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"Solarization can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        if invert:
            warnings.warn("Trying to invert a Solarization Filter, which is not invertible.")
            return pixel

        do_apply = jax.random.bernoulli(rng, self.probability)
        solarized = jnp.where((pixel > self.threshold) & do_apply,
                1.0 - pixel,
                pixel
        )
        return solarized


class ChannelDrop(ColorspaceTransformation):
    """Randomly drops a channelf from the image
   
    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray, invert=False) -> jnp.ndarray:
        k1, k2 = jax.random.split(rng)
        C, = pixel.shape
        do_apply = jax.random.bernoulli(k1, self.probability)
        apply_channel = jax.random.randint(k1, [], minval=0, maxval=C)

        return jnp.where(do_apply & (jnp.arange(C) == apply_channel), 0.0, pixel)

