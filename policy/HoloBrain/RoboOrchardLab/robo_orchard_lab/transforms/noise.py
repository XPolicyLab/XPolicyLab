# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Generic, Sequence, TypeVar

import torch
from robo_orchard_core.utils.config import Config, TorchTensor

from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "GaussianNoiseConfig",
    "UniformNoiseConfig",
    "AddNoise",
    "AddNoiseConfig",
]


class NoiseConfig(Config, metaclass=ABCMeta):
    @abstractmethod
    def generate_noise(
        self, target_shape: Sequence[int] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement generate_noise method."
        )


NoiseConfigT_co = TypeVar("NoiseConfigT_co", bound=NoiseConfig, covariant=True)


class GaussianNoiseConfig(NoiseConfig):
    """Configuration for Gaussian noise transform."""

    mean: TorchTensor
    """Mean of the Gaussian noise."""

    std: TorchTensor
    """Standard deviation of the Gaussian noise."""

    def generate_noise(
        self, target_shape: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Generate Gaussian noise with the specified shape."""

        if target_shape is None:
            target_shape = self.mean.shape

        mean = self.mean
        if target_shape != self.mean.shape:
            if len(target_shape) > self.mean.dim():
                # broadcast the mean  to the target shape
                to_expand_shape = [
                    1 for _ in range(len(target_shape) - self.mean.dim())
                ] + list(self.mean.shape)

                mean = self.mean.view(to_expand_shape)
                mean = mean.expand(*target_shape)
            else:
                raise ValueError(
                    f"Target shape {target_shape} does not match mean shape "
                    f"{self.mean.shape}. Please ensure the target shape is "
                    "compatible with the mean shape."
                )

        return torch.normal(mean=mean, std=self.std)


class UniformNoiseConfig(NoiseConfig):
    """Configuration for Uniform noise transform."""

    low: TorchTensor
    """Lower bound of the uniform noise."""
    high: TorchTensor
    """Upper bound of the uniform noise."""

    def generate_noise(
        self, target_shape: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Generate Uniform noise with the specified shape."""
        if target_shape is None:
            target_shape = self.low.shape

        return torch.rand(target_shape) * (self.high - self.low) + self.low

    @staticmethod
    def from_min_max_stats(
        min: torch.Tensor,
        max: torch.Tensor,
        scale: float,
        repeat: Sequence[int] | None = None,
    ) -> UniformNoiseConfig:
        """Create a UniformNoiseConfig from min and max statistics.

        The noise will be uniformly distributed in the range
        [- scale * (max - min)/2, scale * (max - min)/2].
        The scale factor is actually the percentage of the range
        that the noise will cover.

        Args:
            min (torch.Tensor): Minimum values.
            max (torch.Tensor): Maximum values.
            scale (float): Scale factor to determine the range of the noise.
        """

        r = scale / -2.0 * (max - min)
        if repeat is not None:
            r = r.repeat(*repeat)

        return UniformNoiseConfig(low=r, high=-r)


class AddNoise(DictTransform, Generic[NoiseConfigT_co]):
    cfg: AddNoiseConfig[NoiseConfigT_co]

    def __init__(self, cfg: AddNoiseConfig[NoiseConfigT_co]) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs) -> dict:
        """Apply noise to the input columns."""

        ret = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                noise = self.cfg.noise.generate_noise(target_shape=v.shape)
                if not self.cfg.inplace:
                    ret[k] = v + noise
                else:
                    v += noise
                    ret[k] = v
            else:
                raise TypeError(
                    f"Expected TorchTensor for column '{k}', "
                    f"but got {type(v).__name__}."
                )
        return ret


class AddNoiseConfig(DictTransformConfig[AddNoise], Generic[NoiseConfigT_co]):
    class_type: ClassType[AddNoise] = AddNoise

    noise: NoiseConfigT_co
    """Configuration for the noise to be added."""

    inplace: bool = True
