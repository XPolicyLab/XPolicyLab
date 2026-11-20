#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import collections
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class Pi05StyleAugment(Transform):
    """Approximate openpi pi0.5 training image augmentation.

    Applies a 95% crop-resize and small rotation, then brightness, contrast, and
    saturation jitter. Parameters mirror openpi's pi0.5 preprocessing defaults.
    """

    def __init__(
        self,
        crop_scale: float = 0.95,
        degrees: Sequence[float] = (-5.0, 5.0),
        brightness: Sequence[float] = (0.7, 1.3),
        contrast: Sequence[float] = (0.6, 1.4),
        saturation: Sequence[float] = (0.5, 1.5),
        apply_geometric: bool = True,
        wrist_camera_keywords: Sequence[str] = ("wrist", "hand", "left", "right", "image1", "image2"),
    ) -> None:
        super().__init__()
        if not 0.0 < crop_scale <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}.")
        self.crop_scale = float(crop_scale)
        self.degrees = self._check_range(degrees, "degrees")
        self.brightness = self._check_range(brightness, "brightness")
        self.contrast = self._check_range(contrast, "contrast")
        self.saturation = self._check_range(saturation, "saturation")
        self.apply_geometric = bool(apply_geometric)
        self.wrist_camera_keywords = tuple(keyword.lower() for keyword in wrist_camera_keywords)
        self._camera_key: str | None = None

    def set_current_key(self, camera_key: str | None) -> None:
        self._camera_key = camera_key

    @staticmethod
    def _check_range(value: Sequence[float], name: str) -> tuple[float, float]:
        if not isinstance(value, collections.abc.Sequence) or len(value) != 2:
            raise TypeError(f"{name} must be a sequence of two numbers.")
        low, high = (float(value[0]), float(value[1]))
        if low > high:
            raise ValueError(f"{name} lower bound must be <= upper bound, got {value}.")
        return low, high

    @staticmethod
    def _sample_uniform(bounds: tuple[float, float]) -> float:
        low, high = bounds
        return torch.empty(1).uniform_(low, high).item()

    @staticmethod
    def _get_height_width(inpt: Any) -> tuple[int, int]:
        if hasattr(inpt, "shape"):
            return int(inpt.shape[-2]), int(inpt.shape[-1])
        if hasattr(inpt, "size"):
            width, height = inpt.size
            return int(height), int(width)
        raise TypeError(f"Unsupported image input type for Pi05StyleAugment: {type(inpt)!r}")

    def _is_wrist_camera(self) -> bool:
        if self._camera_key is None:
            return False
        camera_key = self._camera_key.lower()
        return any(keyword in camera_key for keyword in self.wrist_camera_keywords)

    def forward(self, inpt: Any) -> Any:
        if self.apply_geometric and not self._is_wrist_camera():
            height, width = self._get_height_width(inpt)
            crop_height = max(1, int(height * self.crop_scale))
            crop_width = max(1, int(width * self.crop_scale))
            max_top = height - crop_height
            max_left = width - crop_width
            top = torch.randint(0, max_top + 1, (1,)).item() if max_top > 0 else 0
            left = torch.randint(0, max_left + 1, (1,)).item() if max_left > 0 else 0
            inpt = F.resized_crop(
                inpt,
                top=int(top),
                left=int(left),
                height=crop_height,
                width=crop_width,
                size=[height, width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            angle = self._sample_uniform(self.degrees)
            if abs(angle) > 0.1:
                inpt = F.rotate(
                    inpt,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )

        inpt = F.adjust_brightness(inpt, self._sample_uniform(self.brightness))
        inpt = F.adjust_contrast(inpt, self._sample_uniform(self.contrast))
        inpt = F.adjust_saturation(inpt, self._sample_uniform(self.saturation))
        if isinstance(inpt, torch.Tensor) and torch.is_floating_point(inpt):
            inpt = inpt.clamp(0.0, 1.0)
        return inpt


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # Optional named preset for common augmentation policies.
    preset: str | None = None
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def resolve_image_transforms_preset(cfg: ImageTransformsConfig) -> ImageTransformsConfig:
    resolved_cfg = copy.deepcopy(cfg)
    preset = resolved_cfg.preset
    if preset in (None, "", "default"):
        return resolved_cfg

    if preset == "lightly":
        resolved_cfg.max_num_transforms = 1
        resolved_cfg.random_order = False
        resolved_cfg.tfs = {
            "identity": ImageTransformConfig(
                weight=1.0,
                type="Identity",
                kwargs={},
            ),
            "brightness": ImageTransformConfig(
                weight=0.6,
                type="ColorJitter",
                kwargs={"brightness": (0.85, 1.15)},
            ),
            "contrast": ImageTransformConfig(
                weight=0.6,
                type="ColorJitter",
                kwargs={"contrast": (0.85, 1.15)},
            ),
            "saturation": ImageTransformConfig(
                weight=0.6,
                type="ColorJitter",
                kwargs={"saturation": (0.9, 1.1)},
            ),
            "hue": ImageTransformConfig(
                weight=0.3,
                type="ColorJitter",
                kwargs={"hue": (-0.02, 0.02)},
            ),
            "sharpness": ImageTransformConfig(
                weight=0.3,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.9, 1.1)},
            ),
            "affine": ImageTransformConfig(
                weight=0.2,
                type="RandomAffine",
                kwargs={"degrees": (-2.0, 2.0), "translate": (0.02, 0.02)},
            ),
        }
        return resolved_cfg

    if preset in {"pi05", "pi0.5", "pi05_style"}:
        resolved_cfg.max_num_transforms = 1
        resolved_cfg.random_order = False
        resolved_cfg.tfs = {
            "pi05_style": ImageTransformConfig(
                weight=1.0,
                type="Pi05StyleAugment",
                kwargs={},
            ),
        }
        return resolved_cfg

    raise ValueError(f"Unknown image transforms preset: {preset}")


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "Pi05StyleAugment":
        return Pi05StyleAugment(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        cfg = resolve_image_transforms_preset(cfg)
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def set_current_key(self, camera_key: str | None) -> None:
        for transform in self.transforms.values():
            if hasattr(transform, "set_current_key"):
                transform.set_current_key(camera_key)

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
