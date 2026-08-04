"""Data augmentation transforms for improving policy generalization.

These transforms are designed for bimanual robot manipulation and are safe
to use without breaking the consistency between observations and actions.

Image augmentations help the policy generalize to:
- Different object positions (random crop/resize simulates camera shifts)
- Different lighting conditions (color jitter)
- Partial occlusions (random erasing)
- Different visual appearances (blur, noise)

These augmentations specifically target the `_random` eval variants
(randomized object positions) and improve overall robustness.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import cv2
import numpy as np

from openpi import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class ImageAugmentation(_transforms.DataTransformFn):
    """Apply visual augmentations to all camera images.

    Operates on data["image"] dict with HWC uint8 arrays.
    All cameras receive independent augmentation draws for diversity.

    Args:
        color_jitter_prob: Probability of applying color jitter per image.
        brightness: Max brightness shift factor (e.g. 0.2 = +/- 20%).
        contrast: Max contrast shift factor.
        saturation: Max saturation shift factor.
        random_crop_prob: Probability of random crop-and-resize per image.
        crop_scale_min: Minimum crop scale (fraction of image area).
        random_erase_prob: Probability of random erasing per image.
        erase_scale_range: (min, max) fraction of image area to erase.
        gaussian_noise_prob: Probability of adding Gaussian noise.
        noise_std: Standard deviation of Gaussian noise (on 0-255 scale).
        blur_prob: Probability of Gaussian blur.
        seed: Random seed for reproducibility (None = use global numpy rng).
    """

    color_jitter_prob: float = 0.8
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    random_crop_prob: float = 0.5
    crop_scale_min: float = 0.8
    random_erase_prob: float = 0.3
    erase_scale_range: tuple[float, float] = (0.02, 0.08)
    gaussian_noise_prob: float = 0.2
    noise_std: float = 10.0
    blur_prob: float = 0.1
    seed: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    def __call__(self, data: dict) -> dict:
        if "image" not in data:
            return data

        augmented_images = {}
        for cam_name, img in data["image"].items():
            augmented_images[cam_name] = self._augment_image(np.asarray(img))
        data["image"] = augmented_images
        return data

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single image (HWC, uint8)."""
        is_batched = img.ndim == 4
        if is_batched:
            return np.stack([self._augment_single(frame) for frame in img])
        return self._augment_single(img)

    def _augment_single(self, img: np.ndarray) -> np.ndarray:
        """Augment a single HWC uint8 image."""
        assert img.ndim == 3
        h, w = img.shape[:2]
        was_uint8 = img.dtype == np.uint8

        # Work in float for precision
        img_f = img.astype(np.float32)

        # 1. Random crop and resize (simulates camera position shifts)
        if self._rng.random() < self.random_crop_prob:
            img_f = self._random_crop_resize(img_f, h, w)

        # 2. Color jitter (brightness, contrast, saturation)
        if self._rng.random() < self.color_jitter_prob:
            img_f = self._color_jitter(img_f)

        # 3. Gaussian noise
        if self._rng.random() < self.gaussian_noise_prob:
            noise = self._rng.normal(0, self.noise_std, img_f.shape).astype(np.float32)
            img_f = img_f + noise

        # 4. Gaussian blur
        if self._rng.random() < self.blur_prob:
            ksize = self._rng.choice([3, 5])
            img_f = cv2.GaussianBlur(img_f.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)

        # 5. Random erasing (simulates partial occlusion)
        if self._rng.random() < self.random_erase_prob:
            img_f = self._random_erase(img_f, h, w)

        # Clip and convert back
        img_f = np.clip(img_f, 0, 255)
        if was_uint8:
            return img_f.astype(np.uint8)
        return img_f

    def _random_crop_resize(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Random crop and resize back to original size."""
        scale = self._rng.uniform(self.crop_scale_min, 1.0)
        new_h = int(h * np.sqrt(scale))
        new_w = int(w * np.sqrt(scale))
        top = self._rng.integers(0, h - new_h + 1)
        left = self._rng.integers(0, w - new_w + 1)
        cropped = img[top : top + new_h, left : left + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, and saturation jitter."""
        # Brightness
        b_factor = 1.0 + self._rng.uniform(-self.brightness, self.brightness)
        img = img * b_factor

        # Contrast (relative to mean)
        c_factor = 1.0 + self._rng.uniform(-self.contrast, self.contrast)
        mean = img.mean()
        img = (img - mean) * c_factor + mean

        # Saturation (convert to HSV-like space)
        s_factor = 1.0 + self._rng.uniform(-self.saturation, self.saturation)
        gray = np.mean(img, axis=-1, keepdims=True)
        img = gray + (img - gray) * s_factor

        return img

    def _random_erase(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Random erasing: fill a random rectangle with random values."""
        area = h * w
        erase_area = area * self._rng.uniform(*self.erase_scale_range)
        aspect_ratio = self._rng.uniform(0.3, 3.3)
        eh = int(np.sqrt(erase_area * aspect_ratio))
        ew = int(np.sqrt(erase_area / aspect_ratio))
        eh = min(eh, h)
        ew = min(ew, w)
        top = self._rng.integers(0, max(1, h - eh + 1))
        left = self._rng.integers(0, max(1, w - ew + 1))
        erase_val = self._rng.uniform(0, 255, size=(eh, ew, img.shape[-1])).astype(np.float32)
        img[top : top + eh, left : left + ew] = erase_val
        return img


@dataclasses.dataclass(frozen=True)
class PromptAugmentation(_transforms.DataTransformFn):
    """Augment the text prompt with minor variations for language generalization.

    This helps the policy generalize to `_by_language` eval variants by
    exposing it to different phrasings of the same task instruction during training.

    Args:
        prob: Probability of augmenting the prompt on each sample.
        augmentations: Dict mapping original prompt substrings to lists of alternatives.
            If the prompt contains a key, it may be replaced with a random alternative.
        simplify_prob: Probability of simplifying the prompt (removing extra clauses).
        seed: Random seed.
    """

    prob: float = 0.3
    augmentations: dict[str, Sequence[str]] | None = None
    simplify_prob: float = 0.1
    seed: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            return data

        prompt = data["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.item()
        if not isinstance(prompt, str):
            return data

        if self._rng.random() < self.prob:
            prompt = self._augment_prompt(prompt)

        data["prompt"] = np.asarray(prompt)
        return data

    def _augment_prompt(self, prompt: str) -> str:
        """Apply random prompt augmentation."""
        # Try substring replacements from augmentation dict
        if self.augmentations:
            for original, alternatives in self.augmentations.items():
                if original in prompt:
                    replacement = self._rng.choice(list(alternatives))
                    prompt = prompt.replace(original, replacement, 1)
                    break

        # Simplify: keep only the first sentence/clause
        if self._rng.random() < self.simplify_prob:
            for sep in [", then", ", and", ", remember"]:
                if sep in prompt:
                    prompt = prompt.split(sep)[0] + "."
                    break

        return prompt
