"""Configurable camera-wise augmentation with replayable temporal parameters.

AdaIN here means pixel-channel moment transfer, not a learned style network.
FFT jitter uses a positive, conjugate-symmetric frequency gain, preserving phase
before pixel clipping. Neither operation needs another episode or future frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as F


_DEFAULTS = {
    "width": 320,
    "height": 160,
    "frame_batch_size": 16,
    "probability": 1.0,
    "independent_cameras": True,
    "input_color_order": "RGB",
    "crop": {"enabled": False, "probability": 1.0, "scale": 0.95},
    "brightness": {"enabled": False, "probability": 1.0, "range": [0.7, 1.3]},
    "contrast": {"enabled": False, "probability": 1.0, "range": [0.6, 1.4]},
    "saturation": {"enabled": False, "probability": 1.0, "range": [0.5, 1.5]},
    "hue": {"enabled": False, "probability": 1.0, "strength": 0.08},
    "gamma": {"enabled": False, "probability": 1.0, "range": [0.7, 1.4]},
    "exposure": {"enabled": False, "probability": 1.0, "stops": [-0.5, 0.5]},
    "temperature": {
        "enabled": False, "probability": 1.0,
        "red_gain": [0.85, 1.15], "blue_gain": [0.85, 1.15],
    },
    "blur": {
        "enabled": False, "probability": 0.3, "kernel_size": 3, "sigma": [0.1, 1.5],
    },
    "noise": {"enabled": False, "probability": 1.0, "std": 0.03},
    "adain": {
        "enabled": False, "probability": 0.5,
        "mean": [0.3, 0.7], "std": [0.1, 0.3], "blend": 0.5, "eps": 1e-6,
    },
    "fft": {
        "enabled": False, "probability": 0.5,
        "amplitude": [0.5, 1.5], "preserve_dc": True,
    },
}


def _merge(base: Mapping, override: Mapping, path: str = "augmentation") -> dict:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"unknown {path} option: {key}")
        if isinstance(base[key], Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}.{key} must be a mapping")
            result[key] = _merge(base[key], value, f"{path}.{key}")
        else:
            result[key] = deepcopy(value)
    return result


def _number(value: Any, name: str, low: float, high: float = float("inf")) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def _range(value: Any, name: str, low: float, high: float = float("inf")) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a [min, max] pair")
    start, stop = (_number(v, name, low, high) for v in value)
    if start > stop:
        raise ValueError(f"{name} min must not exceed max")


def _validate(config: dict) -> None:
    for name in ("width", "height", "frame_batch_size"):
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["independent_cameras"]) is not bool:
        raise ValueError("independent_cameras must be bool")
    if config["input_color_order"] not in {"RGB", "BGR"}:
        raise ValueError("input_color_order must be RGB or BGR")
    _number(config["probability"], "probability", 0, 1)
    for name, options in config.items():
        if not isinstance(options, dict):
            continue
        if type(options["enabled"]) is not bool:
            raise ValueError(f"{name}.enabled must be bool")
        _number(options["probability"], f"{name}.probability", 0, 1)
    _number(config["crop"]["scale"], "crop.scale", 1e-6, 1)
    for name in ("brightness", "contrast", "saturation"):
        _range(config[name]["range"], f"{name}.range", 0)
    _number(config["hue"]["strength"], "hue.strength", 0, 0.5)
    _range(config["gamma"]["range"], "gamma.range", 1e-6)
    _range(config["exposure"]["stops"], "exposure.stops", -16, 16)
    for channel in ("red_gain", "blue_gain"):
        _range(config["temperature"][channel], f"temperature.{channel}", 1e-6)
    _range(config["blur"]["sigma"], "blur.sigma", 1e-6)
    kernel = config["blur"]["kernel_size"]
    if type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
        raise ValueError("blur.kernel_size must be a positive odd integer")
    if kernel // 2 >= min(config["width"], config["height"]):
        raise ValueError("blur.kernel_size is too large for the output size")
    _number(config["noise"]["std"], "noise.std", 0, 1)
    _range(config["adain"]["mean"], "adain.mean", 0, 1)
    _range(config["adain"]["std"], "adain.std", 0, 1)
    _number(config["adain"]["blend"], "adain.blend", 0, 1)
    _number(config["adain"]["eps"], "adain.eps", 1e-12, 1)
    _range(config["fft"]["amplitude"], "fft.amplitude", 1e-6)
    if type(config["fft"]["preserve_dc"]) is not bool:
        raise ValueError("fft.preserve_dc must be bool")


@dataclass(frozen=True)
class CameraAugmentationParameters:
    """One camera's draws, reused for every frame, including head history."""

    crop: tuple[float, float] | None = None
    color: tuple[tuple[str, float], ...] = ()
    gamma: float | None = None
    exposure: float | None = None
    temperature: tuple[float, float, float] | None = None
    blur_sigma: float | None = None
    noise_seed: int | None = None
    adain_mean: tuple[float, ...] | None = None
    adain_std: tuple[float, ...] | None = None
    fft_seed: int | None = None


class ConfiguredVideoAugmentation:
    """Resize and augment uint8 videos, with optional exact-task overrides.

    Parameters are sampled once per camera, not per frame. Noise uses a shared
    spatial realization and FFT uses a shared frequency mask across time.
    Processing is microbatched on CPU to bound memory for long video chunks.
    """

    def __init__(self, config: Mapping[str, Any], *, training: bool = True) -> None:
        # Hydra may pass DictConfig/ListConfig; resolve before validation and
        # retain only plain, picklable data for DataLoader workers.
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config, Mapping):
            raise ValueError("video_augmentation_config must be a mapping")
        options = dict(config)
        overrides = options.pop("task_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("task_overrides must be a mapping keyed by exact task_id")
        self.config = _merge(_DEFAULTS, options)
        _validate(self.config)
        self.training = training
        self._task_policies = {}
        for task_id, override in overrides.items():
            if not isinstance(task_id, str) or not task_id or not isinstance(override, Mapping):
                raise ValueError("each task override must have a nonempty task_id and mapping")
            merged = _merge(self.config, override)
            # Preserve the collator's shape and channel contract across tasks.
            for key in ("width", "height", "input_color_order"):
                if merged[key] != self.config[key]:
                    raise ValueError(f"task overrides cannot change {key}")
            self._task_policies[task_id] = ConfiguredVideoAugmentation(merged, training=training)

    def for_task(self, task_id: str) -> ConfiguredVideoAugmentation:
        return self._task_policies.get(task_id, self)

    def sample_parameters(
        self, num_views: int, *, rng: np.random.Generator
    ) -> tuple[CameraAugmentationParameters, ...]:
        if num_views < 1:
            raise ValueError("num_views must be positive")
        if not self.training or rng.random() >= self.config["probability"]:
            return (CameraAugmentationParameters(),) * num_views
        if not self.config["independent_cameras"]:
            return (self._sample_camera(rng),) * num_views
        return tuple(self._sample_camera(rng) for _ in range(num_views))

    def _sample_camera(self, rng: np.random.Generator) -> CameraAugmentationParameters:
        config = self.config

        def active(name: str) -> bool:
            return config[name]["enabled"] and rng.random() < config[name]["probability"]

        def uniform(name: str, field: str = "range") -> float:
            return float(rng.uniform(*config[name][field]))

        crop = tuple(float(v) for v in rng.random(2)) if active("crop") else None
        color = [
            (name, uniform(name))
            for name in ("brightness", "contrast", "saturation") if active(name)
        ]
        if active("hue") and config["hue"]["strength"] > 0:
            strength = config["hue"]["strength"]
            color.append(("hue", float(rng.uniform(-strength, strength))))
        rng.shuffle(color)
        gamma = uniform("gamma") if active("gamma") else None
        exposure = 2.0 ** uniform("exposure", "stops") if active("exposure") else None
        temperature = (
            (uniform("temperature", "red_gain"), 1.0, uniform("temperature", "blue_gain"))
            if active("temperature") else None
        )
        blur_sigma = uniform("blur", "sigma") if active("blur") else None
        noise_seed = int(rng.integers(0, 2**63)) if active("noise") else None
        adain_mean = adain_std = None
        if active("adain"):
            adain_mean = tuple(float(v) for v in rng.uniform(*config["adain"]["mean"], size=3))
            adain_std = tuple(float(v) for v in rng.uniform(*config["adain"]["std"], size=3))
        fft_seed = int(rng.integers(0, 2**63)) if active("fft") else None
        return CameraAugmentationParameters(
            crop, tuple(color), gamma, exposure, temperature, blur_sigma,
            noise_seed, adain_mean, adain_std, fft_seed,
        )

    def __call__(
        self, video: np.ndarray, *, rng: np.random.Generator | None = None,
        parameters: tuple[CameraAugmentationParameters, ...] | None = None,
    ) -> np.ndarray:
        values = np.asarray(video)
        if values.dtype != np.uint8 or values.ndim not in {4, 5} or values.shape[-1] != 3:
            raise ValueError("video must be uint8 [T,H,W,3] or [T,V,H,W,3]")
        views = values[:, None] if values.ndim == 4 else values
        if min(views.shape[1:4]) < 1:
            raise ValueError("video must have nonempty views and spatial dimensions")
        if parameters is None:
            parameters = self.sample_parameters(views.shape[1], rng=rng or np.random.default_rng())
        if len(parameters) != views.shape[1]:
            raise ValueError("augmentation parameters must match the video view count")
        cfg = self.config
        output = np.empty((len(views), views.shape[1], cfg["height"], cfg["width"], 3), dtype=np.uint8)
        for index, params in enumerate(parameters):
            noise, fft_gain = self._spatial_fields(params)
            for start in range(0, len(views), cfg["frame_batch_size"]):
                frames = views[start : start + cfg["frame_batch_size"], index]
                tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
                tensor = self._apply(tensor, params, noise, fft_gain)
                output[start : start + len(frames), index] = tensor.permute(0, 2, 3, 1).numpy()
        return output[:, 0] if values.ndim == 4 else output

    def _spatial_fields(
        self, params: CameraAugmentationParameters
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cfg = self.config
        height, width = cfg["height"], cfg["width"]
        noise = fft_gain = None
        if params.noise_seed is not None:
            rng = np.random.default_rng(params.noise_seed)
            noise = torch.from_numpy(rng.standard_normal((3, height, width), dtype=np.float32))
            noise = noise * cfg["noise"]["std"]
        if params.fft_seed is not None:
            rng = np.random.default_rng(params.fft_seed)
            # Assign one iid uniform gain to each conjugate pair. Averaging two
            # iid gains would change the requested uniform distribution.
            indices = np.arange(height * width).reshape(height, width)
            partner = indices[np.ix_((-np.arange(height)) % height, (-np.arange(width)) % width)]
            canonical = np.minimum(indices, partner)
            draws = rng.uniform(*cfg["fft"]["amplitude"], size=(3, height * width))
            gain = draws[:, canonical].astype(np.float32)
            if cfg["fft"]["preserve_dc"]:
                gain[:, 0, 0] = 1.0
            fft_gain = torch.from_numpy(gain[:, :, : width // 2 + 1].copy())
        return noise, fft_gain

    def _apply(
        self, tensor: torch.Tensor, params: CameraAugmentationParameters,
        noise: torch.Tensor | None, fft_gain: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.config
        size = [cfg["height"], cfg["width"]]
        if params.crop is not None:
            height, width = tensor.shape[-2:]
            ch, cw = max(1, int(height * cfg["crop"]["scale"])), max(1, int(width * cfg["crop"]["scale"]))
            top = min(int(params.crop[0] * (height - ch + 1)), height - ch)
            left = min(int(params.crop[1] * (width - cw + 1)), width - cw)
            tensor = F.resized_crop(tensor, top, left, ch, cw, size, antialias=True)
        else:
            tensor = F.resize(tensor, size, antialias=True)
        tensor = tensor.to(torch.float32) / 255.0
        if cfg["input_color_order"] == "BGR":
            tensor = tensor.flip(1)
        for name, factor in params.color:
            tensor = getattr(F, f"adjust_{name}")(tensor, factor)
        if params.gamma is not None:
            tensor = F.adjust_gamma(tensor, params.gamma)
        if params.exposure is not None:
            tensor = (tensor * params.exposure).clamp(0, 1)
        if params.temperature is not None:
            gain = tensor.new_tensor(params.temperature).view(1, 3, 1, 1)
            tensor = (tensor * gain).clamp(0, 1)
        if params.blur_sigma is not None:
            kernel = cfg["blur"]["kernel_size"]
            tensor = F.gaussian_blur(tensor, [kernel, kernel], [params.blur_sigma] * 2)
        if noise is not None:
            tensor = (tensor + noise).clamp(0, 1)
        if params.adain_mean is not None:
            mean = tensor.mean(dim=(-2, -1), keepdim=True)
            std = tensor.var(dim=(-2, -1), correction=0, keepdim=True).add(cfg["adain"]["eps"]).sqrt()
            target_mean = tensor.new_tensor(params.adain_mean).view(1, 3, 1, 1)
            target_std = tensor.new_tensor(params.adain_std).view(1, 3, 1, 1)
            styled = (tensor - mean) / std * target_std + target_mean
            tensor = tensor.lerp(styled, cfg["adain"]["blend"]).clamp(0, 1)
        if fft_gain is not None:
            spectrum = torch.fft.rfft2(tensor)
            tensor = torch.fft.irfft2(spectrum * fft_gain, s=tuple(size)).clamp(0, 1)
        if cfg["input_color_order"] == "BGR":
            tensor = tensor.flip(1)
        return (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
