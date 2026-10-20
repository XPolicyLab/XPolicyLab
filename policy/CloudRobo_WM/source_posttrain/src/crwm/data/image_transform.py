"""Clip-consistent CR-WM augmentation for CR-WM's per-camera sample.

RGB is float32 in [-1, 1]; depth stays integer millimeters. Geometry uses
CR-WM's normalized output-to-input affine grid and updates K at the depth
grid. RGB teachers consume the same augmented per_cam as the video/VLM.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def boolean(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be true or false")


class ClipImageAugmentation:
    """One affine per selected camera and one color transform per entire clip.

    As in CR-WM, camera_keys selects geometry; color jitter applies to all
    cameras. Validation and disabled augmentation consume no random draws.
    """

    def __init__(
        self, *, enabled=False, is_training=False, camera_keys=None,
        crop_scale=0.95, rotate_degrees=5.0, brightness=0.3,
        contrast=0.4, saturation=0.5, hue=0.0,
    ):
        self.enabled = boolean(enabled, "enabled") and boolean(is_training, "is_training")
        self.camera_keys = tuple(
            str(key).removeprefix("observation.images.")
            for key in (["cam_high"] if camera_keys is None else camera_keys)
        )
        self.crop_scale = float(crop_scale)
        self.rotate_degrees = float(rotate_degrees)
        self.brightness, self.contrast = float(brightness), float(contrast)
        self.saturation, self.hue = float(saturation), float(hue)
        if self.enabled:
            values = (self.crop_scale, self.rotate_degrees, self.brightness,
                      self.contrast, self.saturation, self.hue)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("augmentation parameters must be finite")
            if not 0.5 <= self.crop_scale <= 1.0 or self.rotate_degrees < 0:
                raise ValueError("crop_scale must be in [0.5,1]; rotate_degrees must be nonnegative")
            if min(self.brightness, self.contrast, self.saturation) < 0 or not 0 <= self.hue <= 0.5:
                raise ValueError("color amplitudes must be nonnegative; hue must be in [0,0.5]")
            if len(set(self.camera_keys)) != len(self.camera_keys):
                raise ValueError("duplicate augmentation camera keys")

    def metadata(self):
        return {
            "version": "crwm.egowm_image_augmentation.v1",
            "geometry_cameras": list(self.camera_keys),
            "crop_scale": self.crop_scale, "rotate_degrees": self.rotate_degrees,
            "brightness": self.brightness, "contrast": self.contrast,
            "saturation": self.saturation, "hue": self.hue,
            "temporal_policy": "same_parameters_for_whole_clip",
            "color_cameras": "all", "rgb_interpolation": "bilinear",
            "depth_interpolation": "nearest", "align_corners": False,
            "intrinsics": "inverse_pixel_affine_times_K_per_example",
        }

    def sample_theta(self):
        scale = float(np.random.uniform(self.crop_scale, 1.0))
        radians = math.radians(float(np.random.uniform(-self.rotate_degrees, self.rotate_degrees)))
        cos, sin = math.cos(radians) * scale, math.sin(radians) * scale
        shift = 1.0 - scale
        tx, ty = float(np.random.uniform(-shift, shift)), float(np.random.uniform(-shift, shift))
        return torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)

    def sample_color(self):
        if max(self.brightness, self.contrast, self.saturation, self.hue) == 0.0:
            return None
        def factor(amount):
            return 1.0 if amount <= 0 else float(np.random.uniform(max(0.0, 1.0-amount), 1.0+amount))
        # Preserve CR-WM's draw order, including its hue draw before factors.
        hue = 0.0 if self.hue <= 0 else float(np.random.uniform(-self.hue, self.hue))
        return {"brightness": factor(self.brightness), "contrast": factor(self.contrast),
                "saturation": factor(self.saturation), "hue": hue}

    @staticmethod
    def warp(tensor, theta, *, mode="bilinear", fill=-1.0):
        if tensor.ndim != 4:
            raise ValueError("warp expects [frames,channels,height,width]")
        theta = theta.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(0).expand(tensor.shape[0], -1, -1)
        grid = F.affine_grid(theta, tensor.shape, align_corners=False)
        return F.grid_sample(tensor - fill, grid, mode=mode, padding_mode="zeros", align_corners=False) + fill

    @staticmethod
    def transformed_intrinsics(K, theta, height, width):
        """Conjugate the half-pixel affine into the native depth pixel grid."""
        K = K.float()
        P = K.new_tensor([[width/2, 0, (width-1)/2], [0, height/2, (height-1)/2], [0, 0, 1]])
        affine = torch.eye(3, device=K.device, dtype=K.dtype)
        affine[:2] = theta.to(device=K.device, dtype=K.dtype)
        pixel_out_to_in = P @ affine @ torch.linalg.inv(P)
        return torch.linalg.solve(pixel_out_to_in, K)

    def __call__(self, data, *, camera_order):
        if not self.enabled:
            return data
        per_cam = data["per_cam"]
        if not isinstance(per_cam, Mapping):
            raise TypeError("per_cam must be a camera mapping")
        if any(key not in per_cam for key in self.camera_keys):
            raise KeyError("augmentation camera is missing from per_cam")
        depth = data.get("per_cam_depth")
        K = data.get("camera_intrinsics")
        if depth is not None and self.camera_keys:
            if not isinstance(K, torch.Tensor) or K.shape != (len(camera_order), 3, 3):
                raise ValueError("depth augmentation requires per-sample [cameras,3,3] intrinsics")
            if any(key not in depth or key not in camera_order for key in self.camera_keys):
                raise KeyError("augmentation camera is missing from depth or intrinsics order")
            K = K.clone()
        thetas = {key: self.sample_theta() for key in self.camera_keys}
        color = self.sample_color()
        for key, rgb in per_cam.items():
            if rgb.ndim != 4 or rgb.shape[0] != 3 or not rgb.is_floating_point():
                raise ValueError("per_cam RGB must be floating [3,frames,height,width]")
            frames = rgb.permute(1, 0, 2, 3)
            if color is not None:
                frames = self.color_jitter(frames, color)
            if key in thetas:
                frames = self.warp(frames, thetas[key])
            per_cam[key] = frames.permute(1, 0, 2, 3).contiguous()
        if depth is not None:
            for key, theta in thetas.items():
                value = depth[key]
                if value.ndim != 3 or value.shape[0] != per_cam[key].shape[1]:
                    raise ValueError("depth and RGB clip frames must match")
                height, width = value.shape[-2:]
                depth[key] = self.warp(value.float().unsqueeze(1), theta, mode="nearest", fill=0.0).squeeze(1).to(value.dtype)
                index = camera_order.index(key)
                K[index] = self.transformed_intrinsics(K[index], theta, height, width)
            if self.camera_keys:
                data["camera_intrinsics"] = K
        return data

    @staticmethod
    def color_jitter(tensor: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor [N,C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[1] < 3:
            brightness = float(params.get("brightness", 1.0))
            contrast = float(params.get("contrast", 1.0))
            image = ((tensor.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
            image = image * brightness
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean
            return (image.clamp(0.0, 1.0) * 2.0 - 1.0).to(dtype=tensor.dtype)

        image = ((tensor.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
        leading = image[:, :3]
        trailing = image[:, 3:] if image.shape[1] > 3 else None

        leading = leading * float(params.get("brightness", 1.0))
        gray = (
            0.299 * leading[:, 0:1]
            + 0.587 * leading[:, 1:2]
            + 0.114 * leading[:, 2:3]
        )
        mean = gray.mean(dim=(-2, -1), keepdim=True)
        leading = (leading - mean) * float(params.get("contrast", 1.0)) + mean
        gray = (
            0.299 * leading[:, 0:1]
            + 0.587 * leading[:, 1:2]
            + 0.114 * leading[:, 2:3]
        )
        saturation = float(params.get("saturation", 1.0))
        leading = (leading - gray) * saturation + gray

        hue = float(params.get("hue", 0.0))
        if abs(hue) > 0.0:
            leading = ClipImageAugmentation._adjust_hue_rgb(leading.clamp(0.0, 1.0), hue)

        leading = leading.clamp(0.0, 1.0)
        if trailing is not None:
            image = torch.cat([leading, trailing], dim=1)
        else:
            image = leading
        return (image.clamp(0.0, 1.0) * 2.0 - 1.0).to(dtype=tensor.dtype)

    @staticmethod
    def _adjust_hue_rgb(rgb: torch.Tensor, hue_factor: float) -> torch.Tensor:
        hsv = ClipImageAugmentation._rgb_to_hsv(rgb)
        hsv[:, 0:1] = torch.remainder(hsv[:, 0:1] + float(hue_factor), 1.0)
        return ClipImageAugmentation._hsv_to_rgb(hsv)

    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        maxc, _ = rgb.max(dim=1, keepdim=True)
        minc, _ = rgb.min(dim=1, keepdim=True)
        delta = maxc - minc
        eps = torch.finfo(rgb.dtype).eps
        safe_delta = torch.where(delta > eps, delta, torch.ones_like(delta))

        hue_r = torch.remainder((g - b) / safe_delta, 6.0)
        hue_g = (b - r) / safe_delta + 2.0
        hue_b = (r - g) / safe_delta + 4.0
        hue = torch.where(maxc == r, hue_r, torch.where(maxc == g, hue_g, hue_b)) / 6.0
        hue = torch.where(delta > eps, hue, torch.zeros_like(hue))
        saturation = torch.where(maxc > eps, delta / maxc.clamp_min(eps), torch.zeros_like(maxc))
        value = maxc
        return torch.cat([hue, saturation, value], dim=1)

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        h6 = h * 6.0
        i = torch.floor(h6).to(torch.int64)
        f = h6 - i.to(dtype=hsv.dtype)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = torch.remainder(i, 6)

        out = torch.zeros_like(hsv)
        cases = (
            torch.cat([v, t, p], dim=1),
            torch.cat([q, v, p], dim=1),
            torch.cat([p, v, t], dim=1),
            torch.cat([p, q, v], dim=1),
            torch.cat([t, p, v], dim=1),
            torch.cat([v, p, q], dim=1),
        )
        for idx, value in enumerate(cases):
            mask = (i == idx).expand_as(out)
            out = torch.where(mask, value, out)
        return out
