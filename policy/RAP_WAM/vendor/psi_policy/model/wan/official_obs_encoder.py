from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin
from psi_policy.model.wan.runtime import Wan2_2_VAE


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def _normalize_obs_shapes(obs_shapes: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    normalized: dict[str, tuple[int, ...]] = {}
    for raw_key, raw_shape in dict(obs_shapes).items():
        normalized[str(raw_key)] = tuple(int(value) for value in raw_shape)
    return normalized


def _load_transformer_config(checkpoint_dir: str) -> dict[str, object]:
    root = Path(checkpoint_dir)
    candidates = [
        root / "config.json",
        root / "transformer" / "config.json",
        root / "model" / "config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def _resolve_file(path_or_dir: str, explicit_path: Optional[str], candidates: list[str]) -> str:
    if explicit_path is not None:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        return str(resolved)

    root = Path(path_or_dir).expanduser().resolve()
    for candidate in candidates:
        full_path = root / candidate
        if full_path.exists():
            return str(full_path)
    raise FileNotFoundError(
        f"Could not resolve file under {root}. Tried: {', '.join(candidates)}"
    )


class WanOfficialObsEncoder(ModuleAttrMixin):
    """Wan 5B TI2V 官方预处理 helper。"""

    conditioning_mode = "prefix_latent"

    def __init__(
        self,
        *,
        obs_shapes: Mapping[str, Sequence[int]],
        obs_horizon: int,
        wan_checkpoint_dir: str,
        target_horizon: int,
        image_key: str = "rgb_head",
        target_key: str = "rgb_head_future",
        fallback_target_key: str = "rgb_head_last",
        vae_checkpoint_path: str | None = None,
        vae_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.obs_shapes = _normalize_obs_shapes(obs_shapes)
        self.obs_horizon = int(obs_horizon)
        self.target_horizon = int(target_horizon)
        self.wan_checkpoint_dir = str(Path(wan_checkpoint_dir).expanduser().resolve())
        self.image_key = str(image_key)
        self.target_key = str(target_key)
        self.fallback_target_key = str(fallback_target_key)
        self.vae_dtype = _parse_torch_dtype(vae_dtype)

        if self.obs_horizon <= 0:
            raise ValueError("obs_horizon must be > 0")
        if self.target_horizon <= 0:
            raise ValueError("target_horizon must be > 0")

        transformer_config = _load_transformer_config(self.wan_checkpoint_dir)
        model_type = str(transformer_config.get("model_type", "ti2v")).lower()
        if model_type != "ti2v":
            raise ValueError(
                f"WanOfficialObsEncoder currently supports only TI2V-5B checkpoints, got {model_type!r}"
            )

        self.latent_channels = int(transformer_config.get("out_dim", 48))
        self.spatial_stride = 16
        self.total_video_frames = 1 + self.target_horizon
        if (self.total_video_frames - 1) % 4 != 0:
            raise ValueError(
                "Official Wan requires total_video_frames to satisfy 4n+1. "
                f"Got total_video_frames={self.total_video_frames}."
            )

        if self.image_key not in self.obs_shapes:
            raise KeyError(f"obs_shapes missing image key: {self.image_key}")
        image_shape = self.obs_shapes[self.image_key]
        if len(image_shape) != 3:
            raise ValueError(f"Wan expects image shape (C,H,W), got {image_shape}")
        self.image_channels = int(image_shape[0])
        self.image_height = int(image_shape[1])
        self.image_width = int(image_shape[2])
        if self.image_channels != 3:
            raise ValueError(f"Wan expects RGB input, got {image_shape}")
        if self.image_height % self.spatial_stride != 0 or self.image_width % self.spatial_stride != 0:
            raise ValueError(
                f"Image size {(self.image_height, self.image_width)} must be divisible by {self.spatial_stride}."
            )

        self.vae_checkpoint_path = _resolve_file(
            self.wan_checkpoint_dir,
            vae_checkpoint_path,
            candidates=["Wan2.2_VAE.pth"],
        )

        self._vae: Wan2_2_VAE | None = None
        self._runtime_device: str | None = None

    def _runtime_device_str(self) -> str:
        if self.device.type != "cuda":
            raise RuntimeError("Official Wan training currently requires CUDA.")
        if self.device.index is None:
            return "cuda"
        return f"cuda:{self.device.index}"

    def _ensure_runtime_modules(self) -> None:
        runtime_device = self._runtime_device_str()
        if self._runtime_device == runtime_device and self._vae is not None:
            return

        self._vae = Wan2_2_VAE(
            vae_pth=self.vae_checkpoint_path,
            dtype=self.vae_dtype,
            device=runtime_device,
        )
        self._runtime_device = runtime_device

    def _get_condition_frame(self, nobs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.image_key not in nobs:
            raise KeyError(f"Missing condition key: {self.image_key}")
        condition = nobs[self.image_key]
        if condition.ndim != 5:
            raise ValueError(f"Expected {self.image_key} to be [B,T,C,H,W], got {tuple(condition.shape)}")
        return condition[:, -1:]

    def _get_future_frames(self, nobs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.target_key in nobs:
            future = nobs[self.target_key]
        elif self.fallback_target_key in nobs:
            future = nobs[self.fallback_target_key]
        else:
            raise KeyError(
                f"Neither {self.target_key!r} nor {self.fallback_target_key!r} exists in obs."
            )
        if future.ndim != 5:
            raise ValueError(f"Expected future frames to be [B,T,C,H,W], got {tuple(future.shape)}")
        return future

    def _to_wan_video(self, frames_btchw: torch.Tensor) -> torch.Tensor:
        return frames_btchw.permute(0, 2, 1, 3, 4).contiguous().mul(2.0).sub(1.0)

    def _encode_video_batch(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            raise RuntimeError("VAE runtime is not initialized.")
        video_bcthw = video_bcthw.to(device=self.device, dtype=self.vae_dtype)
        return self._vae.encode(video_bcthw).float()

    def _full_latent_shape(self, batch_size: int) -> torch.Size:
        latent_t = 1 + self.target_horizon // 4
        latent_h = self.image_height // self.spatial_stride
        latent_w = self.image_width // self.spatial_stride
        return torch.Size([batch_size, self.latent_channels, latent_t, latent_h, latent_w])

    def _build_prefix_latent_condition(
        self,
        prefix_latents: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        full_shape = self._full_latent_shape(batch_size)
        condition_latents = torch.zeros(
            full_shape,
            device=prefix_latents.device,
            dtype=prefix_latents.dtype,
        )
        prefix_steps = int(prefix_latents.shape[2])
        if prefix_steps > condition_latents.shape[2]:
            raise ValueError(
                f"Prefix latent steps {prefix_steps} exceed full latent horizon {condition_latents.shape[2]}"
            )
        condition_latents[:, :, :prefix_steps] = prefix_latents
        latent_condition_mask = torch.ones_like(condition_latents, dtype=torch.float32)
        latent_condition_mask[:, :, :prefix_steps] = 0.0
        return condition_latents, latent_condition_mask, prefix_steps

    @torch.no_grad()
    def prepare_batch(
        self,
        nobs: Mapping[str, torch.Tensor],
        training: bool = True,
    ) -> dict[str, torch.Tensor | str | int]:
        del training
        self._ensure_runtime_modules()

        condition_frame = self._get_condition_frame(nobs)
        future_frames = self._get_future_frames(nobs)
        if future_frames.shape[1] != self.target_horizon:
            raise ValueError(
                f"Expected {self.target_key} horizon={self.target_horizon}, got {future_frames.shape[1]}"
            )

        full_video = torch.cat([condition_frame, future_frames], dim=1)
        clean_latents = self._encode_video_batch(self._to_wan_video(full_video))

        prefix_latents = self._encode_video_batch(self._to_wan_video(condition_frame))
        condition_latents, latent_condition_mask, prefix_steps = self._build_prefix_latent_condition(
            prefix_latents,
            batch_size=clean_latents.shape[0],
        )
        return {
            "conditioning_mode": self.conditioning_mode,
            "condition_frame": condition_frame,
            "future_frames": future_frames,
            "clean_latents": clean_latents,
            "condition_latents": condition_latents,
            "latent_condition_mask": latent_condition_mask,
            "num_condition_latent_steps": int(prefix_steps),
        }

    @torch.no_grad()
    def prepare_condition_only(self, nobs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | str | int]:
        self._ensure_runtime_modules()
        condition_frame = self._get_condition_frame(nobs)
        prefix_latents = self._encode_video_batch(self._to_wan_video(condition_frame))
        condition_latents, latent_condition_mask, prefix_steps = self._build_prefix_latent_condition(
            prefix_latents,
            batch_size=condition_frame.shape[0],
        )
        return {
            "conditioning_mode": self.conditioning_mode,
            "condition_frame": condition_frame,
            "condition_latents": condition_latents,
            "latent_condition_mask": latent_condition_mask,
            "latent_shape": condition_latents.shape,
            "num_condition_latent_steps": int(prefix_steps),
        }

    @torch.no_grad()
    def decode_full_latents(self, full_latents: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime_modules()
        if self._vae is None:
            raise RuntimeError("VAE runtime is not initialized.")
        if full_latents.ndim != 5:
            raise ValueError(f"Expected full_latents to be [B,C,T,H,W], got {tuple(full_latents.shape)}")
        decoded = self._vae.decode(
            [latent.to(device=self.device, dtype=self.vae_dtype) for latent in full_latents]
        )
        decoded = torch.stack([video.float() for video in decoded], dim=0)
        decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()
        return decoded.add(1.0).div(2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def output_shape(self) -> tuple[torch.Size, dict[str, object]]:
        return self._full_latent_shape(batch_size=1), {}


__all__ = ["WanOfficialObsEncoder"]
