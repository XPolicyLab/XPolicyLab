from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn


class LingBotWanVAEWrapper(nn.Module):
    """Adapter exposing the local FlexWAM VAE interface for diffusers AutoencoderKLWan."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.temporal_downsample_factor = 4
        self.upsampling_factor = 8
        self.model = SimpleNamespace(z_dim=int(getattr(vae.config, "z_dim", 16)))

        mean = torch.tensor(getattr(vae.config, "latents_mean", [0.0] * self.model.z_dim), dtype=torch.float32)
        std = torch.tensor(getattr(vae.config, "latents_std", [1.0] * self.model.z_dim), dtype=torch.float32)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_std", std.view(1, -1, 1, 1, 1), persistent=False)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latents_mean.to(device=latents.device, dtype=latents.dtype)) / self.latents_std.to(
            device=latents.device, dtype=latents.dtype
        )

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std.to(device=latents.device, dtype=latents.dtype) + self.latents_mean.to(
            device=latents.device, dtype=latents.dtype
        )

    @staticmethod
    def _stack_video_list(video: torch.Tensor | Iterable[torch.Tensor], device: torch.device | str | None) -> torch.Tensor:
        if isinstance(video, torch.Tensor):
            out = video
        else:
            out = torch.stack(list(video), dim=0)
        if device is not None:
            out = out.to(device=device)
        return out

    def _set_tiling(self, enabled: bool) -> None:
        if enabled and hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        elif not enabled and hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()

    @torch.no_grad()
    def encode(
        self,
        video: torch.Tensor | Iterable[torch.Tensor],
        device: torch.device | str | None = None,
        tiled: bool = False,
        tile_size=None,
        tile_stride=None,
    ) -> torch.Tensor:
        del tile_size, tile_stride
        sample = self._stack_video_list(video, device)
        self._set_tiling(tiled)
        posterior = self.vae.encode(sample).latent_dist
        latents = posterior.mode()
        return self._normalize_latents(latents)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        device: torch.device | str | None = None,
        tiled: bool = False,
        tile_size=None,
        tile_stride=None,
    ) -> torch.Tensor:
        del tile_size, tile_stride
        if device is not None:
            latents = latents.to(device=device)
        self._set_tiling(tiled)
        sample = self._denormalize_latents(latents)
        return self.vae.decode(sample).sample
