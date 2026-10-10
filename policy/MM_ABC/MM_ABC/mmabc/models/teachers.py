"""Frozen visual teachers that produce future-frame representation targets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input normalisation shared by the visual teachers.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass
class TeacherConfig:
    name: str
    repo_path: str
    ckpt_path: str
    image_size: int = 256
    feature_dim: int = 2048
    patch_size: int = 16
    block_grid: int = 8


def _to_teacher_batch(
    images: torch.Tensor, size: int, *, normalize: bool, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """(B, V, H, W, 3) uint8 -> (B, V, 3, size, size) in the teacher's input space."""
    b, v, h, w, c = images.shape
    x = images.reshape(b * v, h, w, c).permute(0, 3, 1, 2).float().div_(255.0)
    if (h, w) != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if normalize:
        mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x.view(b, v, 3, size, size).to(dtype)


def _pool_to_grid(tokens: torch.Tensor, grid: int, out_grid: int) -> torch.Tensor:
    """(N, grid*grid, D) -> (N, out_grid*out_grid, D) by average pooling."""
    n, p, d = tokens.shape
    if p != grid * grid:
        raise ValueError(f"expected {grid * grid} patch tokens, got {p}")
    x = tokens.view(n, grid, grid, d).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, (out_grid, out_grid))
    return x.permute(0, 2, 3, 1).reshape(n, out_grid * out_grid, d)


class VGGTOmegaTeacher(nn.Module):
    """Geometry-aware patch tokens from an external VGGT-Omega checkout."""

    def __init__(self, cfg: TeacherConfig) -> None:
        super().__init__()
        self.cfg = cfg
        repo = str(Path(cfg.repo_path).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega(enable_alignment=True)
        state = torch.load(self._checkpoint(), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _checkpoint(self) -> str:
        path = Path(self.cfg.ckpt_path).expanduser()
        if path.is_file():
            return str(path)
        if path.is_dir():
            preferred = path / "vggt_omega_1b_256_text.pt"
            if preferred.is_file():
                return str(preferred)
            found = sorted(path.glob("*.pt"))
            if found:
                return str(found[0])
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found under {self.cfg.ckpt_path}")

    @property
    def feature_dim(self) -> int:
        return self.cfg.feature_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, V, T, H, W, 3) uint8 -> T tensors of (B, V * block_grid^2, D).

        One sequence is one reconstruction: VGGT fixes scale per call. The
        policy therefore extracts each future horizon as its own sequence
        (see ``MMABCPolicy._extract_future_targets``) so the near and far
        targets are not forced onto a shared scale.

        A 5-dim input (no timestep axis) is treated as a single timestep.
        """
        if images.dim() == 5:
            images = images[:, :, None]
        b, v, t = images.shape[:3]
        flat = images.permute(0, 2, 1, 3, 4, 5).reshape(b, t * v, *images.shape[3:])

        dtype = next(self.model.parameters()).dtype
        x = _to_teacher_batch(flat, self.cfg.image_size, normalize=False, dtype=dtype)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            tokens_list, patch_start = self.model.aggregator(x)
        tokens = tokens_list[-1][:, :, patch_start:]  # (B, T*V, P, D)

        grid = self.cfg.image_size // self.cfg.patch_size
        pooled = _pool_to_grid(
            tokens.reshape(b * t * v, tokens.shape[2], tokens.shape[3]).float(),
            grid,
            self.cfg.block_grid,
        )
        per_view = self.cfg.block_grid**2
        pooled = pooled.view(b, t, v * per_view, -1)
        return pooled


class DinoV3Teacher(nn.Module):
    """Semantic patch tokens from a DINOv3 ViT checkpoint."""

    def __init__(self, cfg: TeacherConfig) -> None:
        super().__init__()
        self.cfg = cfg
        repo = str(Path(cfg.repo_path).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch.hub

        self.model = torch.hub.load(
            repo, "dinov3_vitl16", source="local", weights=cfg.ckpt_path
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.cfg.feature_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, V, T, H, W, 3) uint8 -> (B, T, V * block_grid^2, D)."""
        if images.dim() == 5:
            images = images[:, :, None]
        b, v, t = images.shape[:3]
        flat_imgs = images.permute(0, 2, 1, 3, 4, 5).reshape(b, t * v, *images.shape[3:])

        dtype = next(self.model.parameters()).dtype
        x = _to_teacher_batch(flat_imgs, self.cfg.image_size, normalize=True, dtype=dtype)
        flat = x.view(b * t * v, *x.shape[2:])
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self.model.forward_features(flat)
        tokens = out["x_norm_patchtokens"].float()
        grid = self.cfg.image_size // self.cfg.patch_size
        pooled = _pool_to_grid(tokens, grid, self.cfg.block_grid)
        return pooled.view(b, t, v * self.cfg.block_grid**2, -1)


def build_teacher(name: str, cfg: TeacherConfig) -> nn.Module:
    if name == "vggt":
        return VGGTOmegaTeacher(cfg)
    if name == "dino":
        return DinoV3Teacher(cfg)
    raise ValueError(f"unknown teacher {name!r}")
