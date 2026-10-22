"""V-JEPA 2.1 online targets, masking, predictor, and latent loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TransformerBlock
from .types import VJEPATargets


class LocalVJEPABackend(nn.Module):
    def __init__(self, teacher: nn.Module, patch_grid: int = 16, tubelet_size: int = 2) -> None:
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.patch_grid = int(patch_grid)
        self.tubelet_size = int(tubelet_size)
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def infer(self, frames: torch.Tensor) -> VJEPATargets:
        batch_size, num_frames = frames.shape[:2]
        normalized = frames.permute(0, 2, 1, 3, 4)
        normalized = (normalized - self._mean) / self._std
        with torch.inference_mode():
            tokens = self.teacher(normalized)
        expected = (num_frames // self.tubelet_size) * self.patch_grid**2
        if tokens.shape[1] != expected:
            raise RuntimeError(f"V-JEPA teacher returned {tokens.shape[1]} tokens, expected {expected}")
        tokens = tokens.reshape(batch_size, num_frames // self.tubelet_size, self.patch_grid**2, -1)
        return VJEPATargets(tokens.detach().clone())


class SpatialBlockMaskGenerator:
    """Curriculum over alternating V-JEPA small/large tube-mask profiles."""

    def __init__(
        self,
        patch_grid: int = 16,
        curriculum_steps: int = 0,
        start_union_min: float = 0.35,
        start_union_max: float = 0.55,
        union_min: float = 0.5,
        union_max: float = 0.9,
        small_area_start: float = 0.07,
        small_area_end: float = 0.15,
        large_area_start: float = 0.25,
        large_area_end: float = 0.70,
        max_attempts: int = 10,
    ) -> None:
        self.patch_grid = int(patch_grid)
        self.curriculum_steps = int(curriculum_steps)
        self.start_union_min = float(start_union_min)
        self.start_union_max = float(start_union_max)
        self.union_min = float(union_min)
        self.union_max = float(union_max)
        self.profile_blocks = (8, 2)
        self.profile_area_starts = (float(small_area_start), float(large_area_start))
        self.profile_area_ends = (float(small_area_end), float(large_area_end))
        self.max_attempts = int(max_attempts)

    def curriculum_progress(self, update_step: int) -> float:
        if self.curriculum_steps <= 0:
            return 1.0
        return min(max(int(update_step), 0) / self.curriculum_steps, 1.0)

    def schedule(self, update_step: int) -> tuple[int, float, float, float, float]:
        progress = self.curriculum_progress(update_step)
        profile_index = int(update_step) % len(self.profile_blocks)
        area = self.profile_area_starts[profile_index] + progress * (
            self.profile_area_ends[profile_index] - self.profile_area_starts[profile_index]
        )
        union_min = self.start_union_min + progress * (self.union_min - self.start_union_min)
        union_max = self.start_union_max + progress * (self.union_max - self.start_union_max)
        return self.profile_blocks[profile_index], area, union_min, union_max, progress

    def _rectangle(self, area_fraction: float, device: torch.device) -> tuple[int, int, int, int]:
        area = area_fraction * self.patch_grid**2
        aspect = torch.empty((), device=device).uniform_(0.75, 1.5).item()
        height = min(self.patch_grid, max(1, round(math.sqrt(area / aspect))))
        width = min(self.patch_grid, max(1, round(math.sqrt(area * aspect))))
        top = int(torch.randint(self.patch_grid - height + 1, (), device=device).item())
        left = int(torch.randint(self.patch_grid - width + 1, (), device=device).item())
        return top, left, height, width

    def _one(
        self,
        num_blocks: int,
        area: float,
        union_min: float,
        union_max: float,
        device: torch.device,
    ) -> torch.Tensor:
        for _ in range(self.max_attempts):
            mask = torch.zeros(self.patch_grid, self.patch_grid, dtype=torch.bool, device=device)
            for _ in range(num_blocks):
                top, left, height, width = self._rectangle(area, device)
                mask[top : top + height, left : left + width] = True
            coverage = mask.float().mean().item()
            if union_min <= coverage <= union_max:
                return mask.flatten()
        raise RuntimeError(f"Could not sample a block mask with union coverage in [{union_min}, {union_max}]")

    def __call__(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        update_step: int,
    ) -> torch.Tensor:
        num_blocks, area, union_min, union_max, _ = self.schedule(update_step)
        spatial = torch.stack([self._one(num_blocks, area, union_min, union_max, device) for _ in range(batch_size)])
        return spatial[:, None].expand(-1, num_frames, -1).clone()


class VJEPAPredictor(nn.Module):
    LEGACY_MASKING_SEMANTICS = 1
    CURRENT_MASKING_SEMANTICS = 2

    def __init__(
        self,
        input_dim: int = 1024,
        width: int = 384,
        depth: int = 4,
        num_heads: int = 12,
        patch_grid: int = 16,
    ) -> None:
        super().__init__()
        self.patch_grid = int(patch_grid)
        self.tubelet_projection = nn.Conv3d(
            input_dim,
            width,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        self.mask_embedding = nn.Parameter(torch.zeros(width))
        head_dim = width // num_heads
        axis_dim = 2 * ((head_dim // 3) // 2)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width,
                    num_heads,
                    (axis_dim, axis_dim, axis_dim),
                    10000.0,
                    4.0,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, input_dim)
        self.register_buffer(
            "masking_semantics_version",
            torch.tensor(self.CURRENT_MASKING_SEMANTICS, dtype=torch.int64),
        )
        self._use_legacy_additive_masking = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        projection_key = prefix + "tubelet_projection.weight"
        semantics_key = prefix + "masking_semantics_version"
        projection_weight = state_dict.get(projection_key)
        legacy_projection = projection_weight is not None and projection_weight.ndim == 2
        if semantics_key not in state_dict:
            semantics_version = self.LEGACY_MASKING_SEMANTICS if legacy_projection else self.CURRENT_MASKING_SEMANTICS
            state_dict[semantics_key] = self.masking_semantics_version.new_tensor(semantics_version)
        else:
            semantics_version = int(state_dict[semantics_key])
        if legacy_projection:
            state_dict[projection_key] = projection_weight[:, :, None, None, None].repeat(1, 1, 2, 1, 1) * 0.5
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._use_legacy_additive_masking = semantics_version == self.LEGACY_MASKING_SEMANTICS

    def _tubelet_tokens(self, student_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, channels = student_tokens.shape
        tokens = student_tokens.reshape(batch_size, num_frames, self.patch_grid, self.patch_grid, channels)
        tokens = tokens.permute(0, 4, 1, 2, 3)
        tokens = self.tubelet_projection(tokens)
        return tokens.permute(0, 2, 3, 4, 1).flatten(2, 3)

    def _positions(
        self,
        batch_size: int,
        num_tubelets: int,
        device: torch.device,
    ) -> torch.Tensor:
        time, height, width = torch.meshgrid(
            torch.arange(num_tubelets, device=device),
            torch.arange(self.patch_grid, device=device),
            torch.arange(self.patch_grid, device=device),
            indexing="ij",
        )
        positions = torch.stack([time, height, width], dim=-1).reshape(1, -1, 3)
        return positions.expand(batch_size, -1, -1)

    def forward(self, student_tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_patches, _ = student_tokens.shape
        if num_frames % 2:
            raise ValueError(f"V-JEPA tubelets require even T, got {num_frames}")
        if num_patches != self.patch_grid**2:
            raise ValueError(f"Expected {self.patch_grid**2} patches, got {num_patches}")
        tokens = self._tubelet_tokens(student_tokens)
        tubelet_mask = patch_mask[:, 0::2]
        mask_tokens = self.mask_embedding.to(dtype=tokens.dtype).view(1, 1, 1, -1)
        if self._use_legacy_additive_masking:
            tokens = tokens + tubelet_mask.unsqueeze(-1) * mask_tokens
        else:
            tokens = torch.where(tubelet_mask.unsqueeze(-1), mask_tokens, tokens)
        tokens = tokens.flatten(1, 2)
        positions = self._positions(batch_size, num_frames // 2, tokens.device)
        for block in self.blocks:
            tokens = block(tokens, positions)
        tokens = self.output_projection(self.output_norm(tokens))
        return tokens.reshape(batch_size, num_frames // 2, num_patches, -1)


class VJEPAObjective(nn.Module):
    def __init__(self, predictor: VJEPAPredictor, patch_grid: int = 16) -> None:
        super().__init__()
        self.predictor = predictor
        self.patch_grid = int(patch_grid)
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(patch_grid), torch.arange(patch_grid), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2)
        self.register_buffer("coordinates", coordinates.float(), persistent=False)

    def _context_weights(self, spatial_mask: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(spatial_mask, dtype=torch.float32)
        for batch_index in range(spatial_mask.shape[0]):
            masked_coordinates = self.coordinates[spatial_mask[batch_index]]
            distances = torch.cdist(self.coordinates, masked_coordinates).amin(dim=-1)
            weights[batch_index] = torch.where(
                spatial_mask[batch_index],
                0.0,
                torch.rsqrt(distances + 1.0),
            )
        return weights

    def forward(
        self,
        student_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        targets: VJEPATargets,
        context_weight: float,
    ) -> dict[str, torch.Tensor]:
        prediction = self.predictor(student_tokens, patch_mask)
        target = targets.tokens.detach()
        prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        target = F.layer_norm(target.float(), (target.shape[-1],))
        token_loss = (prediction - target).abs().mean(dim=-1)
        tubelet_mask = patch_mask[:, 0::2]
        masked = token_loss[tubelet_mask].mean()

        spatial_mask = tubelet_mask[:, 0]
        spatial_weights = self._context_weights(spatial_mask)
        weights = spatial_weights[:, None].expand_as(token_loss)
        context = (token_loss * weights).sum() / weights.sum()
        loss = masked + float(context_weight) * context
        return {
            "loss": loss,
            "loss_vjepa_masked": masked,
            "loss_vjepa_context": context,
            "vjepa_context_weight": loss.new_tensor(float(context_weight)),
            "mask_coverage": tubelet_mask.float().mean(),
        }
