"""Deployable DINOv3-L/16 plus shared spatiotemporal adapter."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .attention import TransformerBlock
from .types import AdapterStates, StudentOutput


class SharedAdapterBlock(nn.Module):
    def __init__(
        self,
        width: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        head_dim = width // num_heads
        self.frame = TransformerBlock(
            width,
            num_heads,
            (head_dim // 2, head_dim // 2),
            100.0,
            mlp_ratio,
            layer_scale_init=layer_scale_init,
        )
        self.patch_temporal = TransformerBlock(
            width,
            num_heads,
            (head_dim,),
            10000.0,
            mlp_ratio,
            layer_scale_init=layer_scale_init,
        )
        self.special_inter = TransformerBlock(
            width,
            num_heads,
            (head_dim,),
            10000.0,
            mlp_ratio,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        frame_positions: torch.Tensor,
        spatial_positions: torch.Tensor,
        num_special: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_tokens, width = tokens.shape
        frame_tokens = tokens.flatten(0, 1)
        frame_positions_2d = spatial_positions.expand(batch_size, num_frames, -1, -1).flatten(0, 1)
        local = self.frame(frame_tokens, frame_positions_2d).unflatten(0, (batch_size, num_frames))

        special = local[:, :, :num_special].reshape(batch_size, num_frames * num_special, width)
        special_times = frame_positions.unsqueeze(-1).expand(-1, -1, num_special).reshape(batch_size, -1, 1)
        special = self.special_inter(special, special_times).reshape(batch_size, num_frames, num_special, width)

        patches = local[:, :, num_special:].permute(0, 2, 1, 3).flatten(0, 1)
        patch_times = frame_positions[:, None, :, None].expand(-1, num_tokens - num_special, -1, -1).flatten(0, 1)
        patches = self.patch_temporal(patches, patch_times)
        patches = patches.unflatten(0, (batch_size, num_tokens - num_special)).permute(0, 2, 1, 3)
        return local, torch.cat([special, patches], dim=2)


class SharedSpatiotemporalAdapter(nn.Module):
    num_camera_tokens = 1

    def __init__(
        self,
        input_dim: int = 1024,
        width: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        patch_grid: int = 16,
        supervision_taps: tuple[int, ...] | list[int] | None = None,
        use_activation_checkpointing: bool = False,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError(f"Adapter depth must be positive, got {self.depth}")
        self.num_register_tokens = int(num_register_tokens)
        self.num_special_tokens = self.num_camera_tokens + self.num_register_tokens
        self.patch_grid = int(patch_grid)
        self.supervision_taps = self._resolve_supervision_taps(supervision_taps)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.layer_scale_init = None if layer_scale_init is None else float(layer_scale_init)
        self.input_projection = nn.Linear(input_dim, width)
        self.cls_projection = nn.Linear(input_dim, width)
        self.camera_embedding = nn.Parameter(torch.zeros(width))
        self.camera_frame_type = nn.Parameter(torch.zeros(2, width))
        self.register_templates = nn.Parameter(torch.empty(2, num_register_tokens, width))
        nn.init.normal_(self.register_templates, std=1e-3)
        self.blocks = nn.ModuleList(
            [SharedAdapterBlock(width, num_heads, mlp_ratio, self.layer_scale_init) for _ in range(depth)]
        )
        head_dim = width // num_heads
        self.final_fusion = TransformerBlock(
            width,
            num_heads,
            (head_dim // 2, head_dim // 2),
            100.0,
            mlp_ratio,
            layer_scale_init=self.layer_scale_init,
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, input_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("spatial_positions", self._make_spatial_positions(), persistent=False)

    def _resolve_supervision_taps(
        self,
        supervision_taps: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        if supervision_taps is None:
            count = min(4, self.depth)
            supervision_taps = tuple((index + 1) * self.depth // count - 1 for index in range(count))
        taps = tuple(int(index) for index in supervision_taps)
        if taps != tuple(sorted(set(taps))):
            raise ValueError(f"supervision_taps must be sorted and unique, got {taps}")
        if not taps or taps[0] < 0 or taps[-1] >= self.depth:
            raise ValueError(f"supervision_taps {taps} must index adapter depth {self.depth}")
        return taps

    def _make_spatial_positions(self) -> torch.Tensor:
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(self.patch_grid),
                torch.arange(self.patch_grid),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        special = torch.zeros(self.num_special_tokens, 2, dtype=coordinates.dtype)
        return torch.cat([special, coordinates], dim=0).unsqueeze(0)

    def _special_tokens(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = cls_tokens.shape
        frame_types = torch.ones(num_frames, dtype=torch.long, device=cls_tokens.device)
        frame_types[0] = 0
        camera = self.cls_projection(cls_tokens)
        camera = camera + self.camera_embedding + self.camera_frame_type[frame_types][None]
        registers = self.register_templates[frame_types][None].expand(batch_size, -1, -1, -1)
        return torch.cat([camera.unsqueeze(2), registers], dim=2)

    def _run_block(
        self,
        block: SharedAdapterBlock,
        tokens: torch.Tensor,
        frame_positions: torch.Tensor,
        spatial_positions: torch.Tensor,
        collect_local: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not (self.use_activation_checkpointing and self.training and torch.is_grad_enabled()):
            local, inter = block(tokens, frame_positions, spatial_positions, self.num_special_tokens)
            return (local if collect_local else None), inter

        if collect_local:
            return activation_checkpoint(
                block,
                tokens,
                frame_positions,
                spatial_positions,
                self.num_special_tokens,
                use_reentrant=False,
            )

        def inter_only(
            value: torch.Tensor,
            temporal_positions: torch.Tensor,
            spatial: torch.Tensor,
        ) -> torch.Tensor:
            return block(value, temporal_positions, spatial, self.num_special_tokens)[1]

        inter = activation_checkpoint(
            inter_only,
            tokens,
            frame_positions,
            spatial_positions,
            use_reentrant=False,
        )
        return None, inter

    def _fuse_final(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_activation_checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(self.final_fusion, tokens, positions, use_reentrant=False)
        return self.final_fusion(tokens, positions)

    def _forward(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: torch.Tensor,
        frame_positions: torch.Tensor,
        collect_states: bool,
    ) -> tuple[torch.Tensor, AdapterStates | None]:
        batch_size, num_frames, num_patches, _ = patch_tokens.shape
        if num_patches != self.patch_grid**2:
            raise ValueError(f"Expected {self.patch_grid**2} patches, got {num_patches}")
        if frame_positions.shape != (batch_size, num_frames):
            raise ValueError(f"Expected frame_positions {(batch_size, num_frames)}, got {tuple(frame_positions.shape)}")
        tokens = torch.cat([self._special_tokens(cls_tokens), self.input_projection(patch_tokens)], dim=2)
        local_states = []
        inter_states = []
        spatial = self.spatial_positions.to(device=tokens.device)
        taps = set(self.supervision_taps) if collect_states else set()
        for block_index, block in enumerate(self.blocks):
            collect_local = block_index in taps
            local, tokens = self._run_block(block, tokens, frame_positions, spatial, collect_local)
            if collect_local:
                if local is None:
                    raise RuntimeError("Tapped adapter block did not return its local state")
                local_states.append(local)
                inter_states.append(tokens)

        fused = tokens.flatten(0, 1)
        positions = spatial.expand(batch_size, num_frames, -1, -1).flatten(0, 1)
        fused = self._fuse_final(fused, positions).unflatten(0, (batch_size, num_frames))
        final_patches = fused[:, :, self.num_special_tokens :]
        output = patch_tokens + self.output_projection(self.output_norm(final_patches))
        states = AdapterStates(tuple(local_states), tuple(inter_states)) if collect_states else None
        return output, states

    def forward_with_states(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: torch.Tensor,
        frame_positions: torch.Tensor,
    ) -> StudentOutput:
        output, states = self._forward(patch_tokens, cls_tokens, frame_positions, collect_states=True)
        if states is None:
            raise RuntimeError("Adapter state collection unexpectedly returned no states")
        return StudentOutput(output, states)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: torch.Tensor,
        frame_positions: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self._forward(patch_tokens, cls_tokens, frame_positions, collect_states=False)
        return output


class StudentVideoEncoder(nn.Module):
    """The sole deployable module and its deliberately small interface."""

    valid_lengths = (4, 8, 12, 16)

    def __init__(
        self,
        dino: nn.Module,
        adapter: SharedSpatiotemporalAdapter,
        use_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dino = dino
        self.adapter = adapter
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.register_buffer(
            "_dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _dino_features(
        self,
        frames: torch.Tensor,
        patch_mask: torch.Tensor | None,
        require_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames = frames.shape[:2]
        normalized = ((frames.flatten(0, 1) - self._dino_mean) / self._dino_std).contiguous()
        masks = patch_mask.flatten(0, 1) if patch_mask is not None else None
        grad_context = nullcontext() if require_grad else torch.no_grad()
        with grad_context:
            if self.use_activation_checkpointing and self.training and require_grad:
                features = activation_checkpoint(
                    self.dino.forward_features,
                    normalized,
                    masks,
                    use_reentrant=False,
                )
            else:
                features = self.dino.forward_features(normalized, masks)
        patches = features["x_norm_patchtokens"].unflatten(0, (batch_size, num_frames))
        cls = features["x_norm_clstoken"].unflatten(0, (batch_size, num_frames))
        return patches, cls

    def _validate(
        self,
        frames: torch.Tensor,
        frame_positions: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> None:
        if frames.ndim != 5 or frames.shape[2:] != (3, 256, 256):
            raise ValueError(f"Expected frames [B,T,3,256,256], got {tuple(frames.shape)}")
        batch_size, num_frames = frames.shape[:2]
        if num_frames not in self.valid_lengths:
            raise ValueError(f"T must be one of {self.valid_lengths}, got {num_frames}")
        if frame_positions.shape != (batch_size, num_frames):
            raise ValueError(f"Expected frame_positions {(batch_size, num_frames)}, got {frame_positions.shape}")
        if patch_mask is not None and patch_mask.shape != (batch_size, num_frames, 256):
            raise ValueError(f"Expected patch_mask {(batch_size, num_frames, 256)}, got {patch_mask.shape}")

    def _forward_with_states(
        self,
        frames: torch.Tensor,
        frame_positions: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        *,
        dino_grad: bool = True,
    ) -> StudentOutput:
        self._validate(frames, frame_positions, patch_mask)
        patches, cls = self._dino_features(frames, patch_mask, require_grad=dino_grad)
        return self.adapter.forward_with_states(patches, cls, frame_positions)

    def forward(
        self,
        frames: torch.Tensor,
        frame_positions: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(frames, frame_positions, patch_mask)
        patches, cls = self._dino_features(frames, patch_mask)
        return self.adapter(patches, cls, frame_positions)


class DINOOnlyVideoEncoder(nn.Module):
    """Raw DINOv3 ablation with the same deployed tensor interface."""

    valid_lengths = StudentVideoEncoder.valid_lengths

    def __init__(self, dino: nn.Module) -> None:
        super().__init__()
        self.dino = dino
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        frames: torch.Tensor,
        frame_positions: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2:] != (3, 256, 256):
            raise ValueError(f"Expected frames [B,T,3,256,256], got {tuple(frames.shape)}")
        batch_size, num_frames = frames.shape[:2]
        if num_frames not in self.valid_lengths or frame_positions.shape != (batch_size, num_frames):
            raise ValueError("Invalid DINO-only video shape or frame positions")
        normalized = (frames.flatten(0, 1) - self._mean) / self._std
        masks = patch_mask.flatten(0, 1) if patch_mask is not None else None
        output = self.dino.forward_features(normalized, masks)["x_norm_patchtokens"]
        return output.unflatten(0, (batch_size, num_frames))
