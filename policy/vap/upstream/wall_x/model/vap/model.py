"""Token-level VAP model used before Wan VAE unpatchify integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from .block import VAPBlock
from .initialization import copy_linear, resize_vector
from .kv_cache import VAPActionPrefixKVCache, VAPAttentionKVCache
from .layout import (
    ActionStageLayout,
    AttentionLayout,
    InterleaveLayout,
    ParallelTeacherForcingLayout,
)
from .memory import FirstFrameMemoryProjector, FirstFrameMemoryTokens
from .rope import (
    HIERARCHICAL_ROPE_AXES_DIM,
    action_position_ids as build_action_position_ids,
    build_rope_4d_cache,
    text_position_ids as build_text_position_ids,
)


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor, dim: int, *, max_period: float = 10000.0
) -> torch.Tensor:
    if timestep.ndim not in (1, 2):
        raise ValueError(
            f"timestep must be [B] or per-token [B,S], got {tuple(timestep.shape)}"
        )
    half = dim // 2
    frequencies = torch.pow(
        torch.tensor(max_period, device=timestep.device, dtype=torch.float64),
        -torch.arange(half, device=timestep.device, dtype=torch.float64) / max(half, 1),
    )
    phase = timestep.to(torch.float64)[..., None] * frequencies
    embedding = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[..., :1])), dim=-1)
    return embedding.to(torch.float32)


class ExpertTimeConditioner(nn.Module):
    """An expert-local timestep path producing six AdaLN parameters."""

    def __init__(
        self,
        dim: int,
        frequency_dim: int = 256,
        *,
        time_embedding: nn.Module | None = None,
        time_projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.time_embedding = (
            nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            if time_embedding is None
            else time_embedding
        )
        self.time_projection = (
            nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
            if time_projection is None
            else time_projection
        )

    def embed(self, timestep: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        embedding = sinusoidal_timestep_embedding(timestep, self.frequency_dim)
        # DMuon exposes fp32 master parameters outside the module's forward
        # hook, then materializes bf16 live parameters at the hook boundary.
        # Deriving the activation dtype from ``weight.dtype`` here therefore
        # observes storage dtype rather than compute dtype. The caller owns the
        # live-forward dtype and is the only reliable source at this point.
        embedding = embedding.to(dtype=dtype)
        return self.time_embedding(embedding).to(dtype)

    def project(self, embedding: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        modulation = self.time_projection(embedding.to(dtype=dtype))
        return modulation.unflatten(-1, (6, -1)).to(dtype)

    def forward(self, timestep: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.project(self.embed(timestep, dtype=dtype), dtype=dtype)


class ExpertOutputHead(nn.Module):
    """Wan-compatible timestep-conditioned output head for one expert."""

    def __init__(self, dim: int, output_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, output_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self, hidden: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        if time_embedding.ndim == 3:
            if time_embedding.shape[:2] != hidden.shape[:2]:
                raise ValueError("per-token output timestep must align with hidden")
            parameters = (
                self.modulation[:, None].to(time_embedding)
                + time_embedding[:, :, None, :]
            )
            shift, scale = parameters.unbind(dim=2)
        elif time_embedding.ndim == 2:
            if time_embedding.shape[0] != hidden.shape[0]:
                raise ValueError("output timestep batch must align with hidden")
            shift, scale = (
                self.modulation.to(time_embedding) + time_embedding[:, None, :]
            ).unbind(dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            raise ValueError("output timestep embedding must be [B,D] or [B,S,D]")
        return self.head(self.norm(hidden) * (1 + scale) + shift)


@dataclass(frozen=True)
class VAPTokenOutput:
    video_hidden: torch.Tensor
    action_hidden: torch.Tensor
    video_velocity: torch.Tensor
    action_velocity: torch.Tensor
    layout: AttentionLayout


class VideoIdentityEmbedding(nn.Module):
    """Categorical camera/modality identity, kept outside metric MRoPE axes."""

    def __init__(self, max_views: int, dim: int) -> None:
        super().__init__()
        self.camera = nn.Embedding(max_views, dim)
        self.modality = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.zeros_(self.camera.weight)

    def forward(self, tokens: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        if tuple(camera_ids.shape) != tuple(tokens.shape[:2]):
            raise ValueError("video camera_ids must align with [B,S] video tokens")
        return tokens + self.camera(camera_ids) + self.modality.to(tokens)


class ActionIdentityEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.modality = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.modality.to(tokens)


class ChunkStateFiLM(nn.Module):
    """Apply window/chunk-start proprioception to Action tokens with FiLM."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("State FiLM dimensions must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if not 0.0 <= float(dropout_probability) < 1.0:
            raise ValueError("State FiLM dropout probability must be in [0, 1)")
        self.dropout_probability = float(dropout_probability)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * self.action_dim),
        )
        # Identity initialization makes switching State FiLM on safe when the
        # Action expert is initialized from Wan rather than a State checkpoint.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        action: torch.Tensor,
        states: torch.Tensor,
        *,
        layout: AttentionLayout,
    ) -> torch.Tensor:
        if states.ndim == 2:
            states = states[:, None, :]
        if states.ndim != 3 or states.shape[0] != action.shape[0]:
            raise ValueError("state_condition must be [B,D], [B,1,D], or [B,E,D]")
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"state_condition dim {states.shape[-1]} != configured {self.state_dim}"
            )
        if states.shape[1] == 1:
            states = states.expand(-1, layout.num_chunks, -1)
        elif states.shape[1] != layout.num_chunks:
            raise ValueError(
                "state_condition chunk axis must be 1 or layout.num_chunks="
                f"{layout.num_chunks}, got {states.shape[1]}"
            )
        if not torch.isfinite(states).all():
            raise ValueError("state_condition contains non-finite values")

        film = self.net(states.to(device=action.device, dtype=action.dtype))
        if self.training and self.dropout_probability > 0.0:
            # Conditioning dropout, not feature dropout: one decision per
            # sample is locked across both A0/A1 chunks. Do not use inverted
            # dropout scaling; kept conditioning must match inference strength.
            keep = (
                torch.rand((film.shape[0], 1, 1), device=film.device)
                >= self.dropout_probability
            )
            film = film * keep.to(dtype=film.dtype)
        scale, shift = film.chunk(2, dim=-1)
        scale = torch.tanh(scale)
        scale = scale.repeat_interleave(layout.action_tokens_per_chunk, dim=1)
        shift = shift.repeat_interleave(layout.action_tokens_per_chunk, dim=1)
        if isinstance(layout, ParallelTeacherForcingLayout):
            scale = layout.expand_action(scale, scale)
            shift = layout.expand_action(shift, shift)
        elif isinstance(layout, ActionStageLayout):
            scale = scale[:, : layout.num_action_tokens]
            shift = shift[:, : layout.num_action_tokens]
        if scale.shape != action.shape or shift.shape != action.shape:
            raise ValueError(
                "expanded State FiLM shape does not match Action tokens: "
                f"{tuple(scale.shape)} vs {tuple(action.shape)}"
            )
        return action * (1.0 + scale) + shift


class GlobalTimeAnchor(nn.Module):
    """Text token fixed at the current planning-window origin ``(0,0,0,0)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        anchor = self.token.to(context).expand(context.shape[0], -1, -1)
        return torch.cat((anchor, context), dim=1)


class VAPTokenModel(nn.Module):
    """Functional two-stream core with independent Video/Action diffusion time.

    Video inputs are already Wan patch tokens ``[B,Sv,Dv]``. Raw RGB/latent
    patchify and unpatchify remain owned by the Wan2.2 pipeline integration.
    Actions enter as normalized continuous values ``[B,Sa,action_input_dim]``.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        action_dim: int,
        action_input_dim: int,
        video_output_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        video_ffn_dim: int,
        action_ffn_dim: int,
        num_chunks: int = 2,
        rope_axes_dim: Sequence[int] = HIERARCHICAL_ROPE_AXES_DIM,
        max_views: int = 64,
        frequency_dim: int = 256,
        attention_backend: str = "sdpa",
        state_input_dim: int | None = None,
        state_film_hidden_dim: int = 512,
        state_film_dropout_probability: float = 0.0,
        first_frame_memory_enabled: bool = False,
        first_frame_memory_reader_layers: Sequence[int] = (3, 9, 15, 21, 27),
        wan_dit: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        self.video_dim = video_dim
        self.action_dim = action_dim
        self.action_input_dim = action_input_dim
        self.num_chunks = num_chunks
        self.max_views = max_views
        self.head_dim = int(head_dim)
        self.rope_axes_dim = tuple(int(value) for value in rope_axes_dim)
        self.attention_backend = attention_backend
        self.first_frame_memory_enabled = bool(first_frame_memory_enabled)
        self.first_frame_memory_reader_layers = tuple(
            int(index) for index in first_frame_memory_reader_layers
        )
        if self.first_frame_memory_enabled and (
            not self.first_frame_memory_reader_layers
            or any(
                index < 0 or index >= num_layers
                for index in self.first_frame_memory_reader_layers
            )
        ):
            raise ValueError(
                "first-frame memory reader layers must exist in the loaded DiT"
            )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )
        self.video_identity = VideoIdentityEmbedding(max_views, video_dim)
        self.action_identity = ActionIdentityEmbedding(action_dim)
        self.state_film = (
            None
            if state_input_dim is None
            else ChunkStateFiLM(
                state_input_dim,
                action_dim,
                state_film_hidden_dim,
                state_film_dropout_probability,
            )
        )
        self.global_time_anchor = GlobalTimeAnchor(video_dim)
        if wan_dit is not None:
            if getattr(wan_dit, "dim", None) != video_dim:
                raise ValueError("wan_dit.dim must equal video_dim")
            if len(wan_dit.blocks) != num_layers:
                raise ValueError("wan_dit layer count must equal num_layers")
        self.video_time = ExpertTimeConditioner(
            video_dim,
            frequency_dim,
            time_embedding=None if wan_dit is None else wan_dit.time_embedding,
            time_projection=None if wan_dit is None else wan_dit.time_projection,
        )
        self.action_time = ExpertTimeConditioner(action_dim, frequency_dim)
        wan_blocks = [None] * num_layers if wan_dit is None else wan_dit.blocks
        memory_reader_layers = (
            set(self.first_frame_memory_reader_layers)
            if self.first_frame_memory_enabled
            else set()
        )
        self.blocks = nn.ModuleList(
            [
                VAPBlock(
                    video_dim=video_dim,
                    action_dim=action_dim,
                    video_ffn_dim=video_ffn_dim,
                    action_ffn_dim=action_ffn_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    rope_axes_dim=tuple(rope_axes_dim),
                    attention_backend=attention_backend,
                    first_frame_memory_reader=index in memory_reader_layers,
                    wan_block=wan_block,
                )
                for index, wan_block in enumerate(wan_blocks)
            ]
        )
        self.video_head = (
            ExpertOutputHead(video_dim, video_output_dim)
            if wan_dit is None
            else wan_dit.head
        )
        if self.video_head.head.out_features != video_output_dim:
            raise ValueError(
                f"video_output_dim {video_output_dim} does not match Wan head "
                f"{self.video_head.head.out_features}"
            )
        self.text_embedding = None if wan_dit is None else wan_dit.text_embedding
        self.text_input_projector = None  # Opt-in; no parameters/RNG in T5 mode.
        self.video_patch_embedding = (
            None if wan_dit is None else wan_dit.patch_embedding
        )
        self.video_patch_size = None if wan_dit is None else tuple(wan_dit.patch_size)
        self.action_head = ExpertOutputHead(action_dim, action_input_dim)
        nn.init.zeros_(self.action_head.head.weight)
        nn.init.zeros_(self.action_head.head.bias)
        # Allocate this after all historical modules so enabling Memory cannot
        # change RNG-initialized baseline weights in synthetic/no-Wan tests.
        self.first_frame_memory = (
            FirstFrameMemoryProjector(
                dim=video_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                rope_axes_dim=rope_axes_dim,
                wan_attention=(
                    None if wan_dit is None else wan_dit.blocks[0].cross_attn
                ),
            )
            if self.first_frame_memory_enabled
            else None
        )
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpoint_num_layers: int | None = None
        if wan_dit is not None:
            self._initialize_action_time_from_wan(wan_dit)
            self._initialize_action_head_from_wan(wan_dit)

    def gradient_checkpointing_enable(self, num_layers: int | None = None) -> None:
        self.gradient_checkpointing_enabled = True
        self.gradient_checkpoint_num_layers = num_layers

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpoint_num_layers = None

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        if self.video_patch_embedding is None:
            raise RuntimeError("patchify requires VAPTokenModel.from_wan_dit")
        parameter = self.video_patch_embedding.weight
        # Do not use ``parameter.dtype`` here: under DMuon it is the fp32
        # master dtype before the Conv3d hook installs bf16 live parameters.
        # Latents already carry the model's declared live-forward dtype; the
        # DMuon patch boundary additionally validates/casts them defensively.
        return self.video_patch_embedding(latents.to(device=parameter.device))

    def unpatchify(
        self, patch_tokens: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        if self.video_patch_size is None:
            raise RuntimeError("unpatchify requires VAPTokenModel.from_wan_dit")
        patch_t, patch_h, patch_w = self.video_patch_size
        frames, height, width = grid
        return rearrange(
            patch_tokens,
            "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=frames,
            h=height,
            w=width,
            pt=patch_t,
            ph=patch_h,
            pw=patch_w,
        )

    @classmethod
    def from_wan_dit(
        cls,
        wan_dit: nn.Module,
        *,
        action_input_dim: int,
        action_dim: int = 1536,
        action_ffn_dim: int | None = None,
        num_chunks: int = 2,
        rope_axes_dim: Sequence[int] = HIERARCHICAL_ROPE_AXES_DIM,
        max_views: int = 64,
        expected_num_heads: int | None = None,
        expected_head_dim: int | None = None,
        attention_backend: str = "sdpa",
        state_input_dim: int | None = None,
        state_film_hidden_dim: int = 512,
        state_film_dropout_probability: float = 0.0,
        first_frame_memory_enabled: bool = False,
        first_frame_memory_reader_layers: Sequence[int] = (3, 9, 15, 21, 27),
    ) -> "VAPTokenModel":
        """Adopt loaded Wan modules without allocating a second Video Expert."""

        video_dim = int(wan_dit.dim)
        video_ffn_dim = int(wan_dit.blocks[0].ffn[0].out_features)
        if action_ffn_dim is None:
            action_ffn_dim = round(video_ffn_dim * action_dim / video_dim)
        num_heads = int(wan_dit.num_heads)
        head_dim = video_dim // num_heads
        if expected_num_heads is not None and int(expected_num_heads) != num_heads:
            raise ValueError(
                "configured joint_num_heads does not match the loaded Wan DiT: "
                f"{expected_num_heads} != {num_heads}"
            )
        if expected_head_dim is not None and int(expected_head_dim) != head_dim:
            raise ValueError(
                "configured joint_head_dim does not match the loaded Wan DiT: "
                f"{expected_head_dim} != {head_dim}"
            )
        model = cls(
            video_dim=video_dim,
            action_dim=action_dim,
            action_input_dim=action_input_dim,
            video_output_dim=int(wan_dit.head.head.out_features),
            num_layers=len(wan_dit.blocks),
            num_heads=num_heads,
            head_dim=head_dim,
            video_ffn_dim=video_ffn_dim,
            action_ffn_dim=action_ffn_dim,
            num_chunks=num_chunks,
            rope_axes_dim=rope_axes_dim,
            max_views=max_views,
            frequency_dim=int(wan_dit.freq_dim),
            attention_backend=attention_backend,
            state_input_dim=state_input_dim,
            state_film_hidden_dim=state_film_hidden_dim,
            state_film_dropout_probability=state_film_dropout_probability,
            first_frame_memory_enabled=first_frame_memory_enabled,
            first_frame_memory_reader_layers=first_frame_memory_reader_layers,
            wan_dit=wan_dit,
        )
        reference = next(wan_dit.parameters())
        return model.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def _initialize_action_time_from_wan(self, wan_dit: nn.Module) -> None:
        for dst, src in zip(self.action_time.time_embedding, wan_dit.time_embedding):
            if isinstance(dst, nn.Linear) and isinstance(src, nn.Linear):
                copy_linear(dst, src, allow_resize=True)
        for dst, src in zip(self.action_time.time_projection, wan_dit.time_projection):
            if isinstance(dst, nn.Linear) and isinstance(src, nn.Linear):
                copy_linear(dst, src, allow_resize=True)

    @torch.no_grad()
    def _initialize_action_head_from_wan(self, wan_dit: nn.Module) -> None:
        source = wan_dit.head.modulation.squeeze(0)
        resized = torch.stack(
            [resize_vector(row, self.action_dim) for row in source], dim=0
        ).unsqueeze(0)
        self.action_head.modulation.copy_(
            resized.to(
                device=self.action_head.modulation.device,
                dtype=self.action_head.modulation.dtype,
            )
        )

    @torch.no_grad()
    def initialize_from_wan_dit(self, wan_dit: nn.Module) -> None:
        """Initialize the complete Video path and resized Action transformer path.

        The copy is deliberately strict for all Video tensors. Action attention,
        FFN and timestep weights are interpolated from the corresponding Wan
        tensors; the continuous action encoder/head remain VAP-specific.
        """

        if getattr(wan_dit, "dim", None) != self.video_dim:
            raise ValueError(
                f"Wan dim {getattr(wan_dit, 'dim', None)} != video_dim {self.video_dim}"
            )
        if len(wan_dit.blocks) != len(self.blocks):
            raise ValueError(
                f"Wan layers {len(wan_dit.blocks)} != VAP layers {len(self.blocks)}"
            )
        self.video_time.time_embedding.load_state_dict(
            wan_dit.time_embedding.state_dict(), strict=True
        )
        self.video_time.time_projection.load_state_dict(
            wan_dit.time_projection.state_dict(), strict=True
        )
        self._initialize_action_time_from_wan(wan_dit)
        self._initialize_action_head_from_wan(wan_dit)
        for vap_block, wan_block in zip(self.blocks, wan_dit.blocks):
            vap_block.initialize_from_wan_block(wan_block)
        if self.first_frame_memory is not None:
            self.first_frame_memory.initialize_from_wan_cross_attention(
                wan_dit.blocks[0].cross_attn
            )
        copy_linear(self.video_head.head, wan_dit.head.head, allow_resize=False)
        self.video_head.modulation.copy_(wan_dit.head.modulation)

    def forward(
        self,
        video_tokens: torch.Tensor,
        noisy_actions: torch.Tensor,
        *,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        video_position_ids: torch.Tensor,
        condition_video_tokens: int,
        text_context: torch.Tensor | None = None,
        action_position_ids: torch.Tensor | None = None,
        video_camera_ids: torch.Tensor | None = None,
        layout: AttentionLayout | None = None,
        capture_action_prefix_rank: int | None = None,
        action_prefix_embedding_dropout_scale: torch.Tensor | None = None,
        state_condition: torch.Tensor | None = None,
        first_frame_memory: FirstFrameMemoryTokens | None = None,
        memory_prefix_valid: torch.Tensor | None = None,
    ) -> VAPTokenOutput | tuple[VAPTokenOutput, VAPActionPrefixKVCache]:
        if video_tokens.ndim != 3 or video_tokens.shape[-1] != self.video_dim:
            raise ValueError(
                f"video_tokens must be [B,S,{self.video_dim}], got {video_tokens.shape}"
            )
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_input_dim:
            raise ValueError(
                f"noisy_actions must be [B,S,{self.action_input_dim}], "
                f"got {noisy_actions.shape}"
            )
        batch_size = video_tokens.shape[0]
        if noisy_actions.shape[0] != batch_size:
            raise ValueError("video/action batch sizes must match")
        if video_camera_ids is None:
            raise ValueError(
                "video_camera_ids are required for categorical view identity"
            )
        if tuple(video_camera_ids.shape) != tuple(video_tokens.shape[:2]):
            raise ValueError("video_camera_ids must be [B,Sv]")
        if (
            video_camera_ids.min().item() < 0
            or video_camera_ids.max().item() >= self.max_views
        ):
            raise ValueError(f"video_camera_ids must be in [0,{self.max_views})")
        valid_video_timestep_shapes = {
            (batch_size,),
            (batch_size, video_tokens.shape[1]),
        }
        if tuple(video_timestep.shape) not in valid_video_timestep_shapes:
            raise ValueError("video_timestep must be [B] or per-token [B,Sv]")
        valid_action_timestep_shapes = {
            (batch_size,),
            (batch_size, noisy_actions.shape[1]),
        }
        if tuple(action_timestep.shape) not in valid_action_timestep_shapes:
            raise ValueError("action_timestep must be [B] or per-token [B,Sa]")
        if condition_video_tokens <= 0:
            raise ValueError("condition_video_tokens must be positive")
        if layout is None:
            future_video_tokens = video_tokens.shape[1] - condition_video_tokens
            if future_video_tokens <= 0 or future_video_tokens % self.num_chunks:
                raise ValueError(
                    "future video token length after I must be positive and divisible "
                    "by num_chunks"
                )
            if noisy_actions.shape[1] % self.num_chunks:
                raise ValueError(
                    "action token length must be divisible by configured num_chunks"
                )
            layout = InterleaveLayout(
                num_chunks=self.num_chunks,
                condition_video_tokens=condition_video_tokens,
                video_tokens_per_chunk=future_video_tokens // self.num_chunks,
                action_tokens_per_chunk=noisy_actions.shape[1] // self.num_chunks,
            )
        else:
            if layout.num_chunks != self.num_chunks:
                raise ValueError("attention layout chunk count does not match model")
            if layout.condition_video_tokens != condition_video_tokens:
                raise ValueError(
                    "attention layout condition length does not match input"
                )
            if video_tokens.shape[1] != layout.num_video_tokens:
                raise ValueError("video token length does not match attention layout")
            if noisy_actions.shape[1] != layout.num_action_tokens:
                raise ValueError("action token length does not match attention layout")
        if capture_action_prefix_rank is not None:
            if self.training:
                raise RuntimeError("prefix KV capture is inference-only")
            if not isinstance(layout, (ActionStageLayout, InterleaveLayout)):
                raise TypeError("prefix KV capture requires an IVAVA causal layout")
            if int(capture_action_prefix_rank) not in (1, 3):
                raise ValueError("Action prefix rank must be A0=1 or A1=3")
        if self.first_frame_memory is None:
            if first_frame_memory is not None:
                raise ValueError(
                    "first-frame memory inputs were provided but the module is disabled"
                )
        else:
            if first_frame_memory is None:
                raise ValueError(
                    "first-frame memory is enabled but no Anchor Bank was provided"
                )
            if first_frame_memory.tokens.shape[0] != batch_size:
                raise ValueError("first-frame memory batch size does not match query")
            if first_frame_memory.tokens.shape[-1] != self.video_dim:
                raise ValueError("first-frame memory token width does not match Video")
            if (
                first_frame_memory.camera_ids.min().item() < 0
                or first_frame_memory.camera_ids.max().item() >= self.max_views
            ):
                raise ValueError(
                    f"first-frame camera ids must be in [0,{self.max_views})"
                )
            future_camera_ids = video_camera_ids[:, condition_video_tokens:]
            if future_camera_ids.numel() and not bool(
                (
                    future_camera_ids[:, :, None]
                    == first_frame_memory.camera_ids[:, None, :]
                )
                .any(dim=-1)
                .any(dim=-1)
                .all()
                .item()
            ):
                raise ValueError(
                    "each sample must have at least one future Video query "
                    "with a same-view first-frame anchor"
                )
        if action_position_ids is None:
            base_action_positions = build_action_position_ids(
                batch_size,
                self.num_chunks,
                layout.action_tokens_per_chunk,
                chunk_stride=layout.action_tokens_per_chunk,
                device=noisy_actions.device,
            )
            action_positions = (
                layout.expand_action(base_action_positions, base_action_positions)
                if isinstance(layout, ParallelTeacherForcingLayout)
                else base_action_positions
            )
        else:
            action_positions = action_position_ids
        video_time_embedding = self.video_time.embed(
            video_timestep, dtype=video_tokens.dtype
        )
        video_mod = self.video_time.project(
            video_time_embedding, dtype=video_tokens.dtype
        )
        action = self.action_identity(self.action_encoder(noisy_actions))
        if action_prefix_embedding_dropout_scale is not None:
            if not self.training:
                raise RuntimeError("action prefix embedding dropout is training-only")
            if not isinstance(layout, ParallelTeacherForcingLayout):
                raise TypeError(
                    "action prefix embedding dropout requires a parallel "
                    "teacher-forcing layout"
                )
            expected_scale_shape = (batch_size, 1, self.action_dim)
            if (
                tuple(action_prefix_embedding_dropout_scale.shape)
                != expected_scale_shape
            ):
                raise ValueError(
                    "action prefix embedding dropout scale must be "
                    f"{expected_scale_shape}, got "
                    f"{tuple(action_prefix_embedding_dropout_scale.shape)}"
                )
            prefix_tokens = int(layout.action_tokens_per_chunk)
            if not layout.robust_a0:
                # Preserve the historical tensor expression exactly when the
                # feature is disabled.
                clean_a0 = action[:, :prefix_tokens] * (
                    action_prefix_embedding_dropout_scale.to(action)
                )
                action = torch.cat((clean_a0, action[:, prefix_tokens:]), dim=1)
            else:
                # Physical Action storage is [A0_clean,A0_robust,A0_noisy,
                # A1_clean,A1_noisy], so only the robust copy is weakened.
                prefix_start = prefix_tokens
                prefix_end = prefix_start + prefix_tokens
                dropped_prefix = action[:, prefix_start:prefix_end] * (
                    action_prefix_embedding_dropout_scale.to(action)
                )
                action = torch.cat(
                    (
                        action[:, :prefix_start],
                        dropped_prefix,
                        action[:, prefix_end:],
                    ),
                    dim=1,
                )
        if self.state_film is None:
            if state_condition is not None:
                raise ValueError(
                    "state_condition was provided but State FiLM is disabled"
                )
        else:
            if state_condition is None:
                raise ValueError(
                    "State FiLM is enabled but state_condition was not provided"
                )
            action = self.state_film(action, state_condition, layout=layout)
        action_time_embedding = self.action_time.embed(
            action_timestep, dtype=action.dtype
        )
        action_mod = self.action_time.project(action_time_embedding, dtype=action.dtype)
        context = text_context
        if self.text_input_projector is not None:
            if context is None:
                raise ValueError("Qwen text conditioning requires raw cached context")
            context = self.text_input_projector(context)
        if context is not None and self.text_embedding is not None:
            expected_raw_dim = self.text_embedding[0].in_features
            if context.shape[-1] == expected_raw_dim:
                context = self.text_embedding(context)
            elif context.shape[-1] != self.video_dim:
                raise ValueError(
                    f"text_context dim must be {expected_raw_dim} raw or "
                    f"{self.video_dim} projected"
                )
        text_positions = None
        if context is not None:
            context = self.global_time_anchor(context)
            text_positions = build_text_position_ids(
                batch_size, context.shape[1], device=context.device
            )
        first_frame_memory_context = None
        if first_frame_memory is None:
            # Keep this exact call as the historical Memory-off path.
            video = self.video_identity(video_tokens, video_camera_ids)
        else:
            # One identity-module invocation preserves DMuon boundary semantics
            # while applying the same physical camera embedding to I/V and M0.
            combined_video = torch.cat((video_tokens, first_frame_memory.tokens), dim=1)
            combined_camera_ids = torch.cat(
                (video_camera_ids, first_frame_memory.camera_ids), dim=1
            )
            combined_video = self.video_identity(combined_video, combined_camera_ids)
            video = combined_video[:, : video_tokens.shape[1]]
            memory_tokens = combined_video[:, video_tokens.shape[1] :]
            first_frame_memory_context = self.first_frame_memory(
                memory_tokens,
                position_ids=first_frame_memory.position_ids,
            )
        video_rope_cache = None
        action_rope_cache = None
        text_rope_cache = None
        if capture_action_prefix_rank is not None:
            # Prefix capture is already guarded as inference-only above. Build
            # every immutable phase exactly once and share it across all blocks.
            video_rope_cache = build_rope_4d_cache(
                video_position_ids,
                head_dim=self.head_dim,
                axes_dim=self.rope_axes_dim,
                dtype=video.dtype,
            )
            action_rope_cache = build_rope_4d_cache(
                action_positions,
                head_dim=self.head_dim,
                axes_dim=self.rope_axes_dim,
                dtype=action.dtype,
            )
            if context is not None:
                if text_positions is None:
                    raise AssertionError("text positions were not constructed")
                text_rope_cache = build_rope_4d_cache(
                    text_positions,
                    head_dim=self.head_dim,
                    axes_dim=self.rope_axes_dim,
                    dtype=context.dtype,
                )
        if memory_prefix_valid is not None:
            if capture_action_prefix_rank is not None:
                raise ValueError("History Memory requires full-forward sampling; KV capture is not supported")
            from .history_mask import MemoryValidityLayout
            if memory_prefix_valid.shape[0] != batch_size:
                raise ValueError("History mask batch differs from model batch")
            layout = MemoryValidityLayout(layout, memory_prefix_valid)
            if self.attention_backend == "flex_attention":
                layout.flex_block_mask(video.device)
        checkpoint_layers = self.gradient_checkpoint_num_layers
        if checkpoint_layers is None:
            checkpoint_layers = len(self.blocks)
        layer_caches = []
        for index, block in enumerate(self.blocks):
            if (
                self.training
                and self.gradient_checkpointing_enabled
                and index < checkpoint_layers
            ):

                if block.first_frame_memory_reader:
                    if first_frame_memory_context is None or first_frame_memory is None:
                        raise AssertionError("Memory reader has no Anchor Bank")

                    def block_forward(
                        video_input,
                        action_input,
                        anchor_key,
                        anchor_value,
                        module=block,
                    ):
                        return module(
                            video_input,
                            action_input,
                            video_time_mod=video_mod,
                            action_time_mod=action_mod,
                            video_position_ids=video_position_ids,
                            action_position_ids=action_positions,
                            layout=layout,
                            context=context,
                            text_position_ids=text_positions,
                            video_camera_ids=video_camera_ids,
                            first_frame_memory=VAPAttentionKVCache(
                                key=anchor_key, value=anchor_value
                            ),
                            first_frame_memory_camera_ids=(
                                first_frame_memory.camera_ids
                            ),
                            condition_video_tokens=condition_video_tokens,
                        )

                    video, action = torch.utils.checkpoint.checkpoint(
                        block_forward,
                        video,
                        action,
                        first_frame_memory_context.key,
                        first_frame_memory_context.value,
                        use_reentrant=False,
                    )
                else:

                    def block_forward(video_input, action_input, module=block):
                        return module(
                            video_input,
                            action_input,
                            video_time_mod=video_mod,
                            action_time_mod=action_mod,
                            video_position_ids=video_position_ids,
                            action_position_ids=action_positions,
                            layout=layout,
                            context=context,
                            text_position_ids=text_positions,
                        )

                    video, action = torch.utils.checkpoint.checkpoint(
                        block_forward, video, action, use_reentrant=False
                    )
            else:
                block_result = block(
                    video,
                    action,
                    video_time_mod=video_mod,
                    action_time_mod=action_mod,
                    video_position_ids=video_position_ids,
                    action_position_ids=action_positions,
                    layout=layout,
                    context=context,
                    text_position_ids=text_positions,
                    capture_prefix_rank=capture_action_prefix_rank,
                    video_rope_cache=video_rope_cache,
                    action_rope_cache=action_rope_cache,
                    text_rope_cache=text_rope_cache,
                    video_camera_ids=video_camera_ids,
                    first_frame_memory=(
                        first_frame_memory_context
                        if block.first_frame_memory_reader
                        else None
                    ),
                    first_frame_memory_camera_ids=(
                        None
                        if first_frame_memory is None
                        else first_frame_memory.camera_ids
                    ),
                    condition_video_tokens=condition_video_tokens,
                )
                if capture_action_prefix_rank is None:
                    video, action = block_result
                else:
                    video, action, layer_cache = block_result
                    layer_caches.append(layer_cache)
            if memory_prefix_valid is not None:
                m = memory_prefix_valid.shape[1]
                video = torch.cat((video[:, :m].masked_fill(~memory_prefix_valid[..., None], 0), video[:, m:]), dim=1)
        output = VAPTokenOutput(
            video_hidden=video,
            action_hidden=action,
            video_velocity=self.video_head(video, video_time_embedding),
            action_velocity=self.action_head(action, action_time_embedding),
            layout=layout,
        )
        if capture_action_prefix_rank is None:
            return output
        active_tokens = int(layout.action_tokens_per_chunk)
        if action_rope_cache is None:
            raise AssertionError("prefix inference did not construct Action RoPE")
        active_chunk = int(capture_action_prefix_rank) // 2
        active_start = active_chunk * active_tokens
        return output, VAPActionPrefixKVCache(
            layers=tuple(layer_caches),
            active_rank=int(capture_action_prefix_rank),
            active_tokens=active_tokens,
            batch_size=batch_size,
            active_action_rope=action_rope_cache.token_slice(
                active_start, active_start + active_tokens
            ),
        )

    def forward_action_stage_with_prefix_cache(
        self,
        video_tokens: torch.Tensor,
        noisy_actions: torch.Tensor,
        *,
        active_action_chunk: int,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        video_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        condition_video_tokens: int,
        video_camera_ids: torch.Tensor,
        text_context: torch.Tensor | None = None,
        state_condition: torch.Tensor | None = None,
        first_frame_memory: FirstFrameMemoryTokens | None = None,
    ) -> tuple[torch.Tensor, VAPActionPrefixKVCache]:
        """Evaluate one Action stage without any causally invisible future.

        For chunk 0 this executes ``I,A0``.  For chunk 1 it executes
        ``I,A0,V0,A1``.  The returned velocity is only the active Action chunk;
        all earlier clean stages are captured as immutable per-layer K/V.
        """

        if self.training:
            raise RuntimeError("prefix-only Action evaluation is inference-only")
        if not 0 <= int(active_action_chunk) < self.num_chunks:
            raise ValueError(
                f"active_action_chunk must be in [0,{self.num_chunks}), "
                f"got {active_action_chunk}"
            )
        future_video_tokens = video_tokens.shape[1] - condition_video_tokens
        if future_video_tokens <= 0 or future_video_tokens % self.num_chunks:
            raise ValueError("full video tokens must contain equal future chunks")
        if noisy_actions.shape[1] <= 0 or noisy_actions.shape[1] % self.num_chunks:
            raise ValueError("full Action tokens must contain equal chunks")
        video_tokens_per_chunk = future_video_tokens // self.num_chunks
        action_tokens_per_chunk = noisy_actions.shape[1] // self.num_chunks
        chunk = int(active_action_chunk)
        video_end = condition_video_tokens + chunk * video_tokens_per_chunk
        action_end = (chunk + 1) * action_tokens_per_chunk
        layout = ActionStageLayout(
            num_chunks=self.num_chunks,
            active_action_chunk=chunk,
            condition_video_tokens=condition_video_tokens,
            video_tokens_per_chunk=video_tokens_per_chunk,
            action_tokens_per_chunk=action_tokens_per_chunk,
        )

        def _prefix_tokens(tensor: torch.Tensor, end: int) -> torch.Tensor:
            if tensor.ndim < 2:
                raise ValueError("per-token prefix input must have a sequence axis")
            return tensor[:, :end]

        prefix_video_timestep = (
            video_timestep
            if video_timestep.ndim == 1
            else _prefix_tokens(video_timestep, video_end)
        )
        prefix_action_timestep = (
            action_timestep
            if action_timestep.ndim == 1
            else _prefix_tokens(action_timestep, action_end)
        )
        output, cache = self(
            _prefix_tokens(video_tokens, video_end),
            _prefix_tokens(noisy_actions, action_end),
            video_timestep=prefix_video_timestep,
            action_timestep=prefix_action_timestep,
            video_position_ids=_prefix_tokens(video_position_ids, video_end),
            action_position_ids=_prefix_tokens(action_position_ids, action_end),
            condition_video_tokens=condition_video_tokens,
            video_camera_ids=_prefix_tokens(video_camera_ids, video_end),
            text_context=text_context,
            layout=layout,
            capture_action_prefix_rank=layout.active_rank,
            state_condition=state_condition,
            first_frame_memory=first_frame_memory,
        )
        active_start = chunk * action_tokens_per_chunk
        return output.action_velocity[:, active_start:action_end], cache

    def forward_active_action_with_prefix_cache(
        self,
        active_actions: torch.Tensor,
        *,
        action_timestep: torch.Tensor,
        prefix_cache: VAPActionPrefixKVCache,
        state_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return velocity for one Action stage without recomputing its prefix."""

        if self.training:
            raise RuntimeError("prefix KV decode is inference-only")
        if (
            active_actions.ndim != 3
            or active_actions.shape[-1] != self.action_input_dim
        ):
            raise ValueError(
                f"active_actions must be [B,S,{self.action_input_dim}], "
                f"got {tuple(active_actions.shape)}"
            )
        batch_size, active_tokens = active_actions.shape[:2]
        if batch_size != prefix_cache.batch_size:
            raise ValueError("active action batch does not match prefix cache")
        if active_tokens != prefix_cache.active_tokens:
            raise ValueError("active action length does not match prefix cache")
        if len(prefix_cache.layers) != len(self.blocks):
            raise ValueError("prefix cache layer count does not match model")
        valid_timestep_shapes = {(batch_size,), (batch_size, active_tokens)}
        if tuple(action_timestep.shape) not in valid_timestep_shapes:
            raise ValueError("active action timestep must be [B] or [B,S]")

        action = self.action_identity(self.action_encoder(active_actions))
        if self.state_film is None:
            if state_condition is not None:
                raise ValueError(
                    "state_condition was provided but State FiLM is disabled"
                )
        else:
            if state_condition is None:
                raise ValueError(
                    "State FiLM is enabled but state_condition was not provided"
                )
            active_chunk = int(prefix_cache.active_rank) // 2
            if state_condition.ndim == 2:
                active_state = state_condition
            elif state_condition.ndim == 3 and state_condition.shape[1] == 1:
                active_state = state_condition[:, 0]
            elif (
                state_condition.ndim == 3
                and state_condition.shape[1] == self.num_chunks
            ):
                active_state = state_condition[:, active_chunk]
            else:
                raise ValueError("state_condition must be [B,D], [B,1,D], or [B,E,D]")
            # Reuse the same FiLM implementation for the active chunk only.
            active_layout = ActionStageLayout(
                num_chunks=1,
                active_action_chunk=0,
                condition_video_tokens=1,
                video_tokens_per_chunk=1,
                action_tokens_per_chunk=active_tokens,
            )
            action = self.state_film(
                action, active_state[:, None, :], layout=active_layout
            )
        action_time_embedding = self.action_time.embed(
            action_timestep, dtype=action.dtype
        )
        action_mod = self.action_time.project(action_time_embedding, dtype=action.dtype)
        for block, layer_cache in zip(self.blocks, prefix_cache.layers):
            action = block.forward_active_action(
                action,
                action_time_mod=action_mod,
                action_rope_cache=prefix_cache.active_action_rope,
                cache=layer_cache,
            )
        return self.action_head(action, action_time_embedding)


__all__ = [
    "ChunkStateFiLM",
    "ExpertTimeConditioner",
    "ExpertOutputHead",
    "GlobalTimeAnchor",
    "VAPTokenModel",
    "VAPTokenOutput",
    "sinusoidal_timestep_embedding",
    "VideoIdentityEmbedding",
    "ActionIdentityEmbedding",
]
