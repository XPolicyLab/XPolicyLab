"""Shared protocols and data-only types for policy inference."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NotRequired
from typing import Protocol
from typing import TypeAlias
from typing import TypedDict

import numpy as np
import torch

ImageInput: TypeAlias = np.ndarray | Mapping[str, object]
DiTOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
Velocity: TypeAlias = tuple[torch.Tensor | None, torch.Tensor]
TextEncoding: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
VelocityForward: TypeAlias = Callable[[bool], Velocity]
VelocityThunk: TypeAlias = Callable[[], Velocity]


class RopeAxisConfig(TypedDict):
    """Configuration for one axis in a precomputed RoPE cache."""

    dim_range: tuple[int, int]
    base: float
    interpolate_factor: NotRequired[float]


@dataclass(frozen=True)
class ActionOutput:
    """Model-space and controller-space forms of one action chunk."""

    raw_action: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class SparseVideoMemory:
    """Three jointly encoded history latents per view, readable as keys."""

    latents: list[torch.Tensor]


@dataclass(frozen=True)
class RemoteActionCondition:
    """Reference to one observation encoded by the session's model server."""

    cache_key: str


@dataclass(frozen=True)
class DiffusionCondition:
    """Encoded video views and instruction for one diffusion request."""

    video_latents: list[torch.Tensor]
    text_embedding: torch.Tensor
    action_conditions: dict[str, dict[int, torch.Tensor] | RemoteActionCondition] | None = None
    sparse_memory: SparseVideoMemory | None = None


@dataclass(frozen=True)
class DiffusionSample:
    """Model-space outputs produced by one diffusion trajectory."""

    action: torch.Tensor
    video_latents: list[torch.Tensor] | None


@dataclass(frozen=True)
class DiTInput:
    """Model inputs for one joint video-action velocity prediction."""

    video_tokens: torch.Tensor
    video_patches: Sequence[torch.Tensor]
    action: torch.Tensor
    text_embedding: torch.Tensor
    video_timestep: torch.Tensor
    action_timestep: torch.Tensor
    action_conditions: dict[str, dict[int, torch.Tensor] | RemoteActionCondition] | None = None
    video_key_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PipelineActivations:
    """Activation bundle transferred between inference pipeline stages."""

    hidden_states: torch.Tensor
    action_hidden_states: torch.Tensor
    caption: torch.Tensor
    adaln: torch.Tensor


class VideoVAE(Protocol):
    """Video VAE operations required by policy inference."""

    dtype: torch.dtype

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video batch."""
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent batch."""
        ...


class TextEncoder(Protocol):
    """Text encoder operation required by policy inference."""

    def __call__(self, prompts: Sequence[str]) -> TextEncoding:
        """Encode a batch of prompts."""
        ...


class DiTModel(Protocol):
    """DiT invocation required by the model runtime."""

    def __call__(
        self,
        *,
        v: torch.Tensor,
        caption: torch.Tensor,
        t: torch.Tensor,
        a: torch.Tensor,
        t_a: torch.Tensor,
        **metadata: object,
    ) -> DiTOutput:
        """Predict video and action velocities."""
        ...


class DiTForwardRuntime(Protocol):
    """Minimal model runtime required by a velocity predictor."""

    device: torch.device

    def forward_dit(
        self,
        *,
        video_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        video_timestep: torch.Tensor,
        action: torch.Tensor,
        action_timestep: torch.Tensor,
        metadata: Mapping[str, object],
    ) -> DiTOutput:
        """Execute one DiT velocity prediction."""
        ...


class VelocityPredictor(Protocol):
    """Prediction operations required by a diffusion trajectory runner."""

    @property
    def device(self) -> torch.device:
        """Device on which trajectory tensors are allocated."""
        ...

    def reset(self) -> None:
        """Clear state before a new trajectory."""
        ...

    def predict(
        self,
        model_input: DiTInput,
        *,
        timestep: torch.Tensor,
        step_index: int,
        skip_video: bool,
        forward: VelocityForward | None = None,
    ) -> Velocity:
        """Predict video and action velocity for one solver step."""
        ...

    def finish(self) -> None:
        """Finalize predictor state after a trajectory."""
        ...
