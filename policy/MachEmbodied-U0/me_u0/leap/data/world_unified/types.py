"""Canonical, model-independent tensors for world-unified training.

The legacy data boundary uses 33 RGB frames and 32 action steps. Temporal
sampling configs may use a different RGB/action length, while every batch
remains source-homogeneous and therefore has one fixed geometry. Text remains
raw until the model packer applies the existing chat template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor

from leap.data.world_unified.metadata import Metadata


WorldUnifiedTask = Literal["video_action_pred"]

RGB_CHANNELS = 3
RGB_FRAMES = 33
ACTION_STEPS = 32
VAE_TEMPORAL_DOWNSAMPLE = 4
LATENT_FRAMES = 9
# The default world-unified route remains a 26-D contract. Schemas wider than
# this must opt in through their own wrapper rather than changing this global
# legacy boundary.
MAX_ACTION_DIM = 26
MAX_STATE_DIM = 26
NUM_LOGICAL_DOMAINS = 17


def _bool_vector(
    value: Optional[Tensor], *, length: int, name: str
) -> Tensor:
    if value is None:
        return torch.ones(length, dtype=torch.bool)
    value = torch.as_tensor(value)
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(
            f"`{name}` must have shape [{length}], got {tuple(value.shape)}"
        )
    return value.to(dtype=torch.bool)


def _require_float_tensor(name: str, value: Tensor, rank: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(
            f"`{name}` must be rank {rank}, got shape {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"`{name}` must have a floating-point dtype")


def _require_finite(name: str, value: Tensor, valid: Tensor) -> None:
    if valid.any() and not torch.isfinite(value[valid]).all():
        raise ValueError(f"`{name}` contains non-finite values in valid entries")


def _require_zero(name: str, value: Tensor, invalid: Tensor) -> None:
    if invalid.any() and torch.count_nonzero(value[invalid]).item() != 0:
        raise ValueError(f"`{name}` must be exactly zero at invalid padding entries")


@dataclass(frozen=True)
class WorldUnifiedSample:
    """One canonical, normalized trajectory before task planning.

    Shapes are unbatched.  ``video_time_valid_mask`` and
    ``action_time_valid_mask`` describe temporal padding, while the dimension
    masks describe a source's semantic dimensions.  Invalid entries are
    required to be zero so padding can never leak into VAE/action features.
    """

    video: Tensor
    action: Tensor
    state: Tensor
    instruction: str
    domain_id: int
    num_cameras: int
    text_metadata: Optional[Metadata] = None
    video_time_valid_mask: Optional[Tensor] = None
    action_time_valid_mask: Optional[Tensor] = None
    action_dim_valid_mask: Optional[Tensor] = None
    state_dim_valid_mask: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    main_image: Optional[Tensor] = None
    max_action_dim: int = MAX_ACTION_DIM
    max_state_dim: int = MAX_STATE_DIM

    def __post_init__(self) -> None:
        _require_float_tensor("video", self.video, 4)
        _require_float_tensor("action", self.action, 2)
        _require_float_tensor("state", self.state, 1)

        rgb_frames = int(self.video.shape[1])
        if self.video.shape[0] != RGB_CHANNELS or rgb_frames < 2:
            raise ValueError(
                "`video` must have shape [3,T,H,W] with T >= 2, got "
                f"{tuple(self.video.shape)}"
            )
        if (rgb_frames - 1) % VAE_TEMPORAL_DOWNSAMPLE:
            raise ValueError(
                "video future frames must be divisible by Wan's temporal stride"
            )
        if self.video.shape[2] % 16 or self.video.shape[3] % 16:
            raise ValueError(
                "`video` height and width must be divisible by 16, got "
                f"{tuple(self.video.shape[2:])}"
            )
        action_steps = int(self.action.shape[0])
        if action_steps < 1:
            raise ValueError("`action` must contain at least one step")
        if (
            isinstance(self.max_action_dim, bool)
            or not isinstance(self.max_action_dim, int)
            or self.max_action_dim < 1
        ):
            raise ValueError("`max_action_dim` must be a positive integer")
        if not 1 <= self.action.shape[1] <= self.max_action_dim:
            raise ValueError(
                f"`action` dimension must be in [1,{self.max_action_dim}], got "
                f"{self.action.shape[1]}"
            )
        if (
            isinstance(self.max_state_dim, bool)
            or not isinstance(self.max_state_dim, int)
            or self.max_state_dim < 1
        ):
            raise ValueError("`max_state_dim` must be a positive integer")
        if not 1 <= self.state.shape[0] <= self.max_state_dim:
            raise ValueError(
                f"`state` dimension must be in [1,{self.max_state_dim}], got "
                f"{self.state.shape[0]}"
            )
        if not isinstance(self.instruction, str):
            raise TypeError("`instruction` must be the raw instruction string")
        if self.text_metadata is not None and not isinstance(
            self.text_metadata, Metadata
        ):
            raise TypeError("`text_metadata` must be Metadata or None")
        if self.main_image is not None:
            _require_float_tensor("main_image", self.main_image, 3)
            if tuple(self.main_image.shape) != (RGB_CHANNELS, 224, 224):
                raise ValueError(
                    "`main_image` must have shape [3,224,224], got "
                    f"{tuple(self.main_image.shape)}"
                )
            if not torch.isfinite(self.main_image).all():
                raise ValueError("`main_image` contains non-finite values")
        if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int):
            raise TypeError("`domain_id` must be an integer")
        if not 0 <= self.domain_id < NUM_LOGICAL_DOMAINS:
            raise ValueError(
                f"`domain_id` must be in [0,{NUM_LOGICAL_DOMAINS}), got "
                f"{self.domain_id}"
            )
        video_valid = _bool_vector(
            self.video_time_valid_mask,
            length=rgb_frames,
            name="video_time_valid_mask",
        ).to(device=self.video.device)
        action_time_valid = _bool_vector(
            self.action_time_valid_mask,
            length=action_steps,
            name="action_time_valid_mask",
        ).to(device=self.action.device)
        action_dim_valid = _bool_vector(
            self.action_dim_valid_mask,
            length=self.action.shape[1],
            name="action_dim_valid_mask",
        ).to(device=self.action.device)
        state_dim_valid = _bool_vector(
            self.state_dim_valid_mask,
            length=self.state.shape[0],
            name="state_dim_valid_mask",
        ).to(device=self.state.device)

        object.__setattr__(self, "video_time_valid_mask", video_valid)
        object.__setattr__(self, "action_time_valid_mask", action_time_valid)
        object.__setattr__(self, "action_dim_valid_mask", action_dim_valid)
        object.__setattr__(self, "state_dim_valid_mask", state_dim_valid)

        video_entry_valid = video_valid.view(1, rgb_frames, 1, 1).expand_as(
            self.video
        )
        action_entry_valid = (
            action_time_valid[:, None] & action_dim_valid[None, :]
        )
        _require_finite("video", self.video, video_entry_valid)
        _require_finite("action", self.action, action_entry_valid)
        _require_finite("state", self.state, state_dim_valid)
        _require_zero("video", self.video, ~video_entry_valid)
        _require_zero("action", self.action, ~action_entry_valid)
        _require_zero("state", self.state, ~state_dim_valid)

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def rgb_frames(self) -> int:
        return int(self.video.shape[1])

    @property
    def action_steps(self) -> int:
        return int(self.action.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[0])
