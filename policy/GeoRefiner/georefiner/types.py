"""Typed data contracts and validation utilities for GeoRefiner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from torch import Tensor

from georefiner.config import GeoRefinerConfig


Instruction = str | Sequence[str]


@dataclass(slots=True)
class GeoRefinerInput:
    """Input bundle for GeoRefiner.

    Either raw instructions or precomputed language tokens must be supplied.
    For each visual view, either an RGB image or precomputed visual tokens must
    be supplied. This keeps the main model debuggable without real backbones.
    """

    nominal_action: Tensor
    state: Tensor
    global_rgb: Tensor | None = None
    wrist_rgb: Tensor | None = None
    instruction: Instruction | None = None
    language_tokens: Tensor | None = None
    global_visual_tokens: Tensor | None = None
    wrist_visual_tokens: Tensor | None = None
    language_padding_mask: Tensor | None = None
    global_visual_padding_mask: Tensor | None = None
    wrist_visual_padding_mask: Tensor | None = None
    action_padding_mask: Tensor | None = None

    def validate(self, config: GeoRefinerConfig | None = None) -> None:
        """Validate tensor ranks, leading dimensions, and configured feature dims."""

        action_dim = config.action_dim if config is not None else None
        state_dim = config.state_dim if config is not None else None
        text_dim = config.text_feature_dim if config is not None else None
        visual_dim = config.visual_feature_dim if config is not None else None

        _require_tensor("nominal_action", self.nominal_action)
        _require_tensor("state", self.state)
        _expect_rank("nominal_action", self.nominal_action, 3)
        batch_size, horizon, actual_action_dim = self.nominal_action.shape

        if action_dim is not None and actual_action_dim != action_dim:
            raise ValueError(
                "nominal_action must have shape [B, T, action_dim]; "
                f"expected action_dim={action_dim}, got shape "
                f"{tuple(self.nominal_action.shape)}."
            )

        if horizon <= 0:
            raise ValueError("nominal_action must have a non-empty time dimension.")

        if config is not None and horizon > config.max_action_horizon:
            raise ValueError(
                "nominal_action time dimension exceeds max_action_horizon; "
                f"got T={horizon}, max_action_horizon={config.max_action_horizon}."
            )

        self._validate_state(batch_size, horizon, state_dim)
        self._validate_language(batch_size, text_dim)
        self._validate_visual_sources(batch_size, visual_dim)
        self._validate_masks(batch_size, horizon)

    def _validate_state(
        self, batch_size: int, horizon: int, state_dim: int | None
    ) -> None:
        if self.state.ndim not in (2, 3):
            raise ValueError(
                "state must have shape [B, S] or [B, T, S]; "
                f"got shape {tuple(self.state.shape)}."
            )
        if self.state.shape[0] != batch_size:
            raise ValueError(
                "state batch size must match nominal_action; "
                f"got state B={self.state.shape[0]}, action B={batch_size}."
            )
        if self.state.ndim == 3 and self.state.shape[1] != horizon:
            raise ValueError(
                "state with rank 3 must have shape [B, T, S] with matching T; "
                f"got state T={self.state.shape[1]}, action T={horizon}."
            )
        if state_dim is not None and self.state.shape[-1] != state_dim:
            raise ValueError(
                "state last dimension must match config.state_dim; "
                f"expected {state_dim}, got shape {tuple(self.state.shape)}."
            )

    def _validate_language(self, batch_size: int, text_dim: int | None) -> None:
        if self.instruction is None and self.language_tokens is None:
            raise ValueError("Either instruction or language_tokens must be provided.")

        if self.language_tokens is not None:
            _require_tensor("language_tokens", self.language_tokens)
            _expect_rank("language_tokens", self.language_tokens, 3)
            if self.language_tokens.shape[0] != batch_size:
                raise ValueError(
                    "language_tokens batch size must match nominal_action; "
                    f"got language B={self.language_tokens.shape[0]}, "
                    f"action B={batch_size}."
                )
            if text_dim is not None and self.language_tokens.shape[-1] != text_dim:
                raise ValueError(
                    "language_tokens last dimension must match "
                    f"config.text_feature_dim={text_dim}; got shape "
                    f"{tuple(self.language_tokens.shape)}."
                )

        if self.instruction is not None and not isinstance(self.instruction, str):
            if len(self.instruction) != batch_size:
                raise ValueError(
                    "instruction sequence length must match batch size; "
                    f"got {len(self.instruction)} instructions for B={batch_size}."
                )

    def _validate_visual_sources(
        self, batch_size: int, visual_dim: int | None
    ) -> None:
        if self.global_rgb is None and self.global_visual_tokens is None:
            raise ValueError(
                "Provide global_rgb or global_visual_tokens for the global view."
            )
        if self.wrist_rgb is None and self.wrist_visual_tokens is None:
            raise ValueError("Provide wrist_rgb or wrist_visual_tokens for the wrist view.")

        _validate_rgb("global_rgb", self.global_rgb, batch_size)
        _validate_rgb("wrist_rgb", self.wrist_rgb, batch_size)
        _validate_visual_tokens(
            "global_visual_tokens",
            self.global_visual_tokens,
            batch_size,
            visual_dim,
        )
        _validate_visual_tokens(
            "wrist_visual_tokens",
            self.wrist_visual_tokens,
            batch_size,
            visual_dim,
        )

    def _validate_masks(self, batch_size: int, horizon: int) -> None:
        _validate_mask(
            "language_padding_mask",
            self.language_padding_mask,
            batch_size,
            expected_second_dim=(
                self.language_tokens.shape[1]
                if self.language_tokens is not None
                else None
            ),
        )
        _validate_mask(
            "global_visual_padding_mask",
            self.global_visual_padding_mask,
            batch_size,
            expected_second_dim=(
                self.global_visual_tokens.shape[1]
                if self.global_visual_tokens is not None
                else None
            ),
        )
        _validate_mask(
            "wrist_visual_padding_mask",
            self.wrist_visual_padding_mask,
            batch_size,
            expected_second_dim=(
                self.wrist_visual_tokens.shape[1]
                if self.wrist_visual_tokens is not None
                else None
            ),
        )
        _validate_mask(
            "action_padding_mask",
            self.action_padding_mask,
            batch_size,
            expected_second_dim=horizon,
        )


@dataclass(slots=True)
class GeoRefinerOutput:
    """Output bundle returned by a complete GeoRefiner forward pass."""

    refined_action: Tensor
    canonical_nominal_action: Tensor
    gate: Tensor
    residual: Tensor
    canonical_refined_action: Tensor | None = None
    applied_correction: Tensor | None = None
    intermediates: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, config: GeoRefinerConfig | None = None) -> None:
        action_dim = config.action_dim if config is not None else None
        _expect_action_tensor("refined_action", self.refined_action, action_dim)
        expected_shape = tuple(self.refined_action.shape)
        for name, value in (
            ("canonical_nominal_action", self.canonical_nominal_action),
            ("gate", self.gate),
            ("residual", self.residual),
        ):
            _expect_action_tensor(name, value, action_dim)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must match refined_action shape {expected_shape}; "
                    f"got {tuple(value.shape)}."
                )
        for name, value in (
            ("canonical_refined_action", self.canonical_refined_action),
            ("applied_correction", self.applied_correction),
        ):
            if value is None:
                continue
            _expect_action_tensor(name, value, action_dim)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must match refined_action shape {expected_shape}; "
                    f"got {tuple(value.shape)}."
                )

    @property
    def language_tokens(self) -> Tensor:
        """Projected CLIP or precomputed language tokens."""

        return self._intermediate("language_tokens")

    @property
    def geometry_tokens(self) -> Tensor:
        """Cross-view geometry tokens."""

        return self._intermediate("geometry_tokens")

    @property
    def region_tokens(self) -> Tensor:
        """Language-guided region-query tokens."""

        return self._intermediate("region_tokens")

    @property
    def trajectory_tokens(self) -> Tensor:
        """Temporal action trajectory tokens."""

        return self._intermediate("trajectory_tokens")

    @property
    def fused_action_tokens(self) -> Tensor:
        """Geometry-conditioned fused action representation."""

        return self._intermediate("fused_action_tokens")

    def _intermediate(self, name: str) -> Tensor:
        value = self.intermediates.get(name)
        if not isinstance(value, Tensor):
            raise AttributeError(
                f"{name} is unavailable because return_intermediates is disabled."
            )
        return value


def validate_georefiner_input(
    data: GeoRefinerInput, config: GeoRefinerConfig | None = None
) -> None:
    """Functional wrapper for callers that prefer explicit validation."""

    data.validate(config)


def _require_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(value).__name__}.")


def _expect_rank(name: str, value: Tensor, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {tuple(value.shape)}.")


def _expect_action_tensor(
    name: str, value: Tensor, action_dim: int | None = None
) -> None:
    _require_tensor(name, value)
    _expect_rank(name, value, 3)
    if action_dim is not None and value.shape[-1] != action_dim:
        raise ValueError(
            f"{name} last dimension must equal action_dim={action_dim}; "
            f"got shape {tuple(value.shape)}."
        )


def _validate_rgb(name: str, value: Tensor | None, batch_size: int) -> None:
    if value is None:
        return
    _require_tensor(name, value)
    _expect_rank(name, value, 4)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size must match nominal_action; "
            f"got {value.shape[0]} and {batch_size}."
        )
    if value.shape[1] != 3:
        raise ValueError(f"{name} must have 3 channels; got shape {tuple(value.shape)}.")
    if value.shape[2] <= 0 or value.shape[3] <= 0:
        raise ValueError(f"{name} spatial dimensions must be positive.")


def _validate_visual_tokens(
    name: str,
    value: Tensor | None,
    batch_size: int,
    visual_dim: int | None,
) -> None:
    if value is None:
        return
    _require_tensor(name, value)
    _expect_rank(name, value, 3)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size must match nominal_action; "
            f"got {value.shape[0]} and {batch_size}."
        )
    if visual_dim is not None and value.shape[-1] != visual_dim:
        raise ValueError(
            f"{name} last dimension must match config.visual_feature_dim="
            f"{visual_dim}; got shape {tuple(value.shape)}."
        )


def _validate_mask(
    name: str,
    value: Tensor | None,
    batch_size: int,
    expected_second_dim: int | None,
) -> None:
    if value is None:
        return
    _require_tensor(name, value)
    _expect_rank(name, value, 2)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size must match nominal_action; "
            f"got {value.shape[0]} and {batch_size}."
        )
    if expected_second_dim is not None and value.shape[1] != expected_second_dim:
        raise ValueError(
            f"{name} second dimension must be {expected_second_dim}; "
            f"got shape {tuple(value.shape)}."
        )
