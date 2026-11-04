"""Action dynamics and temporal action encoder."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from georefiner.config import GeoRefinerConfig
from georefiner.modules.common import (
    make_transformer_encoder,
    sanitize_padding_mask,
    sinusoidal_position_embedding,
)


@dataclass(slots=True)
class ActionDynamics:
    """Finite-difference action dynamics used by the temporal action encoder."""

    velocity: Tensor
    acceleration: Tensor
    gripper_transition: Tensor
    concatenated: Tensor


@dataclass(slots=True)
class ActionEncoderOutput:
    """Trajectory tokens and debug dynamics from `ActionEncoder`."""

    trajectory_tokens: Tensor
    dynamics: ActionDynamics


class ActionDynamicsEncoder(nn.Module):
    """Compute nominal action dynamics and project them to hidden space."""

    def __init__(self, config: GeoRefinerConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.action_dim * 3 + 1
        self.projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

    def compute_dynamics(self, nominal_action: Tensor) -> ActionDynamics:
        if nominal_action.ndim != 3:
            raise ValueError(
                "nominal_action must have shape [B, T, action_dim]; "
                f"got {tuple(nominal_action.shape)}."
            )
        if nominal_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                "nominal_action last dimension must match config.action_dim="
                f"{self.config.action_dim}; got {tuple(nominal_action.shape)}."
            )

        velocity = nominal_action.new_zeros(nominal_action.shape)
        velocity[:, 1:] = nominal_action[:, 1:] - nominal_action[:, :-1]

        acceleration = nominal_action.new_zeros(nominal_action.shape)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]

        gripper = nominal_action[..., -1:]
        gripper_transition = gripper.new_zeros(gripper.shape)
        gripper_transition[:, 1:] = gripper[:, 1:] - gripper[:, :-1]

        concatenated = torch_cat_action_dynamics(
            nominal_action, velocity, acceleration, gripper_transition
        )
        return ActionDynamics(
            velocity=velocity,
            acceleration=acceleration,
            gripper_transition=gripper_transition,
            concatenated=concatenated,
        )

    def forward(self, nominal_action: Tensor) -> tuple[Tensor, ActionDynamics]:
        dynamics = self.compute_dynamics(nominal_action)
        return self.projection(dynamics.concatenated), dynamics


class ActionEncoder(nn.Module):
    """Fuse action/state calibration features with action dynamics."""

    def __init__(self, config: GeoRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self.dynamics_encoder = ActionDynamicsEncoder(config)
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.temporal_transformer = make_transformer_encoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.temporal_layers,
            dropout=config.dropout,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        canonical_nominal_action: Tensor,
        action_feature: Tensor,
        state_feature: Tensor,
        action_padding_mask: Tensor | None = None,
    ) -> ActionEncoderOutput:
        self._validate_feature("action_feature", action_feature, canonical_nominal_action)
        self._validate_feature("state_feature", state_feature, canonical_nominal_action)

        dynamics_feature, dynamics = self.dynamics_encoder(canonical_nominal_action)
        fused = self.fusion(
            torch_cat_features(action_feature, state_feature, dynamics_feature)
        )
        position = sinusoidal_position_embedding(
            fused.shape[1], self.config.hidden_dim, fused.device
        ).to(dtype=fused.dtype)
        fused = fused + position
        mask = sanitize_padding_mask(action_padding_mask)
        if mask is not None and mask.shape != fused.shape[:2]:
            raise ValueError(
                "action_padding_mask must have shape [B, T]; "
                f"got {tuple(mask.shape)} for tokens {tuple(fused.shape)}."
            )
        trajectory_tokens = self.temporal_transformer(fused, src_key_padding_mask=mask)
        return ActionEncoderOutput(
            trajectory_tokens=self.output_norm(trajectory_tokens),
            dynamics=dynamics,
        )

    def _validate_feature(
        self,
        name: str,
        feature: Tensor,
        canonical_nominal_action: Tensor,
    ) -> None:
        if feature.ndim != 3:
            raise ValueError(f"{name} must have shape [B, T, D]; got {tuple(feature.shape)}.")
        if feature.shape[:2] != canonical_nominal_action.shape[:2]:
            raise ValueError(
                f"{name} must share [B, T] with canonical_nominal_action; "
                f"got {tuple(feature.shape)} and {tuple(canonical_nominal_action.shape)}."
            )
        if feature.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"{name} last dimension must be hidden_dim={self.config.hidden_dim}; "
                f"got {tuple(feature.shape)}."
            )


def torch_cat_action_dynamics(
    action: Tensor,
    velocity: Tensor,
    acceleration: Tensor,
    gripper_transition: Tensor,
) -> Tensor:
    from torch import cat

    return cat([action, velocity, acceleration, gripper_transition], dim=-1)


def torch_cat_features(
    action_feature: Tensor,
    state_feature: Tensor,
    dynamics_feature: Tensor,
) -> Tensor:
    from torch import cat

    return cat([action_feature, state_feature, dynamics_feature], dim=-1)
