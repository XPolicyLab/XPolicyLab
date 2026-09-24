"""Benchmark adapter modules for canonical action/state feature calibration."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from georefiner.adapters.converters import (
    ActionConverterBase,
    IdentityActionConverter,
    IdentityStateConverter,
    StateConverterBase,
)
from georefiner.config import GeoRefinerConfig


@dataclass(slots=True)
class ActionAdapterOutput:
    """Canonical action and calibrated action feature from `ActionAdapter`."""

    canonical_action: Tensor
    action_feature: Tensor


@dataclass(slots=True)
class StateAdapterOutput:
    """Canonical state sequence and calibrated state feature from `StateAdapter`."""

    canonical_state: Tensor
    state_feature: Tensor


@dataclass(slots=True)
class BenchmarkAdapterOutput:
    """Combined action/state adapter outputs for the core model."""

    canonical_nominal_action: Tensor
    action_feature: Tensor
    state_feature: Tensor
    canonical_state: Tensor


class CalibrationMLP(nn.Module):
    """Configurable MLP that maps the final input dimension to `hidden_dim`.

    The module preserves all leading dimensions, so `[B, T, C]` becomes
    `[B, T, D]` and `[B, C]` becomes `[B, D]`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive; got {num_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        modules: list[nn.Module] = []
        current_dim = input_dim
        for layer_index in range(num_layers):
            linear = nn.Linear(current_dim, hidden_dim)
            modules.append(linear)
            if layer_index < num_layers - 1:
                modules.extend(
                    [
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
            current_dim = hidden_dim
        self.net = nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"CalibrationMLP expected last dimension {self.input_dim}; "
                f"got shape {tuple(inputs.shape)}."
            )
        return self.net(inputs)


class ActionAdapter(nn.Module):
    """Convert native actions to canonical actions and calibrated features."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        converter: ActionConverterBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.converter = converter or IdentityActionConverter(config.action_dim)
        self.calibration = CalibrationMLP(
            input_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.action_mlp_layers,
            dropout=config.dropout,
        )

    def forward(self, native_action: Tensor) -> ActionAdapterOutput:
        canonical_action = self.converter.to_canonical(native_action)
        self._validate_canonical_action(canonical_action)
        action_feature = self.calibration(canonical_action)
        return ActionAdapterOutput(
            canonical_action=canonical_action,
            action_feature=action_feature,
        )

    def inverse_action(self, canonical_action: Tensor) -> Tensor:
        self._validate_canonical_action(canonical_action)
        return self.converter.from_canonical(canonical_action)

    def _validate_canonical_action(self, canonical_action: Tensor) -> None:
        if canonical_action.ndim != 3:
            raise ValueError(
                "canonical_action must have shape [B, T, action_dim]; "
                f"got {tuple(canonical_action.shape)}."
            )
        if canonical_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                "canonical_action last dimension must match config.action_dim="
                f"{self.config.action_dim}; got shape {tuple(canonical_action.shape)}."
            )


class StateAdapter(nn.Module):
    """Convert native states and calibrate them to temporal state features."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        converter: StateConverterBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.converter = converter or IdentityStateConverter()
        self.calibration = CalibrationMLP(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.state_mlp_layers,
            dropout=config.dropout,
        )

    def forward(self, native_state: Tensor, horizon: int) -> StateAdapterOutput:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive; got {horizon}.")

        canonical_state = self.converter.to_canonical(native_state)
        canonical_state = self._expand_state(canonical_state, horizon)
        state_feature = self.calibration(canonical_state)
        return StateAdapterOutput(
            canonical_state=canonical_state,
            state_feature=state_feature,
        )

    def _expand_state(self, canonical_state: Tensor, horizon: int) -> Tensor:
        if canonical_state.ndim == 2:
            if canonical_state.shape[-1] != self.config.state_dim:
                raise ValueError(
                    "canonical_state last dimension must match config.state_dim="
                    f"{self.config.state_dim}; got shape {tuple(canonical_state.shape)}."
                )
            return canonical_state.unsqueeze(1).expand(-1, horizon, -1)

        if canonical_state.ndim == 3:
            if canonical_state.shape[1] != horizon:
                raise ValueError(
                    "canonical_state with rank 3 must have matching horizon; "
                    f"got state T={canonical_state.shape[1]}, action T={horizon}."
                )
            if canonical_state.shape[-1] != self.config.state_dim:
                raise ValueError(
                    "canonical_state last dimension must match config.state_dim="
                    f"{self.config.state_dim}; got shape {tuple(canonical_state.shape)}."
                )
            return canonical_state

        raise ValueError(
            "canonical_state must have shape [B, S] or [B, T, S]; "
            f"got {tuple(canonical_state.shape)}."
        )


class BenchmarkAdapter(nn.Module):
    """Compose action/state converters with MLP feature calibration."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        action_converter: ActionConverterBase | None = None,
        state_converter: StateConverterBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_adapter = ActionAdapter(config, action_converter)
        self.state_adapter = StateAdapter(config, state_converter)

    def forward(self, native_nominal_action: Tensor, native_state: Tensor) -> BenchmarkAdapterOutput:
        action_output = self.action_adapter(native_nominal_action)
        horizon = action_output.canonical_action.shape[1]
        state_output = self.state_adapter(native_state, horizon=horizon)

        if state_output.state_feature.shape[:2] != action_output.action_feature.shape[:2]:
            raise ValueError(
                "state_feature and action_feature must share [B, T]; "
                f"got {tuple(state_output.state_feature.shape)} and "
                f"{tuple(action_output.action_feature.shape)}."
            )

        return BenchmarkAdapterOutput(
            canonical_nominal_action=action_output.canonical_action,
            action_feature=action_output.action_feature,
            state_feature=state_output.state_feature,
            canonical_state=state_output.canonical_state,
        )

    def inverse_action(self, canonical_action: Tensor) -> Tensor:
        return self.action_adapter.inverse_action(canonical_action)
