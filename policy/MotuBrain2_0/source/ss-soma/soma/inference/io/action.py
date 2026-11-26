"""Morph-backed conversion between controller and model action spaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from scipy.signal import savgol_filter

from soma.inference.config import ActionConfig
from soma.inference.types import ActionOutput


class ActionCodec:
    """Apply the configured Morph profile at the policy I/O boundary."""

    def __init__(
        self,
        config: ActionConfig,
        device: str | torch.device,
        *,
        smoothing_window: int = 0,
    ) -> None:
        self._profile = config.profile
        self._state_key = config.state_key
        self._camera_key = config.camera_key
        self._embodiment = config.embodiment
        self._state_field_order = config.state_field_order
        self._output_field_order = config.output_field_order
        self._device = torch.device(device)
        self._smoothing_window = smoothing_window
        if self._profile is not None:
            self._validate_field_order(
                self._state_field_order,
                space="input",
                label="state_field_order",
            )
            self._validate_field_order(
                self._output_field_order,
                space="output",
                label="output_field_order",
            )

    @property
    def model_mask(self) -> torch.Tensor | None:
        """Static mask over model action dimensions, if a profile is set."""
        if self._profile is None:
            return None
        return self._profile.model_mask.to(self._device)

    def encode_constraint(
        self,
        constraint: object | None,
        context: Mapping[str, Any],
    ) -> torch.Tensor | None:
        """Convert an optional controller-space RTC constraint to model space."""
        if constraint is None:
            return None
        action = torch.as_tensor(
            constraint,
            device=self._device,
            dtype=torch.float32,
        )
        if action.numel() == 0 or self._profile is None:
            return action
        state, frame = self._action_context(action, context)
        semantic_action = self._unpack_fields(
            action,
            self._state_field_order,
            space="input",
        )
        encoded = self._profile.encode(
            semantic_action,
            embodiment=self._embodiment,
            ref_state=state,
            ref_frame=frame,
        ).action
        return torch.as_tensor(
            encoded,
            device=self._device,
            dtype=torch.float32,
        )

    def decode(
        self,
        action: torch.Tensor,
        context: Mapping[str, Any],
    ) -> ActionOutput:
        """Convert a model-space prediction to the controller representation."""
        action = self._smooth(action.float())
        raw_action = action.detach().cpu().numpy()
        if self._profile is None:
            return ActionOutput(raw_action=raw_action, action=raw_action)

        state, frame = self._action_context(action, context)
        decoded = self._profile.decode(
            action,
            embodiment=self._embodiment,
            ref_state=state,
            ref_frame=frame,
            ignore_unused=self._output_field_order is None,
        ).action
        if self._output_field_order is not None:
            decoded = self._pack_fields(
                decoded,
                self._output_field_order,
                space="output",
            )
        decoded_tensor = torch.as_tensor(
            decoded,
            device=action.device,
        )
        return ActionOutput(
            raw_action=raw_action,
            action=decoded_tensor.detach().cpu().numpy(),
        )

    def _action_context(
        self,
        action: torch.Tensor,
        context: Mapping[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        state = context.get("state")
        if isinstance(state, Mapping):
            state = state[self._state_key]
        if state is not None:
            state = torch.as_tensor(
                state,
                device=action.device,
                dtype=action.dtype,
            )
            state = self._unpack_fields(
                state,
                self._state_field_order,
                space="input",
            )

        frame = None
        if self._camera_key is not None:
            if self._camera_key not in context:
                raise KeyError(
                    "Camera-frame action conversion requires observation key "
                    f"{self._camera_key!r}"
                )
            frame = torch.as_tensor(
                context[self._camera_key],
                device=action.device,
                dtype=action.dtype,
            )
            if frame.shape[-2:] != (4, 4):
                raise ValueError(
                    f"Camera extrinsic {self._camera_key!r} must end with "
                    f"shape [4, 4], got {tuple(frame.shape)}"
                )
        return state, frame

    def _validate_field_order(
        self,
        order: tuple[str, ...] | None,
        *,
        space: str,
        label: str,
    ) -> None:
        if order is None:
            return
        if not order or len(order) != len(set(order)):
            raise ValueError(f"{label} must be non-empty and unique")
        fields = {
            str(field.name): field
            for field in self._profile.fields
            if getattr(field, f"{space}_used")
        }
        if set(order) != set(fields):
            raise ValueError(
                f"{label} must contain every used {space} field exactly once; "
                f"expected {sorted(fields)}, got {list(order)}"
            )

    def _unpack_fields(
        self,
        value: torch.Tensor,
        order: tuple[str, ...] | None,
        *,
        space: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if order is None:
            return value
        fields = {str(field.name): field for field in self._profile.fields}
        expected = sum(
            getattr(fields[name], f"{space}_dim") for name in order
        )
        if value.ndim == 0 or value.shape[-1] != expected:
            raise ValueError(
                f"ordered {space} value must end with dimension {expected}, "
                f"got {tuple(value.shape)}"
            )
        result = {}
        offset = 0
        for name in order:
            dimension = getattr(fields[name], f"{space}_dim")
            result[name] = value[..., offset : offset + dimension]
            offset += dimension
        return result

    def _pack_fields(
        self,
        value: torch.Tensor,
        order: tuple[str, ...],
        *,
        space: str,
    ) -> torch.Tensor:
        fields = {str(field.name): field for field in self._profile.fields}
        parts = {}
        offset = 0
        for field in self._profile.fields:
            dimension = getattr(field, f"{space}_dim")
            parts[str(field.name)] = value[..., offset : offset + dimension]
            offset += dimension
        return torch.cat([parts[name] for name in order], dim=-1)

    def _smooth(self, action: torch.Tensor) -> torch.Tensor:
        window = min(self._smoothing_window, len(action))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return action
        smoothed = savgol_filter(
            action.detach().cpu().numpy(),
            window_length=window,
            polyorder=2,
            axis=0,
        )
        return torch.as_tensor(
            smoothed,
            device=action.device,
            dtype=action.dtype,
        )
