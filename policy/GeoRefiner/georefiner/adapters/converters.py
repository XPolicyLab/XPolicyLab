"""Converter interfaces for native and canonical benchmark spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class ActionConverterBase(ABC):
    """Interface for native/canonical action conversion.

    Concrete benchmark-specific converters are deferred to later stages.
    """

    @abstractmethod
    def to_canonical(self, native_action: Tensor) -> Tensor:
        """Convert a native action tensor into canonical action coordinates."""

    @abstractmethod
    def from_canonical(self, canonical_action: Tensor) -> Tensor:
        """Convert a canonical action tensor back into native coordinates."""

    def to_native(self, canonical_action: Tensor) -> Tensor:
        """Backward-compatible alias for :meth:`from_canonical`."""

        return self.from_canonical(canonical_action)


class StateConverterBase(ABC):
    """Interface for native/canonical state conversion."""

    @abstractmethod
    def to_canonical(self, native_state: Tensor) -> Tensor:
        """Convert a native state tensor into canonical state coordinates."""


class IdentityActionConverter(ActionConverterBase):
    """Identity converter for already-canonical `[B, T, 7]` action chunks."""

    def __init__(self, action_dim: int = 7) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive; got {action_dim}.")
        self.action_dim = action_dim

    def to_canonical(self, native_action: Tensor) -> Tensor:
        self._validate_action("native_action", native_action)
        return native_action

    def from_canonical(self, canonical_action: Tensor) -> Tensor:
        self._validate_action("canonical_action", canonical_action)
        return canonical_action

    def _validate_action(self, name: str, action: Tensor) -> None:
        if action.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, T, {self.action_dim}]; "
                f"got {tuple(action.shape)}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"{name} last dimension must be {self.action_dim}; "
                f"got shape {tuple(action.shape)}."
            )


class IdentityStateConverter(StateConverterBase):
    """Identity converter for state tensors.

    Supports native state shapes `[B, S]` and `[B, T, S]`.
    """

    def to_canonical(self, native_state: Tensor) -> Tensor:
        if native_state.ndim not in (2, 3):
            raise ValueError(
                "native_state must have shape [B, S] or [B, T, S]; "
                f"got {tuple(native_state.shape)}."
            )
        return native_state
