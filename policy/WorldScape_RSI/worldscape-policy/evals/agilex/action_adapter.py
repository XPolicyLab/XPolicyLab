from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from evals.agilex.observation_adapter import action_fields_to_trajectories
from evals.common.checkpoint_runtime import CheckpointTransformBundle
from evals.common.protocols import ActionAdapter
from worldscape_policy.action_space import (
    compose_rotation6d,
    parse_action_mode,
)
from worldscape_policy.types import WorldActionOutput


class AgileXActionAdapter(
    ActionAdapter[tuple[list[list[float]], list[list[float]]]]
):
    """Denormalize native action tensors with checkpoint-owned transforms."""

    def __init__(
        self,
        bundle: CheckpointTransformBundle,
        *,
        action_mode: str = "eef",
    ) -> None:
        parse_action_mode(action_mode)
        self.bundle = bundle
        self.action_mode = action_mode

    def __call__(
        self,
        output: WorldActionOutput,
        *,
        current_state: Mapping[str, Any] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        action = output.require_action().detach()
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError("AgileX native runtime supports batch size 1 only")
            action = action[0]
        if action.ndim != 2:
            raise ValueError(f"Expected action [H,D], got {tuple(action.shape)}")
        if action.shape[-1] > self.bundle.max_action_dim:
            raise ValueError("Action exceeds checkpoint max_action_dim")
        transformed = self.bundle.transform.unapply(
            {"action": action.float().cpu()}
        )
        action_fields = {
            key: _as_numpy(value)
            for key, value in transformed.items()
            if key.startswith("action.")
        }
        if not action_fields:
            raise ValueError("Checkpoint transform did not split native action")
        relative_keys = _relative_action_keys(self.bundle.transform)
        for key in relative_keys:
            if key not in action_fields:
                continue
            state_key = f"state.{key.removeprefix('action.')}"
            if current_state is None or state_key not in current_state:
                raise ValueError(
                    f"Relative checkpoint action {key!r} requires {state_key!r}"
                )
            reference = np.asarray(current_state[state_key]).reshape(-1)
            value = action_fields[key]
            if value.shape[-1] != reference.shape[-1]:
                raise ValueError(
                    f"Relative action/state dimensions differ for {key}: "
                    f"{value.shape[-1]} != {reference.shape[-1]}"
                )
            if key.endswith("_rot6d"):
                action_fields[key] = compose_rotation6d(reference, value)
            else:
                action_fields[key] = value + reference

        return action_fields_to_trajectories(
            action_fields,
            action_mode=self.action_mode,
        )

    def action(
        self,
        output: WorldActionOutput,
        *,
        current_state: Mapping[str, Any] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Adapt through the backend-neutral action protocol."""

        return self(output, current_state=current_state)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _relative_action_keys(transform: Any) -> set[str]:
    keys = set(getattr(transform, "relative_action_keys", ()))
    metadata = getattr(transform, "modality_metadata", None)
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if str(key).startswith("action.") and getattr(value, "absolute", True) is False:
                keys.add(str(key))
    for child in getattr(transform, "transforms", ()):
        keys.update(_relative_action_keys(child))
    return keys


__all__ = ["AgileXActionAdapter"]
