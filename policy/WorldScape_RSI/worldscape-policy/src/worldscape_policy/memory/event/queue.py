from __future__ import annotations

import torch
from torch import Tensor

from worldscape_policy.types import EventMemoryState


class EventMemoryQueue:
    """Bounded FIFO for executed event representations.

    The queue stores event steps as ``[B, H, tokens, dim]`` and never conflates
    them with the WAM KV cache. Call ``reset_episode`` at every episode boundary.
    """

    def __init__(self, max_steps: int, *, detach: bool = True) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.max_steps = int(max_steps)
        self.detach = detach
        self._state = EventMemoryState()

    @property
    def state(self) -> EventMemoryState:
        return self._state

    def append(
        self,
        perception_tokens: Tensor,
        planning_tokens: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> EventMemoryState:
        self._validate_step("perception_tokens", perception_tokens)
        if planning_tokens is not None:
            self._validate_step("planning_tokens", planning_tokens)
            if planning_tokens.shape[0] != perception_tokens.shape[0]:
                raise ValueError("Planning and perception token batch sizes must match")
            if planning_tokens.shape[-1] != perception_tokens.shape[-1]:
                raise ValueError("Planning and perception token dimensions must match")

        previous_has_planning = self._state.planning_tokens is not None
        if self.length > 0 and previous_has_planning != (planning_tokens is not None):
            raise ValueError(
                "Planning-token presence must remain consistent within one episode"
            )

        batch_size = perception_tokens.shape[0]
        if valid is None:
            valid = torch.ones(batch_size, device=perception_tokens.device, dtype=torch.bool)
        if valid.ndim != 1 or valid.shape[0] != batch_size:
            raise ValueError(f"valid must have shape [{batch_size}], got {tuple(valid.shape)}")

        perception_step = self._prepare(perception_tokens).unsqueeze(1)
        planning_step = (
            self._prepare(planning_tokens).unsqueeze(1)
            if planning_tokens is not None
            else None
        )
        valid_step = self._prepare(valid.to(dtype=torch.bool)).unsqueeze(1)

        self._state = EventMemoryState(
            perception_tokens=self._append_and_trim(
                self._state.perception_tokens, perception_step
            ),
            planning_tokens=self._append_and_trim(
                self._state.planning_tokens, planning_step
            ),
            valid_mask=self._append_and_trim(self._state.valid_mask, valid_step),
        )
        return self._state

    @property
    def length(self) -> int:
        if self._state.valid_mask is None:
            return 0
        return self._state.valid_mask.shape[1]

    def reset_episode(self) -> EventMemoryState:
        self._state = EventMemoryState()
        return self._state

    def load_state(self, state: EventMemoryState) -> None:
        if state.valid_mask is None:
            if state.perception_tokens is not None or state.planning_tokens is not None:
                raise ValueError("Token history requires a valid_mask")
            self._state = EventMemoryState()
            return
        if state.perception_tokens is None:
            raise ValueError("valid_mask requires perception_tokens")
        if state.perception_tokens.ndim != 4:
            raise ValueError("Stored perception_tokens must have shape [B, H, L, D]")
        if state.valid_mask.shape != state.perception_tokens.shape[:2]:
            raise ValueError("valid_mask must match stored perception batch/history dimensions")
        if state.planning_tokens is not None:
            if state.planning_tokens.ndim != 4:
                raise ValueError("Stored planning_tokens must have shape [B, H, K, D]")
            if state.planning_tokens.shape[:2] != state.perception_tokens.shape[:2]:
                raise ValueError("Planning and perception histories must align")
        if state.valid_mask.shape[1] > self.max_steps:
            raise ValueError(
                f"State history length {state.valid_mask.shape[1]} exceeds max_steps "
                f"{self.max_steps}"
            )
        self._state = state

    def _prepare(self, tensor: Tensor) -> Tensor:
        return tensor.detach() if self.detach else tensor

    def _append_and_trim(
        self,
        previous: Tensor | None,
        step: Tensor | None,
    ) -> Tensor | None:
        if step is None:
            return previous
        if previous is None:
            result = step
        else:
            if previous.shape[0] != step.shape[0] or previous.shape[2:] != step.shape[2:]:
                raise ValueError(
                    "Event-memory token shape changed within an episode: "
                    f"{tuple(previous.shape)} vs {tuple(step.shape)}"
                )
            result = torch.cat([previous, step], dim=1)
        return result[:, -self.max_steps :]

    @staticmethod
    def _validate_step(name: str, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, tokens, dim], got {tuple(tokens.shape)}")
        if tokens.shape[1] == 0 or tokens.shape[2] == 0:
            raise ValueError(f"{name} cannot have an empty token or feature dimension")
