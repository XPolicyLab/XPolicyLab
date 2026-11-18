from __future__ import annotations

import torch

from worldscape_policy.policy import WorldScapePolicy
from worldscape_policy.types import (
    EventMemoryState,
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    VisualMemoryState,
    WorldActionOutput,
)


class PolicyRuntime:
    """Own per-episode memory with explicit predict/commit semantics."""

    def __init__(self, policy: WorldScapePolicy) -> None:
        self.policy = policy
        self._mode: InteractionMode | None = None
        self._event_memory: EventMemoryState | None = None
        self._visual_memory: VisualMemoryState | None = None
        self._pending: WorldActionOutput | None = None

    @property
    def mode(self) -> InteractionMode | None:
        return self._mode

    @property
    def event_memory(self) -> EventMemoryState | None:
        return self._event_memory

    @property
    def visual_memory(self) -> VisualMemoryState | None:
        return self._visual_memory

    @property
    def has_pending_prediction(self) -> bool:
        return self._pending is not None

    def reset(self, mode: InteractionMode | str) -> None:
        self.policy.reset_episode()
        self._mode = InteractionMode.parse(mode)
        self._event_memory = None
        self._visual_memory = None
        self._pending = None

    @torch.no_grad()
    def predict(
        self,
        *,
        observation: ObservationBatch,
        prompts: PromptBatch,
        generator: torch.Generator,
    ) -> WorldActionOutput:
        if self._mode is None:
            raise RuntimeError("Call reset(mode) before starting an episode")
        if self._pending is not None:
            raise RuntimeError("Commit or discard the pending prediction before predicting again")
        output = self.policy.sample(
            mode=self._mode,
            observation=observation,
            prompts=prompts,
            generator=generator,
            event_memory=self._event_memory,
            visual_memory=self._visual_memory,
        )
        self._pending = output
        return output

    def commit(self, output: WorldActionOutput | None = None) -> None:
        if self._pending is None:
            raise RuntimeError("There is no pending prediction to commit")
        if output is not None and output is not self._pending:
            raise ValueError("Only the current pending prediction can be committed")
        self._event_memory = self._pending.next_memory
        self._visual_memory = self._pending.next_visual_memory
        self._pending = None

    def discard(self) -> None:
        """Discard an unexecuted action without advancing either memory."""

        self._pending = None
