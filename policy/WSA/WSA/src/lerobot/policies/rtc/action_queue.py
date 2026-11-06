from __future__ import annotations

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue that tracks both model-space and robot-space actions."""

    def __init__(self, cfg: RTCConfig):
        self.queue: Tensor | None = None
        self.original_queue: Tensor | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        return self.last_index

    def get(self) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def get_left_over(self) -> Tensor | None:
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> None:
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
            else:
                self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int) -> None:
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()
        self.last_index = 0

        logger.debug(
            "RTC queue replaced: original_shape=%s processed_shape=%s real_delay=%d clamped_delay=%d",
            tuple(original_actions.shape),
            tuple(processed_actions.shape),
            real_delay,
            clamped_delay,
        )

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor) -> None:
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])[self.last_index :]
        self.queue = torch.cat([self.queue, processed_actions.clone()])[self.last_index :]
        self.last_index = 0

    def _check_and_resolve_delays(
        self,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> int:
        effective_delay = max(0, int(real_delay))

        if action_index_before_inference is not None and self.queue is not None:
            indexes_diff = max(0, self.last_index - int(action_index_before_inference))
            if indexes_diff != effective_delay:
                logger.warning(
                    "Action queue observed consumed steps (%d) do not match reported delay (%d).",
                    indexes_diff,
                    effective_delay,
                )
        return effective_delay
