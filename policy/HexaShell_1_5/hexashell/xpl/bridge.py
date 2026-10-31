"""Hand-over between XPolicyLab's request threads and the episode thread.

RoboDojo's client asks for one action at a time.  While the model thinks, there is no action:
``next_actions`` then answers WAIT after ``poll_s`` and the client asks again, so the simulation
stands still and no step is used.  When the episode thread has an action, it parks it here and
blocks until the observation that follows it arrives.

Episode end: the client reports its verdict (``trial_end``) when RoboDojo has ended the episode
(success, step limit) or after we answered "no more actions" (``finish``).  A waiting episode
thread then gets :class:`EpisodeOver`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..motion.driver import EpisodeOver

#: What get_action answers while no action is ready yet.
WAIT = [{"hexashell": "wait"}]


class Bridge:
    def __init__(self, poll_s: float = 20.0):
        self.poll_s = poll_s
        self._cv = threading.Condition()
        self._obs: dict[str, Any] | None = None
        self._obs_seq = 0
        self._pending: dict | None = None
        self._finished = False
        self._verdict: dict | None = None
        self._abandoned = False
        self.actions_out = 0
        self.waits = 0
        self.created = time.time()

    # -- XPolicyLab side ---------------------------------------------------------------------------
    def offer_obs(self, obs: dict[str, Any]) -> int:
        with self._cv:
            self._obs = obs
            self._obs_seq += 1
            self._cv.notify_all()
            return self._obs_seq

    def next_actions(self) -> list:
        deadline = time.monotonic() + self.poll_s
        with self._cv:
            while True:
                if self._pending is not None:
                    action, self._pending = self._pending, None
                    self.actions_out += 1
                    self._cv.notify_all()
                    return [action]
                if self._finished or self._abandoned or self._verdict is not None:
                    return []
                left = deadline - time.monotonic()
                if left <= 0:
                    self.waits += 1
                    return WAIT
                self._cv.wait(left)

    def conclude(self, verdict: dict[str, Any]) -> None:
        with self._cv:
            self._verdict = dict(verdict or {})
            self._cv.notify_all()

    def abandon(self) -> None:
        with self._cv:
            self._abandoned = True
            self._cv.notify_all()

    # -- episode side ----------------------------------------------------------------------------
    @property
    def verdict(self) -> dict | None:
        with self._cv:
            return None if self._verdict is None else dict(self._verdict)

    @property
    def over(self) -> bool:
        with self._cv:
            return self._verdict is not None or self._abandoned

    def latest(self) -> tuple[int, dict[str, Any] | None]:
        with self._cv:
            return self._obs_seq, self._obs

    def act(self, action: dict) -> tuple[int, dict[str, Any]]:
        """Send one action; block until the observation after it arrives."""
        with self._cv:
            self._raise_if_over()
            seq = self._obs_seq
            self._pending = action
            self._cv.notify_all()
            while self._obs_seq == seq:
                self._raise_if_over()
                self._cv.wait(1.0)
            return self._obs_seq, self._obs

    def finish(self) -> None:
        """No more actions: the client ends the episode and RoboDojo judges the scene as it stands."""
        with self._cv:
            self._finished = True
            self._pending = None
            self._cv.notify_all()

    def wait_verdict(self, timeout: float) -> dict | None:
        end = time.monotonic() + timeout
        with self._cv:
            while self._verdict is None and not self._abandoned:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return None if self._verdict is None else dict(self._verdict)

    def _raise_if_over(self) -> None:
        if self._verdict is not None:
            raise EpisodeOver(dict(self._verdict))
        if self._abandoned:
            raise EpisodeOver({"abandoned": True})
