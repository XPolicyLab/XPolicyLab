"""Hand-over between XPolicyLab's request threads and the Auriga episode thread.

The RoboDojo client asks for one action chunk at a time. While the model reasons there is no
action: ``next_actions`` answers WAIT after ``poll_s`` and the client asks again, so physics does
not advance and no native step is spent. When the episode thread has joint rows to execute it
parks them here and blocks until the observation after the last row arrives.
"""

from __future__ import annotations

import threading
import time
from typing import Any

WAIT = [{"auriga_wait": True}]


class EpisodeOver(Exception):
    """RoboDojo ended the episode (success, step limit) or the client left the trial."""


def hold_still(obs: dict) -> dict:
    """An action that keeps every joint and gripper at its observed value (debug wiring check)."""
    state = (obs or {}).get("state") or {}
    action = {}
    for key, value in state.items():
        if key.endswith("_arm_joint_state") or key.endswith("_ee_joint_state"):
            values = value.tolist() if hasattr(value, "tolist") else list(value)
            action[key] = [float(v) for v in values]
    if not action:
        action = {"left_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [1.0],
                  "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]}
    return action


class Bridge:
    def __init__(self, poll_s: float = 20.0):
        self.poll_s = poll_s
        self._cv = threading.Condition()
        self._obs: dict | None = None
        self._obs_seq = 0
        self._pending: list | None = None
        self._finished = False
        self._verdict: dict | None = None
        self._abandoned = False

    # -- XPolicyLab side --------------------------------------------------------------------------
    def offer_obs(self, obs: dict) -> int:
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
                    rows, self._pending = self._pending, None
                    self._cv.notify_all()
                    return rows
                if self._finished or self._abandoned or self._verdict is not None:
                    return []
                left = deadline - time.monotonic()
                if left <= 0:
                    return WAIT
                self._cv.wait(left)

    def conclude(self, verdict: dict) -> None:
        with self._cv:
            self._verdict = dict(verdict or {})
            self._cv.notify_all()

    def abandon(self) -> None:
        with self._cv:
            self._abandoned = True
            self._cv.notify_all()

    # -- episode side -----------------------------------------------------------------------------
    def latest(self) -> tuple[int, dict | None]:
        with self._cv:
            return self._obs_seq, self._obs

    def act(self, rows: list[dict]) -> tuple[int, dict]:
        """Send one chunk of native joint rows; block until the observation after it arrives.

        The demo-aligned loop sends an observation after every row, so callers that need
        per-row measurements (arrival, stall and grip guards) pass one row per call.
        """
        if not rows:
            raise ValueError("empty action chunk")
        with self._cv:
            self._raise_if_over()
            seq = self._obs_seq
            self._pending = list(rows)
            self._cv.notify_all()
            while self._obs_seq == seq:
                self._raise_if_over()
                self._cv.wait(1.0)
            return self._obs_seq, self._obs

    def finish(self) -> None:
        with self._cv:
            self._finished = True
            self._pending = None
            self._cv.notify_all()

    @property
    def verdict(self) -> dict | None:
        with self._cv:
            return None if self._verdict is None else dict(self._verdict)

    @property
    def over(self) -> bool:
        """The client ended the trial (verdict received) or left it (reset of a new trial)."""
        with self._cv:
            return self._verdict is not None or self._abandoned

    def wait_verdict(self, timeout_s: float) -> dict | None:
        """After ``finish``: wait for the client's final check to come back through ``trial_end``."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cv:
            while self._verdict is None and not self._abandoned:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)
            return None if self._verdict is None else dict(self._verdict)

    def _raise_if_over(self) -> None:
        if self._verdict is not None:
            raise EpisodeOver(dict(self._verdict))
        if self._abandoned:
            raise EpisodeOver({"abandoned": True})
