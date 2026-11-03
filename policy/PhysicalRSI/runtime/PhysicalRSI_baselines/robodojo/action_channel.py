"""One actor's observation/action rendezvous for XPolicyLab-style rollouts.

The environment thread publishes observations and takes one-step action chunks.
The isolated policy thread observes and steps; a step completes only after a fresh
post-action observation. Reset creates a new channel, never reuses old waiters.
"""

import math
import time
import uuid
from copy import deepcopy
from threading import Condition

from .expert_probe import validate_actions


class ActionChannel:
    def __init__(self, dimensions, *, cameras=None):
        if not dimensions or any(
            type(n) is not int or n < 1 for n in dimensions.values()
        ):
            raise ValueError("Explicit positive action dimensions required")
        self.dimensions = dict(dimensions)
        self.cameras = dict(cameras or {})
        for name, shape in self.cameras.items():
            if (
                not isinstance(name, str)
                or not name
                or len(shape) != 3
                or any(type(n) is not int or n < 1 for n in shape)
                or shape[2] != 3
                or shape[0] * shape[1] > 4096 * 4096
            ):
                raise ValueError(
                    "Declared RGB camera shape required (at most 4096² pixels)"
                )
        self.cameras = {key: tuple(shape) for key, shape in self.cameras.items()}
        self._channel_id = uuid.uuid4().hex
        self._vision = {}
        self._received_at = None
        self._condition = Condition()
        self._state = None
        self._sequence = 0
        self._pending = None
        self._delivered = False
        self._closed = None
        self._finished = False
        self._hold_limit = 0
        self._hold_count = 0
        self._hold_action = None

    def _wait(self, predicate, deadline):
        if not math.isfinite(deadline):
            raise ValueError("Finite absolute deadline required")
        while True:
            if self._closed is not None:
                raise RuntimeError("Actor channel closed: " + self._closed)
            if time.monotonic() >= deadline:
                self._closed = "observation/action deadline exceeded"
                self._condition.notify_all()
                raise TimeoutError(self._closed)
            if predicate():
                return
            self._condition.wait(deadline - time.monotonic())

    def update_obs(self, observation):
        # Whitelist observable joint fields; do not retain task score or layouts.
        state = validate_actions(
            [{key: observation["state"][key] for key in self.dimensions}],
            self.dimensions,
        )[0]
        vision = {}
        if self.cameras:
            import numpy as np

            for name, shape in self.cameras.items():
                color = observation["vision"][name]["color"]
                if (
                    not isinstance(color, np.ndarray)
                    or color.dtype != np.uint8
                    or color.shape != shape
                ):
                    raise ValueError(
                        "Camera RGB array differs from declared uint8 shape"
                    )
                vision[name] = {"color": color.copy()}
        with self._condition:
            if self._closed is not None:
                raise RuntimeError("Actor channel is closed")
            if self._sequence and not self._delivered:
                raise RuntimeError("Observation has no preceding dispatched action")
            self._state = state
            self._vision = vision
            self._received_at = time.monotonic()
            self._sequence += 1
            self._pending = None
            self._delivered = False
            self._condition.notify_all()

    def observe(self, *, deadline):
        with self._condition:
            self._wait(lambda: self._state is not None, deadline)
            return deepcopy(self._state)

    def snapshot(self, *, deadline):
        """Trusted perception view: selected RGB and joints from the same update."""
        with self._condition:
            self._wait(lambda: self._state is not None, deadline)
            return dict(
                channel_id=self._channel_id,
                sequence=self._sequence,
                received_at=self._received_at,
                state=deepcopy(self._state),
                vision=deepcopy(self._vision),
            )

    def require_current(self, snapshot, *, max_age_s, deadline):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Positive observation age limit required")
        with self._condition:
            self._wait(lambda: self._state is not None, deadline)
            if (
                snapshot["channel_id"] != self._channel_id
                or snapshot["sequence"] != self._sequence
                or snapshot["received_at"] != self._received_at
            ):
                raise RuntimeError("Perception refers to a replaced observation")
            if self._pending is not None:
                raise RuntimeError("Perception cannot qualify an outstanding action")
            if time.monotonic() - self._received_at > max_age_s:
                raise TimeoutError("Observation exceeded local receipt age limit")

    def step(self, action, *, deadline):
        action = validate_actions([action], self.dimensions)[0]
        with self._condition:
            self._wait(lambda: self._state is not None, deadline)
            if self._finished:
                raise RuntimeError("Policy has already completed")
            if self._pending is not None:
                raise RuntimeError("Actor already has an outstanding action")
            sequence = self._sequence
            self._pending = action
            self._condition.notify_all()
            self._wait(lambda: self._sequence > sequence, deadline)
            return deepcopy(self._state)

    def get_action(self, *, deadline):
        with self._condition:
            if self._delivered:
                raise RuntimeError(
                    "Action already dispatched; fresh observation required"
                )
            self._wait(
                lambda: (
                    self._pending is not None
                    or (self._finished and self._state is not None)
                ),
                deadline,
            )
            if self._pending is None:
                if self._hold_count >= self._hold_limit:
                    self.close("completion hold budget exhausted")
                    raise RuntimeError("Completion hold budget exhausted")
                if self._hold_action is None:
                    self._hold_action = deepcopy(self._state)
                self._pending = deepcopy(self._hold_action)
                self._hold_count += 1
            if self._delivered:
                raise RuntimeError(
                    "Action already dispatched; fresh observation required"
                )
            self._delivered = True
            return [deepcopy(self._pending)]

    def finish(self, *, hold_steps=0):
        """Bounded position hold after successful code return; not task success.

        The first available post-policy observed position becomes a fixed hold target.
        The environment still supplies every post-action observation and termination.
        Only enable for position-command adapters whose units match observed state.
        """
        if type(hold_steps) is not int or not 0 <= hold_steps <= 10000:
            raise ValueError("Bounded completion hold steps required")
        with self._condition:
            if self._closed is not None:
                raise RuntimeError("Actor channel is closed")
            if self._finished or self._pending is not None:
                raise RuntimeError("Cannot finish twice or with an outstanding action")
            self._finished = True
            self._hold_limit = hold_steps
            if hold_steps == 0:
                self._closed = "policy execution ended"
            self._condition.notify_all()

    def close(self, reason):
        with self._condition:
            if self._closed is None:
                self._closed = str(reason)
            self._condition.notify_all()

    def status(self):
        with self._condition:
            return dict(
                observation_sequence=self._sequence,
                outstanding_action=self._pending is not None,
                action_dispatched=self._delivered,
                closed=self._closed,
                policy_finished=self._finished,
                completion_hold_steps=self._hold_count,
                completion_hold_limit=self._hold_limit,
            )
