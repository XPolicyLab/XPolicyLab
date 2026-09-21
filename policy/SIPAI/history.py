"""Exact action-step history, isolated by evaluation and environment."""

from collections import deque

import numpy as np


class CameraHistory:
    def __init__(self, frames=20, action_interval=25):
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (frames, action_interval)):
            raise ValueError("History frame count and interval must be positive integers")
        self.frames = frames
        self.action_interval = action_interval
        self._states = {}

    def update(self, image, *, evaluation_id, env_idx):
        # Called exactly once per control step, including the initial observation.
        key = (evaluation_id, env_idx)
        if key not in self._states:
            self._states[key] = deque(maxlen=self.frames * self.action_interval + 1)
        self._states[key].append(np.asarray(image).copy())

    def frames_for(self, *, evaluation_id, env_idx):
        images = self._states.get((evaluation_id, env_idx), ())
        return [
            images[-offset - 1] if len(images) > offset else None
            for offset in range(self.frames * self.action_interval, 0, -self.action_interval)
        ]

    def reset(self, evaluation_id=None):
        if evaluation_id is None:
            self._states.clear()
        else:
            for key in list(self._states):
                if key[0] == evaluation_id:
                    del self._states[key]
