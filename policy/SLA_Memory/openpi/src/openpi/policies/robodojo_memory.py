"""Online history buffer that rebuilds the training-time history frames at inference.

Call ``update`` once per camera/control frame, including frames for which no new
action chunk is requested. Call ``build_observation`` only when inference is
needed. Reset the buffer at every episode boundary.
"""

from collections import deque
import copy
import dataclasses

import numpy as np


@dataclasses.dataclass
class RobodojoMemoryBuffer:
    """Build the cam_high stack used in training: [episode start, t-8s, ..., t-1s, current]."""

    fps: int
    history_seconds: int = 8

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.history_seconds <= 0:
            raise ValueError(f"history_seconds must be positive, got {self.history_seconds}")
        self.reset()

    def reset(self) -> None:
        """Start a new episode and discard every frame from the previous one."""
        self._cam_high: deque[np.ndarray] = deque(maxlen=self.history_seconds * self.fps + 1)
        self._episode_start: np.ndarray | None = None
        self._latest_observation: dict | None = None

    def update(self, observation: dict) -> None:
        """Record one raw observation; this must run at ``fps`` rather than action-chunk rate."""
        image = np.asarray(observation["images"]["cam_high"])
        if image.ndim != 3:
            raise ValueError(f"Expected one cam_high image [H,W,C] or [C,H,W], got {image.shape}")
        image = np.array(image, copy=True)
        if self._episode_start is None:
            self._episode_start = image
        self._cam_high.append(image)
        self._latest_observation = observation

    def build_observation(self, observation: dict | None = None) -> dict:
        """Return an inference observation with [start, t-8s..t-1s, current]."""
        if observation is not None:
            self.update(observation)
        if self._latest_observation is None or self._episode_start is None:
            raise RuntimeError("No frame is available; call update after resetting the episode")

        frames = list(self._cam_high)
        current = frames[-1]
        memory = [self._episode_start]
        for seconds_ago in range(self.history_seconds, 0, -1):
            index = len(frames) - 1 - seconds_ago * self.fps
            memory.append(frames[index] if index >= 0 else self._episode_start)

        result = copy.copy(self._latest_observation)
        result["images"] = dict(self._latest_observation["images"])
        result["images"]["cam_high"] = np.stack([*memory, current])
        return result

