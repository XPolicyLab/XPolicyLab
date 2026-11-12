"""Policy-local image delay, measured in observation updates per environment."""
from collections import deque
import os
import re


def observation_lag(value=None):
    """Resolve the same nonnegative update count in server and client tools."""
    if value is None:
        value = os.environ.get('HY_OBS_LAG', '0')
    if isinstance(value, bool) or not re.fullmatch(r'[0-9]+', str(value).strip()):
        raise ValueError('HY_OBS_LAG must be a nonnegative integer')
    return int(str(value).strip())


def legacy_render_setting(lag):
    """Use SDK rendering with policy lag; reject two simultaneous delays."""
    value = os.environ.get('ROBODOJO_EVAL_LEGACY_RENDER', '0').strip().lower()
    if value not in {'1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'}:
        raise ValueError('ROBODOJO_EVAL_LEGACY_RENDER must be 1/0 or true/false')
    enabled = value in {'1', 'true', 'yes', 'on'}
    if lag and enabled:
        raise ValueError('HY_OBS_LAG cannot be combined with legacy rendering; set ROBODOJO_EVAL_LEGACY_RENDER=0')
    return enabled


class ImageObservationLag:
    """Own image snapshots; leave current state, language and other fields intact."""

    def __init__(self, lag):
        self.lag = observation_lag(lag)
        self._by_env = {}

    def reset(self):
        self._by_env.clear()

    def apply(self, env_idx, batch):
        if not self.lag:
            return batch
        frames = {key: value.copy() for key, value in batch.items()
                  if key.startswith('raw_images.')}
        pending = self._by_env.setdefault(env_idx, deque(maxlen=self.lag + 1))
        pending.append(frames)
        # Before k earlier updates exist, repeat this environment's first frame.
        for key, image in pending[0].items():
            batch[key] = image
            name = key.removeprefix('raw_images.')
            batch['observation.images.' + name] = (
                image.transpose(2, 0, 1)[None].astype('float32') / 255.0
            )
        return batch
