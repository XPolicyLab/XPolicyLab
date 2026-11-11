"""Episode-isolated, dense 25 Hz first-frame and CD-LAM memory."""
from collections import deque
from dataclasses import asdict

import numpy as np

from starVLA.visual_history import (
    VisualHistoryConfig, image_metadata, prepare_history_images,
)
from starVLA.lam_memory import LAMMemoryBank

VIEWS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


class MemoryInputs:
    def __init__(self, data_config):
        self.spec = VisualHistoryConfig.from_data_config(data_config, for_inference=True)
        self.lam = LAMMemoryBank()
        self.last = -1
        self.first = None
        self.current = None
        self.teacher_frames = deque(maxlen=4)
        self.pending_pairs = []
        self.clock_mode = None
        self.last_action_frame = None

    def observation_tick(self, obs, head):
        """MR0 already streams every step; old clients need no extra wire fields."""
        mode = 'explicit' if 'memory_frame_index' in obs else 'counted'
        if self.clock_mode is not None and mode != self.clock_mode:
            raise ValueError('Memory clock mode changed within an episode; reset before switching clients')
        if mode == 'counted' and 'memory_timestamp_sec' in obs:
            raise ValueError('memory_timestamp_sec requires memory_frame_index')
        frame = obs.get('memory_frame_index', self.last + 1)
        fresh = self.validate_tick(frame, float(obs.get('memory_timestamp_sec', frame / 25)), head)
        return frame, fresh, mode

    def validate_action_tick(self, execute_horizon):
        if self.clock_mode != 'counted':
            return
        expected = 0 if self.last_action_frame is None else self.last_action_frame + execute_horizon
        if self.last != expected:
            raise ValueError(
                f'MR0 memory requires one update_obs per control step: expected counted frame '
                f'{expected}, received {self.last}. Sparse/chunk-only observations cannot supply '
                'exact CD-LAM history; use the MR0 per-step deploy loop.'
            )

    def validate_tick(self, frame, timestamp, head):
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise ValueError('memory_frame_index must be a nonnegative integer')
        if not np.isfinite(timestamp) or abs(timestamp - frame / 25) > 1e-6:
            raise ValueError('Memory observation clock must be exactly 25 Hz from episode start')
        if frame not in (self.last, self.last + 1):
            raise ValueError(f'Missing observation tick: {self.last} -> {frame}; send every control step')
        if frame == self.last and (self.current is None or not np.array_equal(head, self.current)):
            raise ValueError('A repeated observation tick must have the same main-camera frame')
        return frame != self.last

    def pair(self, frame, teacher_image):
        if frame < 4:
            return None
        if len(self.teacher_frames) != 4 or self.teacher_frames[0][0] != frame - 4:
            raise ValueError('Missing causal CD-LAM source frame')
        # Share immutable preprocessed frames; stack only the next GPU batch.
        return (self.teacher_frames[0][1], teacher_image)

    def append(self, frame, head, teacher_image):
        if frame == self.last:
            return
        if frame == 0:
            self.first = self._history_image(head)
        if frame >= 4:
            self.pending_pairs.append((frame, self.pair(frame, teacher_image)))
        self.teacher_frames.append((frame, teacher_image))
        self.current = head.copy()
        self.last = frame

    def _history_image(self, head):
        # Identical to training's role-specific resize, performed once at insert.
        # Frozen prepare_history_images then sees the target size and skips resize.
        return np.asarray(prepare_history_images(
            [head], [{'role': 'history'}], self.spec, None, None,
        )[0])

    def example(self, images, language, state):
        example = dict(image=images, lang=language, state=state)
        example['image'] = ([self.first] if self.last else []) + images
        example['image_metas'] = image_metadata(self.spec, self.last, 25, VIEWS)
        example['lam_memory'] = self.lam.select(self.last)
        example['visual_history'] = asdict(self.spec)
        return example
