"""Bounded, episode-local calibrated RGB history for support reconstruction."""
import numpy as np
from . import Camera, GeometryUnavailable


class SupportViewHistory:
    def __init__(self, *, max_references=16, min_translation=.005,
                 min_rotation=.02, source_name='cam_head',
                 reference_names=('cam_left_wrist', 'cam_right_wrist')):
        if (not reference_names or len(set(reference_names)) != len(reference_names)
                or source_name in reference_names or max_references < len(reference_names)
                or not np.isfinite([min_translation, min_rotation]).all()
                or min_translation <= 0 or min_rotation <= 0):
            raise ValueError('Finite positive sampling thresholds and sufficient capacity required')
        self.max_references = max_references
        self.min_translation, self.min_rotation = min_translation, min_rotation
        self.source_name, self.reference_names = source_name, tuple(reference_names)
        self.reset()

    def reset(self):
        self.source = None
        self._anchors = {}
        self._recent = []

    @staticmethod
    def _snapshot(camera):
        saved = Camera(camera.image.copy(), camera.k.copy(), camera.transform.copy())
        for array in (saved.image, saved.k, saved.transform):
            array.setflags(write=False)
        return saved

    def _near(self, first, second):
        delta = np.linalg.norm(first.transform[:3, 3] - second.transform[:3, 3])
        relative = first.transform[:3, :3].T @ second.transform[:3, :3]
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        return delta < self.min_translation and angle < self.min_rotation

    def observe(self, observation):
        # Validate the whole observation before modifying episode state.
        source = Camera.from_observation(observation, self.source_name)
        references = [(name, Camera.from_observation(observation, name))
                      for name in self.reference_names]
        if self.source is None:
            self.source = self._snapshot(source)
        added = 0
        for name, camera in references:
            if name not in self._anchors:
                self._anchors[name] = self._snapshot(camera)
                added += 1
            elif not any(self._near(camera, old)
                         for old in [*self._anchors.values(), *self._recent]):
                self._recent.append(self._snapshot(camera))
                added += 1
        capacity = self.max_references - len(self._anchors)
        self._recent = self._recent[-capacity:] if capacity else []
        return added

    @property
    def cameras(self):
        if self.source is None:
            raise GeometryUnavailable('No calibrated RGB observations in this episode')
        return (self.source, *self._anchors.values(), *self._recent)
