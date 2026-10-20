from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .schema import atomic_json, file_sha256, fingerprint, within

class WindowIndex:

    def __init__(self, manifest, split='train', window_stride=1, language=None):
        if manifest['version'] != 1 or window_stride < 1:
            raise ValueError('Unsupported manifest/stride')
        self.manifest, self.stride, self.split = (manifest, int(window_stride), split)
        from .language import language_policy, training_intervals
        self.language = language_policy(manifest.get('source', {}), language)
        self.intervals, counts = ([], [])
        for ep in manifest['episodes']:
            if ep['split'] != split:
                continue
            previous = -1
            for start, end in ep['intervals']:
                if not 0 <= start < end <= ep['length'] or start < previous:
                    raise ValueError('Invalid or overlapping episode intervals')
                previous = end
            for view, start, end in training_intervals(ep, self.language):
                self.intervals.append((view, start, end))
                counts.append((end - start + self.stride - 1) // self.stride)
        self.ends = np.cumsum(counts, dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = int(np.searchsorted(self.ends, index, side='right'))
        ep, start, end = self.intervals[i]
        offset = index - (int(self.ends[i - 1]) if i else 0)
        return (ep, start + offset * self.stride, end)

    @property
    def digest(self):
        from crwm.checkpoint import portable_manifest
        value = dict(manifest=portable_manifest(self.manifest), split=self.split, window_stride=self.stride)
        if self.language['mode'] != 'episode':
            from .language import VERSION
            value['language_view'] = dict(version=VERSION, **self.language)
        return fingerprint(value)
