"""Distribute a shuffled layout cycle across simultaneous RLinf environment groups."""

import numpy as np


class LayoutSampler:
    def __init__(self, layout_ids, *, seed, groups, rank, world_size):
        self.ids = np.asarray(layout_ids, dtype=np.int64)
        if not self.ids.size:
            raise ValueError("The selected task has no layouts.")
        self.groups = groups
        self.start = rank * groups
        self.stride = world_size * groups
        self.reseed(seed)

    def reseed(self, seed):
        self.order = np.random.default_rng(seed).permutation(self.ids)
        self.offset = self.start

    def take(self):
        # All ranks read slices of the same global batch. Carry the remainder
        # into the next batch so even a pool smaller than the batch stays balanced.
        indices = (self.offset + np.arange(self.groups)) % self.order.size
        self.offset = (self.offset + self.stride) % self.order.size
        return self.order[indices]
