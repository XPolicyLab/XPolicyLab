"""Shared causal fixed-offset Memory selection (control-step coordinates)."""
from __future__ import annotations


def validate_history_offsets(offsets):
    values = tuple(offsets)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("history_offsets must contain positive integer control steps")
    if values != tuple(sorted(set(values))):
        raise ValueError("history_offsets must be sorted and unique")
    return values


def select_history(query_step: int, offsets, *, start_step: int = 0):
    """Return fixed [anchor, oldest..newest] indices/valid/ages; never repeat O0."""
    offsets = validate_history_offsets(offsets)
    if query_step < start_step:
        raise ValueError("Memory query precedes episode start")
    indices = [start_step] + [query_step - d for d in reversed(offsets)]
    valid = [True] + [i > start_step for i in indices[1:]]
    ages = [query_step - i if ok else 0 for i, ok in zip(indices, valid)]
    return [i if ok else start_step for i, ok in zip(indices, valid)], valid, ages


class HistoryObservationBuffer:
    """Per-session bounded observations. Caller supplies executed-step IDs, not calls.

    Keeps the episode anchor separately. Missing exact observations stay missing;
    no unbounded stale-frame substitution and no cross-episode carryover.
    """
    def __init__(self, offsets):
        self.offsets = validate_history_offsets(offsets)
        self.reset()

    def reset(self):
        self.anchor = None
        self.start = None
        self.latest = None
        self.frames = {}

    def append(self, step, value):
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("Memory observation step must be an integer")
        if self.latest is not None and step < self.latest:
            raise ValueError("Memory observation time moved backwards; reset episode")
        if self.start is None:
            self.start, self.anchor = step, value
        self.latest = step
        self.frames[step] = value
        cutoff = step - max(self.offsets, default=0)
        self.frames = {k: v for k, v in self.frames.items() if k >= cutoff}

    def snapshot(self, query_step):
        if self.start is None or query_step not in self.frames:
            raise ValueError("Current observation must be appended before Memory snapshot")
        indices, valid, ages = select_history(query_step, self.offsets, start_step=self.start)
        values = [self.anchor]
        for slot, i in enumerate(indices[1:], 1):
            ok = valid[slot] and i in self.frames
            valid[slot] = ok
            ages[slot] = query_step - i if ok else 0
            values.append(self.frames[i] if ok else None)
        return values, valid, ages
