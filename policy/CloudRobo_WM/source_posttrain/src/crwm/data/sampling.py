from __future__ import annotations
from copy import deepcopy
from collections import OrderedDict
import math
import numpy as np
from .schema import fingerprint

class CoverageSampler:
    VERSION = 2
    SHUFFLE = 'global_feistel8_cyclewalk_v1'

    def __init__(self, leaf_sizes, *, weights=None, samples_per_epoch=None, seed=42, batch_size=1, num_processes=1, accumulation_steps=1, block_size=65536, manifest_hash=''):
        self.sizes = np.asarray(leaf_sizes, np.int64)
        if not len(self.sizes) or (self.sizes <= 0).any():
            raise ValueError('All sampling leaves must contain windows')
        w = self.sizes.astype(float) if weights is None else np.asarray(weights, float)
        if w.shape != self.sizes.shape or not np.isfinite(w).all() or (w < 0).any() or (w.sum() <= 0):
            raise ValueError('Invalid leaf weights')
        self.weights, self.seed = (w / w.sum(), int(seed))
        self.offsets = np.r_[0, self.sizes.cumsum()[:-1]]
        self.quantum = int(batch_size * num_processes * accumulation_steps)
        self.block_size = int(block_size)
        if self.quantum <= 0 or self.block_size <= 0:
            raise ValueError('Batch and block sizes must be positive')
        self.samples_per_epoch = math.ceil((samples_per_epoch or int(self.sizes.sum())) / self.quantum) * self.quantum
        self.contract = dict(sizes=self.sizes.tolist(), weights=self.weights.tolist(), seed=self.seed, samples_per_epoch=self.samples_per_epoch, batch_size=batch_size, world_size=num_processes, accumulation_steps=accumulation_steps, block_size=block_size, manifest_hash=manifest_hash, shuffle=self.SHUFFLE)
        self.committed = dict(position=0, counts=[0] * len(self.sizes), deficit=[0.0] * len(self.sizes))
        self.scheduled = deepcopy(self.committed)
        self.epoch = 0
        self.epoch_offset = 0
        self._cache = OrderedDict()

    def _permutation(self, leaf, cycle, block):
        key = (leaf, cycle, block)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        n = int(self.sizes[leaf])
        start = block * self.block_size
        if not 0 <= start < n:
            raise IndexError('Permutation cache block outside sampling leaf')
        values = np.arange(start, min(start + self.block_size, n), dtype=np.uint64)
        half_bits = max(1, ((n - 1).bit_length() + 1) // 2)
        mask = np.uint64((1 << half_bits) - 1)
        keys = np.random.SeedSequence([self.seed, leaf, cycle]).generate_state(8, dtype=np.uint64)

        def permute(x):
            left, right = (x >> half_bits, x & mask)
            for round_key in keys:
                mixed = right ^ round_key
                mixed = (mixed ^ mixed >> 30) * np.uint64(13787848793156543929)
                mixed = (mixed ^ mixed >> 27) * np.uint64(10723151780598845931)
                mixed ^= mixed >> 31
                left, right = (right, left ^ mixed & mask)
            return left << half_bits | right
        values = permute(values)
        outside = values >= n
        while outside.any():
            values[outside] = permute(values[outside])
            outside = values >= n
        result = values.astype(np.int64)
        self._cache[key] = result
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return result

    def _window(self, leaf, count):
        size = int(self.sizes[leaf])
        cycle, cursor = divmod(count, size)
        block, within = divmod(cursor, self.block_size)
        return int(self._permutation(leaf, cycle, block)[within])

    def _next(self, state):
        deficit = np.asarray(state['deficit']) + self.weights
        eligible = np.where(self.weights > 0, deficit, -np.inf)
        leaf = int(np.argmax(eligible))
        deficit[leaf] -= 1
        occurrence = state['counts'][leaf]
        index = int(self.offsets[leaf]) + self._window(leaf, occurrence)
        position = state['position']
        state['counts'][leaf] += 1
        state['position'] += 1
        state['deficit'] = deficit.tolist()
        return (index, position)

    def __iter__(self):
        self.scheduled = deepcopy(self.committed)
        end = (self.epoch + 1) * self.samples_per_epoch
        while self.scheduled['position'] < end:
            yield self._next(self.scheduled)

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch) + self.epoch_offset

    def set_epoch_offset(self, epoch):
        self.epoch_offset = int(epoch)

    def set_resume_batch_offset(self, batch):
        self.set_start_batch(batch)

    def clear_resume_batch_offset(self):
        pass

    def set_start_batch(self, batch):
        if batch and (not self.committed['position']):
            raise ValueError('Coverage resume requires sampler state, not a batch offset')

    def commit(self, count, observed=None):
        if count < 0 or self.committed['position'] + count > self.scheduled['position']:
            raise ValueError('Cannot commit windows that were not scheduled')
        actual = None if observed is None else sorted(observed, key=lambda x: x[1])
        if actual is not None and len(actual) != count:
            raise ValueError('Consumed batch size differs from the global optimizer batch')
        candidate = deepcopy(self.committed)
        for i in range(count):
            expected = self._next(candidate)
            if actual is not None and tuple(actual[i]) != expected:
                raise ValueError(f'Consumed sequence differs from global sampling stream: {actual[i]} != {expected}')
        self.committed = candidate

    def state_dict(self):
        return dict(version=self.VERSION, contract=self.contract, epoch=self.epoch, committed=deepcopy(self.committed))

    def load_state_dict(self, state):
        if state['version'] != self.VERSION or fingerprint(state['contract']) != fingerprint(self.contract):
            raise ValueError('Strict resume rejected: sampling algorithm/version/data/world/batch contract changed')
        self.epoch = int(state['epoch'])
        self.epoch_offset = self.epoch
        self.committed = deepcopy(state['committed'])
        self.scheduled = deepcopy(self.committed)

    def report(self, budget_samples=None):
        counts = np.asarray(self.committed['counts'])
        report = dict(shuffle=self.SHUFFLE, sampler_version=self.VERSION, scheduled=self.scheduled['position'], consumed=self.committed['position'], leaves=[dict(windows=int(n), consumed=int(c), unique=int(min(n, c)), uncovered=int(max(n - c, 0)), repeats=int(max(c - n, 0)), weight=float(w)) for n, c, w in zip(self.sizes, counts, self.weights)])
        report['windows_per_optimizer_step'] = self.quantum
        for i, leaf in enumerate(report['leaves']):
            leaf['leaf_index'] = i
            leaf['first_coverage_additional_steps_upper_bound'] = None if leaf['weight'] == 0 else math.ceil((leaf['uncovered'] + len(self.sizes)) / leaf['weight'] / self.quantum)
        if budget_samples is not None:
            if budget_samples > 100000:
                lower = counts + np.maximum(0, np.floor(budget_samples * self.weights).astype(np.int64) - len(self.sizes))
                positive = self.weights > 0
                bound = math.ceil(np.max((np.maximum(self.sizes[positive] - counts[positive], 0) + len(self.sizes)) / self.weights[positive]))
                report.update(budget_consumption_lower_bound=lower.tolist(), first_coverage_additional_samples_upper_bound=bound, first_coverage_additional_steps_upper_bound=math.ceil(bound / self.quantum), budget_covers_all_positive_leaves=bool(np.all(lower[positive] >= self.sizes[positive])))
                return report
            future = deepcopy(self.committed)
            first_coverage = None
            for step in range(int(budget_samples)):
                d = np.asarray(future['deficit']) + self.weights
                i = int(np.argmax(np.where(self.weights > 0, d, -np.inf)))
                d[i] -= 1
                future['deficit'] = d.tolist()
                future['counts'][i] += 1
                if first_coverage is None and np.all(np.asarray(future['counts'])[self.weights > 0] >= self.sizes[self.weights > 0]):
                    first_coverage = step + 1
            report['budget_consumption'] = future['counts']
            report['first_coverage_additional_samples'] = first_coverage
            report['first_coverage_additional_steps'] = None if first_coverage is None else math.ceil(first_coverage / self.quantum)
            report['budget_covers_all_positive_leaves'] = first_coverage is not None or all((c >= n for c, n, w in zip(counts, self.sizes, self.weights) if w > 0))
        return report
