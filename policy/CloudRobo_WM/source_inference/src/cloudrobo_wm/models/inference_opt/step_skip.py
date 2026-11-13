from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

class StepSkipController:
    VALID_AGGREGATIONS = ('mean', 'min', 'concat')

    def __init__(self, enabled: bool, thresholds: Sequence[Tuple[float, int]]=((0.95, 4), (0.93, 2)), sim_aggregation: str='mean'):
        if sim_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(f'sim_aggregation must be one of {self.VALID_AGGREGATIONS}; got {sim_aggregation!r}.')
        self.enabled = bool(enabled)
        self.thresholds: List[Tuple[float, int]] = [(float(t), int(n)) for t, n in thresholds]
        self.sim_aggregation = sim_aggregation
        self._prev_refs: List[List[torch.Tensor]] = []
        self._skip_countdown: int = 0
        self._cached_preds: Any = None
        self.steps_run: int = 0
        self.last_sim: Optional[float] = None
        self.sim_history: List[float] = []
        self.per_stream_history: List[List[float]] = []
        self.decision_history: List[str] = []

    def should_run(self) -> bool:
        if not self.enabled:
            return True
        if len(self._prev_refs) < 2:
            return True
        if self._skip_countdown > 1:
            self._skip_countdown -= 1
            return False
        if self._skip_countdown == 1:
            self._skip_countdown = 0
            return True
        last_streams = self._prev_refs[-1]
        prev_streams = self._prev_refs[-2]
        per_stream = [F.cosine_similarity(l, p, dim=1).mean().item() for l, p in zip(last_streams, prev_streams)]
        self.per_stream_history.append(per_stream)
        if self.sim_aggregation == 'concat':
            last = torch.cat(last_streams, dim=1)
            prev = torch.cat(prev_streams, dim=1)
            sim = F.cosine_similarity(last, prev, dim=1).mean().item()
        elif self.sim_aggregation == 'mean':
            sim = sum(per_stream) / len(per_stream)
        else:
            sim = min(per_stream)
        self.last_sim = sim
        self.sim_history.append(sim)
        for threshold, count in self.thresholds:
            if sim > threshold:
                self._skip_countdown = count
                self.decision_history.append(f'skip{count}')
                return False
        self.decision_history.append('run')
        return True

    def record(self, sim_refs: Sequence[torch.Tensor], preds: Any) -> None:
        self._cached_preds = preds
        self.steps_run += 1
        if not self.enabled or not sim_refs:
            return
        flat_streams = [r.flatten(1).float().detach() for r in sim_refs]
        self._prev_refs.append(flat_streams)
        if len(self._prev_refs) > 2:
            self._prev_refs.pop(0)

    def cached(self) -> Any:
        if self._cached_preds is None:
            raise RuntimeError('StepSkipController.cached() called before any record() — no cached prediction to reuse. This indicates a logic error in the denoising loop (should_run() must return True on the first iteration).')
        return self._cached_preds

def resolve_sim_sources(explicit: Optional[Sequence[str]], denoised_flags: dict[str, bool]) -> List[str]:
    valid_names = list(denoised_flags.keys())
    if explicit is None:
        return [name for name in valid_names if denoised_flags[name]]
    if len(explicit) == 0:
        raise ValueError('`step_skip_sim_sources` must be None (auto) or a non-empty list; got an empty list.')
    resolved: List[str] = []
    for name in explicit:
        if name not in denoised_flags:
            raise ValueError(f'Unknown step_skip_sim_source {name!r}; valid for this model: {valid_names}.')
        if not denoised_flags[name]:
            raise ValueError(f'step_skip_sim_source {name!r} requires the corresponding joint flag to be True (stream is not being denoised).')
        resolved.append(name)
    return resolved
