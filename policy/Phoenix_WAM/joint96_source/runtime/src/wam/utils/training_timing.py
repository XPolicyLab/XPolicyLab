"""Opt-in, bounded, per-rank timing. No distributed collectives or RNG draws.

CUDA events measure stream elapsed time (including launch gaps and waits), not
kernel-only time. Nested ranges are inclusive and must not be added together.
Host durations without synchronization are submission/wait time, not GPU time.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import socket
import time

import torch


_CURRENT = ContextVar("wam_training_timing", default=None)
_ACTIVE_TIMER = None


def current_timer():
    # CUDA autograd may recompute checkpoints on an engine worker thread that
    # does not inherit ContextVars. Training supports one active timer/process.
    return _CURRENT.get() or _ACTIVE_TIMER


def timing_section(name):
    timer = current_timer()
    return timer.section(name) if timer is not None else nullcontext()


def timing_count(name, amount=1):
    timer = current_timer()
    if timer is not None:
        timer.counts[name] += amount


def timing_attention_call(backend):
    timer = current_timer()
    if timer is not None:
        phase = "backward_recompute" if "backward_and_engine_step" in timer.stack else "forward"
        timer.counts[f"{backend}_calls_{phase}"] += 1


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with timing_section(name):
                return function(*args, **kwargs)
        return wrapper
    return decorate


@contextmanager
def time_methods(specs):
    """Temporarily instrument existing bound methods; restore even on failure."""
    if current_timer() is None:
        yield
        return
    restored = []
    try:
        for obj, attribute, label in specs:
            if obj is None or not callable(getattr(obj, attribute, None)):
                continue
            had_instance_value = attribute in vars(obj)
            previous = vars(obj).get(attribute)
            original = getattr(obj, attribute)
            setattr(obj, attribute, timed(label)(original))
            restored.append((obj, attribute, had_instance_value, previous))
        yield
    finally:
        for obj, attribute, existed, previous in reversed(restored):
            if existed:
                setattr(obj, attribute, previous)
            else:
                delattr(obj, attribute)


class TrainingTimer:
    def __init__(self, output_dir, *, rank, world_size, device, steps=0, start_step=5):
        self.steps, self.start_step = int(steps), int(start_step)
        if self.steps < 0 or self.start_step < 0:
            raise ValueError("Timing steps and start_step must be nonnegative")
        self.rank, self.world_size = int(rank), int(world_size)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.active = False
        self.rows, self.stack, self.counts = [], [], Counter()
        self.context_token = None

    @classmethod
    def from_environment(cls, output_dir, *, rank, world_size, device):
        return cls(
            os.environ.get("WAM_TIMING_DIR", str(Path(output_dir) / "timing")),
            rank=rank, world_size=world_size, device=device,
            steps=int(os.environ.get("WAM_TIMING_STEPS", "0")),
            start_step=int(os.environ.get("WAM_TIMING_START_STEP", "5")),
        )

    def begin(self, step):
        global _ACTIVE_TIMER
        if self.active:
            raise RuntimeError("Previous timing step was not finished")
        if not self.start_step <= step < self.start_step + self.steps:
            return
        if current_timer() is not None:
            raise RuntimeError("Nested training timers are not supported")
        # Drain preceding work before attributing time to this iteration.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step = int(step)
        self.rows, self.stack, self.counts = [], [], Counter()
        self.started = time.perf_counter()
        self.active = True
        _ACTIVE_TIMER = self
        self.context_token = _CURRENT.set(self)

    @contextmanager
    def section(self, name):
        if not self.active:
            yield
            return
        path = "/".join([*self.stack, name])
        start_event = end_event = None
        if self.device.type == "cuda":
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        self.stack.append(name)
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
            self.stack.pop()
            self.rows.append((path, elapsed, start_event, end_event))

    def cancel(self):
        global _ACTIVE_TIMER
        if self.context_token is not None:
            _CURRENT.reset(self.context_token)
        self.context_token, self.active = None, False
        if _ACTIVE_TIMER is self:
            _ACTIVE_TIMER = None
        self.rows, self.stack, self.counts = [], [], Counter()

    def finish(self, *, sample=None):
        if not self.active:
            return None
        try:
            if self.stack:
                raise RuntimeError("Unclosed timing ranges")
            drain_start = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            drain_ms = (time.perf_counter() - drain_start) * 1000
            elapsed_ms = (time.perf_counter() - self.started) * 1000
            sections = defaultdict(lambda: dict(calls=0, host_ms=0.0, cuda_ms=None))
            for path, host_ms, start, end in self.rows:
                value = sections[path]
                value["calls"] += 1
                value["host_ms"] += host_ms
                if start is not None:
                    value["cuda_ms"] = (value["cuda_ms"] or 0.0) + start.elapsed_time(end)
            record = dict(schema="wam.training_timing.v1", step=self.step,
                          rank=self.rank, world_size=self.world_size, host=socket.gethostname(),
                          pid=os.getpid(), device=str(self.device), unix_time=time.time(),
                          iteration_wall_ms=elapsed_ms, end_drain_ms=drain_ms,
                          sections=dict(sections), counts=dict(self.counts))
            if sample is not None:
                record["profiles"] = dict(Counter(sample.get("profile_id", [])))
                pad = sample.get("camera_is_pad")
                if pad is not None:
                    record["real_views_per_sample"] = (~pad.detach().cpu()).sum(dim=1).tolist()
            if self.device.type == "cuda":
                record["cuda_memory"] = dict(
                    allocated=torch.cuda.memory_allocated(self.device),
                    reserved=torch.cuda.memory_reserved(self.device),
                    lifetime_peak_allocated=torch.cuda.max_memory_allocated(self.device),
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"rank_{self.rank:05d}_pid_{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            return record
        finally:
            self.cancel()
