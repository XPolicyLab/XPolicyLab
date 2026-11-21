"""Opt-in, bounded, asynchronous host timing traces; never synchronizes CUDA/DDP.

Set WSP_TIMING_TRACE_DIR to enable. Step summaries are recorded on every rank;
expensive loader details are emitted only above WSP_TIMING_TRACE_MIN_SECONDS.
Each process owns a file, so workers never contend on a shared log handle.
"""
from __future__ import annotations

import atexit
import contextvars
import json
import math
import os
from pathlib import Path
import queue
import resource
import socket
import threading
import time
import warnings

_current = contextvars.ContextVar('wsp_timing_span', default=None)
_writer = None
_directory = None
_threshold = 2.0


def configure_trace() -> bool:
    """Validate configuration before model allocation, without starting a thread."""
    global _directory, _threshold
    value = os.environ.get('WSP_TIMING_TRACE_DIR', '')
    _directory = None
    if not value:
        return False
    path = Path(value).expanduser().resolve()
    from worldscape_policy.data.robodojo_snapshots import validate_writable_path
    path = validate_writable_path(path)
    threshold = float(os.environ.get('WSP_TIMING_TRACE_MIN_SECONDS', '2'))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('WSP_TIMING_TRACE_MIN_SECONDS must be finite and nonnegative')
    _directory, _threshold = path, threshold
    return True


def enabled() -> bool:
    return _directory is not None


class _Writer:
    def __init__(self, directory):
        self.pid = os.getpid()
        self.directory = directory
        self.items = queue.Queue(maxsize=256)
        self.dropped = 0
        self.failed = False
        self.closed = False
        self.identity = dict(host=socket.gethostname(), rank=os.environ.get('RANK', '0'),
                             local_rank=os.environ.get('LOCAL_RANK', '0'), pid=self.pid)
        self.thread = threading.Thread(target=self._run, name='wsp-timing-writer', daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def submit(self, event):
        if self.failed:
            return
        try:
            self.items.put_nowait({**self.identity, 'dropped_events': self.dropped, **event})
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            host = ''.join(c for c in self.identity['host'] if c.isalnum() or c in '-_.')
            rank = ''.join(c for c in self.identity['rank'] if c.isalnum() or c in '-_.')
            filename = f"{host}-rank{rank}-pid{self.pid}-{time.time_ns()}.jsonl"
            with (self.directory / filename).open('x') as stream:
                startup = dict(event='process', **self.identity, time_ns=time.time_ns(),
                    affinity=sorted(os.sched_getaffinity(0)), cpu_count=os.cpu_count(),
                    settings={key: os.environ.get(key) for key in (
                        'DATALOADER_NUM_WORKERS', 'DATALOADER_PREFETCH_FACTOR',
                        'DATALOADER_GLOBAL_BUCKET_SAMPLE_WORKERS', 'OMP_NUM_THREADS',
                        'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'ROBODOJO_INDEX_PATH', 'DATA_ROOT')})
                stream.write(json.dumps(startup) + '\n')
                size = 0
                while True:
                    event = self.items.get()
                    if event is None:
                        return
                    line = json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'
                    size += len(line.encode())
                    if size > 256 * 1024 * 1024:
                        raise RuntimeError('256 MiB per-process trace limit reached')
                    stream.write(line)
                    stream.flush()
        except Exception as exc:
            self.failed = True
            warnings.warn(f'WSP timing trace disabled for pid {self.pid}: {exc}', RuntimeWarning)

    def close(self):
        if os.getpid() != self.pid or not self.thread.is_alive():
            return
        # Never make a filesystem stall become a training shutdown stall.
        try:
            self.items.put_nowait(None)
        except queue.Full:
            return
        self.closed = True
        self.thread.join(timeout=2)


def emit(event: str, **fields) -> None:
    global _writer
    if not enabled():
        return
    # DataLoader fork inherits module globals, but must own a fresh queue/thread.
    if (_writer is None or _writer.pid != os.getpid()
            or _writer.directory != _directory or _writer.closed):
        _writer = _Writer(_directory)
    _writer.submit(dict(event=event, time_ns=time.time_ns(), **fields))


def flush_trace() -> None:
    if _writer is not None:
        _writer.close()


class TraceSpan:
    """Nested host spans, including fast children when their parent is slow."""
    def __init__(self, event: str, *, parent=None, **fields):
        self.active = enabled()
        self.event, self.fields, self.parent = event, fields, parent
        self.details = []

    def __enter__(self):
        if self.active:
            if self.parent is None:
                self.parent = _current.get()
            self.started = time.perf_counter()
            self.wall_ns = time.time_ns()
            self.usage = resource.getrusage(resource.RUSAGE_SELF)
            self.token = _current.set(self)
        return self

    def add(self, **fields):
        if self.active:
            self.fields.update(fields)

    def __exit__(self, typ, value, traceback):
        if not self.active:
            return
        _current.reset(self.token)
        elapsed = time.perf_counter() - self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record = dict(event=self.event, start_ns=self.wall_ns, seconds=elapsed,
                      **self.fields, details=self.details)
        if typ is not None:
            record['error'] = typ.__name__
        if self.parent is not None:
            # Ordered sample outputs are unchanged; diagnostic child order may
            # reflect thread completion. CPython list append is atomic.
            if len(self.parent.details) < 256:
                self.parent.details.append(record)
        elif elapsed >= _threshold or typ is not None:
            record['process_usage'] = dict(
                user_s=usage.ru_utime - self.usage.ru_utime,
                system_s=usage.ru_stime - self.usage.ru_stime,
                max_rss_kib=usage.ru_maxrss,
                major_faults=usage.ru_majflt - self.usage.ru_majflt,
                input_blocks=usage.ru_inblock - self.usage.ru_inblock,
                voluntary_switches=usage.ru_nvcsw - self.usage.ru_nvcsw,
                involuntary_switches=usage.ru_nivcsw - self.usage.ru_nivcsw)
            emit(**record)


# Spawned DataLoader workers import modules afresh and must enable their traces.
configure_trace()
