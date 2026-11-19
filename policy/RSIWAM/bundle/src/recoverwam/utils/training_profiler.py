"""Opt-in wall-clock profiling for distributed training runs.

The profiler is deliberately configuration driven and disabled by default.  It
uses synchronization only when explicitly requested, because synchronizing at
stage boundaries perturbs normal asynchronous CUDA execution.  Diagnostic
runs can enable synchronization to obtain attributable forward/backward/
optimizer timings; formal runs retain the original execution path.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, MutableMapping

import numpy as np
import torch


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


class TrainingProfiler:
    """Collect bounded startup and optimizer-step timing samples per rank."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        skip_steps: int = 0,
        collect_steps: int = 200,
        log_every: int = 50,
        synchronize_cuda: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.skip_steps = max(int(skip_steps), 0)
        self.collect_steps = max(int(collect_steps), 0)
        self.log_every = max(int(log_every), 0)
        self.synchronize_cuda = bool(synchronize_cuda)
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = (
            int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else int(world_size)
        )
        self.started_at_unix = time.time()
        self.started_at_perf = time.perf_counter()
        self.startup_ms: dict[str, float] = {}
        self.step_samples: list[dict[str, float | int]] = []
        self.events: dict[str, list[dict[str, float | int]]] = defaultdict(list)

    @classmethod
    def from_config(cls, cfg) -> "TrainingProfiler":
        settings = cfg.get("profiling")
        enabled = bool(settings is not None and settings.get("enabled", False))
        profile_dir = None if settings is None else settings.get("output_dir")
        if profile_dir in (None, "", "null"):
            profile_dir = Path(str(cfg.output_dir)) / "profiling"
        return cls(
            enabled=enabled,
            output_dir=profile_dir,
            skip_steps=(0 if settings is None else settings.get("skip_steps", 0)),
            collect_steps=(200 if settings is None else settings.get("collect_steps", 200)),
            log_every=(50 if settings is None else settings.get("log_every", 50)),
            synchronize_cuda=(True if settings is None else settings.get("synchronize_cuda", True)),
        )

    @classmethod
    def disabled(cls, output_dir: str | Path = ".") -> "TrainingProfiler":
        return cls(enabled=False, output_dir=output_dir, collect_steps=0)

    def _cuda_synchronize(self) -> None:
        if not self.enabled or not self.synchronize_cuda or not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
        device = self.local_rank if 0 <= self.local_rank < device_count else None
        torch.cuda.synchronize(device=device)

    @contextmanager
    def measure(
        self,
        target: MutableMapping[str, float],
        name: str,
        *,
        cuda: bool = False,
        active: bool = True,
    ) -> Iterator[None]:
        """Accumulate one phase duration into ``target`` in milliseconds."""
        if not self.enabled or not active:
            yield
            return
        if cuda:
            self._cuda_synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                self._cuda_synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            target[name] = float(target.get(name, 0.0)) + elapsed_ms

    @contextmanager
    def measure_startup(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        with self.measure(self.startup_ms, name, cuda=cuda):
            yield

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at_perf) * 1000.0

    def wants_step_sample(self, completed_steps: int | None = None) -> bool:
        if not self.enabled or len(self.step_samples) >= self.collect_steps:
            return False
        if completed_steps is None:
            return True
        return int(completed_steps) >= self.skip_steps

    def record_step(
        self,
        *,
        step: int,
        epoch: int,
        timings_ms: MutableMapping[str, float],
    ) -> None:
        if not self.wants_step_sample():
            return
        sample: dict[str, float | int] = {
            "step": int(step),
            "epoch": int(epoch),
            "recorded_at_unix": float(time.time()),
        }
        for key, value in timings_ms.items():
            sample[str(key)] = float(value)
        stage_sum = sum(
            float(value)
            for key, value in sample.items()
            if key.endswith("_ms") and key != "total_step_ms"
        )
        if "total_step_ms" in sample:
            sample["unattributed_ms"] = max(
                float(sample["total_step_ms"]) - stage_sum,
                0.0,
            )
        self.step_samples.append(sample)

    def record_event(self, name: str, *, value_ms: float, step: int = -1) -> None:
        if not self.enabled:
            return
        self.events[str(name)].append({"step": int(step), "value_ms": float(value_ms)})

    @staticmethod
    def _summarize_steps(samples: list[dict]) -> dict[str, dict[str, float | int]]:
        values_by_metric: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            for key, value in sample.items():
                if key.endswith("_ms"):
                    values_by_metric[key].append(float(value))
        return {key: _summary(values) for key, values in sorted(values_by_metric.items())}

    @staticmethod
    def _summarize_distributed_steps(ranks: list[dict]) -> dict[str, dict[str, float | int]]:
        """Summarize the slowest rank for each metric and optimizer step.

        Synchronous distributed training advances at the speed of the slowest
        rank. Pooling every rank sample would hide a rare rank straggler, so
        the cluster-facing p99 is computed from per-step rank maxima.
        """
        values_by_metric_and_step: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rank in ranks:
            for sample in rank.get("step_samples", []):
                step = int(sample["step"])
                for key, value in sample.items():
                    if key.endswith("_ms"):
                        values_by_metric_and_step[key][step].append(float(value))
        return {
            metric: _summary([max(values) for _, values in sorted(values_by_step.items())])
            for metric, values_by_step in sorted(values_by_metric_and_step.items())
        }

    def rank_payload(self, *, status: str, final_step: int) -> dict:
        return {
            "schema": "recoverwam_training_profile_v1",
            "status": str(status),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "started_at_unix": self.started_at_unix,
            "elapsed_ms": self.elapsed_ms(),
            "final_step": int(final_step),
            "settings": {
                "skip_steps": self.skip_steps,
                "collect_steps": self.collect_steps,
                "log_every": self.log_every,
                "synchronize_cuda": self.synchronize_cuda,
            },
            "startup_ms": dict(sorted(self.startup_ms.items())),
            "step_samples": self.step_samples,
            "step_summary": self._summarize_steps(self.step_samples),
            "events": dict(sorted(self.events.items())),
        }

    def write_rank_report(self, *, status: str, final_step: int) -> Path | None:
        if not self.enabled:
            return None
        path = self.output_dir / f"rank_{self.rank:03d}.json"
        _atomic_write_json(path, self.rank_payload(status=status, final_step=final_step))
        return path

    def write_global_report(self) -> Path | None:
        if not self.enabled or self.rank != 0:
            return None
        rank_paths = sorted(self.output_dir.glob("rank_*.json"))
        ranks = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
        all_steps = [sample for rank in ranks for sample in rank.get("step_samples", [])]

        startup_names = sorted({name for rank in ranks for name in rank.get("startup_ms", {})})
        startup_summary = {
            name: _summary(
                [
                    float(rank["startup_ms"][name])
                    for rank in ranks
                    if name in rank.get("startup_ms", {})
                ]
            )
            for name in startup_names
        }
        event_summary = {}
        event_names = sorted({name for rank in ranks for name in rank.get("events", {})})
        for name in event_names:
            values = [
                float(event["value_ms"])
                for rank in ranks
                for event in rank.get("events", {}).get(name, [])
            ]
            event_summary[name] = _summary(values)

        rank_sample_summary = self._summarize_steps(all_steps)
        step_summary = self._summarize_distributed_steps(ranks)
        total_mean = float(step_summary.get("total_step_ms", {}).get("mean_ms", 0.0))
        if total_mean > 0:
            for name, metric in step_summary.items():
                if name not in {"total_step_ms", "unattributed_ms"} and "mean_ms" in metric:
                    metric["mean_fraction_of_step"] = float(metric["mean_ms"]) / total_mean

        payload = {
            "schema": "recoverwam_training_profile_summary_v1",
            "world_size_expected": self.world_size,
            "rank_reports_found": len(ranks),
            "complete_rank_set": len(ranks) == self.world_size,
            "rank_files": [path.name for path in rank_paths],
            "startup_summary": startup_summary,
            "step_summary": step_summary,
            "rank_sample_summary": rank_sample_summary,
            "event_summary": event_summary,
            "per_rank_step_summary": {
                str(rank["rank"]): rank.get("step_summary", {}) for rank in ranks
            },
        }
        path = self.output_dir / "summary.json"
        _atomic_write_json(path, payload)
        return path
