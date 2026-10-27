from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch


_INFERENCE_PROFILE_ACTIVE: ContextVar[bool] = ContextVar(
    "inference_profile_active", default=False
)

_PROFILE_TRUTHY = frozenset({"1", "true", "yes", "on"})


def stage_profile_enabled() -> bool:
    return os.environ.get("PROFILE", "").strip().lower() in _PROFILE_TRUTHY


@contextmanager
def inference_profile_scope() -> Iterator[None]:
    token = _INFERENCE_PROFILE_ACTIVE.set(True)
    try:
        yield
    finally:
        _INFERENCE_PROFILE_ACTIVE.reset(token)


def inference_profile_active() -> bool:
    return _INFERENCE_PROFILE_ACTIVE.get()


class StageProfiler:
    def __init__(
        self,
        tag: str,
        device: torch.device | torch.Tensor | str | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.tag = tag
        self.enabled = (
            stage_profile_enabled() if enabled is None else bool(enabled)
        )
        self.device = self._resolve_device(device)
        self._stages_ms: dict[str, float] = {}
        self._finished = False
        self._total_ms = 0.0
        started_at = self._now()
        self._started_at = started_at
        self._last_at = started_at

    @staticmethod
    def _resolve_device(
        device: torch.device | torch.Tensor | str | None,
    ) -> torch.device | None:
        if isinstance(device, torch.Tensor):
            return device.device
        if device is None:
            return None
        return torch.device(device)

    def _synchronize(self) -> None:
        if (
            self.enabled
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _now(self) -> float:
        if not self.enabled:
            return 0.0
        self._synchronize()
        return time.perf_counter()

    def mark(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        if self._finished:
            raise RuntimeError("Cannot mark a finished StageProfiler.")
        now = self._now()
        elapsed_ms = (now - self._last_at) * 1000.0
        self._stages_ms[name] = self._stages_ms.get(name, 0.0) + elapsed_ms
        self._last_at = now
        return elapsed_ms

    def get_ms(self, name: str, default: float = 0.0) -> float:
        return self._stages_ms.get(name, default)

    def sum_ms(self, names: Iterable[str]) -> float:
        return sum(self.get_ms(name) for name in names)

    def finish(self) -> float:
        if not self.enabled:
            return 0.0
        if not self._finished:
            self._total_ms = (self._now() - self._started_at) * 1000.0
            self._finished = True
        return self._total_ms

    def report(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        order: Iterable[str] | None = None,
        extra_ms: Mapping[str, float] | None = None,
    ) -> float:
        total_ms = self.finish()
        if not self.enabled:
            return total_ms

        measurements = dict(self._stages_ms)
        if extra_ms is not None:
            measurements.update(extra_ms)

        parts = []
        if metadata is not None:
            parts.extend(f"{name}={value}" for name, value in metadata.items())

        names = list(order) if order is not None else list(measurements)
        if extra_ms is not None:
            names.extend(name for name in extra_ms if name not in names)
        parts.extend(
            f"{name}={measurements.get(name, 0.0):.2f} ms"
            for name in names
        )
        parts.append(f"total={total_ms:.2f} ms")
        print(f"[{self.tag}] " + " | ".join(parts), flush=True)
        return total_ms
