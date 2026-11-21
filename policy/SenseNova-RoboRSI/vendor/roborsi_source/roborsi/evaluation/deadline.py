# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monotonic case deadline and cooperative cancellation shared by runtimes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Event


class CaseTimeout(TimeoutError):
    """The whole case exceeded its deadline."""


class CaseCancelled(RuntimeError):
    """The caller cancelled the whole case."""


class CaseDeadline:
    """Return remaining wall time before each blocking operation."""

    def __init__(
        self,
        duration_s: float,
        *,
        cancel_event: Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("case_timeout_s must be finite and positive")
        self._expires_at = clock() + duration_s
        self._clock = clock
        self._cancel_event = cancel_event

    def remaining(self, limit_s: float | None = None) -> float:
        """Raise on cancellation or expiry; otherwise cap an operation timeout."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise CaseCancelled("case cancelled")
        remaining = self._expires_at - self._clock()
        if remaining <= 0:
            raise CaseTimeout("case deadline exceeded")
        return min(remaining, limit_s) if limit_s is not None else remaining

    def check(self) -> None:
        """Reject a result returned after cancellation or expiry."""
        self.remaining()
