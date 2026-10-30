"""Bounded in-process resource allocation; no hardware capacity is inferred."""

from contextlib import contextmanager
from threading import Condition
from time import monotonic


class ResourcePool:
    """Share one pool across local jobs. Capacities are explicitly configured.

    Leases last until the worker actually returns, including cancellation. This
    allocator does not coordinate other processes or promise GPU isolation.
    """

    def __init__(self, capacities: dict[str, int]):
        if not capacities or any(
            type(v) is not int or v < 1 for v in capacities.values()
        ):
            raise ValueError("Resource capacities must be positive integers")
        self._capacity = dict(capacities)
        self._used = dict.fromkeys(capacities, 0)
        self._condition = Condition()

    def status(self) -> dict:
        with self._condition:
            return {
                k: {"capacity": v, "used": self._used[k]}
                for k, v in self._capacity.items()
            }

    def require_capacity(self, request):
        """Check total capacity without reserving resources or inferring hardware fit."""
        if any(
            type(amount) is not int
            or amount < 1
            or key not in self._capacity
            or amount > self._capacity[key]
            for key, amount in request.items()
        ):
            raise ValueError("Resource request exceeds configured capacity")

    @contextmanager
    def lease(self, request: dict[str, int], *, context=None, timeout=30.0):
        if timeout < 0:
            raise ValueError("Timeout must be nonnegative")
        request = dict(request)
        if any(
            type(v) is not int
            or v < 1
            or k not in self._capacity
            or v > self._capacity[k]
            for k, v in request.items()
        ):
            raise ValueError("Invalid or impossible resource request")
        until = monotonic() + timeout
        with self._condition:
            while True:
                if context is not None:
                    context.check()
                if all(
                    self._used[k] + v <= self._capacity[k] for k, v in request.items()
                ):
                    for k, v in request.items():
                        self._used[k] += v
                    break
                left = until - monotonic()
                if left <= 0:
                    raise TimeoutError("Resource lease unavailable")
                self._condition.wait(min(left, 0.05))
        try:
            yield
        finally:
            with self._condition:
                for k, v in request.items():
                    self._used[k] -= v
                self._condition.notify_all()
