"""Bounded CPU concurrency, retaining leases until all workers terminate."""

from concurrent.futures import ThreadPoolExecutor
from threading import local

from PhysicalRSI_core.contracts import Cancelled

from .resources import ResourcePool

_worker = local()


class Executor:
    def __init__(self, workers=1, pool=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers
        self.pool = pool if pool is not None else ResourcePool({"cpu": workers})

    def map(self, function, values, *, context):
        """Preserve input order; failures cancel sibling work cooperatively."""
        # Nested compositions sharing a saturated pool run inside their parent's
        # lease. Waiting for more capacity here would deadlock a bounded pool.
        if getattr(_worker, "pool", None) is self.pool:
            results = []
            for value in values:
                context.check()
                results.append(function(value))
            return results

        def invoke(value):
            try:
                with self.pool.lease({"cpu": 1}, context=context):
                    context.check()
                    previous = getattr(_worker, "pool", None)
                    _worker.pool = self.pool
                    try:
                        return function(value)
                    finally:
                        _worker.pool = previous
            except BaseException:
                context.cancelled.set()
                raise

        with ThreadPoolExecutor(max_workers=self.workers) as workers:
            futures = [workers.submit(invoke, value) for value in values]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as error:
                    errors.append(error)
            if errors:
                # Surface the originating failure rather than a sibling's
                # cooperative cancellation, regardless of input order.
                raise next(
                    (error for error in errors if not isinstance(error, Cancelled)),
                    errors[0],
                )
            return results
