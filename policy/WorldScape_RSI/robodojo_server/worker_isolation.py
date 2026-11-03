"""Project-owned failure boundary for independent, exclusive GPU pipes."""

from concurrent.futures import wait
from threading import Lock


class IsolatedFanoutMixin:
    def __init__(self, *args, **kwargs):
        self._faulted_workers = set()
        self._worker_close_locks = {}
        super().__init__(*args, **kwargs)

    def worker_available(self, index):
        return (not self._closed and index not in self._faulted_workers
                and self._workers[index].process.is_alive())

    def _quarantine(self, index):
        self._faulted_workers.add(index)
        lock = self._worker_close_locks.setdefault(index, Lock())
        with lock:
            self._workers[index].close()

    def _fanout(self, calls):
        if self._closed:
            raise RuntimeError('Model pool closed')
        for index in calls:
            if not self.worker_available(index):
                raise RuntimeError('GPU worker unavailable; session lost')
        jobs = {}
        for index, (method, args) in calls.items():
            worker = self._workers[index]
            jobs[index] = (self._pool.submit(worker.ready) if method == '__ready__'
                           else self._pool.submit(worker.call, method, *args))
        _, pending = wait(jobs.values(), timeout=self.timeout)
        failures = [index for index, job in jobs.items() if job in pending or job.exception() is not None]
        if failures:
            # Closing these pipes interrupts their reads; other GPU calls keep
            # their weights, state and executor threads.
            for index in failures:
                self._quarantine(index)
            raise RuntimeError('GPU worker failed or timed out; session lost')
        return {index: job.result() for index, job in jobs.items()}
