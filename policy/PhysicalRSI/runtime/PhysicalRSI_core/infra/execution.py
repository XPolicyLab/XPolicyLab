"""PhysicalRSI execution lifecycle derived from upstream Toolkit dispatch.

Record before invocation, preserve action and observation errors independently,
and retain exclusive effects until the operation actually returns.
"""

import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Lock

from PhysicalRSI_core.contracts import Cancelled

from .artifacts import Artifacts
from .events import Event, NullEvents
from .storage import atomic_json


class Execution:
    def __init__(self, root, *, observe=None, events=None):
        self.root = Path(root)
        self.observe = observe
        self.events = events or NullEvents()
        self._condition = Condition()
        self._effects = {}
        self._active = {}
        self.artifacts = Artifacts(self.root / "artifacts")

    def cancel_and_wait(self, episode, timeout=5):
        until = time.monotonic() + timeout
        with self._condition:
            for context in self._active.values():
                if context.episode == episode:
                    context.cancelled.set()
            while any(context.episode == episode for context in self._active.values()):
                left = until - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Operation still active; effects remain owned")
                self._condition.wait(left)

    def __call__(self, operation, value, context):
        context.check()
        call_id = uuid.uuid4().hex
        path = self.root / context.episode / (call_id + ".json")
        record = dict(
            id=call_id,
            episode=context.episode,
            operation=operation.name,
            revision=operation.revision,
            harness_revision=context.harness_revision,
            input_contract=asdict(operation.input),
            output_contract=asdict(operation.output),
            effects=sorted(operation.effects),
            state="started",
        )
        started = time.monotonic()
        with self._condition:
            self._active[call_id] = context
            locks = [
                self._effects.setdefault(effect, Lock())
                for effect in sorted(operation.effects)
            ]
        error = None
        result = None
        try:
            record["input"] = self.artifacts.encode(value)
            atomic_json(path, record)
            self.events.emit(Event("operation.started", dict(record)))
            with ExitStack() as stack:
                # Composite nodes carry union effects for planning. Their leaves
                # acquire the actual locks, so nested parallel work cannot deadlock.
                if not operation.children:
                    for lock in locks:
                        while not lock.acquire(timeout=0.05):
                            context.check()
                        stack.callback(lock.release)
                context.check()
                try:
                    result = operation.call(value, context)
                    record["output"] = self.artifacts.encode(result)
                    context.check()
                    record["state"] = "completed"
                except BaseException as caught:
                    error = caught
                    record["state"] = (
                        "cancelled" if isinstance(caught, Cancelled) else "failed"
                    )
                    record["error"] = type(caught).__name__ + ": " + str(caught)
                    if request_id := getattr(caught, "request_id", None):
                        record["request_id"] = request_id
                        record["state"] = "uncertain"
                # Observe even after failure/cancellation, while effects are held.
                if operation.effects and not operation.children and self.observe:
                    try:
                        record["observation"] = self.artifacts.encode(
                            self.observe(operation, context)
                        )
                    except Exception as caught:
                        record["observation_error"] = str(caught)
                        if error is None:
                            error = caught
                            record["state"] = "uncertain"
                record["elapsed_seconds"] = time.monotonic() - started
                atomic_json(path, record)
                self.events.emit(Event("operation.finished", dict(record)))
            if error is not None:
                raise error
            return result
        finally:
            with self._condition:
                self._active.pop(call_id, None)
                self._condition.notify_all()
