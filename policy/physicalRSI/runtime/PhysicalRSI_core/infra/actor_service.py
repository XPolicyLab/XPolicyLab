"""One frozen actor, explicit capabilities and durable uncertain-action gate."""

from pathlib import Path
from threading import Lock
import fcntl

from .rpc.journal import RequestJournal
from .storage import atomic_json, digest, read_json


class ActorService:
    def __init__(self, identity, handlers, root):
        self.identity = dict(identity)
        if not all(
            self.identity.get(k) for k in ("actor", "episode", "harness_revision")
        ):
            raise ValueError("Actor, episode and frozen harness identity required")
        self.handlers = dict(handlers)
        if not self.handlers or any(not callable(v) for v in self.handlers.values()):
            raise ValueError("Explicit callable actor capabilities required")
        self.root = Path(root)
        self.journal = RequestJournal(self.root / "requests")
        self._lock = Lock()
        self.state_path = self.root / "actor.json"
        if self.state_path.exists():
            if read_json(self.state_path)["identity"] != self.identity:
                raise ValueError("Actor state belongs to another frozen identity")
        else:
            atomic_json(self.state_path, dict(identity=self.identity, state="idle"))

    @property
    def executing(self):
        """Whether this service currently owns a primitive invocation."""
        return self._lock.locked()

    def dispatch(self, method, args, kwargs):
        if args:
            raise ValueError("Actor RPC accepts named arguments only")
        if method == "identity" and not kwargs:
            return dict(self.identity)
        if method == "request.status" and set(kwargs) == {"request_id"}:
            return self.journal.status(kwargs["request_id"])
        if method != "request.execute" or set(kwargs) != {
            "request_id",
            "method",
            "kwargs",
        }:
            raise ValueError("Unknown actor RPC")
        request_id, name, payload = (
            kwargs["request_id"],
            kwargs["method"],
            kwargs["kwargs"],
        )
        if (
            name not in self.handlers
            or not isinstance(payload, dict)
            or set(payload) != {"actor_binding", "arguments"}
            or payload["actor_binding"] != self.identity
            or not isinstance(payload["arguments"], dict)
        ):
            raise ValueError("Actor capability or frozen identity mismatch")
        binding = dict(
            identity=self.identity, method=name, arguments=payload["arguments"]
        )
        if self.journal.status(request_id)["state"] == "completed":

            def missing_replay():
                raise RuntimeError("Completed request record disappeared")

            # Reading a committed result is not a new physical action. It must
            # remain available while the actor publishes idle or starts other work.
            return self.journal.execute(request_id, binding, missing_replay)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Actor is executing; reconcile before sending another action"
            )
        try:
            ownership = (self.root / ".actor.lock").open("a")
            try:
                fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                ownership.close()
                raise RuntimeError("Actor is owned by another execution") from None

            def action():
                state = read_json(self.state_path)
                if state["state"] != "idle":
                    raise RuntimeError(
                        "Actor requires reconciliation: " + state["state"]
                    )
                atomic_json(
                    self.state_path,
                    dict(
                        identity=self.identity,
                        state="started",
                        request_id=request_id,
                        binding=digest(binding),
                    ),
                )
                try:
                    result = self.handlers[name](**payload["arguments"])
                except BaseException:
                    atomic_json(
                        self.state_path,
                        dict(
                            identity=self.identity,
                            state="uncertain",
                            request_id=request_id,
                        ),
                    )
                    raise
                return result

            result = self.journal.execute(request_id, binding, action)
            # Publish idle only after the completed request is durably committed.
            state = read_json(self.state_path)
            if state.get("request_id") == request_id and state["state"] == "started":
                atomic_json(self.state_path, dict(identity=self.identity, state="idle"))
            return result
        finally:
            if "ownership" in locals():
                ownership.close()
            self._lock.release()
