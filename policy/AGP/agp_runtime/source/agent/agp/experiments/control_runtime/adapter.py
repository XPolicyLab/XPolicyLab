"""Persistent access to the existing robot server's session file protocol."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from agp.experiments.control_runtime.errors import NotDispatched


class RobotAdapter:
    """Publish each physical action at most once, including across process failure.

    A durable reservation without a response is uncertain even if its request
    file is absent. Neither restart nor timeout can authorize replay.
    """

    def __init__(self, session, *, timeout_s=240.0, bridge_name="bridge"):
        self.session = Path(session).resolve()
        if bridge_name not in {"bridge", "bridge_right"}:
            raise ValueError("Unknown session bridge")
        self.bridge = self.session / bridge_name
        if not self.bridge.is_dir():
            raise FileNotFoundError(f"No robot server bridge directory at {self.bridge}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.timeout_s = timeout_s
        self.database = self.session / ("adapter_right.sqlite3" if bridge_name == "bridge_right" else "adapter.sqlite3")
        with self._database() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                request_id INTEGER NOT NULL UNIQUE,
                dispatched_ns INTEGER NOT NULL,
                response TEXT,
                response_received_ns INTEGER
            )""")

    @contextmanager
    def _database(self):
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _locked(self, action_id=None):
        # Every entry point opens its own descriptor so concurrent threads and
        # processes contend on the same lock until the response wait finishes.
        with (self.bridge / ".native_transport.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                if action_id is not None:
                    with self._database() as db:
                        reserved = db.execute("SELECT 1 FROM actions WHERE action_id=?", (action_id,)).fetchone()
                    if reserved:
                        raise TimeoutError("Reserved action is busy; reconcile its outcome") from exc
                raise NotDispatched("Robot session transport is locked by another caller") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _response_path(self, request_id):
        return self.bridge / f"resp_{request_id:06d}.json"

    def _read_response(self, request_id):
        path = self._response_path(request_id)
        if not path.exists():
            return None
        response = json.loads(path.read_text())
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise RuntimeError(f"Invalid robot response identity for request {request_id}")
        return response

    def _action_result_locked(self, row):
        if row["response"] is not None:
            response = json.loads(row["response"])
            received_ns = row["response_received_ns"]
        else:
            request_path = self.bridge / f"req_{row['request_id']:06d}.json"
            if not request_path.exists():
                return None
            request = json.loads(request_path.read_text())
            payload = json.loads(row["payload"])
            if (request.get("id") != row["request_id"] or request.get("cmd") != payload["command"]
                    or request.get("args") != payload["args"]):
                raise RuntimeError(f"Robot request identity collision for action {row['action_id']!r}")
            response = self._read_response(row["request_id"])
            if response is None:
                return None
            received_ns = time.monotonic_ns()
            with self._database() as db:
                db.execute("UPDATE actions SET response=?, response_received_ns=? WHERE action_id=?",
                           (json.dumps(response, allow_nan=False), received_ns, row["action_id"]))
        return dict(response, request_id=row["request_id"],
                    dispatched_monotonic_ns=row["dispatched_ns"],
                    response_received_monotonic_ns=received_ns)

    def action_result(self, action_id):
        """Read a known or late response without writing any robot request."""
        with self._locked(), self._database() as db:
            row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return None if row is None else self._action_result_locked(row)

    def _allocate(self):
        request_ids = []
        for path in self.bridge.glob("req_*.json"):
            try:
                request_id = int(path.stem[4:])
            except ValueError:
                continue
            if not self._response_path(request_id).exists():
                raise NotDispatched(f"Preexisting pending robot request {request_id}; refusing to enqueue")
            request_ids.append(request_id)
        with self._database() as db:
            rows = db.execute("SELECT * FROM actions").fetchall()
        for row in rows:
            if self._action_result_locked(row) is None:
                raise NotDispatched(f"Unresolved action {row['action_id']!r}; refusing to enqueue")
            request_ids.append(row["request_id"])
        # A surviving response also reserves its sequence number after an
        # external request-file removal. Never associate it with a new action.
        for path in self.bridge.glob("resp_*.json"):
            try:
                request_ids.append(int(path.stem[5:]))
            except ValueError:
                continue
        return max(request_ids, default=0) + 1

    def _publish(self, request_id, command, args):
        path = self.bridge / f"req_{request_id:06d}.json"
        request = {"id": request_id, "cmd": command, "args": args, "t": time.time()}
        descriptor, temporary = tempfile.mkstemp(prefix=".native_request_", dir=self.bridge)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(request, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard link publishes a complete file atomically and fails if an
            # uncooperative writer has already claimed this request number.
            os.link(temporary, path)
            directory_fd = os.open(self.bridge, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.unlink(temporary)

    def _wait(self, request_id):
        deadline = time.monotonic() + self.timeout_s
        while True:
            response = self._read_response(request_id)
            if response is not None:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Robot request {request_id} has no response; outcome is unknown")
            time.sleep(min(.02, remaining))

    def _call(self, command, args):
        with self._locked():
            request_id = self._allocate()
            dispatched_ns = time.monotonic_ns()
            self._publish(request_id, command, args)
            response = self._wait(request_id)
            return dict(response, request_id=request_id,
                        dispatched_monotonic_ns=dispatched_ns,
                        response_received_monotonic_ns=time.monotonic_ns())

    def act(self, action_id, command, args):
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a nonempty string")
        if command not in {"move_ee", "move_delta", "move_joints", "home", "gripper", "fold_home"}:
            raise ValueError(f"Unsupported physical command {command!r}")
        args = dict(args)
        if args.get("speed_profile", "precision") not in {"precision", "transit"}:
            raise ValueError("Use precision or transit speed")
        if command in {"move_ee", "move_delta"}:
            args.setdefault("speed_profile", "precision")
        payload = json.dumps({"command": command, "args": args}, sort_keys=True, allow_nan=False)
        with self._locked(action_id):
            with self._database() as db:
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if row is not None:
                if row["payload"] != payload:
                    raise ValueError("action_id was already used with a different command or arguments")
                result = self._action_result_locked(row)
                if result is None:
                    raise TimeoutError(f"Action {action_id!r} outcome is unknown; it will not be resent")
                return result
            request_id = self._allocate()
            dispatched_ns = time.monotonic_ns()
            # This commit must precede any request file becoming visible.
            with self._database() as db:
                db.execute("INSERT INTO actions (action_id,payload,request_id,dispatched_ns) VALUES (?,?,?,?)",
                           (action_id, payload, request_id, dispatched_ns))
            self._publish(request_id, command, args)
            self._wait(request_id)
            with self._database() as db:
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return self._action_result_locked(row)

    def _saved_path(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.session / path
        path = path.resolve()
        if not path.is_relative_to(self.session):
            raise ValueError("Robot server returned a file outside its session")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
