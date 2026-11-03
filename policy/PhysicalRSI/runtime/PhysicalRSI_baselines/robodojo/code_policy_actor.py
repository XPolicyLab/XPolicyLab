"""One owned Code policy episode: isolated policy, primitive service and action channel.

Physical success remains the environment evaluator's decision. A completed
policy or an actor shutdown never fabricates a successful episode receipt.
"""

import math
import time
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Lock, Thread

from PhysicalRSI_core.infra.actor_service import ActorService
from PhysicalRSI_core.infra.rpc.http_rpc import HttpRpcServer
from PhysicalRSI_core.infra.storage import atomic_json, digest

from .action_channel import ActionChannel
from .code_execution import execute_policy
from .primitive_client import PrimitiveClient


class CodePolicyActor:
    def __init__(
        self,
        *,
        identity,
        source,
        memory,
        primitive_revision,
        primitive_factory,
        dimensions,
        runtime,
        output,
        timeout_s=60,
        owned_services=(),
        cameras=None,
        completion_hold_steps=0,
    ):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive actor timeout required")
        if (
            type(completion_hold_steps) is not int
            or not 0 <= completion_hold_steps <= 10000
        ):
            raise ValueError("Bounded completion hold steps required")
        self.completion_hold_steps = completion_hold_steps
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.owned_services = list(owned_services)
        with ExitStack() as rollback:
            for service in self.owned_services:
                rollback.callback(service.close)
            self.identity = deepcopy(identity)
            self.source, self.memory = source, deepcopy(memory)
            atomic_json(
                self.root / "binding.json",
                dict(
                    identity=self.identity,
                    source_sha256=digest(source),
                    memory_sha256=digest(self.memory),
                    primitive_revision=primitive_revision,
                    timeout_s=timeout_s,
                    completion_hold_steps=completion_hold_steps,
                ),
            )
            self.channel = ActionChannel(dimensions, cameras=cameras)
            self.primitives = primitive_factory(self.channel)
            self.service = ActorService(
                self.identity, self.primitives.handlers, self.root / "primitive-service"
            )
            self.server = HttpRpcServer(("127.0.0.1", 0), self.service.dispatch)
            rollback.callback(self.server.server_close)
            self.server_thread = Thread(
                target=lambda: self.server.serve_forever(poll_interval=0.05),
                daemon=True,
            )
            self.cancelled = Event()
            self._lock = Lock()
            self._status = {"state": "created"}
            self.worker = None
            self.closed = False
            self.runtime, self.timeout_s = runtime, timeout_s
            self.revision = primitive_revision
            rollback.pop_all()

    def start(self):
        if self.worker is not None or self.closed:
            raise RuntimeError("Actor cannot be restarted; create a new episode")
        self.server_thread.start()

        def execute():
            self._record(state="running")
            try:
                client = PrimitiveClient(
                    "http://127.0.0.1:" + str(self.server.server_port),
                    episode=self.identity["episode"],
                    harness_revision=self.identity["harness_revision"],
                    output=self.root / "execution",
                    expected_identity=self.identity,
                    primitives=self.primitives.bindings(
                        revision=self.revision, actor=self.identity["actor"]
                    ),
                )
                result = execute_policy(
                    self.source,
                    self.memory,
                    handlers=client.handlers,
                    runtime=self.runtime,
                    output=self.root / "policy",
                    timeout_s=self.timeout_s,
                    cancelled=self.cancelled,
                )
                self.channel.finish(hold_steps=self.completion_hold_steps)
                self._record(
                    state="completed", result=result, physical_qualification=False
                )
            except BaseException as error:
                try:
                    self._record(
                        state="cancelled" if self.cancelled.is_set() else "failed",
                        error=type(error).__name__ + ": " + str(error),
                    )
                finally:
                    self.channel.close("policy execution failed or cancelled")

        self.worker = Thread(target=execute, daemon=True)
        self.worker.start()
        return self

    def _record(self, **record):
        with self._lock:
            self._status = record
            atomic_json(self.root / "actor.json", record)

    def status(self):
        with self._lock:
            return dict(
                deepcopy(self._status),
                channel=self.channel.status(),
                closed=self.closed,
            )

    def update_obs(self, observation):
        self.channel.update_obs(observation)

    def get_action(self, *, deadline):
        return self.channel.get_action(deadline=deadline)

    def close(self, timeout_s=5):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive actor teardown timeout required")
        if self.closed:
            return
        self.cancelled.set()
        self.channel.close("actor stopping")
        deadline = time.monotonic() + timeout_s
        try:
            for service in reversed(self.owned_services):
                service.close()
            if self.worker is not None:
                self.worker.join(max(0, deadline - time.monotonic()))
                if self.worker.is_alive():
                    raise TimeoutError(
                        "Policy controller still active; ownership retained"
                    )
            if self.service.executing:
                raise TimeoutError("Primitive still executing; ownership retained")
            if self.server_thread.is_alive():
                self.server.shutdown()
                self.server_thread.join(max(0, deadline - time.monotonic()))
                if self.server_thread.is_alive():
                    raise TimeoutError(
                        "Primitive server still active; ownership retained"
                    )
            self.server.server_close()
            self.closed = True
            atomic_json(self.root / "cleanup.json", dict(state="stopped"))
        except BaseException as error:
            atomic_json(
                self.root / "cleanup.json",
                dict(state="cleanup_failed", error=str(error)),
            )
            raise
