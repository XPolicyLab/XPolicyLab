"""Start/attach services as a group, retaining resource leases until teardown."""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .events import Event, NullEvents
from .processes import ManagedProcess
from .rpc import make_rpc_client, wait_for_ready
from .storage import read_json


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    expected: dict
    command: tuple[str, ...] = ()
    endpoint: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    resources: dict[str, int] = field(default_factory=dict)
    sessions: bool = False
    timeout: float = 30.0

    def __post_init__(self):
        from .storage import identifier

        identifier(self.name)
        if bool(self.command) == bool(self.endpoint):
            raise ValueError("Supply exactly one of command or endpoint")
        if not self.expected or self.timeout <= 0:
            raise ValueError("Pinned service identity and positive timeout required")
        if self.endpoint and self.resources:
            raise ValueError("Attached service resources are owned by its operator")


class Services:
    def __init__(self, root, *, pool=None, events=None):
        self.root = Path(root) / uuid.uuid4().hex
        self.pool = pool
        self.events = events or NullEvents()
        self.handles = {}

    def start(self, specifications):
        specs = list(specifications)
        if self.handles or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Fresh service group with unique names required")
        if self.pool:
            requested = {}
            for spec in specs:
                for key, amount in spec.resources.items():
                    requested[key] = requested.get(key, 0) + amount
            self.pool.require_capacity(requested)
        try:
            # All processes are started before readiness polling, overlapping loads.
            for spec in specs:
                handle = dict(
                    spec=spec,
                    process=None,
                    client=None,
                    lease=None,
                    announcement=self.root / (spec.name + ".endpoint.json"),
                )
                self.handles[spec.name] = handle
                self.events.emit(Event("service.starting", {"name": spec.name}))
                if spec.command:
                    if spec.resources:
                        if self.pool is None:
                            raise ValueError("Resource pool required for owned service")
                        lease = self.pool.lease(spec.resources, timeout=spec.timeout)
                        lease.__enter__()
                        handle["lease"] = lease
                    process = ManagedProcess(
                        spec.name,
                        spec.command,
                        log_path=self.root / (spec.name + ".log"),
                        env_overrides={
                            **spec.environment,
                            "PHYSICALRSI_ENDPOINT_FILE": str(handle["announcement"]),
                        },
                    )
                    handle["process"] = process
                    process.start()
            for handle in self.handles.values():
                spec, process = handle["spec"], handle["process"]
                deadline = time.monotonic() + spec.timeout
                endpoint = spec.endpoint
                while endpoint is None:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{spec.name} exited before announcing its endpoint"
                        )
                    if handle["announcement"].exists():
                        announcement = read_json(handle["announcement"])
                        if announcement["pid"] != process.pid:
                            raise ValueError("Endpoint belongs to another process")
                        endpoint = announcement["endpoint"]
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{spec.name} did not announce an endpoint")
                    time.sleep(0.02)
                client = make_rpc_client(endpoint, enable_sessions=spec.sessions)
                handle["client"] = client
                wait_for_ready(
                    client,
                    daemon=process,
                    timeout_s=max(0.001, deadline - time.monotonic()),
                )
                actual = client.call(
                    "service.describe",
                    timeout_s=max(0.001, deadline - time.monotonic()),
                )
                if actual != spec.expected:
                    raise ValueError(
                        f"{spec.name} service identity mismatch: {actual!r}"
                    )
                self.events.emit(
                    Event("service.ready", {"name": spec.name, "owned": bool(process)})
                )
            return {name: handle["client"] for name, handle in self.handles.items()}
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for name, handle in reversed(list(self.handles.items())):
            try:
                if handle["client"]:
                    try:
                        handle["client"].close()
                    except Exception:
                        if handle["process"] is None:
                            raise
                        # An owned process can still be terminated and reaped
                        # if its RPC channel or session has failed.
                if handle["process"]:
                    handle["process"].stop()
                if handle["lease"]:
                    handle["lease"].__exit__(None, None, None)
                    handle["lease"] = None
                del self.handles[name]
                self.events.emit(Event("service.closed", {"name": name}))
            except Exception as error:
                # Keep the handle and lease for explicit retry/reconciliation.
                errors.append(error)
        if errors:
            raise ExceptionGroup("Service teardown incomplete; leases retained", errors)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
