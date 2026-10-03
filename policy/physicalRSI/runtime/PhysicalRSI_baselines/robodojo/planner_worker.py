"""Trusted planner service worker; starts only after pinned dependency checks."""

import importlib
import math
import os
import sys
import time
from threading import Lock

from PhysicalRSI_core.infra.processes import watch_parent_death
from PhysicalRSI_core.infra.rpc.http_rpc import HttpRpcServer
from PhysicalRSI_core.infra.storage import atomic_json, read_json

from .planner_service import check_factory, verify
from .curobo_backend import NoPathFound


def main(path):
    watch_parent_death(lambda: os._exit(70))
    spec = read_json(path)
    identity = verify(spec)
    module, name = spec["entrypoint"].split(":")
    factory = getattr(importlib.import_module(module), name)
    check_factory(factory, spec)
    backend = factory(**spec["configuration"])
    try:
        verify(spec)
        lock = Lock()

        def dispatch(method, args, kwargs):
            if args:
                raise ValueError("Planner uses named arguments")
            if method == "healthz" and not kwargs:
                return {"ready": True}
            if method == "service.describe" and not kwargs:
                return identity
            if method not in {"plan", "tool_pose", "collision_spheres"} or set(
                kwargs
            ) != {"arguments", "timeout_s"}:
                raise ValueError("Unknown planner request")
            timeout = kwargs["timeout_s"]
            if (
                not math.isfinite(timeout)
                or timeout <= 0
                or "deadline" in kwargs["arguments"]
            ):
                raise ValueError("Invalid planner deadline")
            if not lock.acquire(blocking=False):
                raise RuntimeError("Planner already has an active request")
            try:
                deadline = time.monotonic() + timeout
                if method != "plan":
                    if set(kwargs["arguments"]) != {"joints"}:
                        raise ValueError("Kinematics requires only named joints")
                    result = getattr(backend, method)(kwargs["arguments"]["joints"])
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Kinematics returned after deadline")
                    return dict(state="completed", result=result)
                positions = backend.plan(**kwargs["arguments"], deadline=deadline)
                return dict(state="completed", positions=positions)
            except NoPathFound as error:
                return dict(state="no_path", reason=str(error))
            finally:
                lock.release()

        server = HttpRpcServer(("127.0.0.1", 0), dispatch)
        atomic_json(
            os.environ["PHYSICALRSI_ENDPOINT_FILE"],
            dict(
                pid=os.getpid(), endpoint="http://127.0.0.1:" + str(server.server_port)
            ),
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        backend.close()


if __name__ == "__main__":
    main(sys.argv[1])
