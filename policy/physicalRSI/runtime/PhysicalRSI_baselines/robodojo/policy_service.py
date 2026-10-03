"""Owned outer policy WebSocket service for committed or frozen model execution."""

import asyncio
import math
import os
import signal
import sys
import time
from pathlib import Path

from PhysicalRSI_core.infra.processes import ManagedProcess, watch_parent_death
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json

from .policy_model import Model


def verify(spec):
    if spec.get("schema") != "physicalrsi.robodojo.policy-service/v1" or not spec.get(
        "files"
    ):
        raise ValueError("Pinned outer policy service specification required")
    if spec.get("model_adapter", "native") not in {"native", "xpolicylab"}:
        raise ValueError("Unknown policy model adapter")
    for path, sha in spec["files"].items():
        if not Path(path).is_absolute() or file_digest(Path(path)) != sha:
            raise ValueError("Policy service dependency changed")
    for module in (__file__, sys.modules[Model.__module__].__file__):
        if str(Path(module).resolve()) not in spec["files"]:
            raise ValueError(
                "Policy service and model entrypoint sources must be pinned"
            )
    return digest(spec)


class PolicyService:
    def __init__(
        self,
        spec,
        *,
        python,
        environment,
        output,
        startup_timeout_s=600,
        cleanup_timeout_s=30,
        cancelled=None,
    ):
        if any(
            not math.isfinite(v) or v <= 0
            for v in (startup_timeout_s, cleanup_timeout_s)
        ):
            raise ValueError("Positive policy service lifecycle timeouts required")
        self.identity = verify(spec)
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(self.root / "spec.json", spec)
        self.startup_timeout, self.cleanup_timeout = (
            startup_timeout_s,
            cleanup_timeout_s,
        )
        self.ready = None
        self.cancelled = cancelled or (lambda: False)
        self.process = ManagedProcess(
            "policy-server",
            [str(python), "-m", __name__, str(self.root / "spec.json"), str(self.root)],
            env_overrides=dict(environment, PYTHONDONTWRITEBYTECODE="1"),
            cwd=self.root,
            log_path=self.root / "server.log",
        )

    def __enter__(self):
        self.process.start()
        atomic_json(
            self.root / "process.json", dict(state="starting", pid=self.process.pid)
        )
        try:
            deadline = time.monotonic() + self.startup_timeout
            while not (self.root / "ready.json").exists():
                if self.cancelled():
                    raise InterruptedError("Policy server startup cancelled")
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Policy server exited before readiness; inspect server.log"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Policy server startup timed out")
                time.sleep(0.05)
            self.ready = read_json(self.root / "ready.json")
            if (
                self.ready.get("service_identity") != self.identity
                or self.ready.get("pid") != self.process.pid
                or self.process.poll() is not None
            ):
                raise ValueError("Policy server readiness identity mismatch")
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.pid is None:
            return
        atomic_json(self.root / "stop.json", dict(requested=True))
        deadline = time.monotonic() + self.cleanup_timeout
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.process.stop()
        receipt = self.root / "worker-cleanup.json"
        if not receipt.exists() or read_json(receipt).get("state") != "stopped":
            atomic_json(
                self.root / "cleanup.json",
                dict(state="cleanup_unconfirmed", pid=self.process.pid),
            )
            raise RuntimeError(
                "Policy worker stopped but nested cleanup is unconfirmed; inspect owned service evidence"
            )
        atomic_json(
            self.root / "cleanup.json",
            dict(state="stopped", pid=self.process.pid, returncode=self.process.poll()),
        )
        if self.process.poll() != 0:
            raise RuntimeError(
                "Policy worker exited unsuccessfully; cleanup evidence retained"
            )

    def __exit__(self, *args):
        self.close()


async def worker(spec, root):
    identity = verify(spec)
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    import client_server.ws.model_server as transport

    if str(Path(transport.__file__).resolve()) not in spec["files"]:
        raise ValueError("XPolicyLab policy server source must be pinned")
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    watch_parent_death(lambda: loop.call_soon_threadsafe(stopping.set))
    model = None
    server = None
    try:
        model_type = Model
        if spec.get("model_adapter") == "xpolicylab":
            from .xpolicy_adapter import Model as Adapter
            from XPolicyLab.model_template import ModelTemplate
            from XPolicyLab.utils import process_data

            for module in (Adapter, ModelTemplate, process_data):
                source = getattr(module, "__file__", None)
                if source is None:
                    source = sys.modules[module.__module__].__file__
                if str(Path(source).resolve()) not in spec["files"]:
                    raise ValueError("XPolicyLab adapter dependencies must be pinned")
            # The shared dimension helper reads these files outside its checkout.
            # Pin the inputs as well as the helper implementation. This is source
            # inventory only; robot dimensions still come from the shared helper.
            metadata = Path(process_data.__file__).parent / "../../env_cfg"
            for source in (
                metadata / "arx_x5.yml",
                metadata / "robot/_robot_info.json",
            ):
                if str(source.resolve()) not in spec["files"]:
                    raise ValueError("XPolicyLab robot metadata must be pinned")
            model_type = Adapter
        model = model_type(spec["config"])
        verify(spec)

        class Gate:
            def physicalrsi_service_identity(self):
                if stopping.is_set():
                    raise RuntimeError("Policy service stopping")
                return dict(service_identity=identity)

            def __getattr__(self, name):
                method = getattr(model, name)
                if not callable(method):
                    raise AttributeError(name)

                def call(*args, **kwargs):
                    if stopping.is_set():
                        raise RuntimeError("Policy service stopping")
                    return method(*args, **kwargs)

                return call

        server = PolicyServer(Gate(), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        atomic_json(
            root / "ready.json",
            dict(
                service_identity=identity,
                model_identity=model.physicalrsi_identity(),
                pid=os.getpid(),
                endpoint=server.url,
            ),
        )
        while not stopping.is_set():
            if (root / "stop.json").exists():
                stopping.set()
                break
            await asyncio.sleep(0.05)
    finally:
        stopping.set()
        try:
            if model is not None:
                if server is not None:
                    # Same lock as XPolicyLab model calls: do not tear down an
                    # expert while an observation/action call is still active.
                    async with server._model_lock:
                        await asyncio.to_thread(model.close)
                else:
                    model.close()
            if server is not None:
                await server.stop()
            atomic_json(
                root / "worker-cleanup.json",
                dict(
                    state="stopped"
                    if model is not None
                    else "construction_cleanup_unconfirmed"
                ),
            )
        except BaseException as error:
            atomic_json(
                root / "worker-cleanup.json", dict(state="failed", error=str(error))
            )
            raise


if __name__ == "__main__":
    asyncio.run(worker(read_json(sys.argv[1]), Path(sys.argv[2])))
