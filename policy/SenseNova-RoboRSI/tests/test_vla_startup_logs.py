"""Retain owned VLA startup diagnostics while reaping real CPU child processes."""

from __future__ import annotations

import gc
from importlib import import_module
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import time
from types import SimpleNamespace
import weakref

import pytest


pytestmark = pytest.mark.skipif(os.name != "posix", reason="VLA process groups require POSIX")
model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")


def _boundary(backend):
    model = model_module.Model.__new__(model_module.Model)
    model.model_cfg = {"checkpoint_path": "unused-checkpoint", "vla_backend": backend}
    model._vla = model._vla_temporary = None
    model._temporary = model._connector = model._icl = model._obs = model._pending = None
    model._deferred_controller = model._cleanup_issue = None
    model._awaiting_feedback = False
    return model


@pytest.fixture(params=("pi05", "simate_beta"))
def startup(request, tmp_path, monkeypatch):
    """Replace GPU assets only; keep Model, both servers and subprocesses real."""
    backend = request.param
    runtime_name = "simate" if backend == "simate_beta" else "pi05"
    runtime = import_module(f"XPolicyLab.policy.SenseNova-RoboRSI.{runtime_name}_runtime")
    policy = tmp_path / "policy"
    python = policy / f"runtime/{runtime_name}-env/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    children, servers = [], []
    owners = weakref.WeakSet()
    state = SimpleNamespace(
        backend=backend,
        policy=policy,
        children=children,
        servers=servers,
        mode="exit",
        relative_output=False,
        live_before_return=False,
    )
    state.model = _boundary(backend)
    owners.add(state.model)
    monkeypatch.setattr(model_module, "POLICY_DIR", policy)
    monkeypatch.setattr(
        model_module, "resolve_checkpoint_root", lambda *_a, **_k: policy / "checkpoint"
    )
    monkeypatch.delenv("ROBORSI_SHARED_VLA_PORT", raising=False)
    monkeypatch.delenv("ROBORSI_RUNTIME_STATE_DIR", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def environment(_policy=None, _data=None, output=None, _gpu=None):
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
        return {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    if backend == "simate_beta":
        monkeypatch.setattr(runtime, "check_simate_runtime", lambda *_a, **_k: {})
        monkeypatch.setattr(runtime, "runtime_python", lambda *_a: python)
        monkeypatch.setattr(runtime, "model_environment", environment)
        server_name = "SimateServer"
    else:
        for name in (
            "check_pi05_runtime",
            "validate_pi05_checkpoint",
            "validate_pi05_tokenizer",
        ):
            monkeypatch.setattr(runtime, name, lambda *_a, **_k: {})
        monkeypatch.setattr(runtime, "_service_env", environment)
        server_name = "Pi05Server"
    original_server = getattr(runtime, server_name)

    def server_factory(**kwargs):
        # Keep the production lifecycle; shorten only its startup deadline.
        if state.relative_output:
            kwargs["output_dir"] = Path(os.path.relpath(kwargs["output_dir"]))
        server = original_server(**kwargs, startup_timeout_s=0.2)
        servers.append(server)
        return server

    monkeypatch.setattr(runtime, server_name, server_factory)
    original_popen = runtime.subprocess.Popen

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        if state.mode == "exit":
            # Reap a real exit before returning to the readiness loop. This avoids
            # a race between pipe EOF and waitpid observing the same exit.
            assert child.wait(timeout=5) == 17
        else:
            log = Path(kwargs["stderr"].name)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if log.read_text() and child.poll() is None:
                    state.live_before_return = True
                    break
                time.sleep(0.01)
            assert state.live_before_return, "CPU fixture did not start before timeout"
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)

    def program(marker, *, mode="exit", readiness=None):
        state.mode = mode
        state.live_before_return = False
        setup = ""
        ending = "sys.exit(17)" if mode == "exit" else "time.sleep(60)"
        if mode == "readiness":
            ready = "int(sys.argv[sys.argv.index('--ready-fd') + 1])"
            ending = (
                f"os.close({ready})" if readiness is None
                else f"os.write({ready}, {readiness!r})"
            ) + "\ntime.sleep(60)"
        elif mode == "ready":
            # Bind before the fixture's log marker so the short production
            # deadline measures readiness rather than stdlib import latency.
            setup = (
                "import json\n"
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "class HealthHandler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        if self.path != '/healthz':\n"
                "            self.send_error(404)\n"
                "            return\n"
                "        self.send_response(200)\n"
                "        self.end_headers()\n"
                "        self.wfile.write(b'ok')\n"
                "    def log_message(self, *_args):\n"
                "        pass\n"
                "server = HTTPServer(('127.0.0.1', 0), HealthHandler)\n"
            )
            ending = (
                "ready_fd = int(sys.argv[sys.argv.index('--ready-fd') + 1])\n"
                "os.write(ready_fd, (json.dumps({'port': server.server_port}) + '\\n').encode())\n"
                "os.close(ready_fd)\n"
                "server.serve_forever()"
            )
        (policy / f"{runtime_name}_server.py").write_text(
            "import os, sys, time\n"
            f"{setup}"
            f"print({('stdout:' + marker)!r}, flush=True)\n"
            f"print({('stderr:' + marker)!r}, file=sys.stderr, flush=True)\n"
            f"{ending}\n"
        )

    state.program = program
    yield state
    # A failing assertion must never leave this test's CPU children behind.
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
    for server in servers:
        server.close()
    for owner in list(owners):
        owner.shutdown()


def _retained_logs(startup, error, marker):
    server = startup.servers[-1]
    directory = server.output_dir.resolve()
    stdout = directory / "server.stdout.log"
    stderr = directory / "server.stderr.log"
    assert stderr.is_file(), "Model removed the failed child's startup diagnostics"
    assert stdout.read_bytes() == f"stdout:{marker}\n".encode()
    assert stderr.read_bytes() == f"stderr:{marker}\n".encode()
    assert directory.parent == (startup.policy / "outputs/vla").resolve()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert str(stderr) in str(error.value), "Failure must report its resolved stderr path"
    assert server.process is startup.children[-1]
    assert server.process.returncode is not None
    streams = server.streams if startup.backend == "simate_beta" else (server._stdout, server._stderr)
    assert all(stream.closed for stream in streams)
    assert startup.model._vla is None
    return directory


def test_ready_service_is_reused_and_shutdown_retains_logs(startup):
    startup.program("ready", mode="ready")
    port = startup.model._ensure_vla()
    assert 1 <= port <= 65535
    assert startup.live_before_return
    assert len(startup.servers) == len(startup.children) == 1
    server = startup.servers[0]
    child = startup.children[0]
    assert startup.model._vla is server
    assert port == server.port
    assert child.poll() is None
    assert startup.model._ensure_vla() == port
    assert len(startup.servers) == len(startup.children) == 1
    assert startup.model._vla is server
    directory = server.output_dir.resolve()
    assert directory.parent == (startup.policy / "outputs/vla").resolve()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700

    startup.model.shutdown()
    assert startup.model._vla is None
    assert child.poll() == -signal.SIGTERM
    streams = server.streams if startup.backend == "simate_beta" else (server._stdout, server._stderr)
    assert all(stream.closed for stream in streams)
    assert (directory / "server.stdout.log").read_bytes() == b"stdout:ready\n"
    assert (directory / "server.stderr.log").read_bytes() == b"stderr:ready\n"


def test_exit_17_logs_survive_shutdown_and_model_collection(startup):
    startup.program("exit-17")
    with pytest.raises(RuntimeError) as error:
        startup.model._ensure_vla()
    assert startup.children[-1].returncode == 17
    assert "exit 17" in str(error.value)
    directory = _retained_logs(startup, error, "exit-17")
    startup.model.shutdown()
    del error  # Its traceback otherwise keeps the failed Model alive.
    reference = weakref.ref(startup.model)
    startup.model = None
    gc.collect()
    assert reference() is None
    assert (directory / "server.stdout.log").read_text() == "stdout:exit-17\n"
    assert (directory / "server.stderr.log").read_text() == "stderr:exit-17\n"


def test_timeout_reaps_live_child_and_retains_closed_logs(startup):
    startup.program("timeout", mode="timeout")
    with pytest.raises(TimeoutError) as error:
        startup.model._ensure_vla()
    assert startup.live_before_return
    assert startup.children[-1].returncode == -signal.SIGTERM
    assert "exit -" not in str(error.value), "Cleanup signals must not be reported as the startup cause"
    directory = _retained_logs(startup, error, "timeout")
    startup.model.shutdown()
    assert (directory / "server.stderr.log").read_text() == "stderr:timeout\n"


def test_failed_startup_attempts_keep_distinct_diagnostics(startup):
    directories = []
    for marker in ("first-attempt", "second-attempt"):
        startup.program(marker)
        with pytest.raises(RuntimeError) as error:
            startup.model._ensure_vla()
        assert startup.children[-1].returncode == 17
        directories.append(_retained_logs(startup, error, marker))
    assert directories[0] != directories[1]
    startup.model.shutdown()
    for directory, marker in zip(directories, ("first-attempt", "second-attempt")):
        assert (directory / "server.stdout.log").read_text() == f"stdout:{marker}\n"
        assert (directory / "server.stderr.log").read_text() == f"stderr:{marker}\n"


@pytest.mark.parametrize("startup", ("simate_beta",), indirect=True)
@pytest.mark.parametrize(
    ("readiness", "message"),
    (
        pytest.param(None, "closed readiness pipe", id="pipe-eof"),
        pytest.param(b"x" * 129, "Invalid Simate readiness announcement", id="oversized"),
        pytest.param(b"invalid\n", "Invalid Simate readiness announcement", id="invalid-json"),
        pytest.param(b"{}\n", "Invalid Simate readiness announcement", id="missing-port"),
        pytest.param(b"[]\n", "Invalid Simate readiness announcement", id="invalid-shape"),
        pytest.param(b'{"port": true}\n', "Invalid Simate bound port", id="invalid-port"),
    ),
)
def test_simate_readiness_failures_report_resolved_logs(startup, readiness, message):
    startup.relative_output = True
    startup.program("readiness", mode="readiness", readiness=readiness)
    with pytest.raises(RuntimeError, match=message) as error:
        startup.model._ensure_vla()
    assert startup.live_before_return
    assert startup.children[-1].returncode == -signal.SIGTERM
    assert "exit -" not in str(error.value)
    assert not startup.servers[-1].output_dir.is_absolute()
    _retained_logs(startup, error, "readiness")


def test_shared_vla_is_not_owned_or_given_local_startup_logs(startup, monkeypatch):
    with socket.socket() as external:
        external.bind(("127.0.0.1", 0))
        external.listen()
        external.settimeout(2)
        port = external.getsockname()[1]
        monkeypatch.setenv("ROBORSI_SHARED_VLA_PORT", str(port))
        assert startup.model._ensure_vla() == port
        connection, _address = external.accept()
        connection.close()
        startup.model.shutdown()
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            connection, _address = external.accept()
            connection.close()
    assert startup.servers == startup.children == []
    assert not (startup.policy / "outputs/vla").exists()
