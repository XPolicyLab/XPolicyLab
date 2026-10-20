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
runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_runtime")


def _boundary():
    model = model_module.Model.__new__(model_module.Model)
    model.model_cfg = {"checkpoint_path": "unused-checkpoint"}
    model._vla = model._vla_temporary = None
    model._temporary = model._connector = model._icl = model._obs = model._pending = None
    model._deferred_controller = model._cleanup_issue = None
    model._awaiting_feedback = False
    return model


@pytest.fixture
def startup(tmp_path, monkeypatch):
    """Replace GPU assets only; keep Model, Pi05Server and subprocesses real."""
    policy = tmp_path / "policy"
    python = policy / "runtime/pi05-env/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    children, servers = [], []
    owners = weakref.WeakSet()
    state = SimpleNamespace(
        policy=policy,
        children=children,
        servers=servers,
        mode="exit",
        live_before_return=False,
    )
    state.model = _boundary()
    owners.add(state.model)
    monkeypatch.setattr(model_module, "POLICY_DIR", policy)
    monkeypatch.setattr(
        model_module, "resolve_checkpoint_root", lambda *_a, **_k: policy / "checkpoint"
    )
    monkeypatch.delenv("ROBORSI_SHARED_VLA_PORT", raising=False)
    monkeypatch.delenv("ROBORSI_RUNTIME_STATE_DIR", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name in (
        "check_pi05_runtime",
        "validate_pi05_checkpoint",
        "validate_pi05_tokenizer",
    ):
        monkeypatch.setattr(runtime, name, lambda *_a, **_k: {})

    def environment(_policy, _data, output, _gpu):
        output.mkdir(parents=True, exist_ok=True)
        return {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    monkeypatch.setattr(runtime, "_service_env", environment)
    original_server = runtime.Pi05Server

    def server_factory(**kwargs):
        # Keep the production lifecycle; shorten only its startup deadline.
        server = original_server(**kwargs, startup_timeout_s=0.2)
        servers.append(server)
        return server

    monkeypatch.setattr(runtime, "Pi05Server", server_factory)
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

    def program(marker, *, mode="exit"):
        state.mode = mode
        state.live_before_return = False
        ending = "sys.exit(17)" if mode == "exit" else "time.sleep(60)"
        (policy / "pi05_server.py").write_text(
            "import sys, time\n"
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
    assert stdout.read_text() == f"stdout:{marker}\n"
    assert stderr.read_text() == f"stderr:{marker}\n"
    assert directory.parent == (startup.policy / "outputs/vla").resolve()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert str(directory) in str(error.value), "Failure must report its absolute log path"
    assert server.process is startup.children[-1]
    assert server.process.returncode is not None
    assert server._stdout.closed and server._stderr.closed
    assert startup.model._vla is None
    return directory


def test_exit_17_logs_survive_shutdown_and_model_collection(startup):
    startup.program("exit-17")
    with pytest.raises(RuntimeError) as error:
        startup.model._ensure_vla()
    assert startup.children[-1].returncode == 17
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
