"""Startup reservations survive contention, but never consume failed initialization."""

from importlib import import_module

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import pytest

scene = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_scene_resume_identity")
trajectory = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_trajectory_identity")
model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
backend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend")
agent_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
stopped = trajectory.stopped


@pytest.fixture
def startup(stopped, reference, tmp_path, monkeypatch):
    source, _, identity, _ = stopped
    request_path = tmp_path / "request.json"
    request = resume.prepare_resume(source, request_path, reference)
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST", str(request_path))
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST_SHA256", resume.digest(request_path))
    monkeypatch.setenv("ROBODOJO_RUN_ID", identity["run_id"])
    monkeypatch.setenv("ROBORSI_NATIVE_LAYOUT_ID", str(identity["layout"]))
    monkeypatch.delenv("ROBORSI_TRACE_ROOT", raising=False)
    value = trajectory.configured_model(monkeypatch, tmp_path / "attempts")
    value._tool_python = Path(sys.executable)
    value.reset()
    native = scene.observation(scene.receipt(run_id=identity["run_id"]))
    value.update_obs(native)
    captured = scene._prepare_start(monkeypatch, value)
    for name in ("_reserve_resume", "_claim_resume"):
        monkeypatch.setattr(value, name, MethodType(getattr(model_module.Model, name), value))
    try:
        yield SimpleNamespace(
            model=value, path=request_path, request=request, observation=native,
            captured=captured, source=source,
        )
    finally:
        value.close()


def assert_retryable(startup):
    value = startup.model
    assert not startup.path.with_suffix(".json.claimed").exists()
    assert value._resume_start_lock is None
    value.reset()
    assert value._trajectory_id == startup.request["trajectory_id"]
    assert value._resume["recovery_count"] == startup.request["recovery_count"]
    assert resume.load_resume(startup.path, startup.request["identity"]) == startup.request
    value.update_obs(startup.observation)


@pytest.mark.parametrize("failure", ["backend", "agent", "controller", "thread"])
def test_initialization_failure_can_retry_same_request(startup, monkeypatch, failure):
    value = startup.model
    failed_attempt = value._directory

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected startup failure")

    with monkeypatch.context() as injection:
        if failure == "backend":
            injection.setattr(backend, "HybridBackend", fail)
        elif failure == "agent":
            injection.setattr(value._agent_config, "create_agent", fail)
        elif failure == "controller":
            injection.setattr(agent_module, "HybridController", fail)
        else:
            injection.setattr(agent_module, "HybridController", lambda *_: SimpleNamespace(start=fail))
        with pytest.raises(RuntimeError, match="injected startup failure"):
            value.get_action()
    assert not (failed_attempt / "agent/controller/config.json").exists()
    assert_retryable(startup)
    value._start()
    assert not startup.path.with_suffix(".json.claimed").exists()
    startup.captured["context"].before_execution()
    claim = resume.read_json(startup.path.with_suffix(".json.claimed"))
    assert claim["attempt"] == str(value._directory)
    assert claim["attempt"] != str(failed_attempt)
    # The production close path releases only the reservation, never this claim.
    value._connector = None  # The successful start here uses a thread-free fixture.
    value.close()
    assert startup.path.with_suffix(".json.claimed").is_file()
    with pytest.raises(FileExistsError):
        value.reset()


def test_background_initialization_failure_releases_reservation(startup, monkeypatch):
    value = startup.model
    controller_class = agent_module.HybridController
    # _prepare_start uses a thread-free controller; restore the real owner.
    controller_class = import_module(
        "XPolicyLab.policy.SenseNova-RoboRSI.tests.test_original_resume"
    ).hybrid_agent.HybridController if controller_class.__module__ == agent_module.__name__ else REAL_CONTROLLER
    actions = queue.Queue()

    class Agent:
        def bind_reference(self, _context):
            return self

        def run_reference(self):
            raise RuntimeError("injected asynchronous initialization failure")

        def close(self):
            pass

    monkeypatch.setattr(agent_module, "HybridController", controller_class)
    monkeypatch.setattr(value._agent_config, "create_agent", lambda _: Agent())
    monkeypatch.setattr(backend, "HybridBackend", lambda *_a, **_k: SimpleNamespace(
        cancel=lambda: None, next_action=lambda timeout: actions.get(timeout=timeout),
    ))
    with pytest.raises(RuntimeError, match="Original hybrid Agent failed"):
        value.get_action()
    assert_retryable(startup)


REAL_CONTROLLER = agent_module.HybridController

STARTUP_CHILD = r"""
from importlib import import_module
import json, sys
from pathlib import Path
import pytest
fixture = import_module('XPolicyLab.policy.SenseNova-RoboRSI.tests.test_trajectory_identity')
patch = pytest.MonkeyPatch()
value = fixture.configured_model(patch, Path(sys.argv[1]))
try:
    value.reset()
    value._reserve_resume()
    if sys.argv[3] == 'hold':
        Path(sys.argv[2]).write_text('reserved')
        sys.stdin.read()
    else:
        value._claim_resume()
    print(json.dumps({'status': 'reserved', 'trajectory_id': value._trajectory_id}))
except FileExistsError:
    print(json.dumps({'status': 'refused'}))
finally:
    value.close()
    patch.undo()
"""


def test_killed_starter_releases_lock_and_another_process_keeps_trajectory(
    stopped, reference, tmp_path
):
    source, _, identity, _ = stopped
    request_path = tmp_path / "request.json"
    request = resume.prepare_resume(source, request_path, reference)
    ready = tmp_path / "ready"
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": os.pathsep.join(str(Path(p).absolute()) for p in sys.path if isinstance(p, str)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROBORSI_CODEX_MODEL": "fixture-model",
        "ROBORSI_CODEX_BASE_URL": "https://provider.example.invalid/v1",
        "ROBORSI_RESUME_REQUEST": str(request_path),
        "ROBORSI_RESUME_REQUEST_SHA256": resume.digest(request_path),
        "ROBODOJO_RUN_ID": identity["run_id"],
        "ROBORSI_NATIVE_LAYOUT_ID": str(identity["layout"]),
    }
    command = [sys.executable, "-c", STARTUP_CHILD, str(tmp_path / "attempts"), str(ready)]
    first = subprocess.Popen(command + ["hold"], env=environment, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), first.poll()
        contender = subprocess.run(command + ["claim"], env=environment, capture_output=True,
                                   text=True, timeout=20)
        assert contender.returncode == 0, contender.stderr
        assert json.loads(contender.stdout.splitlines()[-1])["status"] == "refused"
        assert not request_path.with_suffix(".json.claimed").exists()
        first.kill()
        first.communicate(timeout=10)
        retry = subprocess.run(command + ["claim"], env=environment, capture_output=True,
                               text=True, timeout=20)
        assert retry.returncode == 0, retry.stderr
        assert json.loads(retry.stdout.splitlines()[-1]) == {
            "status": "reserved", "trajectory_id": request["trajectory_id"],
        }
        assert request_path.with_suffix(".json.claimed").is_file()
    finally:
        if first.poll() is None:
            first.kill()
        first.communicate(timeout=10)
