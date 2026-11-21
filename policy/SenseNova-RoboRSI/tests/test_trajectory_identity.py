"""Stable HTTP trajectory identity across private episode recovery attempts.

These tests use real retained JSON records and production config construction.
No CLI/provider, VLA weights, simulator, robot actions, or network are needed.
"""

from __future__ import annotations

from importlib import import_module

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from roborsi.planner.codex_cli_config import CodexCliConfig

resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
runtime_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")
resume_fixtures = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_original_resume")

bare_model = resume_fixtures.bare_model
stopped = resume_fixtures.stopped

IDENTITY = {"task": "align_blocks", "eval_seed": 0, "layout": 14, "run_id": "fixture"}
HEADER_KEY = "model_providers.runtime.env_http_headers.X-Trajectory-Id"


def retained(tmp_path, name="attempt", *, trajectory_id=None, legacy_id=None,
             source=None, identity=None, scene_sha="a" * 64):
    """Match Model's actual trajectory.json / runtime-*/owner file layout."""
    trajectory = tmp_path / name
    root = trajectory / "runtime-fixture"
    root.mkdir(parents=True)
    native_identity = dict(IDENTITY if identity is None else identity)
    owner = {
        "identity": native_identity,
        "closed": True,
        "native_identity": {"schema": 1, **native_identity,
                            "scene_sha256": scene_sha,
                            "control_step": 1, "step_limit": 200},
    }
    if trajectory_id is not None:
        owner["trajectory_id"] = trajectory_id
    if source is not None:
        owner["source"] = str(source)
    resume.write_json(root / "episode-owner.json", owner)
    if legacy_id is not None:
        resume.write_json(trajectory / "trajectory.json", {
            "schema_version": 1, "task": "align_blocks", "seed": 0,
            "episode_id": legacy_id,
        })
    return root, owner


def test_explicit_trajectory_identity_and_used_records_are_sealed(tmp_path):
    root, owner = retained(tmp_path, trajectory_id="stable-trajectory", legacy_id="attempt-a")
    path = root.parent / "trajectory.json"
    metadata = resume.read_json(path)
    metadata["trajectory_id"] = "stable-trajectory"
    resume.write_json(path, metadata)
    trajectory_id, seals = resume.source_trajectory_id(root, owner)
    assert trajectory_id == "stable-trajectory"
    for record in (root / "episode-owner.json", path):
        assert seals[str(record)] == resume.digest(record)


def test_legacy_trajectory_episode_id_is_restored_from_runtime_parent(tmp_path):
    root, _ = retained(tmp_path, legacy_id="legacy-episode")
    trajectory_id, seals = resume.source_trajectory_id(root)
    assert trajectory_id == "legacy-episode"
    assert seals[str(root.parent / "trajectory.json")] == resume.digest(root.parent / "trajectory.json")


def test_legacy_archive_episode_id_is_restored(tmp_path):
    root, _ = retained(tmp_path)
    root = root.rename(root.with_name("runtime"))
    manifest = root.parent / "manifest.json"
    resume.write_json(manifest, {
        "format": "xpolicylab_episode_archive_v1", "task": "align_blocks",
        "episode_id": "archived-episode", "run_id": "fixture", "files": [],
    })
    trajectory_id, seals = resume.source_trajectory_id(root)
    assert trajectory_id == "archived-episode"
    assert seals[str(manifest)] == resume.digest(manifest)


def test_recovery_ancestry_wins_over_new_attempt_legacy_episode_id(tmp_path):
    original, _ = retained(tmp_path, "original", legacy_id="original-episode")
    resumed, _ = retained(tmp_path, "resumed", legacy_id="second-attempt", source=original)
    next_attempt, _ = retained(tmp_path, "next", legacy_id="third-attempt", source=resumed)
    trajectory_id, seals = resume.source_trajectory_id(next_attempt)
    assert trajectory_id == "original-episode"
    assert str(original / "episode-owner.json") in seals
    assert str(original.parent / "trajectory.json") in seals


@pytest.mark.parametrize("change", [
    {"identity": {**IDENTITY, "run_id": "another-run"}},
    {"scene_sha": "b" * 64},
])
def test_ancestry_cannot_cross_native_episode_or_scene(tmp_path, change):
    original, _ = retained(tmp_path, "original", legacy_id="original", **change)
    resumed, _ = retained(tmp_path, "resumed", legacy_id="new-attempt", source=original)
    with pytest.raises(ValueError):
        resume.source_trajectory_id(resumed)


def test_conflicting_explicit_trajectory_records_are_rejected(tmp_path):
    root, _ = retained(tmp_path, trajectory_id="owner-trajectory", legacy_id="attempt-a")
    metadata = resume.read_json(root.parent / "trajectory.json")
    metadata["trajectory_id"] = "different-trajectory"
    resume.write_json(root.parent / "trajectory.json", metadata)
    with pytest.raises(ValueError):
        resume.source_trajectory_id(root)


@pytest.mark.parametrize("invalid", ["", " ", "bad\nheader", "../escape", "a/b", "非ASCII", "a" * 129, 12, True, [], None])
def test_invalid_explicit_identity_cannot_fall_back_to_valid_legacy_id(tmp_path, invalid):
    root, owner = retained(tmp_path, legacy_id="otherwise-valid")
    owner["trajectory_id"] = invalid
    resume.write_json(root / "episode-owner.json", owner)
    with pytest.raises(ValueError):
        resume.source_trajectory_id(root)


def test_cycle_and_missing_identity_never_generate_a_replacement(tmp_path):
    first, first_owner = retained(tmp_path, "first", legacy_id="first-attempt")
    second, _ = retained(tmp_path, "second", legacy_id="second-attempt", source=first)
    first_owner["source"] = str(second)
    resume.write_json(first / "episode-owner.json", first_owner)
    with pytest.raises(ValueError):
        resume.source_trajectory_id(second)
    empty, _ = retained(tmp_path, "empty")
    with pytest.raises(ValueError):
        resume.source_trajectory_id(empty)


def test_prepare_and_load_keep_the_source_trajectory(stopped, reference, tmp_path):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    assert request["trajectory_id"] == "fixture-trajectory"
    assert resume.load_resume(output, identity)["trajectory_id"] == "fixture-trajectory"
    assert request["source_files"][str(root / "episode-owner.json")] == resume.digest(root / "episode-owner.json")


def test_legacy_request_without_trajectory_field_is_normalized(stopped, reference, tmp_path):
    root, _, identity, _ = stopped
    owner = resume.read_json(root / "episode-owner.json")
    owner.pop("trajectory_id")
    resume.write_json(root / "episode-owner.json", owner)
    resume.write_json(root.parent / "trajectory.json", {
        "task": "align_blocks", "seed": 0, "episode_id": "legacy-source",
    })
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    request.pop("trajectory_id")
    resume.write_json(output, request)
    assert resume.load_resume(output, identity)["trajectory_id"] == "legacy-source"


def test_load_rejects_changed_or_relabelled_trajectory(stopped, reference, tmp_path):
    root, _, identity, _ = stopped
    metadata = root.parent / "trajectory.json"
    resume.write_json(metadata, {
        "task": "align_blocks", "seed": 0,
        "episode_id": "attempt", "trajectory_id": "fixture-trajectory",
    })
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    assert str(metadata) in request["source_files"]
    request["trajectory_id"] = "different-trajectory"
    resume.write_json(output, request)
    with pytest.raises(ValueError):
        resume.load_resume(output, identity)
    request["trajectory_id"] = "fixture-trajectory"
    resume.write_json(output, request)
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError):
        resume.load_resume(output, identity)


def configured_model(monkeypatch, directory):
    value = bare_model(monkeypatch, directory)
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-trajectory-offline-key")
    monkeypatch.setattr(model_module, "load_agent_config", runtime_config.load_agent_config)
    return value


def cli_identity(value):
    cli = CodexCliConfig.from_dict(value._agent_config.document["agent"]["codex_cli"])
    assert cli.config_overrides[HEADER_KEY] == "ROBORSI_TRAJECTORY_ID"
    return cli.subprocess_env()["ROBORSI_TRAJECTORY_ID"]


def test_fresh_episodes_generate_new_ids_without_crossing_instance_headers(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBORSI_RESUME_REQUEST", raising=False)
    monkeypatch.setenv("ROBORSI_TRAJECTORY_ID", "unrelated-parent")
    first = configured_model(monkeypatch, tmp_path / "first")
    second = configured_model(monkeypatch, tmp_path / "second")
    try:
        first.reset()
        first_config = first._agent_config
        original = first._trajectory_id
        assert original == first._episode_id
        second.reset()
        assert second._trajectory_id != original
        assert cli_identity(first) == original
        assert cli_identity(second) == second._trajectory_id
        first.reset()
        assert first._trajectory_id not in {original, second._trajectory_id}
        old_cli = CodexCliConfig.from_dict(first_config.document["agent"]["codex_cli"])
        assert old_cli.subprocess_env()["ROBORSI_TRAJECTORY_ID"] == original
        assert cli_identity(second) == second._trajectory_id
        assert os.environ["ROBORSI_TRAJECTORY_ID"] == "unrelated-parent"
    finally:
        first.close()
        second.close()


def test_model_resume_changes_attempt_but_preserves_header_and_records(stopped, reference, tmp_path, monkeypatch):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    monkeypatch.setenv("ROBODOJO_RUN_ID", identity["run_id"])
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST", str(output))
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST_SHA256", resume.digest(output))
    monkeypatch.setenv("ROBORSI_TRAJECTORY_ID", "unrelated-parent")
    value = configured_model(monkeypatch, tmp_path / "new-attempts")
    continued = None
    try:
        value.reset()
        attempt = value._episode_id
        assert value._trajectory_id == request["trajectory_id"] == "fixture-trajectory"
        assert cli_identity(value) == "fixture-trajectory"
        assert attempt != value._trajectory_id
        assert resume.read_json(value._directory / "episode-owner.json")["trajectory_id"] == value._trajectory_id
        metadata = resume.read_json(value._trajectory_directory / "trajectory.json")
        assert metadata["episode_id"] == attempt
        assert metadata["trajectory_id"] == value._trajectory_id
        value.close()
        continued = configured_model(monkeypatch, tmp_path / "restarted-process-attempts")
        continued.reset()
        assert continued._episode_id != attempt
        assert continued._trajectory_id == cli_identity(continued) == "fixture-trajectory"
        assert os.environ["ROBORSI_TRAJECTORY_ID"] == "unrelated-parent"
    finally:
        value.close()
        if continued is not None:
            continued.close()


def test_invalid_resume_identity_fails_before_creating_attempt(stopped, reference, tmp_path, monkeypatch):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    request["trajectory_id"] = "bad\nheader"
    resume.write_json(output, request)
    monkeypatch.setenv("ROBODOJO_RUN_ID", identity["run_id"])
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST", str(output))
    monkeypatch.delenv("ROBORSI_RESUME_REQUEST_SHA256", raising=False)
    attempts = tmp_path / "new-attempts"
    value = configured_model(monkeypatch, attempts)
    try:
        with pytest.raises(ValueError):
            value.reset()
        assert not attempts.exists()
        assert value._temporary is None
    finally:
        value.close()


CHILD_RESET_PROGRAM = r"""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True
# Only the official robot-metadata import is replaced. The Model lifecycle,
# recovery request validation, AgentConfig and CLI environment are production.
sys.modules["XPolicyLab.utils.process_data"] = SimpleNamespace(get_robot_action_dim_info=lambda _: None)
from importlib import import_module
model = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
hybrid_resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
from roborsi.planner.codex_cli_config import CodexCliConfig

def synchronize(phase):
    gate = os.environ.get("ROBORSI_TEST_PROCESS_BARRIER")
    if not gate:
        return
    folder = Path(gate) / phase
    folder.mkdir(parents=True, exist_ok=True)
    (folder / str(os.getpid())).write_text("ready")
    deadline = time.monotonic() + 10
    while len(list(folder.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("parallel child initialization barrier timed out")
        time.sleep(0.01)

value = model.Model.__new__(model.Model)
value.model_cfg = {"task_name": "align_blocks", "seed": 0, "trajectory_dir": sys.argv[1]}
value._connector = value._temporary = value._icl = value._obs = value._pending = None
value._step, value._resume, value._awaiting_feedback = 0, None, False
try:
    synchronize("before_reset")
    value.reset()
    claiming = os.environ.get("ROBORSI_TEST_CLAIM_RESUME") == "1"
    if claiming:
        # Both reset calls finish before either contender opens .claimed.
        synchronize("before_claim")
        value._claim_resume()
    cli = CodexCliConfig.from_dict(value._agent_config.document["agent"]["codex_cli"])
    owner = hybrid_resume.read_json(value._directory / "episode-owner.json")
    metadata = hybrid_resume.read_json(value._trajectory_directory / "trajectory.json")
    result = {
        "status": "claimed" if claiming else "initialized",
        "pid": os.getpid(), "attempt_id": value._episode_id,
        "attempt_directory": str(value._directory),
        "trajectory_id": value._trajectory_id,
        "header": cli.subprocess_env()["ROBORSI_TRAJECTORY_ID"],
        "owner_trajectory_id": owner["trajectory_id"],
        "metadata_trajectory_id": metadata["trajectory_id"],
        "request_loaded_from_disk": str(value._resume_request_path) if value._resume_request_path else None,
        "source": value._resume["source"] if value._resume else None,
        "parent_environment_unchanged": os.environ["ROBORSI_TRAJECTORY_ID"] == "unrelated-parent",
    }
except FileExistsError:
    if os.environ.get("ROBORSI_TEST_CLAIM_RESUME") != "1":
        raise
    result = {"status": "refused", "pid": os.getpid()}
finally:
    value.close()
print(json.dumps(result))
"""


def child_model(directory, *, request_path=None, identity=None, barrier=None, claim=False):
    directory.mkdir(parents=True)
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": os.pathsep.join(str(Path(entry).absolute()) for entry in sys.path if isinstance(entry, str)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CODEX_API_KEY": "synthetic-child-trajectory-offline-key",
        "ROBORSI_CODEX_BASE_URL": "https://provider.example.invalid/v1",
        "ROBORSI_CODEX_MODEL": "fixture-model",
        "ROBORSI_TRAJECTORY_ID": "unrelated-parent",
    }
    if request_path is not None:
        environment.update({
            "ROBODOJO_RUN_ID": identity["run_id"],
            "ROBORSI_NATIVE_LAYOUT_ID": str(identity["layout"]),
            "ROBORSI_RESUME_REQUEST": str(request_path),
            "ROBORSI_RESUME_REQUEST_SHA256": resume.digest(request_path),
        })
    if barrier is not None:
        environment["ROBORSI_TEST_PROCESS_BARRIER"] = str(barrier)
    if claim:
        environment["ROBORSI_TEST_CLAIM_RESUME"] = "1"
    # Deployment workers may share one trajectory root; each attempt still
    # needs its own directory even when both processes initialize together.
    attempts = (barrier if barrier is not None else directory) / "attempts"
    child = subprocess.run(
        [sys.executable, "-c", CHILD_RESET_PROGRAM, str(attempts)],
        env=environment, cwd=directory, capture_output=True, text=True, timeout=25,
    )
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout.splitlines()[-1])


def test_new_process_loads_saved_request_and_reuses_original_header(stopped, reference, tmp_path):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    result = child_model(tmp_path / "child", request_path=output, identity=identity)
    assert result["pid"] != os.getpid()
    assert result["trajectory_id"] == result["header"] == request["trajectory_id"]
    assert result["owner_trajectory_id"] == result["metadata_trajectory_id"] == request["trajectory_id"]
    assert result["attempt_id"] != request["trajectory_id"]
    assert result["request_loaded_from_disk"] == str(output)
    assert result["source"] == str(root)
    assert result["parent_environment_unchanged"]


@pytest.mark.parametrize("resumed", [False, True])
def test_parallel_processes_keep_episode_headers_and_attempts_isolated(stopped, reference, tmp_path, resumed):
    from concurrent.futures import ThreadPoolExecutor

    output, identity = None, None
    if resumed:
        root, _, identity, _ = stopped
        output = tmp_path / "request.json"
        resume.prepare_resume(root, output, reference)
    barrier = tmp_path / "parallel-barrier"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(child_model, tmp_path / f"child-{index}",
                               request_path=output, identity=identity,
                               barrier=barrier) for index in range(2)]
        results = [future.result() for future in futures]
    assert len({result["pid"] for result in results}) == 2
    assert len({result["attempt_id"] for result in results}) == 2
    assert len({result["attempt_directory"] for result in results}) == 2
    for result in results:
        assert result["trajectory_id"] == result["header"] == result["owner_trajectory_id"] == result["metadata_trajectory_id"]
        assert result["parent_environment_unchanged"]
    if resumed:
        assert {result["header"] for result in results} == {"fixture-trajectory"}
        assert all(result["request_loaded_from_disk"] == str(output) for result in results)
    else:
        assert len({result["header"] for result in results}) == 2
        assert all(result["header"] == result["attempt_id"] for result in results)


def test_parallel_resume_claim_has_exactly_one_process_owner(stopped, reference, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    barrier = tmp_path / "claim-barrier"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(child_model, tmp_path / f"contender-{index}",
                               request_path=output, identity=identity,
                               barrier=barrier, claim=True) for index in range(2)]
        results = [future.result() for future in futures]
    assert sorted(result["status"] for result in results) == ["claimed", "refused"]
    assert len(list((barrier / "before_claim").iterdir())) == 2
    winner = next(result for result in results if result["status"] == "claimed")
    claim = resume.read_json(output.with_suffix(".json.claimed"))
    assert claim["pid"] == winner["pid"]
    assert claim["attempt"] == winner["attempt_directory"]
    assert winner["header"] == "fixture-trajectory"
