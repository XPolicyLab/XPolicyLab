"""Interrupted private replay retries its verified parent under original policy."""

from importlib import import_module

import copy
import shutil
from pathlib import Path

import pytest

resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
resume_fixtures = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_original_resume")

stopped = resume_fixtures.stopped


@pytest.fixture
def replay_interrupted(stopped, reference, tmp_path):
    parent, _, identity, _ = stopped
    first = resume.prepare_resume(parent, tmp_path / "first-request.json", reference)
    failed = tmp_path / "failed-replay"
    run = failed / "agent/controller"
    shutil.copytree(parent / "agent/workspace", failed / "agent/workspace")
    (failed / "agent/workspace/own-attempt-note.md").write_text("Retained failed-attempt workspace")
    native = copy.deepcopy(first["native_identity"])
    native["control_step"] = 0
    resume.write_json(failed / "episode-owner.json", {
        "identity": identity, "closed": True, "native_identity": native,
        "trajectory_id": first["trajectory_id"],
        "recovery_state": first["recovery_state"], "recovery_count": 1,
    })
    resume.write_json(run / "config.json", first["config"])
    resume.write_json(run / "summary.json", {
        "status": "error", "error": "RuntimeError: Simulator exited during replay",
    })
    resume.write_json(run / "segments/evaluation-0001/result.json", {
        "status": "interrupted", "wall_seconds": 125.0,
    })
    return parent, failed, first, identity


def test_original_parent_fallback_deducts_failed_replay_and_retains_workspace(
    replay_interrupted, reference, tmp_path,
):
    parent, failed, first, identity = replay_interrupted
    original_config = resume.digest(parent / "agent/controller/config.json")
    failed_config = resume.digest(failed / "agent/controller/config.json")
    output = tmp_path / "second-request.json"
    request = resume.prepare_resume(failed, output, reference)
    assert request["trajectory_id"] == first["trajectory_id"] == "fixture-trajectory"
    assert request["source"] == str(parent)
    assert request["workspace_source"] == str(failed)
    assert request["recovery_source"] == str(failed)
    assert request["step_offset"] == 1
    assert request["native_identity"]["control_step"] == 1
    assert request["config"]["episode_seconds"] == pytest.approx(first["config"]["episode_seconds"] - 125)
    assert request["config"]["continue_from_run"] == str(parent / "agent/controller")
    assert request["config"]["options"]["replay_fresh_session"] is True
    assert request["config"]["options"]["replay"]["control_step"] == 1
    assert request["recovery_count"] == 2
    assert resume.load_resume(output, identity) == request
    assert resume.digest(parent / "agent/controller/config.json") == original_config
    assert resume.digest(failed / "agent/controller/config.json") == failed_config
    state = resume.read_json(Path(first["recovery_state"]))
    assert state["attempt_history"][-1]["run"] == str(failed / "agent/controller")


def test_fallback_workspace_remains_sealed(replay_interrupted, reference, tmp_path):
    _, failed, _, identity = replay_interrupted
    output = tmp_path / "second-request.json"
    resume.prepare_resume(failed, output, reference)
    (failed / "agent/workspace/own-attempt-note.md").write_text("changed")
    with pytest.raises(ValueError, match="workspace changed"):
        resume.load_resume(output, identity)


def test_replay_validation_failure_cannot_use_technical_parent_fallback(
    replay_interrupted, reference, tmp_path,
):
    _, failed, _, _ = replay_interrupted
    resume.write_json(failed / "agent/controller/summary.json", {
        "error": "RuntimeError: Replayed checkpoint does not match saved RGB/proprioception",
    })
    with pytest.raises(ValueError, match="not eligible"):
        resume.prepare_resume(failed, tmp_path / "second-request.json", reference)


def test_failed_replay_cannot_discard_new_actions(replay_interrupted, reference, tmp_path):
    _, failed, _, _ = replay_interrupted
    record = failed / "agent/controller/episodes/evaluation-0001"
    record.mkdir(parents=True)
    (record / "actions.jsonl").write_text('{"status":"completed"}\n')
    with pytest.raises(ValueError, match="discard new action"):
        resume.prepare_resume(failed, tmp_path / "second-request.json", reference)


def test_parent_fallback_preserves_four_recovery_limit(replay_interrupted, reference, tmp_path):
    _, failed, first, _ = replay_interrupted
    state = resume.read_json(Path(first["recovery_state"]))
    state["recovery_count"] = 4
    resume.write_json(first["recovery_state"], state)
    with pytest.raises(ValueError, match="Four infrastructure"):
        resume.prepare_resume(failed, tmp_path / "second-request.json", reference)


def test_parent_fallback_refuses_different_scene(replay_interrupted, reference, tmp_path):
    _, failed, _, _ = replay_interrupted
    path = failed / "episode-owner.json"
    owner = resume.read_json(path)
    owner["native_identity"]["scene_sha256"] = "b" * 64
    resume.write_json(path, owner)
    with pytest.raises(ValueError, match="scene hash changed"):
        resume.prepare_resume(failed, tmp_path / "second-request.json", reference)


def test_parent_fallback_refuses_different_trajectory(replay_interrupted, reference, tmp_path):
    _, failed, _, _ = replay_interrupted
    path = failed / "episode-owner.json"
    owner = resume.read_json(path)
    owner["trajectory_id"] = "another-episode-trajectory"
    resume.write_json(path, owner)
    with pytest.raises(ValueError, match="trajectory"):
        resume.prepare_resume(failed, tmp_path / "second-request.json", reference)
