"""Fail-closed validation of explicit cross-process resume state."""

from __future__ import annotations

from importlib import import_module

import hashlib
import json
import os
from pathlib import Path

import pytest

ResumeManifestError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.resume_manifest").ResumeManifestError
load_manifest = import_module("XPolicyLab.policy.SenseNova-RoboRSI.resume_manifest").load_manifest
write_manifest = import_module("XPolicyLab.policy.SenseNova-RoboRSI.resume_manifest").write_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def state(tmp_path):
    case = tmp_path / "case"
    trajectory = case / "trajectory"
    workspace = case / "workspace"
    native = case / "native" / "checkpoint.json"
    restore_receipt = case / "native" / "restore-receipt.json"
    actions = trajectory / "episodes" / "evaluation-0001" / "actions.jsonl"
    codex_home = case / "codex-home"
    for directory in (trajectory, workspace, native.parent, actions.parent, codex_home):
        directory.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("keep prior notes\n")
    (codex_home / "session.json").write_text("{}\n")
    native.write_text(json.dumps({"episode_id": "episode-abc", "control_step": 2}) + "\n")
    restore_receipt.write_text(
        json.dumps({"verified": True, "task": "align_blocks", "seed": 0, "episode_id": "episode-abc", "control_step": 2}) + "\n"
    )
    actions.write_text(
        json.dumps({"status": "completed", "control_step": 1, "action": {"x": 1}, "applied_action": {"x": 1}})
        + "\n"
        + json.dumps({"status": "completed", "control_step": 2, "action": {"x": 2}, "applied_action": {"x": 2}})
        + "\n"
    )
    manifest = case / "resume-manifest.json"
    write_manifest(
        manifest,
        {
            "schema_version": 1,
            "case_root": str(case),
            "task": "align_blocks",
            "seed": 0,
            "episode_id": "episode-abc",
            "control_step": 2,
            "step_limit": 42,
            "episode_seconds": 120,
            "elapsed_seconds": 10.5,
            "session_id": "session-abc",
            "paths": {
                "trajectory": str(trajectory),
                "workspace": str(workspace),
                "actions": str(actions),
                "native_checkpoint": str(native),
                "native_restore_receipt": str(restore_receipt),
                "codex_home": str(codex_home),
            },
            "action_sha256": _sha(actions),
            "native_checkpoint_sha256": _sha(native),
            "native_restore_receipt_sha256": _sha(restore_receipt),
        },
    )
    return {
        "case": case,
        "manifest": manifest,
        "trajectory": trajectory,
        "workspace": workspace,
        "native": native,
        "restore_receipt": restore_receipt,
        "actions": actions,
        "codex_home": codex_home,
    }


def test_load_valid_state_binds_actions_workspace_and_step(state):
    loaded = load_manifest(
        state["manifest"], expected_task="align_blocks", expected_seed=0, expected_step_limit=42
    )
    assert loaded.control_step == 2
    assert loaded.steps_remaining == 40
    assert loaded.task == "align_blocks"
    assert loaded.action_records[0]["control_step"] == 1
    assert loaded.action_records[-1]["applied_action"] == {"x": 2}
    assert loaded.source_workspace == state["workspace"]
    assert loaded.codex_home == state["codex_home"]
    assert loaded.session_id == "session-abc"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_task": "pour_balls_into_vase"},
        {"expected_seed": 1},
        {"expected_step_limit": 50},
    ],
)
def test_expected_identity_mismatch_is_rejected(state, kwargs):
    with pytest.raises(ResumeManifestError):
        load_manifest(state["manifest"], **kwargs)


def test_noncontiguous_actions_are_rejected(state):
    rows = state["actions"].read_text().splitlines()
    second = json.loads(rows[1])
    second["control_step"] = 3
    state["actions"].write_text(rows[0] + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ResumeManifestError, match="contiguous"):
        load_manifest(state["manifest"])


def test_missing_applied_action_is_rejected(state):
    rows = [json.loads(line) for line in state["actions"].read_text().splitlines()]
    rows[1].pop("applied_action")
    state["actions"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ResumeManifestError, match="applied_action"):
        load_manifest(state["manifest"])


def test_action_hash_and_native_checkpoint_are_authenticated(state):
    state["actions"].write_text(state["actions"].read_text() + "\n")
    with pytest.raises(ResumeManifestError, match="hash"):
        load_manifest(state["manifest"])
    # Restore the action bytes, then alter the native checkpoint independently.
    state["actions"].write_text(
        json.dumps({"status": "completed", "control_step": 1, "action": {"x": 1}, "applied_action": {"x": 1}})
        + "\n"
        + json.dumps({"status": "completed", "control_step": 2, "action": {"x": 2}, "applied_action": {"x": 2}})
        + "\n"
    )
    state["native"].write_text("changed\n")
    with pytest.raises(ResumeManifestError, match="hash"):
        load_manifest(state["manifest"])


def test_workspace_symlink_is_rejected(state, tmp_path):
    linked = state["workspace"] / "escape"
    linked.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ResumeManifestError, match="symlink"):
        load_manifest(state["manifest"])


def test_codex_auth_file_is_rejected(state):
    (state["codex_home"] / "auth.json").write_text("{}\n")
    with pytest.raises(ResumeManifestError, match="auth.json"):
        load_manifest(state["manifest"])


def test_native_restore_receipt_is_required_and_bound_to_step(state):
    state["restore_receipt"].unlink()
    with pytest.raises(ResumeManifestError, match="missing"):
        load_manifest(state["manifest"])
    state["restore_receipt"].write_text(
        json.dumps({"verified": True, "task": "align_blocks", "seed": 0, "episode_id": "episode-abc", "control_step": 1}) + "\n"
    )
    with pytest.raises(ResumeManifestError, match="does not match"):
        load_manifest(state["manifest"])


def test_path_outside_case_is_rejected(state, tmp_path):
    document = json.loads(state["manifest"].read_text())
    outside = tmp_path / "outside-actions.jsonl"
    outside.write_text(state["actions"].read_text())
    document["paths"]["actions"] = str(outside)
    write_manifest(state["manifest"], document)
    with pytest.raises(ResumeManifestError, match="escapes"):
        load_manifest(state["manifest"])


def test_manifest_replace_is_atomic_and_rejects_destination_symlink(state, tmp_path):
    target = state["case"] / "manifest-copy.json"
    write_manifest(target, json.loads(state["manifest"].read_text()))
    assert json.loads(target.read_text())["schema_version"] == 1
    link = state["case"] / "manifest-link.json"
    link.symlink_to(target)
    with pytest.raises(ResumeManifestError, match="symlink"):
        write_manifest(link, json.loads(state["manifest"].read_text()))
