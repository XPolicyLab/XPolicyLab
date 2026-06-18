import json
from pathlib import Path

from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs, run_dispatch
from robodojo.publish import ArtifactWriter, write_artifacts
from robodojo.schemas import DispatchPayload


def _dispatch_payload() -> DispatchPayload:
    return DispatchPayload.model_validate(platform_dispatch())


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_artifact_writer_creates_layout(tmp_path):
    dispatch = _dispatch_payload()
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    artifact_dir = tmp_path / "artifacts"

    paths = write_artifacts(dispatch, trial_run, artifact_dir, evaluation_id="eval-1")

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    events = _load_jsonl(artifact_dir / "events.jsonl")

    assert manifest["evaluation_id"] == "eval-1"
    assert manifest["status"] == "planned"
    assert manifest["policy_server_url"] == "ws://127.0.0.1:19000"
    assert len(manifest["trials"]) == 1
    assert manifest["files"]["logs"] == "logs/runner.log"
    assert (artifact_dir / "logs" / "runner.log").exists()
    assert metrics["summary"]["trial_count"] == 1
    assert metrics["summary"]["not_executed"] == 1
    assert {event["event"] for event in events} == {
        "run_started",
        "trial_registered",
        "run_finished",
    }
    assert sum(1 for event in events if event["event"] == "trial_registered") == 1

    trial_id = str(trial_run["trial_id"])
    video_path = artifact_dir / f"videos/{trial_id}.mp4"
    assert video_path.exists()
    assert manifest["trials"][0]["video_key"] == f"videos/{trial_id}.mp4"

    assert paths["manifest"].endswith("manifest.json")
    assert paths["events"].endswith("events.jsonl")


def test_write_artifacts_skips_video_when_stage_recordings_disabled(tmp_path, monkeypatch):
    dispatch = _dispatch_payload()
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    artifact_dir = tmp_path / "artifacts"
    called = {"stage": False}

    def fail_if_called(*args, **kwargs):
        called["stage"] = True
        raise AssertionError("_stage_trial_recording should not run")

    monkeypatch.setattr(
        "robodojo.publish.pipeline._stage_trial_recording", fail_if_called
    )

    write_artifacts(
        dispatch,
        trial_run,
        artifact_dir,
        evaluation_id="eval-1",
        stage_recordings=False,
    )

    trial_id = str(trial_run["trial_id"])
    assert not (artifact_dir / f"videos/{trial_id}.mp4").exists()
    assert called["stage"] is False
    assert (artifact_dir / "manifest.json").exists()


def test_run_dispatch_writes_artifacts_without_policy_trials(tmp_path):
    dispatch = _dispatch_payload()
    artifact_dir = tmp_path / "out"

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=artifact_dir,
        upload_s3=False,
        notify_webhook=False,
        run_policy_trials=False,
        trial_index=1,
    )

    assert exit_code == 0
    assert summary["planned_trial_runs"] == 1
    assert summary["artifacts"]["artifact_dir"] == str(artifact_dir)
    assert (artifact_dir / "manifest.json").exists()
    trial_id = str(build_trial_runs(dispatch, evaluation_id="eval-1")[0]["trial_id"])
    assert not (artifact_dir / f"videos/{trial_id}.mp4").exists()


def test_record_trial_lifecycle_updates_metrics(tmp_path):
    dispatch = _dispatch_payload()
    writer = ArtifactWriter(
        tmp_path / "artifacts",
        evaluation_id="eval-1",
        dispatch=dispatch,
    )
    writer.setup()
    try:
        writer.emit_event("run_started")
        writer.register_trial(
            {
                "trial_id": "case-1-r01",
                "action_case_id": "case-1",
                "trial_index": 1,
                "case_meta": {"action_case_id": "case-1", "seed": 1},
            }
        )
        writer.record_trial_start("case-1-r01")
        writer.record_trial_end(
            "case-1-r01",
            status="completed",
            metrics={"success": True, "steps": 12},
        )
        writer.write_video_placeholder("case-1-r01")
        writer.finalize(status="completed")
    finally:
        writer.close()

    metrics = json.loads((writer.root_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["summary"]["completed"] == 1
    assert metrics["trials"][0]["metrics"] == {"success": True, "steps": 12}
    events = _load_jsonl(writer.root_dir / "events.jsonl")
    assert "trial_started" in {event["event"] for event in events}
    assert "trial_finished" in {event["event"] for event in events}


def test_write_artifacts_records_policy_results(tmp_path):
    dispatch = _dispatch_payload()
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    artifact_dir = tmp_path / "artifacts"

    write_artifacts(
        dispatch,
        trial_run,
        artifact_dir,
        evaluation_id="eval-1",
        run_status="done",
        policy_result={
            "trial_id": trial_run["trial_id"],
            "actions": [{"arm_joint_state": [0.0] * 7, "ee_joint_state": [0.0]}],
        },
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    events = _load_jsonl(artifact_dir / "events.jsonl")

    assert manifest["trials"][0]["status"] == "completed"
    assert metrics["summary"]["completed"] == 1
    assert metrics["summary"]["success_rate"] == 100.0
    assert metrics["trials"][0]["metrics"]["action_count"] == 1
    assert "trial_finished" in {event["event"] for event in events}
