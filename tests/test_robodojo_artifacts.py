import json
from pathlib import Path

from robodojo_fixtures import platform_dispatch

from robodojo.artifacts import (
    ArtifactWriter,
)
from robodojo.eval_runner import build_trial_runs, main, write_artifacts
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
    trial_runs = build_trial_runs(dispatch)
    artifact_dir = tmp_path / "artifacts"

    paths = write_artifacts(dispatch, trial_runs, artifact_dir)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    events = _load_jsonl(artifact_dir / "events.jsonl")

    assert manifest["evaluation_id"] == "eval-1"
    assert manifest["status"] == "planned"
    assert manifest["policy_server_url"] == "ws://127.0.0.1:19000"
    assert len(manifest["trials"]) == 4
    assert manifest["files"]["logs"] == "logs/runner.log"
    assert (artifact_dir / "logs" / "runner.log").exists()
    assert metrics["summary"]["trial_count"] == 4
    assert metrics["summary"]["not_executed"] == 4
    assert {event["event"] for event in events} == {
        "run_started",
        "trial_registered",
        "run_finished",
    }
    assert sum(1 for event in events if event["event"] == "trial_registered") == 4

    for trial_run in trial_runs:
        trial_id = str(trial_run["trial_id"])
        video_path = artifact_dir / f"videos/{trial_id}.mp4"
        assert video_path.exists()
        assert (
            manifest["trials"][0]["video_key"]
            == f"videos/{trial_runs[0]['trial_id']}.mp4"
        )

    assert paths["manifest"].endswith("manifest.json")
    assert paths["events"].endswith("events.jsonl")


def test_eval_runner_writes_artifacts_with_flag(tmp_path):
    dispatch = _dispatch_payload()
    dispatch_path = tmp_path / "dispatch.json"
    dispatch_path.write_text(dispatch.model_dump_json(), encoding="utf-8")
    artifact_dir = tmp_path / "out"

    import io

    stdout = io.StringIO()
    exit_code = main(
        [
            "--dispatch-payload",
            str(dispatch_path),
            "--artifact-dir",
            str(artifact_dir),
            "--no-s3",
            "--no-webhook",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["planned_trial_runs"] == 4
    assert summary["artifacts"]["artifact_dir"] == str(artifact_dir)
    assert (artifact_dir / "manifest.json").exists()


def test_record_trial_lifecycle_updates_metrics(tmp_path):
    dispatch = _dispatch_payload()
    writer = ArtifactWriter(tmp_path / "artifacts", dispatch)
    writer.setup()
    try:
        writer.emit_event("run_started")
        writer.register_trials(
            [
                {
                    "trial_id": "case-1-r01",
                    "action_case_id": "case-1",
                    "trial_index": 0,
                    "repeat_index": 1,
                    "case_meta": {"action_case_id": "case-1", "seed": 1},
                }
            ]
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
