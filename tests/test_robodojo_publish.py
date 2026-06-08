import io
import json
from pathlib import Path

from robodojo_fixtures import platform_dispatch

from robodojo.eval_runner import main, publish_artifacts, write_artifacts
from robodojo.schemas import DispatchPayload

FINISH_URL = (
    "https://example.test/api/v1/internal/eval/eval-1/trials/1/finish/"
)


def _dispatch_payload() -> DispatchPayload:
    return DispatchPayload.model_validate(
        platform_dispatch(
            evaluation_plan={
                "repeat_count": 1,
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "action_case_id": "case-1",
                        "trial_index": 1,
                        "finish_url": FINISH_URL,
                    }
                ],
            },
        )
    )


def test_publish_artifacts_uploads_and_webhooks(tmp_path):
    dispatch = _dispatch_payload()
    artifact_dir = tmp_path / "artifacts"
    trial_run = {
        "trial_id": "case-1-r01",
        "action_case_id": "case-1",
        "trial_index": 1,
        "case_meta": {"action_case_id": "case-1"},
    }
    artifact_paths = write_artifacts(dispatch, trial_run, artifact_dir)
    uploads: list[str] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append(key)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 200

    published = publish_artifacts(
        dispatch,
        artifact_paths,
        run_status="completed",
        upload_s3=True,
        notify_webhook=True,
        finish_url=FINISH_URL,
        s3_client=object(),
        upload_file=fake_upload,
        webhook_secret="secret",
        webhook_opener=lambda request, timeout=30: FakeResponse(),
    )

    assert published["s3"]["manifest_s3_key"] == "evaluations/eval-1/manifest.json"
    assert published["webhook"]["status_code"] == 200
    assert uploads[-1] == "evaluations/eval-1/manifest.json"


def test_eval_runner_publishes_with_artifact_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "robodojo.eval_runner.run_policy_trial",
        lambda **_kwargs: {
            "trial_id": "case-1-r01",
            "actions": [{"arm_joint_state": [0.0] * 7, "ee_joint_state": [0.0]}],
        },
    )
    monkeypatch.setattr(
        "robodojo.eval_runner.publish_artifacts",
        lambda *args, **kwargs: {
            "s3": {
                "bucket": "robodojo-artifacts",
                "prefix": "evaluations/eval-1/",
                "manifest_s3_key": "evaluations/eval-1/manifest.json",
                "uploaded_count": 3,
            },
            "webhook": {
                "finish_url": FINISH_URL,
                "status_code": 200,
            },
        },
    )

    dispatch = _dispatch_payload()
    dispatch_path = tmp_path / "dispatch.json"
    dispatch_path.write_text(dispatch.model_dump_json(), encoding="utf-8")
    artifact_dir = tmp_path / "out"
    stdout = io.StringIO()

    exit_code = main(
        [
            "--dispatch-payload",
            str(dispatch_path),
            "--artifact-dir",
            str(artifact_dir),
            "--trial-index",
            "1",
            "--run-policy-trials",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["status"] == "completed"
    assert (
        summary["published"]["s3"]["manifest_s3_key"]
        == "evaluations/eval-1/manifest.json"
    )
    assert summary["published"]["webhook"]["status_code"] == 200


def test_failure_webhook_includes_error_payload(tmp_path):
    dispatch = _dispatch_payload()
    artifact_dir = tmp_path / "artifacts"
    trial_run = {
        "trial_id": "case-1-r01",
        "action_case_id": "case-1",
        "trial_index": 1,
        "case_meta": {"action_case_id": "case-1"},
    }
    artifact_paths = write_artifacts(dispatch, trial_run, artifact_dir)
    captured: dict = {}

    def capture_webhook(request, timeout=30):
        captured["body"] = json.loads(request.data.decode("utf-8"))

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getcode(self):
                return 200

        return FakeResponse()

    publish_artifacts(
        dispatch,
        artifact_paths,
        run_status="failed",
        upload_s3=False,
        notify_webhook=True,
        error={"code": "failed", "message": "webhook down"},
        finish_url=FINISH_URL,
        webhook_secret="secret",
        webhook_opener=capture_webhook,
    )

    assert captured["body"]["status"] == "failed"
    assert captured["body"]["error"] == {
        "code": "failed",
        "message": "webhook down",
    }
