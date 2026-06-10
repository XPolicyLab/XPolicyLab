import json

import pytest
from pydantic import ValidationError
from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import normalize_execution_error, run_dispatch
from robodojo.env_client import TrialRunnerError
from robodojo.protocol.exceptions import ErrorCode, WsError
from robodojo.schemas import DispatchPayload


def _fake_trial_runner(*, result=None, exc=None):
    def runner(_dispatch, trial_run, _evaluation_id):
        if exc is not None:
            raise exc
        return result or {
            "trial_id": trial_run["trial_id"],
            "steps": 3,
            "eval_env": "debug",
            "policy_name": "demo_policy",
            "actions": [],
        }

    return runner


def test_run_dispatch_accepts_platform_dispatch():
    dispatch = DispatchPayload.model_validate(platform_dispatch())

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        trial_index=1,
    )

    assert exit_code == 0
    assert summary["planned_trial_runs"] == 1
    assert summary["trial_runs"][0]["trial_id"] == "case-1-r01"
    assert summary["trial_runs"][0]["trial_index"] == 1
    assert summary["trial_runs"][0]["case_meta"] == {
        "action_case_id": "case-1",
        "trial_index": 1,
        "instruction": "pick up the cube",
    }
    assert summary["trial_runs"][0]["finish_url"].endswith("/trials/1/finish/")


def test_run_dispatch_summary_includes_platform_fields():
    dispatch = DispatchPayload.model_validate(platform_dispatch())

    _, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        trial_index=1,
    )

    assert summary["planned_trial_runs"] == 1
    assert summary["trial_count"] == 4
    assert summary["policy_server_url"] == "ws://127.0.0.1:19000"


def test_dispatch_payload_rejects_malformed_trials():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = {"case": "not-a-list"}

    with pytest.raises(ValidationError, match="trials"):
        DispatchPayload.model_validate(payload)


def test_dispatch_payload_requires_action_case_id_for_each_trial():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = [{}]

    with pytest.raises(ValidationError, match="action_case_id"):
        DispatchPayload.model_validate(payload)


def test_dispatch_payload_accepts_platform_dispatch_shape():
    payload = platform_dispatch(
        evaluation_plan={
            "repeat_count": 1,
            "task": {
                "id": "lift-cube",
                "name": "Lift Cube",
                "env_cfg_type": "default",
                "control_frequency_hz": 30,
            },
            "trials": [
                {
                    "trial_id": "t1-r01",
                    "trial_index": 1,
                    "action_case_id": "t1",
                    "finish_url": (
                        "https://example.test/api/v1/internal/eval/"
                        "eval-django/trials/1/finish/"
                    ),
                }
            ],
        },
    )
    dispatch = DispatchPayload.model_validate(payload)
    assert dispatch.policy_server_url == "ws://127.0.0.1:19000"
    assert dispatch.evaluation_plan.task is not None
    assert dispatch.evaluation_plan.task.id == "lift-cube"
    assert dispatch.evaluation_plan.trials[0].finish_url.endswith("/trials/1/finish/")


def test_dispatch_payload_rejects_empty_trial_plan():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = []

    with pytest.raises(ValidationError, match="trials"):
        DispatchPayload.model_validate(payload)


def test_run_dispatch_rejects_webhook_without_policy_trial(tmp_path):
    dispatch = DispatchPayload.model_validate(platform_dispatch())

    with pytest.raises(ValueError, match="notify_webhook requires run_policy_trials"):
        run_dispatch(
            dispatch,
            evaluation_id="eval-1",
            artifact_dir=tmp_path / "artifacts",
            notify_webhook=True,
            run_policy_trials=False,
            trial_index=1,
        )


def test_run_dispatch_includes_policy_error_in_trial_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        captured["finish_url"] = kwargs.get("finish_url")
        return {"webhook": {"finish_url": kwargs.get("finish_url"), "status_code": 200}}

    monkeypatch.setattr("robodojo.publish.pipeline.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(exc=RuntimeError("policy down")),
    )

    assert exit_code == 1
    assert summary["status"] == "failed"
    assert summary["error_summary"] == "policy down"
    assert summary["error"] == {"code": "internal", "message": "policy down"}
    assert captured["error"] == {"code": "internal", "message": "policy down"}
    assert str(captured["finish_url"]).endswith("/trials/1/finish/")
    manifest = json.loads(
        ((tmp_path / "artifacts") / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error_summary"] == "policy down"


def test_normalize_execution_error_maps_ws_error():
    error = normalize_execution_error(
        WsError(ErrorCode.TIMEOUT, "infer timed out", details={"step": 3})
    )
    assert error == {
        "code": "timeout",
        "message": "infer timed out",
        "details": {"step": 3},
    }


def test_normalize_execution_error_preserves_trial_runner_error():
    error = normalize_execution_error(
        TrialRunnerError(
            "policy unreachable",
            error={"code": "trial_failed", "message": "policy unreachable"},
        )
    )
    assert error == {
        "code": "trial_failed",
        "message": "policy unreachable",
    }


def test_run_dispatch_maps_trial_runner_error_to_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.publish.pipeline.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(
            exc=TrialRunnerError(
                "policy unreachable",
                error={"code": "trial_failed", "message": "policy unreachable"},
            )
        ),
    )

    assert exit_code == 1
    assert summary["error"] == {
        "code": "trial_failed",
        "message": "policy unreachable",
    }
    assert captured["error"] == summary["error"]


def test_run_dispatch_maps_ws_error_to_failed_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.publish.pipeline.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(
            exc=WsError(ErrorCode.INFER_FAILED, "policy rejected frame")
        ),
    )

    assert exit_code == 1
    assert summary["error"] == {
        "code": "infer_failed",
        "message": "policy rejected frame",
    }
    assert captured["error"] == summary["error"]


def test_run_dispatch_maps_not_implemented_error_to_failed_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.publish.pipeline.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(
            exc=NotImplementedError("unsupported RoboDojo model call: set_language")
        ),
    )

    assert exit_code == 1
    assert summary["error"] == {
        "code": "internal",
        "message": "unsupported RoboDojo model call: set_language",
    }
    assert captured["error"] == summary["error"]


def test_run_dispatch_fail_dispatch_still_notifies_on_unexpected_crash(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import robodojo.publish.pipeline as publish_pipeline

    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}
    write_calls = {"count": 0}
    original_write_artifacts = publish_pipeline.write_artifacts

    def flaky_write_artifacts(*args, **kwargs):
        write_calls["count"] += 1
        if write_calls["count"] == 1:
            raise RuntimeError("artifact write crashed")
        return original_write_artifacts(*args, **kwargs)

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.publish.pipeline.write_artifacts", flaky_write_artifacts)
    monkeypatch.setattr("robodojo.publish.pipeline.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(),
    )

    assert exit_code == 1
    assert summary["status"] == "failed"
    assert summary["error"] == {"code": "internal", "message": "artifact write crashed"}
    assert captured["error"] == summary["error"]


def test_run_dispatch_summary_serializes_policy_result():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=None,
        upload_s3=False,
        notify_webhook=False,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=_fake_trial_runner(),
    )

    assert exit_code == 0
    json.dumps(summary)
    assert summary["policy_results"][0]["steps"] == 3