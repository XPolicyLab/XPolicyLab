import io
import json

import numpy as np
import pytest
from pydantic import ValidationError
from robodojo_fixtures import platform_dispatch

from robodojo.eval_runner import main, normalize_execution_error, run_dispatch
from robodojo.protocol.exceptions import ErrorCode, WsError
from robodojo.schemas import DispatchPayload


def test_eval_runner_main_accepts_file_payload(tmp_path):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(platform_dispatch()), encoding="utf-8")
    stdout = io.StringIO()

    exit_code = main(
        ["--dispatch-payload", str(path), "--trial-index", "1"],
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["planned_trial_runs"] == 1
    assert summary["trial_runs"][0]["trial_id"] == "case-1-r01"
    assert summary["trial_runs"][0]["trial_index"] == 1
    assert summary["trial_runs"][0]["case_meta"] == {
        "action_case_id": "case-1",
        "trial_index": 1,
    }
    assert summary["trial_runs"][0]["finish_url"].endswith("/trials/1/finish/")


def test_eval_runner_main_accepts_stdin_payload():
    stdout = io.StringIO()

    exit_code = main(
        ["--dispatch-payload", "-", "--trial-index", "1"],
        stdin=io.StringIO(json.dumps(platform_dispatch())),
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["planned_trial_runs"] == 1
    assert summary["trial_count"] == 4
    assert summary["policy_server_url"] == "ws://127.0.0.1:19000"


def test_eval_runner_rejects_malformed_dispatch_payload():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = {"case": "not-a-list"}

    with pytest.raises(ValidationError, match="trials"):
        main(
            ["--dispatch-payload", "-", "--trial-index", "1"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )


def test_eval_runner_requires_action_case_id_for_each_trial():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = [{}]

    with pytest.raises(ValidationError, match="action_case_id"):
        main(
            ["--dispatch-payload", "-", "--trial-index", "1"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )


def test_eval_runner_accepts_platform_dispatch_shape():
    payload = platform_dispatch(
        evaluation_id="eval-django",
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


def test_eval_runner_rejects_empty_trial_plan():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = []

    with pytest.raises(ValidationError, match="trials"):
        main(
            ["--dispatch-payload", "-", "--trial-index", "1"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )


def test_eval_runner_rejects_webhook_without_policy_trial(tmp_path):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(platform_dispatch()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--dispatch-payload",
                str(path),
                "--artifact-dir",
                str(tmp_path / "artifacts"),
                "--trial-index",
                "1",
            ],
            stdout=io.StringIO(),
        )

    assert exc_info.value.code == 2


def test_run_dispatch_includes_policy_error_in_trial_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fail_policy_trial(**_kwargs):
        raise RuntimeError("policy down")

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        captured["finish_url"] = kwargs.get("finish_url")
        return {"webhook": {"finish_url": kwargs.get("finish_url"), "status_code": 200}}

    monkeypatch.setattr("robodojo.eval_runner.run_policy_trial", fail_policy_trial)
    monkeypatch.setattr("robodojo.eval_runner.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
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


def test_run_dispatch_maps_ws_error_to_failed_webhook(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fail_policy_trial(**_kwargs):
        raise WsError(ErrorCode.INFER_FAILED, "policy rejected frame")

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.eval_runner.run_policy_trial", fail_policy_trial)
    monkeypatch.setattr("robodojo.eval_runner.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
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

    def fail_policy_trial(**_kwargs):
        raise NotImplementedError("unsupported RoboDojo model call: set_language")

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr("robodojo.eval_runner.run_policy_trial", fail_policy_trial)
    monkeypatch.setattr("robodojo.eval_runner.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
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
    import robodojo.eval_runner as eval_runner_module

    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}
    write_calls = {"count": 0}
    original_write_artifacts = eval_runner_module.write_artifacts

    def flaky_write_artifacts(*args, **kwargs):
        write_calls["count"] += 1
        if write_calls["count"] == 1:
            raise RuntimeError("artifact write crashed")
        return original_write_artifacts(*args, **kwargs)

    def fake_publish_artifacts(*_args, **kwargs):
        captured["error"] = kwargs.get("error")
        return {"webhook": {"status_code": 200}}

    monkeypatch.setattr(
        "robodojo.eval_runner.run_policy_trial",
        lambda **_kwargs: {"trial_id": "case-1-r01", "actions": []},
    )
    monkeypatch.setattr("robodojo.eval_runner.write_artifacts", flaky_write_artifacts)
    monkeypatch.setattr("robodojo.eval_runner.publish_artifacts", fake_publish_artifacts)

    exit_code, summary = run_dispatch(
        dispatch,
        artifact_dir=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        run_policy_trials=True,
        trial_index=1,
    )

    assert exit_code == 1
    assert summary["status"] == "failed"
    assert summary["error"] == {"code": "internal", "message": "artifact write crashed"}
    assert captured["error"] == summary["error"]


def test_run_dispatch_summary_serializes_numpy_policy_actions(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_policy_trial(**_kwargs):
        return {
            "trial_id": "case-1-r01",
            "actions": [
                {
                    "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                    "left_ee_joint_state": np.zeros(1, dtype=np.float32),
                }
            ],
        }

    monkeypatch.setattr("robodojo.eval_runner.run_policy_trial", fake_policy_trial)

    dispatch = DispatchPayload.model_validate(platform_dispatch())
    exit_code, summary = run_dispatch(
        dispatch,
        artifact_dir=None,
        upload_s3=False,
        notify_webhook=False,
        run_policy_trials=True,
        trial_index=1,
    )

    assert exit_code == 0
    json.dumps(summary)
    assert summary["policy_results"][0]["actions"][0]["left_arm_joint_state"] == [
        0.0
    ] * 6
