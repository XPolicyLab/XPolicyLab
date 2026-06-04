import io
import json

import pytest
from pydantic import ValidationError
from robodojo_fixtures import platform_dispatch

from robodojo.eval_runner import main
from robodojo.schemas import DispatchPayload


def test_eval_runner_main_accepts_file_payload(tmp_path):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(platform_dispatch()), encoding="utf-8")
    stdout = io.StringIO()

    exit_code = main(["--dispatch-payload", str(path)], stdout=stdout)

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["planned_trial_runs"] == 4
    assert summary["trial_runs"][0]["trial_id"] == "case-1-r01"
    assert summary["trial_runs"][0]["case_meta"] == {
        "action_case_id": "case-1",
        "seed": 1,
    }


def test_eval_runner_main_accepts_stdin_payload():
    stdout = io.StringIO()

    exit_code = main(
        ["--dispatch-payload", "-"],
        stdin=io.StringIO(json.dumps(platform_dispatch())),
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary["planned_trial_runs"] == 4
    assert summary["trial_count"] == 4
    assert summary["policy_server_url"] == "ws://127.0.0.1:19000"


def test_eval_runner_rejects_malformed_dispatch_payload():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = {"case": "not-a-list"}

    with pytest.raises(ValidationError, match="trials"):
        main(
            ["--dispatch-payload", "-"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )


def test_eval_runner_requires_action_case_id_for_each_trial():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = [{"seed": 1}]

    with pytest.raises(ValidationError, match="action_case_id"):
        main(
            ["--dispatch-payload", "-"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )


def test_eval_runner_accepts_platform_dispatch_shape():
    payload = platform_dispatch(
        evaluation_id="eval-django",
        evaluation_plan={
            "repeat_count": 1,
            "trials": [
                {"trial_id": "t1-r01", "action_case_id": "t1", "repeat_index": 1}
            ],
        },
    )
    dispatch = DispatchPayload.model_validate(payload)
    assert dispatch.policy_server_url == "ws://127.0.0.1:19000"
    assert dispatch.finish_url.endswith("/finish/")


def test_eval_runner_rejects_empty_trial_plan():
    payload = platform_dispatch()
    payload["evaluation_plan"]["trials"] = []

    with pytest.raises(ValidationError, match="trials"):
        main(
            ["--dispatch-payload", "-"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )
