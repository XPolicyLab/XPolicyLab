import io
import json

import pytest
from pydantic import ValidationError

from robodojo.eval_runner import main


def _dispatch_payload():
    return {
        "evaluation_id": "eval-1",
        "policy_server": {
            "url": "ws://127.0.0.1:19000",
            "connection_mode": "direct",
        },
        "evaluation_plan": {
            "task": "lift-cube",
            "repeat_count": 2,
            "trials": [
                {"action_case_id": "case-1", "seed": 1},
                {"action_case_id": "case-2", "seed": 2},
            ],
        },
        "artifact": {
            "s3_bucket": "robodojo-artifacts",
            "s3_prefix": "eval-1/",
        },
        "webhook": {
            "finish_url": "https://example.test/finish",
            "hmac_secret_ref": "secret/ref",
        },
    }


def test_eval_runner_main_accepts_file_payload(tmp_path):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(_dispatch_payload()), encoding="utf-8")
    stdout = io.StringIO()

    exit_code = main(["--dispatch-payload", str(path)], stdout=stdout)

    assert exit_code == 0
    assert json.loads(stdout.getvalue())["planned_trial_runs"] == 4


def test_eval_runner_main_accepts_stdin_payload():
    stdout = io.StringIO()

    exit_code = main(
        ["--dispatch-payload", "-"],
        stdin=io.StringIO(json.dumps(_dispatch_payload())),
        stdout=stdout,
    )

    assert exit_code == 0
    summary = json.loads(stdout.getvalue())
    assert summary == {
        "connection_mode": "direct",
        "evaluation_id": "eval-1",
        "planned_trial_runs": 4,
        "policy_server_url": "ws://127.0.0.1:19000",
        "repeat_count": 2,
        "status": "loaded",
        "task": "lift-cube",
        "trial_count": 2,
    }


def test_eval_runner_rejects_malformed_dispatch_payload():
    payload = _dispatch_payload()
    payload["evaluation_plan"]["trials"] = {"case": "not-a-list"}

    with pytest.raises(ValidationError, match="trials"):
        main(
            ["--dispatch-payload", "-"],
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
        )
