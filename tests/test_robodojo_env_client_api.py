from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs
from robodojo.env_client import (
    EnvClientBaselineConfig,
    HealthResponse,
    TrialRunRequest,
    TrialRunResponse,
    trial_request_to_deploy_cfg,
)
from robodojo.schemas import DispatchPayload


def _baseline(**overrides: object) -> EnvClientBaselineConfig:
    return EnvClientBaselineConfig.model_validate(
        {
            "dataset_name": "demo_dataset",
            "task_name": "lift-cube",
            "env_cfg_type": "arx_x5",
            "policy_name": "demo_policy",
            "host": "localhost",
            "port": 19000,
            **overrides,
        }
    )


def _trial_request(**overrides: object) -> TrialRunRequest:
    return TrialRunRequest.model_validate(
        {
            "evaluation_id": "eval-1",
            "trial_id": "case-1-r01",
            "trial_index": 1,
            "action_case_id": "case-1",
            "policy_server_url": "ws://127.0.0.1:19000",
            "case_meta": {"instruction": "pick up the cube", "trial_index": 1},
            **overrides,
        }
    )


def test_health_response_round_trip_json():
    response = HealthResponse(
        policy_name="demo_policy",
        eval_env="debug",
        deploy_yml="/path/to/deploy.yml",
    )
    payload = json.loads(response.model_dump_json())
    assert payload == {
        "ok": True,
        "policy_name": "demo_policy",
        "eval_env": "debug",
        "deploy_yml": "/path/to/deploy.yml",
        "last_trial_id": None,
    }
    assert HealthResponse.model_validate(payload) == response


def test_trial_run_request_requires_core_fields():
    with pytest.raises(ValidationError):
        TrialRunRequest.model_validate({"evaluation_id": "eval-1"})


def test_trial_run_response_completed_and_failed_shapes():
    completed = TrialRunResponse(
        status="completed",
        trial_id="case-1-r01",
        steps=5,
        eval_env="debug",
        policy_name="demo_policy",
    )
    failed = TrialRunResponse(
        status="failed",
        trial_id="case-1-r01",
        error={"code": "internal", "message": "boom"},
    )

    assert completed.model_dump(exclude_none=True) == {
        "status": "completed",
        "trial_id": "case-1-r01",
        "steps": 5,
        "eval_env": "debug",
        "policy_name": "demo_policy",
    }
    assert failed.error == {"code": "internal", "message": "boom"}


def test_trial_request_to_deploy_cfg_uses_baseline_and_request_ids():
    deploy_cfg = trial_request_to_deploy_cfg(_trial_request(), _baseline())
    expected = _baseline().model_dump()
    expected.update(
        {
            "evaluation_id": "eval-1",
            "trial_id": "case-1-r01",
            "action_case_id": "case-1",
            "policy_server_url": "ws://127.0.0.1:19000",
            "host": "127.0.0.1",
            "port": 19000,
            "repeat_index": None,
        }
    )
    assert deploy_cfg == expected


def test_trial_request_to_deploy_cfg_applies_case_meta_and_overrides():
    request = _trial_request(
        case_meta={
            "env_cfg_type": "aloha",
            "task_name": "stack-bowls",
            "policy_name": "demo-policy",
            "eval_batch": True,
            "repeat_index": 2,
        },
        overrides={"eval_episode_num": 1},
    )

    deploy_cfg = trial_request_to_deploy_cfg(request, _baseline())

    assert deploy_cfg["env_cfg_type"] == "aloha"
    assert deploy_cfg["task_name"] == "stack-bowls"
    assert deploy_cfg["policy_name"] == "demo_policy"
    assert deploy_cfg["eval_batch"] is True
    assert deploy_cfg["eval_episode_num"] == 1
    assert deploy_cfg["repeat_index"] == 2


def test_trial_request_to_deploy_cfg_accepts_baseline_mapping():
    deploy_cfg = trial_request_to_deploy_cfg(
        _trial_request(),
        _baseline(
            dataset_name="mapped_dataset",
            task_name="mapped_task",
            policy_name="A1",
            eval_episode_num=3,
        ).model_dump(),
    )

    assert deploy_cfg["dataset_name"] == "mapped_dataset"
    assert deploy_cfg["policy_name"] == "A1"


def test_trial_request_to_deploy_cfg_from_platform_dispatch_trial_run():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    request = TrialRunRequest(
        evaluation_id="eval-1",
        trial_id=str(trial_run["trial_id"]),
        trial_index=int(trial_run["trial_index"]),
        action_case_id=str(trial_run["action_case_id"]),
        policy_server_url=dispatch.policy_server_url,
        case_meta=dict(trial_run["case_meta"]),
        overrides={"eval_episode_num": 1},
    )

    deploy_cfg = trial_request_to_deploy_cfg(request, _baseline(task_name="lift-cube"))
    expected = _baseline(task_name="lift-cube").model_dump()
    expected.update(
        {
            "evaluation_id": "eval-1",
            "trial_id": "case-1-r01",
            "action_case_id": "case-1",
            "policy_server_url": "ws://127.0.0.1:19000",
            "host": "127.0.0.1",
            "port": 19000,
            "eval_episode_num": 1,
            "repeat_index": None,
        }
    )
    assert deploy_cfg == expected
