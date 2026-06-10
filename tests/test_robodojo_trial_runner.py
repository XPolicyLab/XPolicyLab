from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs
from robodojo.schemas import DispatchPayload
from robodojo.trial import build_trial_run_config, normalize_policy_name


def test_normalize_policy_name():
    assert normalize_policy_name("demo-policy") == "demo_policy"


def test_build_trial_runs_propagates_instruction_from_dispatch_trial():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    assert trial_run["case_meta"]["instruction"] == "pick up the cube"


def test_build_trial_run_config_reads_instruction_from_dispatch_trial():
    dispatch = DispatchPayload.model_validate(
        platform_dispatch(
            evaluation_plan={
                "repeat_count": 1,
                "task": {"id": "lift-cube", "env_cfg_type": "arx_x5"},
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "trial_index": 1,
                        "action_case_id": "case-1",
                        "instruction": "stack the bowls",
                        "finish_url": "https://example.test/finish/",
                    }
                ],
            },
        )
    )
    trial_run = {
        "trial_id": "case-1-r01",
        "action_case_id": "case-1",
        "trial_index": 1,
        "case_meta": {"action_case_id": "case-1", "trial_index": 1},
        "env_cfg_type": "arx_x5",
    }
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
    )
    assert config.instruction == "stack the bowls"
    assert config.case_meta["instruction"] == "stack the bowls"


def test_build_trial_run_config_reads_dispatch_fields():
    dispatch = DispatchPayload.model_validate(
        platform_dispatch(
            model_name="demo-policy",
            task_id="lift-cube",
            eval_env="debug",
            evaluation_plan={
                "repeat_count": 1,
                "task": {"id": "lift-cube", "env_cfg_type": "arx_x5"},
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "trial_index": 1,
                        "action_case_id": "case-1",
                        "finish_url": "https://example.test/finish/",
                        "instruction": "pick",
                    }
                ],
            },
        )
    )
    trial_run = {
        "trial_id": "case-1-r01",
        "action_case_id": "case-1",
        "case_meta": {"instruction": "pick"},
        "env_cfg_type": "arx_x5",
    }
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
    )
    assert config.policy_name == "demo_policy"
    assert config.task_name == "lift-cube"
    assert config.eval_env == "debug"
    assert config.instruction == "pick"
