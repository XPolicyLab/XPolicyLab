import numpy as np
import pytest
from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs
from robodojo.schemas import DispatchPayload
from robodojo.trial import (
    ActionRecorder,
    DebugTrialEnv,
    build_mock_observation,
    build_trial_run_config,
    create_trial_env,
    normalize_policy_name,
    run_eval_episode,
)


def test_build_mock_observation_includes_camera_vision():
    obs = build_mock_observation("arx_x5", instruction="pick cube")
    assert "vision" in obs
    assert "cam_head" in obs["vision"]
    assert obs["vision"]["cam_head"]["color"].shape == (480, 640, 3)
    assert "left_arm_joint_state" in obs["state"]


def test_debug_trial_env_steps_until_limit():
    env = DebugTrialEnv(env_cfg_type="arx_x5", episode_step_limit=2)
    env.reset()
    assert not env.is_episode_end()
    action = {
        "left_arm_joint_state": np.zeros(6, dtype=np.float32),
        "left_ee_joint_state": np.zeros(1, dtype=np.float32),
    }
    env.take_action(action)
    assert not env.is_episode_end()
    env.take_action(action)
    assert env.is_episode_end()


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


def test_create_trial_env_passes_instruction_to_observation():
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
                        "instruction": "open the drawer",
                        "finish_url": "https://example.test/finish/",
                    }
                ],
            },
        )
    )
    config = build_trial_run_config(
        dispatch,
        {
            "trial_id": "case-1-r01",
            "action_case_id": "case-1",
            "trial_index": 1,
            "case_meta": {"action_case_id": "case-1", "trial_index": 1},
        },
        evaluation_id="eval-1",
        eval_env="debug",
    )
    env = create_trial_env(config)
    assert env.get_obs()["instruction"] == "open the drawer"


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


class FakeModelClient:
    def __init__(self):
        self.calls: list[tuple[str | None, object]] = []

    def call(self, func_name=None, obs=None, **kwargs):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            return [{"left_arm_joint_state": np.zeros(6, dtype=np.float32)}]
        return None


def test_run_eval_episode_uses_env_observations():
    env = DebugTrialEnv(env_cfg_type="arx_x5", episode_step_limit=1)
    client = ActionRecorder(FakeModelClient())

    run_eval_episode(
        env,
        client,
        policy_name="demo_policy",
        eval_batch=False,
    )

    assert client.steps == 1
    assert len(client.actions) == 1
    assert client._client.calls[0] == ("reset", None)


def test_create_trial_env_sim_requires_factory():
    config = build_trial_run_config(
        DispatchPayload.model_validate(platform_dispatch()),
        {
            "trial_id": "case-1-r01",
            "action_case_id": "case-1",
            "case_meta": {},
        },
        evaluation_id="eval-1",
        eval_env="sim",
    )
    with pytest.raises(RuntimeError, match="simulation factory"):
        create_trial_env(config)
