"""CPU-only contract tests; no checkpoint download or GPU use."""

import copy

import numpy as np
import pytest

from XPolicyLab.policy.WorldLA import deploy
from XPolicyLab.policy.WorldLA.model import (
    Model,
    actions_to_xpolicylab,
    observation_from_xpolicylab,
    state_fields,
)


FIELDS = state_fields({"arm_dim": [6, 6], "ee_dim": [1, 1]})


def observation(env_idx=0, value=0):
    return {
        "env_idx": env_idx,
        "instruction": "put the object in the basket",
        "state": {name: np.full(size, value, np.float32) for name, size in FIELDS},
        "vision": {
            name: {"color": np.zeros((8, 9, 3), np.uint8)}
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
    }


class FakePolicy:
    def __init__(self, root, seed):
        self.root, self.model = root, object()
        self.config = {"deployment": {"execute_k": 10, "old_plan_weight": 0.75}}
        self.reset(seed=seed)

    def reset(self, *, seed):
        self.seed, self.previous_plan = seed, None

    def plan(self, obs):
        value = obs["state"][0]
        plan = np.full((50, 14), value, np.float32)
        plan[:, 0] += np.arange(50)
        if self.previous_plan is not None:
            plan[:10] = 0.75 * self.previous_plan[10:20] + 0.25 * plan[:10]
        self.previous_plan = plan.copy()
        return plan[:10]


@pytest.fixture
def model(monkeypatch, tmp_path):
    monkeypatch.setattr("worldla.policy.WorldLAPolicy", FakePolicy)
    monkeypatch.setattr(
        "XPolicyLab.utils.process_data.get_robot_action_dim_info",
        lambda name: {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    )
    return Model(
        {"model_dir": str(tmp_path), "env_cfg_type": "arx_x5", "action_type": "joint"}
    )


def test_rgb_layout_and_shared_dimensions():
    obs = observation()
    obs["vision"]["cam_head"]["color"][0, 0] = [255, 19, 4]
    obs["vision"]["cam_head"]["color"].flags.writeable = False
    out = observation_from_xpolicylab(obs, FIELDS)
    np.testing.assert_array_equal(out["images"]["cam_high"][:, 0, 0], [255, 19, 4])
    assert out["state"].shape == (sum(size for _, size in FIELDS),)
    expected = np.arange(140, dtype=np.float32).reshape(10, 14)
    actions = actions_to_xpolicylab(expected, 10, FIELDS)
    np.testing.assert_array_equal(
        np.concatenate([actions[0][name] for name, _ in FIELDS]), expected[0]
    )


@pytest.mark.parametrize("bad", ["encoded", "float", "shape", "nan", "instruction"])
def test_invalid_inputs_rejected(bad):
    obs = observation()
    if bad == "encoded":
        obs["vision"]["cam_head"]["color"] = b"jpeg"
    if bad == "float":
        obs["vision"]["cam_head"]["color"] = np.zeros((8, 9, 3), np.float32)
    if bad == "shape":
        obs["state"]["left_arm_joint_state"] = np.zeros(7)
    if bad == "nan":
        obs["state"]["left_arm_joint_state"][0] = np.nan
    if bad == "instruction":
        obs["instruction"] = ""
    with pytest.raises(ValueError):
        observation_from_xpolicylab(obs, FIELDS)


def test_batch_ids_freshness_isolation_and_reset(model):
    model.reset()
    model.update_obs_batch([observation(7, 10), observation(2, 30)])
    first = model.get_action_batch([2, 7])
    assert first[0][0]["left_arm_joint_state"][0] == 30
    assert first[1][0]["left_arm_joint_state"][0] == 10
    assert model._sessions[2].model is model._sessions[7].model
    assert model._sessions[2].seed != model._sessions[7].seed
    with pytest.raises(ValueError):
        model.get_action_batch([7])
    old = model._sessions[7].previous_plan.copy()
    model.update_obs_batch([observation(2, 100)])
    actions = model.get_action_batch()
    assert actions[0][0]["left_arm_joint_state"][0] == 0.75 * 40 + 0.25 * 100
    np.testing.assert_array_equal(old, model._sessions[7].previous_plan)
    model.reset()
    assert not model._sessions and not model._observations


def test_invalid_batch_does_not_mutate_policy(model):
    model.update_obs_batch([observation(3), observation(2)])
    with pytest.raises(ValueError):
        model.get_action_batch([3, 99])
    assert not model._sessions
    with pytest.raises(ValueError):
        model.update_obs_batch([observation(1), observation(1)])
    assert model._order == [3, 2]


@pytest.mark.parametrize(
    "extra",
    [
        {"env_cfg_type": "panda"},
        {"action_type": "ee"},
        {"inference_seed": -1},
        {"old_plan_weight": 2},
    ],
)
def test_bad_config_rejected_before_policy_load(extra, tmp_path):
    with pytest.raises(ValueError):
        Model(
            {
                "model_dir": str(tmp_path),
                "env_cfg_type": "arx_x5",
                "action_type": "joint",
                **extra,
            }
        )


def test_official_keyword_call_and_whole_chunks():
    class Env:
        step = 0

        def is_episode_end(self):
            return self.step >= 23

        def get_obs(self):
            return {"step": self.step}

        def take_action(self, action):
            self.step += 1

    calls = []

    class Client:
        def call(self, func_name, obs=None):
            calls.append((func_name, copy.deepcopy(obs)))
            return [{}] * 10 if func_name == "get_action" else None

    env = Env()
    deploy.eval_one_episode(TASK_ENV=env, model_client=Client())
    assert env.step == 23
    assert [obs["step"] for name, obs in calls if name == "update_obs"] == [0, 10, 20]
