"""Original finish requests use the official reward check, never an idle action."""

from importlib import import_module

from collections import deque
import json
from types import SimpleNamespace

import numpy as np
import pytest

eval_one_episode = import_module("XPolicyLab.policy.SenseNova-RoboRSI.deploy").eval_one_episode
Model = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model").Model


class NativeEnvironment:
    def __init__(self, reward=0.4):
        self.num_envs = 1
        self.success = [True]
        self.end_flag = [False]
        self.take_action_cnt = [0]
        self.reward = reward
        self.reward_checks = []
        self.reward_manager = SimpleNamespace(get_reward=self.get_reward)

    def get_reward(self, *, final_check):
        self.reward_checks.append(final_check)
        return [self.reward]

    def is_episode_end(self):
        return all(self.end_flag)

    def get_obs(self):
        return {"step": self.take_action_cnt[0], "terminated": self.end_flag[0]}

    def take_action(self, action):
        assert "_roborsi_control" not in action
        assert action["left_arm_joint_state"].shape == (6,)
        self.take_action_cnt[0] += 1


class Client:
    def __init__(self, actions):
        self.actions = deque(actions)
        self.calls = []

    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            return self.actions.popleft()


FINISH = [{"_roborsi_control": "finish"}]


@pytest.mark.parametrize("reward, expected_success", [(0.4, False), (0.9995, True), (1.0, True)])
def test_finish_uses_native_final_check_and_terminal_ack_without_spending_step(reward, expected_success):
    env = NativeEnvironment(reward)
    client = Client([FINISH])
    eval_one_episode(env, client)
    assert env.reward_checks == [True]
    assert env.success == [expected_success]
    assert env.end_flag == [True]
    assert env.take_action_cnt == [0]
    assert client.calls[-2:] == [
        ("end_episode", {"step": 0, "terminated": True}), ("close", None)
    ]


def test_joint_action_before_finish_is_preserved_and_counted_once():
    env = NativeEnvironment()
    joint = {"left_arm_joint_state": np.zeros(6, dtype=np.float32)}
    client = Client([[joint], [], FINISH])
    eval_one_episode(env, client)
    assert env.take_action_cnt == [1]
    assert env.reward_checks == [True]
    assert client.calls[-2][1] == {"step": 1, "terminated": True}


@pytest.mark.parametrize("problem", ["batch", "missing_reward", "missing_flags", "non_finite"])
def test_finish_unsupported_environment_fails_closed_and_closes(problem):
    env = NativeEnvironment()
    if problem == "batch":
        env.num_envs = 2
    elif problem == "missing_reward":
        env.reward_manager = object()
    elif problem == "missing_flags":
        env.success = None
    else:
        env.reward = float("nan")
    client = Client([FINISH])
    with pytest.raises(RuntimeError, match="Native finish"):
        eval_one_episode(env, client)
    assert env.take_action_cnt == [0]
    assert env.end_flag == [False]
    assert client.calls[-1] == ("close", None)
    assert all(name != "end_episode" for name, _ in client.calls)


@pytest.mark.parametrize("control", [
    {"_roborsi_control": "reset"},
    {"_roborsi_control": "finish", "left_arm_joint_state": [0] * 6},
])
def test_only_exact_finish_control_is_accepted(control):
    env = NativeEnvironment()
    client = Client([[control]])
    with pytest.raises(ValueError, match="native control"):
        eval_one_episode(env, client)
    assert env.take_action_cnt == [0]
    assert env.reward_checks == []


def cleanup_failure_model(tmp_path):
    """Use the real terminal/close boundary with one owned uncancelled controller."""
    model = Model.__new__(Model)
    model._directory = tmp_path
    model._temporary = SimpleNamespace(_finalizer=SimpleNamespace(detach=lambda: None))
    model._icl = model._obs = None
    model._step = 703
    model._resume = None
    model._ended = False
    model._awaiting_feedback = True
    model._deferred_controller = model._cleanup_issue = None
    model._pending = SimpleNamespace(request_id=704, kind="joint")
    calls = []

    def acknowledge(*args, **kwargs):
        calls.append(("ack", kwargs["terminated"]))

    def close():
        calls.append(("close", None))
        raise RuntimeError("Original hybrid controller survived cancellation")

    controller = SimpleNamespace(done=True, error=None, acknowledge=acknowledge, close=close)
    model._connector = controller
    model.update_obs = lambda obs: setattr(model, "_obs", obs)
    return model, controller, calls


@pytest.mark.parametrize("reward, expected_success", [(0.4, False), (1.0, True)])
def test_terminal_cleanup_failure_allows_native_result_persistence(
    tmp_path, caplog, reward, expected_success
):
    model, controller, calls = cleanup_failure_model(tmp_path)
    env = NativeEnvironment(reward)

    class ModelClient(Client):
        def call(self, func_name, obs=None):
            if func_name == "end_episode":
                return model.end_episode(obs)
            if func_name == "close":
                return model.close()
            return super().call(func_name, obs)

    # The official scorer owns flags and writes its result after eval_one_episode
    # returns. Exercise that ordering without fabricating a native task score.
    eval_one_episode(env, ModelClient([FINISH]))
    result_path = tmp_path / "native-result-fixture.json"
    result_path.write_text(json.dumps({"success": env.success[0]}))
    assert json.loads(result_path.read_text()) == {"success": expected_success}
    assert env.end_flag == [True] and env.take_action_cnt == [0]
    assert calls == [("ack", True), ("close", None)]
    assert model._deferred_controller is controller
    receipt = json.loads((tmp_path / "cleanup-error.json").read_text())
    assert receipt["native_terminal_accepted"] is True
    assert receipt["acknowledged_steps"] == 704
    assert receipt["supervisor_cleanup_required"] is True
    assert "cleanup requires case supervisor" in caplog.text
    with pytest.raises(RuntimeError, match="restart the policy server"):
        model.reset()
    assert calls == [("ack", True), ("close", None)]


def test_execution_error_is_not_masked_by_terminal_cleanup_failure(tmp_path):
    model, controller, calls = cleanup_failure_model(tmp_path)

    def reject_observation(_obs):
        raise ValueError("Terminal observation identity mismatch")

    model.update_obs = reject_observation
    with pytest.raises(ValueError, match="observation identity mismatch"):
        model.end_episode({"fixture": "invalid"})
    assert calls == [("close", None)]
    assert model._deferred_controller is controller
    assert model._cleanup_issue["native_terminal_accepted"] is False


def test_unvalidated_resume_still_fails_when_terminal_cleanup_also_fails(tmp_path):
    model, _, calls = cleanup_failure_model(tmp_path)
    model._resume = {"step_offset": 1}
    with pytest.raises(RuntimeError, match="validated continuation"):
        model.end_episode({"fixture": "terminal"})
    assert calls == [("ack", True), ("close", None)]
    assert model._cleanup_issue["native_terminal_accepted"] is False


def test_nonterminal_cleanup_failure_is_fatal_and_cannot_reuse_controller(tmp_path):
    model, controller, calls = cleanup_failure_model(tmp_path)
    with pytest.raises(RuntimeError, match="survived cancellation"):
        model.close()
    assert model._deferred_controller is controller
    with pytest.raises(RuntimeError, match="restart the policy server"):
        model.reset()
    assert calls == [("close", None)]
