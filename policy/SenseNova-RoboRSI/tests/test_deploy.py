"""Empty inference polls must not become native steps, frames or score checks."""

from __future__ import annotations

from importlib import import_module

from collections import deque

import numpy as np
import pytest

eval_one_episode = import_module("XPolicyLab.policy.SenseNova-RoboRSI.deploy").eval_one_episode


class Environment:
    def __init__(self, event_log, steps=2):
        self.log, self.steps, self.step = event_log, steps, 0

    def is_episode_end(self):
        self.log.append(("is_episode_end", self.step))
        return self.step == self.steps

    def get_obs(self):
        self.log.append(("get_obs", self.step))
        return {"public_observation_index": self.step}

    def take_action(self, action):
        self.log.append(("take_action", self.step))
        assert action["left_arm_joint_state"].shape == (6,)
        self.step += 1


class Client:
    def __init__(self, event_log, actions):
        self.log, self.actions = event_log, deque(actions)

    def call(self, func_name, obs=None):
        self.log.append((func_name, obs))
        if func_name == "get_action":
            value = self.actions.popleft()
            if isinstance(value, Exception):
                raise value
            return value


def action():
    return {
        f"{arm}_{part}_joint_state": np.zeros(size, dtype=np.float32)
        for arm in ("left", "right")
        for part, size in (("arm", 6), ("ee", 1))
    }


def test_empty_chunks_only_poll_the_same_inference_without_native_side_effects():
    events = []
    env = Environment(events)
    client = Client(events, [[], [], [action()], [], [action()]])
    eval_one_episode(env, client)
    names = [name for name, _ in events]
    assert names == [
        "reset",
        "is_episode_end",
        "get_obs",
        "update_obs",
        "get_action",
        "get_action",
        "get_action",
        "take_action",
        "is_episode_end",
        "get_obs",
        "update_obs",
        "get_action",
        "get_action",
        "take_action",
        "is_episode_end",
        "get_obs",
        "end_episode",
        "close",
    ]
    assert env.step == 2 and not client.actions
    assert [value for name, value in events if name == "update_obs"] == [
        {"public_observation_index": 0},
        {"public_observation_index": 1},
    ]
    assert next(value for name, value in events if name == "end_episode") == {
        "public_observation_index": 2
    }


def test_empty_poll_does_not_mask_real_failure_or_execute_a_placeholder():
    events = []
    env = Environment(events)
    client = Client(events, [[], RuntimeError("original CLI failed")])
    with pytest.raises(RuntimeError, match="original CLI failed"):
        eval_one_episode(env, client)
    names = [name for name, _ in events]
    assert names == [
        "reset",
        "is_episode_end",
        "get_obs",
        "update_obs",
        "get_action",
        "get_action",
        "close",
    ]
    assert env.step == 0


@pytest.mark.parametrize("invalid", [None, {}, [action(), action()]])
def test_invalid_chunks_are_rejected_without_native_motion(invalid):
    events = []
    env = Environment(events)
    client = Client(events, [invalid])
    with pytest.raises(ValueError, match="action per chunk"):
        eval_one_episode(env, client)
    assert env.step == 0
    assert [name for name, _ in events][-1] == "close"
