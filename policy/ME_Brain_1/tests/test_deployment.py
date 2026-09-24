"""Exercise per-step history and asynchronous termination without GPUs."""

import numpy as np

from XPolicyLab.policy.ME_Brain_1 import deploy


class Environment:
    def __init__(self, limits):
        self.steps = {key: 0 for key in limits}
        self.limits = limits

    def get_running_env_idx_list(self):
        return [key for key, step in self.steps.items() if step < self.limits[key]]

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_obs_batch(self, ids):
        return [self.observation(key) for key in ids]

    def observation(self, key):
        return {"env_idx": key, "step": self.steps[key]}

    def take_action_batch(self, actions, ids):
        assert len(actions) == len(ids)
        for key in ids:
            assert self.steps[key] < self.limits[key]
            self.steps[key] += 1

    def get_obs(self):
        return self.observation(0)

    def take_action(self, action):
        self.take_action_batch([action], [0])


class Client:
    def __init__(self):
        self.requests = []

    def call(self, func_name, obs=None):
        if func_name.startswith("update_obs"):
            self.observations = obs if isinstance(obs, list) else [obs]
        if func_name.startswith("get_action"):
            self.requests.append([dict(item) for item in self.observations])
            chunks = [np.zeros((16, 14)) for _ in self.observations]
            return chunks if func_name.endswith("_batch") else chunks[0]


class Tracker:
    events = []

    def stamp_obs(self, obs, env_idx):
        self.events.append((env_idx, obs["step"]))
        obs["hist_t0"] = obs["step"]

    def stamp_obs_list(self, observations, ids):
        for observation, env_id in zip(observations, ids):
            self.stamp_obs(observation, env_id)


def test_single_loop_samples_each_environment_step_once(monkeypatch):
    monkeypatch.setattr(deploy, "LiveHeadHistoryTracker", Tracker)
    Tracker.events = []
    env, client = Environment({0: 33}), Client()
    deploy.eval_one_episode(env, client)
    assert Tracker.events == [(0, step) for step in range(33)]
    assert [batch[0]["hist_t0"] for batch in client.requests] == [0, 16, 32]


def test_batch_handles_noncontiguous_ids_and_early_exit(monkeypatch):
    monkeypatch.setattr(deploy, "LiveHeadHistoryTracker", Tracker)
    Tracker.events = []
    env, client = Environment({3: 5, 8: 33}), Client()
    deploy.eval_one_episode_batch(env, client)
    for key, count in env.limits.items():
        assert [step for env_id, step in Tracker.events if env_id == key] == list(
            range(count)
        )
    assert [len(batch) for batch in client.requests] == [2, 1, 1]
