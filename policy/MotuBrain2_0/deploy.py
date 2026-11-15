"""Reuse XPolicyLab's episode loop, adding only a native-action clock."""

from XPolicyLab.policy.demo_policy.deploy import eval_one_episode as _eval_one_episode


class _NativeStepEnv:
    def __init__(self, env):
        self.env = env
        self.step = 0

    def is_episode_end(self):
        return self.env.is_episode_end()

    def get_obs(self):
        obs = dict(self.env.get_obs())
        if "observation_step" in obs and obs["observation_step"] != self.step:
            raise ValueError("Environment observation step disagrees with native action count")
        obs["observation_step"] = self.step
        return obs

    def take_action(self, action):
        self.env.take_action(action)
        self.step += 1


def eval_one_episode(TASK_ENV, model_client):
    return _eval_one_episode(_NativeStepEnv(TASK_ENV), model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("Use eval_batch=false; batch inference is not part of this release")
