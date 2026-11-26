"""Use UniWAM's XPolicyLab episode runner for the official debug client."""

from XPolicyLab.policy.UniWAM.deploy import eval_one_episode, eval_one_episode_batch

__all__ = ["eval_one_episode", "eval_one_episode_batch"]
