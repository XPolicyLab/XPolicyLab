"""RoboDojo rollout loop for HWM_CogWAM.

CogWAM's adapter (``cogwam.eval.robodojo_policy.Model``, re-exported as
``model.Model``) caches its own 25-step action chunk internally and only
ever returns one action per ``get_action``/``get_action_batch`` call — it
replans every ``replan_interval`` control steps on its own schedule. So this
loop just asks for one action per control step, exactly like CogWAM's own
released evaluation driver did (see CogWAM's
``cogwam/eval/xpolicy_overlay/XPolicyLab/policy/cogwam/deploy.py``, which
this file mirrors verbatim for the real XPolicyLab checkout).
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if len(actions) != 1:
            raise RuntimeError(f"HWM_CogWAM must return one cached action per control step, got {len(actions)}.")
        TASK_ENV.take_action(actions[0])


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if not env_idx_list:
            break
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if any(len(env_actions) != 1 for env_actions in actions):
            sizes = [len(env_actions) for env_actions in actions]
            raise RuntimeError(f"HWM_CogWAM must return one cached action per env, got chunk sizes {sizes}.")
        TASK_ENV.take_action_batch([env_actions[0] for env_actions in actions], env_idx_list)


__all__ = ["eval_one_episode", "eval_one_episode_batch"]
