"""Official evaluation loop, identical to policy/demo_policy/deploy.py, plus a notification when the episode ends.

The loop only forwards observations and executes actions: no simulator state reaches the policy.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)
    model_client.call(func_name="episode_over")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("RoboBigWater evaluates one environment at a time (eval_batch: false)")
