"""Standard single-environment observation/action loop."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break
            observation = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=observation)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError(
        "Use eval_batch=false and one environment per policy server."
    )
