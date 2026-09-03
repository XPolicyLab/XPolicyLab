def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    env_idx_list = TASK_ENV.get_running_env_idx_list()
    if len(env_idx_list) != 1:
        raise NotImplementedError("OLA-SEM supports only single-environment inference")
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(
            func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
        )
        action_batch = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        for action_idx, action in enumerate(action_batch[0]):
            TASK_ENV.take_action_batch([action], env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(action_batch[0]):
                break
            model_client.call(
                func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
            )
