def eval_one_episode(TASK_ENV, model_client):
    task_env = TASK_ENV
    model_client.call(func_name="reset")

    while not task_env.is_episode_end():
        model_client.call(func_name="update_obs", obs=task_env.get_obs())
        actions = model_client.call(func_name="get_action")

        for action_index, action in enumerate(actions):
            task_env.take_action(action)
            if task_env.is_episode_end() or action_index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=task_env.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    task_env = TASK_ENV
    model_client.call(func_name="reset")

    while not task_env.is_episode_end():
        env_idx_list = task_env.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=task_env.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_index in range(chunk_size):
            task_env.take_action_batch([chunk[action_index] for chunk in actions], env_idx_list)
            if task_env.is_episode_end() or action_index + 1 == chunk_size:
                break

            running_envs = set(task_env.get_running_env_idx_list())
            active_batch_indices = [
                batch_index for batch_index, env_index in enumerate(env_idx_list) if env_index in running_envs
            ]
            actions = [actions[batch_index] for batch_index in active_batch_indices]
            env_idx_list = [env_idx_list[batch_index] for batch_index in active_batch_indices]
            model_client.call(func_name="update_obs_batch", obs=task_env.get_obs_batch(env_idx_list))
