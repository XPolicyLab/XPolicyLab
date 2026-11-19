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


def eval_one_episode_batch(TASK_ENV, model_client):
    """Run all active RoboDojo environments through the batch policy API.

    ``eval_batch`` is enabled in this policy's deployment configuration.  A
    single-environment fallback leaves the other simulator environments idle,
    while the evaluator waits for every environment to terminate.  Keep each
    action chunk associated with its environment id and compact both lists as
    episodes finish at different times.
    """
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = list(TASK_ENV.get_running_env_idx_list())
        if not env_idx_list:
            raise RuntimeError("Evaluator is not finished but has no running environments")

        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        action_chunks = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if len(action_chunks) != len(env_idx_list):
            raise RuntimeError(
                "Batch policy returned a different number of action chunks than running "
                f"environments: {len(action_chunks)} != {len(env_idx_list)}"
            )
        if not action_chunks or any(not chunk for chunk in action_chunks):
            raise RuntimeError("Batch policy returned an empty action chunk")

        chunk_size = len(action_chunks[0])
        if any(len(chunk) != chunk_size for chunk in action_chunks):
            raise RuntimeError("Batch policy returned action chunks with inconsistent lengths")

        for action_idx in range(chunk_size):
            TASK_ENV.take_action_batch(
                [chunk[action_idx] for chunk in action_chunks], env_idx_list
            )
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            next_env_idx_list = list(TASK_ENV.get_running_env_idx_list())
            previous_pos = {env_idx: pos for pos, env_idx in enumerate(env_idx_list)}
            missing = [env_idx for env_idx in next_env_idx_list if env_idx not in previous_pos]
            if missing:
                raise RuntimeError(
                    "Evaluator reported newly running environments during an action chunk: "
                    f"{missing}"
                )
            action_chunks = [action_chunks[previous_pos[env_idx]] for env_idx in next_env_idx_list]
            env_idx_list = next_env_idx_list
            if not env_idx_list:
                break

            obs_list = TASK_ENV.get_obs_batch(env_idx_list)
            model_client.call(func_name="update_obs_batch", obs=obs_list)
