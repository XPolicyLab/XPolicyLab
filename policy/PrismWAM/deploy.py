"""Execute action chunks while recording one observation per executed action.

Follows the XPolicyLab demo-policy lifecycle. Model history is episode-local;
the policy runtime, not this loop, chooses the checkpoint-bound replanning length.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise RuntimeError("PrismWAM returned an empty action chunk.")
        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        if not indices:
            break
        model_client.call(
            func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices)
        )
        # XPolicyLab's RPC "obs" argument is the called method's positional arg.
        actions = model_client.call(func_name="get_action_batch", obs=indices)
        if len(actions) != len(indices) or not actions or not actions[0]:
            raise RuntimeError("PrismWAM returned an invalid batched action chunk.")
        chunk_size = len(actions[0])
        if any(len(chunk) != chunk_size for chunk in actions):
            raise RuntimeError("Batched action chunks must have equal lengths.")
        for step in range(chunk_size):
            TASK_ENV.take_action_batch([chunk[step] for chunk in actions], indices)
            if TASK_ENV.is_episode_end() or step + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [index for index, env in enumerate(indices) if env in running]
            actions = [actions[index] for index in active]
            indices = [indices[index] for index in active]
            if not indices:
                break
            model_client.call(
                func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices)
            )
