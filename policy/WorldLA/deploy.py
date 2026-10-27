"""Generic XPolicyLab environment loops, preserving WorldLA's K=10 chunks."""


def eval_one_episode(TASK_ENV, model_client):
    task_env = TASK_ENV
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        model_client.call(func_name="update_obs", obs=task_env.get_obs())
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise RuntimeError("WorldLA returned an empty action chunk")
        for action in actions:
            task_env.take_action(action)
            if task_env.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    task_env = TASK_ENV
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list()
        model_client.call(
            func_name="update_obs_batch", obs=task_env.get_obs_batch(indices)
        )
        chunks = model_client.call(func_name="get_action_batch", obs=indices)
        if (
            len(chunks) != len(indices)
            or not chunks
            or any(len(c) != 10 for c in chunks)
        ):
            raise RuntimeError(
                "WorldLA requires one ten-step chunk per active environment"
            )
        for step in range(10):
            task_env.take_action_batch([chunk[step] for chunk in chunks], indices)
            if task_env.is_episode_end():
                break
            running = set(task_env.get_running_env_idx_list())
            active = [
                (index, chunk)
                for index, chunk in zip(indices, chunks, strict=True)
                if index in running
            ]
            if not active:
                break
            indices, chunks = map(list, zip(*active, strict=True))
