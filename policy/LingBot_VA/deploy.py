from XPolicyLab.utils.action_chunks import validate_action_chunk


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        validate_action_chunk(actions, TASK_ENV)

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError(
        "LingBot_VA wan_va_server keeps one global KV/VAE cache and cannot "
        "evaluate multiple envs in one process. Keep eval_batch: false in deploy.yml."
    )
