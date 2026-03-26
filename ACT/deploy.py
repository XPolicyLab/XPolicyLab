def eval_one_episode(TASK_ENV, model_client):

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            
            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs() # Get Observation
                model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation

        model_client.call(func_name="update_obs_batch", obs=obs_list)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list) # Get Action according to observation chunk

        for action_idx in range(len(actions[0])):
            current_action_list = [env_actions[action_idx] for env_actions in actions]

            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            
            if action_idx != len(actions) - 1:
                env_idx_list = TASK_ENV.get_running_env_idx_list()
                obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
                model_client.call(func_name="update_obs_batch", obs=obs_list)