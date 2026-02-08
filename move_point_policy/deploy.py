def get_model(deploy_cfg):
    # import packages and module here
    from policy_lab.move_point_policy.your_policy import Your_Policy
    # Initialize and return your policy_lab model here according to deploy_cfg
    model = Your_Policy(deploy_cfg)
    return model

def eval_one_episode(TASK_ENV, model_client):
    instruction = TASK_ENV.get_instruction()
    model_client.call(func_name="set_language", obs=instruction)

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends

        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs() # Get Observation
                model_client.call(func_name="update_obs", obs=obs)