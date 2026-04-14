import time


def _has_valid_images(obs):
    vision = obs.get("vision", {})
    for camera_name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        camera_data = vision.get(camera_name, {})
        if not isinstance(camera_data, dict) or camera_data.get("color") is None:
            return False
    return True

def _get_valid_obs(task_env, timeout=2.0, interval=0.05):
    deadline = time.monotonic() + timeout
    last_obs = None

    while time.monotonic() < deadline:
        last_obs = task_env.get_obs()
        if _has_valid_images(last_obs):
            return last_obs
        time.sleep(interval)

    raise RuntimeError(f"Timed out waiting for valid camera observations. Last obs keys: {list(last_obs.keys()) if isinstance(last_obs, dict) else type(last_obs)}")


def eval_one_episode(TASK_ENV, model_client):

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = _get_valid_obs(TASK_ENV) # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            
            if action_idx != len(actions) - 1:
                obs = _get_valid_obs(TASK_ENV) # Get Observation
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