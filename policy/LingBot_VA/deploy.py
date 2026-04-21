import time

def _has_valid_images(obs):
    vision = obs.get("vision", {})
    camera_aliases = {
        "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
        "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
        "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
    }

    for candidate_names in camera_aliases.values():
        camera_data = None
        for camera_name in candidate_names:
            if camera_name in vision:
                camera_data = vision.get(camera_name, {})
                break

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
    keyframe_interval = model_client.call(func_name="get_keyframe_interval")
    if not keyframe_interval:
        raise RuntimeError("LingBot_VA model returned invalid keyframe interval.")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = _get_valid_obs(TASK_ENV) # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        if actions is None:
            raise RuntimeError("LingBot_VA server returned no actions. Check the server-side traceback above.")

        key_frame_obs_list = []
        
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if (action_idx + 1) % keyframe_interval == 0:
                key_frame_obs_list.append(_get_valid_obs(TASK_ENV))

        if not TASK_ENV.is_episode_end():
            model_client.call(func_name="update_cache", obs=key_frame_obs_list)

def eval_one_episode_batch(TASK_ENV, model_client):

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation

        model_client.call(func_name="update_obs_batch", obs=obs_list)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action_batch", env_idx_list=env_idx_list) # Get Action according to observation chunk

        for action_idx in range(len(actions[0])):
            current_action_list = [env_actions[action_idx] for env_actions in actions]

            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            
            if action_idx != len(actions) - 1:
                env_idx_list = TASK_ENV.get_running_env_idx_list()
                obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
                model_client.call(func_name="update_obs_batch", obs=obs_list)