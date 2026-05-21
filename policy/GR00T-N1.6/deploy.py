import time


def _has_valid_images(obs):
    vision = obs.get("vision", {})
    if "cam_head" not in vision or vision["cam_head"].get("color") is None:
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
    raise RuntimeError(
        "Timed out waiting for valid camera observations. "
        f"Last obs keys: {list(last_obs.keys()) if isinstance(last_obs, dict) else type(last_obs)}"
    )


def eval_one_episode(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        obs = _get_valid_obs(TASK_ENV)
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = _get_valid_obs(TASK_ENV)
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        for action_idx in range(len(actions[0])):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions[0]):
                break
            env_idx_list = TASK_ENV.get_running_env_idx_list()
            obs_list = TASK_ENV.get_obs_batch(env_idx_list)
            model_client.call(func_name="update_obs_batch", obs=obs_list)
