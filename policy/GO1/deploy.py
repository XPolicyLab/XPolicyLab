import time


def _has_valid_images(obs):
    vision = obs.get("vision", {})
    for camera_name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        camera_data = vision.get(camera_name, {})
        if isinstance(camera_data, dict) and camera_data.get("color") is not None:
            return True
    return False


def _get_valid_obs(task_env, timeout=2.0, interval=0.05):
    deadline = time.monotonic() + timeout
    last_obs = None

    while time.monotonic() < deadline:
        last_obs = task_env.get_obs()
        if _has_valid_images(last_obs):
            return last_obs
        time.sleep(interval)

    raise RuntimeError(
        f"Timed out waiting for valid camera observations. "
        f"Last obs keys: {list(last_obs.keys()) if isinstance(last_obs, dict) else type(last_obs)}"
    )


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

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
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)

        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_idx_list),
            )
