import time


def _has_valid_images(obs):
    vision = obs.get("vision", {})
    for camera_name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        camera_data = vision.get(camera_name, {})
        if not isinstance(camera_data, dict) or (
            camera_data.get("color") is None and camera_data.get("colors") is None
        ):
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
    raise RuntimeError(f"Timed out waiting for valid camera observations. Last observation: {type(last_obs).__name__}")


def _evaluation_scope(task_env):
    deploy_cfg = getattr(task_env, "deploy_cfg", {}) or {}
    evaluation_id = deploy_cfg.get("evaluation_id") or getattr(task_env, "run_id", None)
    if not evaluation_id:
        raise RuntimeError("OpenDM evaluation requires TASK_ENV evaluation_id or run_id")
    return {"evaluation_id": str(evaluation_id)}


def _scope_obs(obs, scope):
    return {**obs, **scope}


def eval_one_episode(TASK_ENV, model_client):
    scope = _evaluation_scope(TASK_ENV)
    model_client.call(func_name="reset_evaluation", obs=scope)
    try:
        while not TASK_ENV.is_episode_end():
            obs = _scope_obs(_get_valid_obs(TASK_ENV), scope)
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action", obs=scope)

            for action_idx, action in enumerate(actions):
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                    break
                obs = _scope_obs(_get_valid_obs(TASK_ENV), scope)
                model_client.call(func_name="update_obs", obs=obs)
    finally:
        model_client.call(func_name="reset_evaluation", obs=scope)


def eval_one_episode_batch(TASK_ENV, model_client):
    scope = _evaluation_scope(TASK_ENV)
    model_client.call(func_name="reset_evaluation", obs=scope)
    try:
        while not TASK_ENV.is_episode_end():
            env_idx_list = TASK_ENV.get_running_env_idx_list()
            obs_list = [
                _scope_obs(obs, scope)
                for obs in TASK_ENV.get_obs_batch(env_idx_list)
            ]
            model_client.call(func_name="update_obs_batch", obs=obs_list)
            actions = model_client.call(
                func_name="get_action_batch",
                obs={**scope, "env_idx_list": env_idx_list},
            )

            chunk_size = min(len(env_actions) for env_actions in actions)
            for action_idx in range(chunk_size):
                TASK_ENV.take_action_batch(
                    [env_actions[action_idx] for env_actions in actions],
                    env_idx_list,
                )
                if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                    break

                running = set(TASK_ENV.get_running_env_idx_list())
                active = [
                    idx for idx, env_idx in enumerate(env_idx_list) if env_idx in running
                ]
                actions = [actions[idx] for idx in active]
                env_idx_list = [env_idx_list[idx] for idx in active]
                obs_list = [
                    _scope_obs(obs, scope)
                    for obs in TASK_ENV.get_obs_batch(env_idx_list)
                ]
                model_client.call(func_name="update_obs_batch", obs=obs_list)
    finally:
        model_client.call(func_name="reset_evaluation", obs=scope)
