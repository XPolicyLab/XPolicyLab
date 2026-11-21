"""Run the Preview action loop with explicit wait and final-verdict messages."""


def episode_status(env, pictures_behind):
    counts = getattr(env, "take_action_cnt", None)
    try:
        step = int(counts[0])
    except (TypeError, IndexError, ValueError):
        step = getattr(env, "episode_step", None)
    limit = getattr(env, "step_lim", None)
    if limit is None:
        limit = getattr(env, "episode_step_limit", None)
    status = {
        "step": step,
        "step_limit": limit,
        "task_name": getattr(env, "task_name", None),
        "pictures_behind": pictures_behind,
    }
    seeds = getattr(env, "env_seeds", None)
    if seeds is not None and len(seeds):
        status["layout"] = int(seeds[0])
    return status


def observation(env, pictures_behind):
    obs = env.get_obs()
    extra = dict(obs.get("additional_info") or {})
    extra["hexashell"] = episode_status(env, pictures_behind)
    return {**obs, "additional_info": extra}


def verdict(env, pictures_behind):
    status = episode_status(env, pictures_behind)
    success = getattr(env, "success", None)
    if success is not None:
        status["success"] = bool(success[0])
        if status["success"]:
            status["progress"] = 100.0
        elif hasattr(env, "get_score"):
            status["progress"] = float(env.reward_manager.get_score()[0])
    return status


def eval_one_episode(TASK_ENV, model_client):
    try:
        from env.camera_manager.capture import render_sync  # noqa: F401
    except ImportError:
        pictures_behind = 1
    else:
        pictures_behind = 0
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = observation(TASK_ENV, pictures_behind)
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        while actions == [{"hexashell": "wait"}]:
            actions = model_client.call(func_name="get_action")
        if not actions:
            if not TASK_ENV.is_episode_end() and hasattr(TASK_ENV, "success"):
                # Ask RoboDojo to do its final reward check on this scene.
                for index in TASK_ENV.get_running_env_idx_list():
                    TASK_ENV.success[index] = False
                TASK_ENV.is_episode_end()
            break
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = observation(TASK_ENV, pictures_behind)
            model_client.call(func_name="update_obs", obs=obs)
    model_client.call(func_name="trial_end", obs=verdict(TASK_ENV, pictures_behind))


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("HexaShell-preview evaluates one environment at a time (eval_batch: false)")
