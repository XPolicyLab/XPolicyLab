"""Official environment owns resets, layouts, action limits and scoring."""


class PolicyFailure(BaseException):
    """Stop on a policy/RPC error; do not silently redraw a scene and retry it."""


def _call(client, name, **kwargs):
    try:
        return client.call(func_name=name, **kwargs)
    except Exception as exc:
        raise PolicyFailure(f"EmbodiedRSI {name} failed: {exc}") from exc


def eval_one_episode(TASK_ENV, model_client):
    # Debug's TestEnv has episode_step_limit; real RoboDojo has step_lim.
    diagnostic = not hasattr(TASK_ENV, "step_lim")
    step_limit = TASK_ENV.episode_step_limit if diagnostic else TASK_ENV.step_lim
    _call(model_client, "reset")
    _call(model_client, "prepare_case", obs={
        "native_step_limit": int(step_limit), "diagnostic_environment": diagnostic,
    })
    while not TASK_ENV.is_episode_end():
        _call(model_client, "update_obs", obs=TASK_ENV.get_obs())
        actions = []
        while not actions:
            actions = _call(model_client, "get_action")
        if len(actions) != 1:
            raise PolicyFailure("EmbodiedRSI requires one action per closed-loop Python transition")
        # Native exceptions remain native: the official client owns its own
        # invalid-scene handling. This adapter does not patch reset or scoring.
        TASK_ENV.take_action(actions[0])
    if diagnostic:
        obs = TASK_ENV.get_obs()
        result = {"success": False, "terminated": False, "truncated": True, "diagnostic": True}
    else:
        obs = TASK_ENV.get_obs_batch([0], last_frame=True)[0]
        success = bool(TASK_ENV.end_flag[0] and TASK_ENV.success[0])
        truncated = not success and TASK_ENV.take_action_cnt[0] >= TASK_ENV.step_lim
        result = {"success": success, "terminated": not truncated, "truncated": bool(truncated)}
    _call(model_client, "update_obs", obs=obs)
    _call(model_client, "trial_end", obs=result)


def eval_one_episode_batch(TASK_ENV, model_client):
    diagnostic = not hasattr(TASK_ENV, "step_lim")
    step_limit = TASK_ENV.episode_step_limit if diagnostic else TASK_ENV.step_lim
    _call(model_client, "reset")
    initial = list(TASK_ENV.get_running_env_idx_list())
    _call(model_client, "prepare_case", obs={
        "native_step_limit": int(step_limit), "diagnostic_environment": diagnostic,
        "environments": [{"env_idx": i, "layout_id": None if diagnostic else int(TASK_ENV.env_seeds[i])}
                         for i in initial],
    })
    pending_end = set(initial)
    while not TASK_ENV.is_episode_end():
        indices = list(TASK_ENV.get_running_env_idx_list())
        _call(model_client, "update_obs_batch", obs=TASK_ENV.get_obs_batch(indices))
        while True:
            actions = _call(model_client, "get_action_batch", obs=indices)
            if len(actions) != len(indices):
                raise PolicyFailure("Action batch and native environment indices disagree")
            if all(len(chunk) == 1 for chunk in actions):
                break
            if any(actions):
                raise PolicyFailure("Batch must wait until every live environment has one action")
        TASK_ENV.take_action_batch([chunk[0] for chunk in actions], indices)
        ended = pending_end - (set() if TASK_ENV.is_episode_end() else set(TASK_ENV.get_running_env_idx_list()))
        if ended:
            _finish_batch(TASK_ENV, model_client, sorted(ended), diagnostic)
            pending_end -= ended
    if pending_end:
        _finish_batch(TASK_ENV, model_client, sorted(pending_end), diagnostic)


def _finish_batch(env, client, indices, diagnostic):
    observations = env.get_obs_batch(indices) if diagnostic else env.get_obs_batch(indices, last_frame=True)
    _call(client, "update_obs_batch", obs=observations)
    results = []
    for index in indices:
        success = False if diagnostic else bool(env.end_flag[index] and env.success[index])
        truncated = True if diagnostic else not success and env.take_action_cnt[index] >= env.step_lim
        results.append({"env_idx": index, "success": success,
                        "terminated": not truncated, "truncated": bool(truncated)})
    _call(client, "trial_end", obs={"results": results})
