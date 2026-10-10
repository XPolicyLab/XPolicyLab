def _eval_robotwin_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())



def _discard_after_error(model_client, error, *, execution_started):
    try:
        model_client.call(func_name="discard_action", obs={"execution_started": execution_started})
    except Exception as cleanup_error:
        # Preserve the execution/transport failure if the server is unreachable.
        error.add_note(f"Could not discard pending WSP prediction: {cleanup_error}")


def _eval_robodojo_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        execution_started = False
        try:
            actions = model_client.call(func_name="get_action")
            if not actions:
                raise ValueError("WorldScapePolicy returned an empty action chunk")
            executed_steps = 0
            for action_idx, action in enumerate(actions):
                # A simulator call can partially execute before raising.
                execution_started = True
                TASK_ENV.take_action(action)
                executed_steps += 1
                if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                    break
                model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
            model_client.call(func_name="commit_action", obs={
                "executed_steps": {0: executed_steps},
                "finished_env_idx_list": [0] if TASK_ENV.is_episode_end() else [],
            })
        except BaseException as error:
            _discard_after_error(model_client, error, execution_started=execution_started)
            raise


def _eval_robodojo_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(
            func_name="update_obs_batch",
            obs=TASK_ENV.get_obs_batch(env_idx_list),
        )
        execution_started = False
        try:
            actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
            if len(actions) != len(env_idx_list) or not actions:
                raise ValueError("WorldScapePolicy returned an invalid action batch")
            horizons = {len(chunk) for chunk in actions}
            if len(horizons) != 1 or 0 in horizons:
                raise ValueError("WorldScapePolicy action chunks must have equal non-zero lengths")
            chunk_size = len(actions[0])
            actions_by_env = dict(zip(env_idx_list, actions, strict=True))
            executed_steps = dict.fromkeys(env_idx_list, 0)
            finished = set()
            for action_idx in range(chunk_size):
                current_action_list = [actions_by_env[env][action_idx] for env in env_idx_list]
                execution_started = True
                TASK_ENV.take_action_batch(current_action_list, env_idx_list)
                for env in env_idx_list:
                    executed_steps[env] += 1
                ended = TASK_ENV.is_episode_end()
                running = set() if ended else set(TASK_ENV.get_running_env_idx_list())
                finished.update(set(env_idx_list) - running)
                env_idx_list = [env for env in env_idx_list if env in running]
                if not env_idx_list or action_idx + 1 == chunk_size:
                    break
                model_client.call(
                    func_name="update_obs_batch",
                    obs=TASK_ENV.get_obs_batch(env_idx_list),
                )
            model_client.call(func_name="commit_action", obs={
                "executed_steps": executed_steps,
                "finished_env_idx_list": sorted(finished),
            })
        except BaseException as error:
            _discard_after_error(model_client, error, execution_started=execution_started)
            raise


def eval_one_episode(TASK_ENV, model_client):
    info = model_client.call(func_name="get_runtime_info")
    if info["requires_explicit_commit"]:
        return _eval_robodojo_episode(TASK_ENV, model_client)
    return _eval_robotwin_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    info = model_client.call(func_name="get_runtime_info")
    if not info["supports_batch"]:
        raise NotImplementedError("robotwin2 requires eval_batch=false and one environment")
    return _eval_robodojo_batch(TASK_ENV, model_client)
