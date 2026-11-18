import os
from pathlib import Path
from uuid import uuid4


def _observation_interval(model_client) -> int:
    info = model_client.call(func_name="get_runtime_info")
    interval = info.get("observation_interval")
    if type(interval) is not int or interval < 1:
        raise ValueError(f"Policy observation_interval must be a positive integer, got {interval!r}")
    return interval


def _robodojo_capture(model_client):
    """Name images by completed simulator actions, independently of RPC counts."""
    info = model_client.call(func_name="get_runtime_info")
    interval = info.get("observation_interval")
    if type(interval) is not int or interval < 1:
        raise ValueError(f"Policy observation_interval must be a positive integer, got {interval!r}")
    memory = info.get("history_memory", False)
    if memory:
        expected_checkpoint = os.environ.get("WORLDSCAPE_CHECKPOINT")
        actual_checkpoint = info.get("checkpoint")
        if expected_checkpoint and actual_checkpoint and (
            Path(expected_checkpoint).expanduser().resolve() != Path(actual_checkpoint).expanduser().resolve()
        ):
            raise ValueError(
                f"Memory server loaded {actual_checkpoint}, but this client expects {expected_checkpoint}; "
                "restart the servers with the selected checkpoint before evaluating"
            )
        capture_mode = info.get("history_capture_mode", "dense")
        if capture_mode == "chunk_boundary":
            if (interval, info.get("action_horizon"), info.get("replan_steps")) != (24, 24, 24):
                raise ValueError("History chunk_boundary capture requires interval=horizon=replan_steps=24")
        elif capture_mode != "dense" or interval != 1:
            raise ValueError("History dense capture requires every real control-step observation")
    episode_id = uuid4().hex

    def stamp(obs, step):
        if not memory:
            return obs
        return {**obs, "wsp_episode_id": episode_id, "wsp_control_step": step}

    return interval, stamp


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
    interval, stamp = _robodojo_capture(model_client)
    model_client.call(func_name="reset")
    control_step = 0
    # The frame after a finished chunk is already stored. Do not read it again.
    skip_leading_obs = False

    while not TASK_ENV.is_episode_end():
        if not skip_leading_obs:
            model_client.call(func_name="update_obs", obs=stamp(TASK_ENV.get_obs(), control_step))
        skip_leading_obs = False
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
                control_step += 1
                episode_done = TASK_ENV.is_episode_end()
                chunk_done = action_idx + 1 == len(actions)
                if executed_steps % interval == 0 and not episode_done:
                    model_client.call(func_name="update_obs", obs=stamp(TASK_ENV.get_obs(), control_step))
                    skip_leading_obs = chunk_done
                if episode_done or chunk_done:
                    break
            model_client.call(func_name="commit_action", obs={
                "executed_steps": {0: executed_steps},
                "finished_env_idx_list": [0] if TASK_ENV.is_episode_end() else [],
            })
        except BaseException as error:
            _discard_after_error(model_client, error, execution_started=execution_started)
            raise


def _eval_robodojo_batch(TASK_ENV, model_client):
    interval, stamp = _robodojo_capture(model_client)
    model_client.call(func_name="reset")
    control_steps = {}
    skip_leading_obs = set()

    def observations(envs):
        raw = TASK_ENV.get_obs_batch(envs)
        return [stamp(obs, control_steps[env]) for env, obs in zip(envs, raw, strict=True)]

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        for env in env_idx_list:
            control_steps.setdefault(env, 0)
        to_send = [env for env in env_idx_list if env not in skip_leading_obs]
        skip_leading_obs = set()
        if to_send:
            model_client.call(
                func_name="update_obs_batch",
                obs=observations(to_send),
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
                current_action_list = [
                    actions_by_env[env][action_idx]
                    for env in env_idx_list
                ]
                execution_started = True
                TASK_ENV.take_action_batch(current_action_list, env_idx_list)
                for env in env_idx_list:
                    executed_steps[env] += 1
                    control_steps[env] += 1
                ended = TASK_ENV.is_episode_end()
                running = set() if ended else set(TASK_ENV.get_running_env_idx_list())
                finished.update(set(env_idx_list) - running)
                stepped = env_idx_list
                chunk_done = action_idx + 1 == chunk_size
                aligned = [
                    env for env in stepped
                    if env in running and executed_steps[env] % interval == 0
                ]
                if aligned:
                    model_client.call(
                        func_name="update_obs_batch",
                        obs=observations(aligned),
                    )
                    if chunk_done:
                        skip_leading_obs.update(aligned)
                env_idx_list = [env for env in stepped if env in running]
                if not env_idx_list or chunk_done:
                    break
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
