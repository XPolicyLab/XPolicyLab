"""Reference XPolicyLab loop with evaluation-scoped action-step memory."""

from uuid import uuid4


def _scope(task_env):
    config = getattr(task_env, "deploy_cfg", {}) or {}
    value = config.get("evaluation_id") or getattr(task_env, "run_id", None) or str(uuid4())
    return {"evaluation_id": str(value)}


def eval_one_episode(TASK_ENV, model_client):
    scope = _scope(TASK_ENV)
    model_client.call(func_name="reset_evaluation", obs=scope)
    try:
        while not TASK_ENV.is_episode_end():
            obs = {**TASK_ENV.get_obs(), **scope, "env_idx": 0}
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action", obs=scope)
            if not actions:
                raise RuntimeError("SIPAI returned an empty action chunk")
            for step, action in enumerate(actions):
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end() or step + 1 == len(actions):
                    break
                obs = {**TASK_ENV.get_obs(), **scope, "env_idx": 0}
                model_client.call(func_name="update_obs", obs=obs)
    finally:
        model_client.call(func_name="reset_evaluation", obs=scope)


def _batch_obs(task_env, indices, scope):
    rows = task_env.get_obs_batch(indices)
    if len(rows) != len(indices):
        raise RuntimeError("Environment observation batch has the wrong length")
    return [{**row, **scope, "env_idx": index} for row, index in zip(rows, indices)]


def eval_one_episode_batch(TASK_ENV, model_client):
    scope = _scope(TASK_ENV)
    model_client.call(func_name="reset_evaluation", obs=scope)
    try:
        while not TASK_ENV.is_episode_end():
            indices = list(TASK_ENV.get_running_env_idx_list())
            if not indices:
                raise RuntimeError("No active environments before episode end")
            model_client.call(func_name="update_obs_batch", obs=_batch_obs(TASK_ENV, indices, scope))
            actions = model_client.call(func_name="get_action_batch", obs={**scope, "env_idx_list": indices})
            if len(actions) != len(indices) or not actions or not actions[0]:
                raise RuntimeError("Invalid SIPAI action batch")
            chunk_size = len(actions[0])
            if any(len(chunk) != chunk_size for chunk in actions):
                raise RuntimeError("SIPAI returned unequal chunk lengths")
            for step in range(chunk_size):
                TASK_ENV.take_action_batch([chunk[step] for chunk in actions], indices)
                if TASK_ENV.is_episode_end() or step + 1 == chunk_size:
                    break
                running = set(TASK_ENV.get_running_env_idx_list())
                active = [i for i, index in enumerate(indices) if index in running]
                actions = [actions[i] for i in active]
                indices = [indices[i] for i in active]
                if not indices:
                    break
                model_client.call(
                    func_name="update_obs_batch",
                    obs=_batch_obs(TASK_ENV, indices, scope),
                )
    finally:
        model_client.call(func_name="reset_evaluation", obs=scope)
