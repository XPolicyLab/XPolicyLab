"""RoboDojo evaluation loop preserving action/observation rendezvous.

The native environment alone decides termination, score and stability. This
adapter executes returned chunks without fabricating terminal outcomes.
"""

def configure_transport(model_client, startup_timeout_s=1800):
    """Compatibility hook for frozen collectors; never mutate an existing client.

    New callers must set request_timeout_s in deployment configuration before
    constructing their client. The obsolete startup-only budget is not applied.
    """
    import warnings
    warnings.warn(
        'configure_transport no longer modifies clients; configure request_timeout_s before client construction',
        DeprecationWarning, stacklevel=2)
    return model_client


def _indices(env):
    indices = list(env.get_running_env_idx_list())
    if (
        not indices
        or len(indices) > 10
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Expected one to ten distinct active environments")
    return indices


def _native_action(action):
    """Preserve the insert-key deploy.py gripper alias at the evaluator boundary."""
    result = dict(action)
    for arm in ('left', 'right'):
        old, new = arm + '_gripper', arm + '_ee_joint_state'
        if old in result:
            if new in result:
                raise ValueError('Ambiguous gripper action: both alias and native field')
            result[new] = result.pop(old)
    return result


def eval_one_episode_batch(TASK_ENV, model_client):
    env = TASK_ENV
    if env.is_episode_end():
        return
    indices = _indices(env)
    model_client.call(func_name="reset")
    model_client.call(func_name="update_obs_batch", obs=env.get_obs_batch(indices))
    while not env.is_episode_end():
        active = _indices(env)
        if not set(active) <= set(indices):
            raise RuntimeError("Environment joined an episode without a fresh reset")
        indices = active
        chunks = model_client.call(func_name="get_action_batch", obs=indices)
        if (
            not isinstance(chunks, list)
            or len(chunks) != len(indices)
            or any(
                not isinstance(chunk, list)
                or not chunk
                or any(not isinstance(a, dict) for a in chunk)
                for chunk in chunks
            )
            or len({len(chunk) for chunk in chunks}) != 1
        ):
            raise ValueError("Policy returned malformed or unequal batch action chunks")
        horizon = len(chunks[0])
        by_index = dict(zip(indices, chunks))
        for step in range(horizon):
            active = _indices(env)
            if not set(active) <= set(indices):
                raise RuntimeError("Unexpected environment during action chunk")
            actions = [_native_action(by_index[i][step]) for i in active]
            env.take_action_batch(actions, active)
            # The public observation API returns the environments still running.
            if env.is_episode_end():
                return
            observations = env.get_obs_batch(env.get_running_env_idx_list())
            model_client.call(func_name="update_obs_batch", obs=observations)
        indices = _indices(env)


def eval_one_episode(TASK_ENV, model_client):
    """Use the upstream single-environment API, independent of batch metadata."""
    env = TASK_ENV
    model_client.call(func_name="reset")
    while not env.is_episode_end():
        model_client.call(func_name="update_obs", obs=env.get_obs())
        actions = model_client.call(func_name="get_action")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Skill returned an empty action chunk")
        for index, action in enumerate(actions):
            env.take_action(_native_action(action))
            if env.is_episode_end() or index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=env.get_obs())
