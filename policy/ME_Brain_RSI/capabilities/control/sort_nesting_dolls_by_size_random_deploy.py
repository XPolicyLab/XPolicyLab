"""Only the evaluator sees TASK_ENV. Policy sees a strict observation whitelist."""
from coap.contracts import restrict_observation


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs=restrict_observation(TASK_ENV.get_obs())
        model_client.call(func_name="update_obs",obs=obs)
        actions=model_client.call(func_name="get_action")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    # Preserve upstream batch policy semantics while keeping one simulator
    # environment per worker. No simulator handle enters the policy process.
    if TASK_ENV.num_envs != 1:
        raise ValueError("COAP development transport supports one environment")
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs={**restrict_observation(TASK_ENV.get_obs()), "env_idx": 0}
        model_client.call(func_name="update_obs_batch",obs=[obs])
        chunks=model_client.call(func_name="get_action_batch")
        if not isinstance(chunks,(list,tuple)) or len(chunks)!=1:
            raise ValueError("Expected exactly one environment action chunk")
        for action in chunks[0]:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
