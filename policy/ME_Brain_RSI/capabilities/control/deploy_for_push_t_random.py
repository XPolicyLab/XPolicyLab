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
    raise NotImplementedError("Baseline deliberately runs one independent episode per worker")
