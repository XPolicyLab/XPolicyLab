"""Unmodified simulator episode loop; no result or layout is sent to the policy."""
def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name='update_obs', obs=TASK_ENV.get_obs())
        for action in model_client.call(func_name='get_action'):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
    model_client.call(func_name='finish_episode')


def eval_one_episode_batch(TASK_ENV, model_client):
    return eval_one_episode(TASK_ENV, model_client)
