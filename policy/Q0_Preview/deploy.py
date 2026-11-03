"""Official environment bridge: observations/actions only; scoring stays in the client."""

def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
            actions = model_client.call(func_name="get_action")
            for action in actions:
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end():
                    break
    except BaseException:
        # Cleanup must not hide the original evaluator/transport error.
        try:
            model_client.call(func_name="trial_end", obs={})
        except Exception:
            pass
        raise
    else:
        # Standard lifecycle notification, with no score/success payload.
        model_client.call(func_name="trial_end", obs={})


def eval_one_episode_batch(TASK_ENV, model_client):
    if int(TASK_ENV.num_envs) != 1:
        raise ValueError("Q0_Preview supports one environment per policy server")
    eval_one_episode(TASK_ENV, model_client)
