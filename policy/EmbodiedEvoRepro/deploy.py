"""Native deployment delegates all evaluation semantics to the pinned adapter."""

from .configuration import interface_probe_enabled, verified_upstream


def eval_one_episode(TASK_ENV, model_client):
    if interface_probe_enabled():
        # The official debug TestEnv has no physics or native success fields.
        # This loop deliberately emits no task score or trial success result.
        model_client.call(func_name="reset")
        while not TASK_ENV.is_episode_end():
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
            actions = model_client.call(func_name="get_action")
            if len(actions) != 1:
                raise ValueError("The interface probe must return exactly one action")
            TASK_ENV.take_action(actions[0])
        return
    if len(TASK_ENV.end_flag) != 1:
        raise ValueError("Native evaluator must run exactly one environment")
    verified_upstream()
    from integrations.robodojo_policy.policy.EmbodiedEvo.deploy import eval_one_episode as native_eval
    native_eval(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise ValueError("This candidate supports num_envs=1 and eval_batch=false only")
