"""
Mem_0 deployment loop.

Ports upstream chunk-based eval: ``begin_episode`` initializes M1/Mn planner
state; ``step`` returns one action at a time with MemoryBank updates between
steps (action smoothing and Mn subtask switching happen on the model server).
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    obs = TASK_ENV.get_obs()
    model_client.call(func_name="begin_episode", obs=obs)

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        action = model_client.call(func_name="step", obs=obs)
        if action is None:
            continue
        TASK_ENV.take_action(action)

    model_client.call(func_name="reset")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError(
        "Mem_0 MemoryBank is per-episode stateful; set eval_batch=false in deploy.yml."
    )
