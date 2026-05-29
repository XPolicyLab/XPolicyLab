"""
Mem_0 deployment loop.

Mem_0's ``get_action`` returns the full predicted action chunk
(``action_horizon`` steps in env layout). We execute the chunk step by step,
re-observing between steps so the MemoryBank keeps a fresh temporal context --
the same rollout shape as policy/DP/deploy.py. Single-environment only:
Mem_0's MemoryBank carries per-episode state, so batch inference is not wired
(``eval_one_episode_batch`` will surface a clear NotImplementedError from the
model server).
"""


def eval_one_episode(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)

        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        for action_idx in range(len(actions[0])):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions[0]):
                break

            env_idx_list = TASK_ENV.get_running_env_idx_list()
            obs_list = TASK_ENV.get_obs_batch(env_idx_list)
            model_client.call(func_name="update_obs_batch", obs=obs_list)
