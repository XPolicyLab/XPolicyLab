"""RoboDojo rollout and explicit adaptation of the shared debug environment."""

import os


def _is_debug():
    return os.environ.get("EVAL_ENV_TYPE") == "debug"


def _prepare(env, client):
    if not _is_debug():
        env.obs_manager.collect_intrinsic_matrix = True
        env.obs_manager.collect_extrinsic_matrix = True
    client.call(func_name="reset")


def _observations(env, indices):
    observations = env.get_obs_batch(indices)
    for obs in observations:
        index = int(obs["env_idx"])
        if _is_debug():
            # Patch TestEnv
            obs["env_origin"] = [0.0, 0.0, 0.0]
            obs["episode_identity"] = {"task": env.deploy_cfg["task_name"], "eval_seed":0,
                                       "layout_id": index}
            obs["control_step"] = int(env.episode_step)
        else:
            obs["env_origin"] = env.robot_manager.scene.env_origins[index].cpu().tolist()
            obs["episode_identity"] = {"task": env.task_name, "eval_seed": int(env.eval_seed),
                                       "layout_id": int(env.env_seeds[index])}
            obs["control_step"] = int(env.take_action_cnt[index])
    return observations


def eval_one_episode(TASK_ENV, model_client):
    _prepare(TASK_ENV, model_client)
    while not TASK_ENV.is_episode_end():
        observations = _observations(TASK_ENV, [0])
        model_client.call(func_name="update_obs", obs=observations[0])
        actions = model_client.call(func_name="get_action")
        for i, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or i + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=_observations(TASK_ENV, [0])[0])


def eval_one_episode_batch(TASK_ENV, model_client):
    _prepare(TASK_ENV, model_client)
    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        observations = _observations(TASK_ENV, indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        actions = model_client.call(func_name="get_action_batch", obs=indices)
        for i in range(len(actions[0])):
            TASK_ENV.take_action_batch([chunk[i] for chunk in actions], indices)
            if TASK_ENV.is_episode_end() or i + 1 == len(actions[0]):
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            remaining = [j for j, index in enumerate(indices) if index in running]
            actions = [actions[j] for j in remaining]
            indices = [indices[j] for j in remaining]
            model_client.call(func_name="update_obs_batch", obs=_observations(TASK_ENV, indices))
