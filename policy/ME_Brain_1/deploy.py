"""Collect full head history at every environment step before websocket inference."""

import os

from .hist_live import LiveHeadHistoryTracker


def _history_tracker():
    mode = os.environ.get("FOCUS_VLWA_HISTORY_MODE", "head_history")
    if mode != "head_history":
        raise ValueError(f"Unsupported history mode: {mode}")
    return LiveHeadHistoryTracker()


def eval_one_episode(TASK_ENV, model_client):
    tracker = _history_tracker()
    model_client.call(func_name="reset")
    observation = TASK_ENV.get_obs()
    while not TASK_ENV.is_episode_end():
        tracker.stamp_obs(observation, 0)
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        if actions is None or len(actions) == 0:
            raise RuntimeError("Policy returned an empty action chunk")
        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
            observation = TASK_ENV.get_obs()
            if index + 1 < len(actions):
                tracker.stamp_obs(observation, 0)


def eval_one_episode_batch(TASK_ENV, model_client):
    tracker = _history_tracker()
    model_client.call(func_name="reset")
    observations = None
    while not TASK_ENV.is_episode_end():
        env_ids = TASK_ENV.get_running_env_idx_list()
        if not env_ids:
            break
        observations = TASK_ENV.get_obs_batch(env_ids)
        tracker.stamp_obs_list(observations, env_ids)
        model_client.call(func_name="update_obs_batch", obs=observations)
        chunks = model_client.call(func_name="get_action_batch", obs=env_ids)
        if len(chunks) != len(env_ids) or any(len(chunk) == 0 for chunk in chunks):
            raise RuntimeError("Policy returned invalid batched action chunks")
        if len({len(chunk) for chunk in chunks}) != 1:
            raise RuntimeError("All environment action chunks must have the same length")
        length = len(chunks[0])
        for index in range(length):
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [(env_id, chunk) for env_id, chunk in zip(env_ids, chunks) if env_id in running]
            if not active:
                break
            env_ids = [env_id for env_id, _ in active]
            chunks = [chunk for _, chunk in active]
            TASK_ENV.take_action_batch([chunk[index] for chunk in chunks], env_ids)
            if TASK_ENV.is_episode_end() or index + 1 == length:
                break
            running_ids = TASK_ENV.get_running_env_idx_list()
            if running_ids:
                tracker.stamp_obs_list(TASK_ENV.get_obs_batch(running_ids), running_ids)
