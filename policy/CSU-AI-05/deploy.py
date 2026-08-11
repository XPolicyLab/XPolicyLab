import collections
import os


G05_HISTORY_OBSERVATIONS_KEY = "__g05_history_observations__"


def _parse_g05_obs_offsets():
    raw = os.environ.get("ROBODOJO_G05_OBS_OFFSETS", "0")
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    offsets = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            offsets.append(int(item))
    return offsets or [0]


def _init_history_buffer():
    offsets = _parse_g05_obs_offsets()
    max_lag = max(0, -min(offsets))
    return offsets, collections.defaultdict(lambda: collections.deque(maxlen=max_lag + 1))


def _obs_with_g05_history(obs, history, offsets):
    """Attach training-aligned sparse history to one current observation.

    RoboDojo calls update_obs after every simulator action, but only calls
    get_action after a policy chunk is exhausted. Server-side buffers therefore
    see chunk-spaced observations, not frame-spaced observations. For G0.5
    temporal models trained with offsets such as [-150, -100, -50, 0], the
    client must keep frame-level history and send the selected frames explicitly.
    Missing early frames are clamped to the first observed frame, matching the
    dataset boundary behavior.
    """
    if offsets == [0]:
        return obs
    seq = list(history)
    if not seq:
        seq = [obs]
    selected = []
    for offset in offsets:
        idx = len(seq) - 1 + int(offset)
        if idx < 0:
            idx = 0
        elif idx >= len(seq):
            idx = len(seq) - 1
        selected.append(seq[idx])
    wrapped = dict(obs)
    wrapped[G05_HISTORY_OBSERVATIONS_KEY] = selected
    return wrapped


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    offsets, history_by_env = _init_history_buffer()

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        history_by_env["single"].append(obs)
        model_client.call(
            func_name="update_obs",
            obs=_obs_with_g05_history(obs, history_by_env["single"], offsets),
        )
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            history_by_env["single"].append(obs)
            model_client.call(
                func_name="update_obs",
                obs=_obs_with_g05_history(obs, history_by_env["single"], offsets),
            )


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    offsets, history_by_env = _init_history_buffer()

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        for env_idx, obs in zip(env_idx_list, obs_list):
            history_by_env[env_idx].append(obs)
        model_client.call(
            func_name="update_obs_batch",
            obs=[
                _obs_with_g05_history(obs, history_by_env[env_idx], offsets)
                for env_idx, obs in zip(env_idx_list, obs_list)
            ],
        )
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            obs_list = TASK_ENV.get_obs_batch(env_idx_list)
            for env_idx, obs in zip(env_idx_list, obs_list):
                history_by_env[env_idx].append(obs)
            model_client.call(
                func_name="update_obs_batch",
                obs=[
                    _obs_with_g05_history(obs, history_by_env[env_idx], offsets)
                    for env_idx, obs in zip(env_idx_list, obs_list)
                ],
            )
