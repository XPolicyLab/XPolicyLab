"""Client-side episode loop for the Awomo05 policy.

One invariant this loop must preserve: the policy pushes ONE frame into its cam_head history
buffer per `update_obs`, and reads the 20-slot history grid off that buffer assuming it advances
one entry per 25 Hz control step. So `update_obs` is called after EVERY executed action -- the
last action of a chunk is followed by the outer loop's own `update_obs`, which keeps the count
exact. Dropping observations to save a round trip would silently stretch the history timeline.
"""

import time


CAMERA_GROUPS = (
    ("cam_head", "cam_high", "head_camera"),
    ("cam_left_wrist", "cam_hand_left", "left_camera"),
    ("cam_right_wrist", "cam_hand_right", "right_camera"),
)


def _has_valid_images(obs):
    vision = obs.get("vision", {})
    for camera_names in CAMERA_GROUPS:
        for camera_name in camera_names:
            camera_data = vision.get(camera_name)
            if isinstance(camera_data, dict):
                if camera_data.get("color") is not None or camera_data.get("rgb") is not None:
                    return True
            elif camera_data is not None:
                return True
    return False


def _get_valid_obs(task_env, timeout=2.0, interval=0.05):
    deadline = time.monotonic() + timeout
    last_obs = None
    while time.monotonic() < deadline:
        last_obs = task_env.get_obs()
        if _has_valid_images(last_obs):
            return last_obs
        time.sleep(interval)
    raise RuntimeError(
        "Timed out waiting for valid camera observations. "
        f"Last obs keys: {list(last_obs.keys()) if isinstance(last_obs, dict) else type(last_obs)}"
    )


def _get_valid_obs_batch(task_env, env_idx_list, timeout=2.0, interval=0.05):
    deadline = time.monotonic() + timeout
    last_obs_list = None
    while time.monotonic() < deadline:
        last_obs_list = task_env.get_obs_batch(env_idx_list)
        if all(_has_valid_images(obs) for obs in last_obs_list):
            return last_obs_list
        time.sleep(interval)
    raise RuntimeError(
        "Timed out waiting for valid batch camera observations. "
        f"Last batch size: {len(last_obs_list) if isinstance(last_obs_list, list) else type(last_obs_list)}"
    )


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = _get_valid_obs(TASK_ENV)
        model_client.call(func_name="update_obs", obs=obs)

        actions = model_client.call(func_name="get_action")
        if actions is None:
            raise RuntimeError("Awomo05 policy server returned None for get_action. "
                               "Check the server-side traceback above.")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = _get_valid_obs(TASK_ENV)
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = _get_valid_obs_batch(TASK_ENV, env_idx_list)

        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if actions is None:
            raise RuntimeError("Awomo05 policy server returned None for get_action_batch. "
                               "Check the server-side traceback above.")

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
            model_client.call(func_name="update_obs_batch", obs=_get_valid_obs_batch(TASK_ENV, env_idx_list))
