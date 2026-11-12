from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROUTES = Path(__file__).resolve().parent / 'routes.json'

def _engine(task_env: Any) -> str | None:
    try:
        return json.loads(ROUTES.read_text()).get(str(task_env.task_name), {}).get('engine')
    except (OSError, ValueError, AttributeError):
        return None

def _pictures_behind() -> int:

    try:
        from env.camera_manager.capture import render_sync
    except Exception:
        return 1
    return 0

def _status(task_env: Any, env_idx: int = 0) -> dict[str, Any]:
    info: dict[str, Any] = {"pictures_behind": _pictures_behind()}

    taken = getattr(task_env, "take_action_cnt", None)
    try:
        info["step"] = int(taken[env_idx])
    except (TypeError, IndexError, ValueError):
        if isinstance(getattr(task_env, "episode_step", None), int):
            info["step"] = task_env.episode_step
    for name in ("step_lim", "episode_step_limit"):
        try:
            info["step_limit"] = int(getattr(task_env, name))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    return info

def _waiting(actions: Any) -> bool:
    return (
        isinstance(actions, (list, tuple))
        and len(actions) == 1
        and isinstance(actions[0], dict)
        and actions[0].get("pyrualean") == "wait"
    )

def _judge_as_it_stands(task_env: Any) -> None:

    if task_env.is_episode_end() or not hasattr(task_env, "success"):
        return
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    for env_idx in running:
        task_env.success[env_idx] = False
    task_env.is_episode_end()

def _pyrua_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs()
            info = dict(obs.get("additional_info") or {})
            info["pyrualean"] = _status(TASK_ENV)
            obs["additional_info"] = info
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            while _waiting(actions):
                actions = model_client.call(func_name="get_action")
            if not actions:
                break
            for action in actions:
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end():
                    break
    finally:
        _judge_as_it_stands(TASK_ENV)
        verdict = dict(_status(TASK_ENV))
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:
            print(f"[REAL_RSI] the verdict did not reach the policy server: {exc}", flush=True)

def _frozen_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset")

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

def eval_one_episode(TASK_ENV, model_client):
    if _engine(TASK_ENV) == 'pyrua':
        _pyrua_episode(TASK_ENV, model_client)
    else:
        _frozen_episode(TASK_ENV, model_client)

def eval_one_episode_batch(TASK_ENV, model_client):

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
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
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
