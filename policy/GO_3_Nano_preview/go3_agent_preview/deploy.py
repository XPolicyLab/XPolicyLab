"""Environment-side evaluation loop for go3_agent_preview.

RoboDojo constructs the websocket client with its protocol defaults and passes it
in; the code branch can plan for longer than that default budget on a few tasks,
which would fail the round with `WsError TIMEOUT`. The budget is the client's own
config, so this adapter raises it from here rather than changing shared transport
code: `PolicyEvalClient` reads `config.request_timeout_s` per request, so editing
the field takes effect immediately.

Both variables are optional and only ever raise the budget:

    XPL_REQUEST_TIMEOUT_S     per-request budget, seconds
    XPL_HANDSHAKE_TIMEOUT_S   handshake budget, seconds

An unset or blank variable leaves the protocol default untouched.
"""
from __future__ import annotations

import os


def _positive_seconds(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _apply_budget(model_client) -> None:
    """Raise the request and handshake budgets on the client we were handed.

    The client owns a `PolicyEvalClient` whose `config` dataclass is read on each
    request. A different transport (a test double, or a future client without that
    structure) simply keeps its own defaults.
    """
    request = _positive_seconds("XPL_REQUEST_TIMEOUT_S")
    handshake = _positive_seconds("XPL_HANDSHAKE_TIMEOUT_S")
    if request is None and handshake is None:
        return
    client = getattr(model_client, "_client", None)
    config = getattr(client, "config", None)
    if config is None:
        return
    if request is not None:
        config.request_timeout_s = request
    if handshake is not None:
        config.handshake_timeout_s = handshake


def eval_one_episode(TASK_ENV, model_client):
    _apply_budget(model_client)
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    _apply_budget(model_client)
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = list(TASK_ENV.get_running_env_idx_list())
        if not env_idx_list:
            break

        model_client.call(
            func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
        )
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        if len(actions) != len(env_idx_list) or not actions:
            raise ValueError(
                "go3_agent_preview returned a batch that does not match active environments"
            )
        chunk_size = len(actions[0])
        if chunk_size == 0 or any(len(chunk) != chunk_size for chunk in actions):
            raise ValueError("go3_agent_preview returned empty or unequal action chunks")

        for action_idx in range(chunk_size):
            TASK_ENV.take_action_batch(
                [chunk[action_idx] for chunk in actions], env_idx_list
            )
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_positions = [
                i for i, env_idx in enumerate(env_idx_list) if env_idx in running
            ]
            if not active_positions:
                break

            actions = [actions[i] for i in active_positions]
            env_idx_list = [env_idx_list[i] for i in active_positions]
            model_client.call(
                func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
            )
