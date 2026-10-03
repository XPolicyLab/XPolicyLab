"""RoboDojo evaluation loop preserving action/observation rendezvous.

The native environment alone decides termination, score and stability. This
adapter executes returned chunks without fabricating terminal outcomes.
"""

import inspect
import math


def configure_transport(model_client, startup_timeout_s=1800):
    """Budget lazy actor loading on the pinned XPolicyLab transport only.

    The first observation after every reset constructs the episode's actors.
    Later observations/actions retain the client's ordinary request budget.
    An uncertain execution poisons this session; cleanup remains available.
    """
    if (
        type(startup_timeout_s) not in (int, float)
        or not math.isfinite(startup_timeout_s)
        or not 0 < startup_timeout_s <= 86400
    ):
        raise ValueError("Startup timeout must be positive and at most 86400 seconds")
    try:
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.protocol.messages import MessageType
    except ImportError:
        return
    if not isinstance(model_client, WsModelClient):
        return
    client = model_client._client
    previous = getattr(client, "_physicalrsi_startup_budget", None)
    if previous is not None:
        if previous != startup_timeout_s:
            raise ValueError("Startup budget changed; create a new client")
        return
    original = client.request
    cold, failed = True, False

    async def request(msg_type, payload, **kwargs):
        nonlocal cold, failed
        execution = msg_type in (MessageType.RESET, MessageType.CALL)
        if execution and failed:
            raise RuntimeError("Interrupted policy session; create a new client")
        first_observation = (
            msg_type == MessageType.CALL
            and payload.get("func_name") in ("update_obs", "update_obs_batch")
            and cold
        )
        kwargs["_reconnect_attempted"] = True
        if first_observation:
            kwargs["timeout_s"] = startup_timeout_s
        try:
            result = await original(msg_type, payload, **kwargs)
        except BaseException:
            if execution:
                failed = True
            raise
        if msg_type == MessageType.RESET:
            cold = True
        elif first_observation:
            cold = False
        return result

    client.request = request
    client._physicalrsi_startup_budget = startup_timeout_s


def _indices(env):
    indices = list(env.get_running_env_idx_list())
    if (
        not indices
        or len(indices) > 10
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Expected one to ten distinct active environments")
    return indices


def eval_one_episode_batch(TASK_ENV, model_client):
    env = TASK_ENV
    if env.is_episode_end():
        return
    indices = _indices(env)
    configure_transport(
        model_client,
        getattr(env, "deploy_cfg", {}).get("physicalrsi_startup_timeout_s", 1800),
    )
    # Native RoboDojo exposes terminal-frame selection; XPolicyLab's debug
    # environment exposes only get_obs_batch(indices). Inspect once instead
    # of catching TypeError, which could hide a failure inside the observer.
    terminal_options = (
        {"last_frame": True}
        if "last_frame" in inspect.signature(env.get_obs_batch).parameters
        else {}
    )
    model_client.call(func_name="reset")
    model_client.call(func_name="update_obs_batch", obs=env.get_obs_batch(indices))
    while not env.is_episode_end():
        active = _indices(env)
        if not set(active) <= set(indices):
            raise RuntimeError("Environment joined an episode without a fresh reset")
        indices = active
        chunks = model_client.call(func_name="get_action_batch", obs=indices)
        if (
            not isinstance(chunks, list)
            or len(chunks) != len(indices)
            or any(
                not isinstance(chunk, list)
                or not chunk
                or any(not isinstance(a, dict) for a in chunk)
                for chunk in chunks
            )
            or len({len(chunk) for chunk in chunks}) != 1
        ):
            raise ValueError("Policy returned malformed or unequal batch action chunks")
        horizon = len(chunks[0])
        by_index = dict(zip(indices, chunks))
        for step in range(horizon):
            active = _indices(env)
            if not set(active) <= set(indices):
                raise RuntimeError("Unexpected environment during action chunk")
            actions = [by_index[i][step] for i in active]
            env.take_action_batch(actions, active)
            # Include environments that just terminated. No second update is
            # sent at the next loop head: every update follows an actual action.
            observations = env.get_obs_batch(active, **terminal_options)
            model_client.call(func_name="update_obs_batch", obs=observations)
            if env.is_episode_end():
                return
        indices = _indices(env)


def eval_one_episode(TASK_ENV, model_client):
    """Use the upstream single-environment API, independent of batch metadata."""
    env = TASK_ENV
    configure_transport(model_client, getattr(env, "deploy_cfg", {}).get("physicalrsi_startup_timeout_s", 1800))
    model_client.call(func_name="reset")
    while not env.is_episode_end():
        model_client.call(func_name="update_obs", obs=env.get_obs())
        actions = model_client.call(func_name="get_action")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Skill returned an empty action chunk")
        for index, action in enumerate(actions):
            env.take_action(action)
            if env.is_episode_end() or index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=env.get_obs())
