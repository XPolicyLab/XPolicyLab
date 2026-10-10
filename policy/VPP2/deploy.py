import os
import socket
import uuid

# VPP2 keeps per-episode observation history on the policy server.  Every
# executed action records exactly one observation through update_obs, and each
# call carries this process's client id so one server can serve several
# simulator clients without mixing their histories.  A single-client official
# evaluation behaves exactly as without the id.
_CLIENT_ID_FIELD = "_vpp2_client_id"
_CLIENT_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _with_client_id(obs):
    tagged = dict(obs)
    tagged[_CLIENT_ID_FIELD] = _CLIENT_ID
    return tagged


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        obs = TASK_ENV.get_obs()  # Get Observation
        model_client.call(func_name="update_obs", obs=_with_client_id(obs))

        actions = model_client.call(func_name="get_action", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=_with_client_id(obs))
