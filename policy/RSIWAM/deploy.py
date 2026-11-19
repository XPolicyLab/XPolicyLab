"""Single-environment loop preserving native per-action observation history."""
import os
import socket
import uuid

_CLIENT_ID_FIELD = "_recoverwam_client_id"
_CLIENT_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _with_client_id(obs):
    tagged = dict(obs)
    tagged[_CLIENT_ID_FIELD] = _CLIENT_ID
    return tagged


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=_with_client_id(TASK_ENV.get_obs()))
        actions = model_client.call(func_name="get_action", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
        if not actions:
            raise ValueError("RSIWAM returned an empty action chunk")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=_with_client_id(TASK_ENV.get_obs()))


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("RSIWAM requires eval_batch=false")
