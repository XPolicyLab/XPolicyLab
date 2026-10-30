"""Official RGB observation forwarding and action execution only."""
import json

def _policy_obs(obs):
    return {"vision": {name: {"color": obs["vision"][name]["color"]} for name in
                       ("cam_head", "cam_left_wrist", "cam_right_wrist")},
            "state": obs["state"]}

def _log_initial_state(index, obs):
    print("[insert_initial_proprio] " + json.dumps({"env_idx": index, "state": obs["state"]}, default=lambda v: v.tolist()), flush=True)
def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name='update_obs', obs=_policy_obs(TASK_ENV.get_obs()))
        actions = model_client.call(func_name='get_action')
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    action_counts = {}
    while not TASK_ENV.is_episode_end():
        indices = list(TASK_ENV.get_running_env_idx_list())
        if not indices:
            break
        observations = TASK_ENV.get_obs_batch(indices)
        for i, observation in zip(indices, observations, strict=True):
            if i not in action_counts:
                _log_initial_state(i, observation)
                action_counts[i] = 0
        model_client.call(func_name='update_obs_batch', obs=[dict(_policy_obs(obs), env_idx=int(i)) for i, obs in zip(indices, observations, strict=True)])
        chunks = model_client.call(func_name='get_action_batch', obs=indices)
        TASK_ENV.take_action_batch([chunk[0] for chunk in chunks], indices)
        for i in indices:
            action_counts[i] += 1
    print('[insert_action_counts] ' + json.dumps(action_counts), flush=True)
