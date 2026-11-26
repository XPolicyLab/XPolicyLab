"""One fixed batch per client/trial/GPU using the standard observation RPCs.

Run independent instances in separate threads/processes to use multiple GPUs.
The caller creates WsModelClient with unique evaluation_id + trial_id and closes
the connection afterwards. No simulator or upstream router is modified.
"""

from .official_client import environment_ids


def eval_one_episode_batch(TASK_ENV, model_client):
    """Run a server-supported batch, shrink on completion, release this trial."""
    task_env = TASK_ENV
    if task_env.is_episode_end():
        model_client.call("trial_end", {})
        return
    ids = environment_ids(task_env.get_running_env_idx_list())
    info = model_client.call("get_runtime_info")
    supported = info.get("supports_batch") if len(ids) > 1 else info.get("supports_single_env_batch", info.get("supports_batch"))
    if not supported or info.get("max_envs_per_session", 0) < len(ids):
        raise ValueError("Server does not support this batch size")
    try:
        model_client.call("reset")
        model_client.call("update_obs_batch", task_env.get_obs_batch(ids))
        while ids:
            chunks = model_client.call("get_action_batch", ids)
            if len(chunks) != len(ids) or not chunks or not chunks[0] or len({len(c) for c in chunks}) != 1:
                raise ValueError("Invalid action batch")
            for step in range(len(chunks[0])):
                task_env.take_action_batch([chunk[step] for chunk in chunks], ids)
                # Native RoboDojo updates end_flag only in this method.
                if task_env.is_episode_end():
                    ids = []
                    break
                running = set(task_env.get_running_env_idx_list())
                if not running.issubset(ids):
                    raise ValueError("Cannot add or restart an environment before resetting the batch")
                positions = [i for i, env in enumerate(ids) if env in running]
                ids = [ids[i] for i in positions]
                chunks = [chunks[i] for i in positions]
                if not ids:
                    break
                model_client.call("update_obs_batch", task_env.get_obs_batch(ids))
    except BaseException:
        # Preserve the original simulation/RPC error if cleanup also fails.
        try:
            model_client.call("trial_end", {})
        except Exception:
            pass
        raise
    else:
        model_client.call("trial_end", {})
