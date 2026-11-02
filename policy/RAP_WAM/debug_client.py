"""Official XPolicyLab debug environment, with a selectable language instruction."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import debug_env_client

if "RAP_WAM_DEBUG_BATCH_SIZE" in os.environ:
    batch_size = int(os.environ["RAP_WAM_DEBUG_BATCH_SIZE"])
    if batch_size < 1:
        raise ValueError("RAP_WAM_DEBUG_BATCH_SIZE must be positive")
    debug_env_client.Batch_Size = batch_size

original = debug_env_client.TestEnv.get_obs


def get_obs(self, env_idx=0):
    obs = original(self, env_idx)
    obs["instruction"] = os.environ.get("RAP_WAM_DEBUG_INSTRUCTION", "Stack the three bowls together.")
    return obs


debug_env_client.TestEnv.get_obs = get_obs
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    for name in ("bench_name", "task_name", "env_cfg_type", "policy_name"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protocol", default="ws")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy_server_url")
    parser.add_argument("--evaluation_id", default="debug-eval")
    parser.add_argument("--trial_id", default="debug-trial")
    parser.add_argument("--action_case_id")
    parser.add_argument("--repeat_index", type=int)
    parser.add_argument("--eval_episode_num", type=int, default=1)
    parser.add_argument("--eval_batch", type=debug_env_client.str2bool, default=False)
    parser.add_argument("--obs_encoded", type=debug_env_client.str2bool, default=os.environ.get("DEBUG_OBS_ENCODED", "0"))
    cfg = vars(parser.parse_args())
    cfg["request_timeout_s"] = 600
    env = debug_env_client.TestEnv(cfg)
    try:
        for idx in range(cfg["eval_episode_num"]):
            env.reset()
            if cfg["eval_batch"]:
                env.eval_one_episode_batch()
            else:
                env.eval_one_episode()
            env.finish_episode()
    finally:
        close = getattr(env.model_client, "close", None)
        if callable(close):
            close()
