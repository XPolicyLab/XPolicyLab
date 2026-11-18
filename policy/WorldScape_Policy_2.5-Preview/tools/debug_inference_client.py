"""Exercise the real policy server and deployment loop without a simulator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

POLICY_DIR = Path(__file__).resolve().parents[1]
XPL_ROOT = POLICY_DIR.parents[1]
BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", str(XPL_ROOT.parent)))
sys.path[:0] = [str(BENCH_ROOT), str(XPL_ROOT)]

import numpy as np
import yaml

from XPolicyLab.utils.debug_env_client import TestEnv, validate_robot_state_dict


class InferenceDebugEnv(TestEnv):
    def __init__(self, config, *, instruction, steps, batch_size):
        super().__init__(config)
        self.instruction = instruction
        self.episode_step_limit = steps
        self.batch_size = batch_size
        self.checked_actions = 0

    def get_obs(self, env_idx=0):
        observation = super().get_obs(env_idx)
        observation["instruction"] = self.instruction
        return observation

    def _check_action(self, action):
        validate_robot_state_dict(action, self.robot_action_dim_info)
        expected = {"left_arm_joint_state", "left_ee_joint_state",
                    "right_arm_joint_state", "right_ee_joint_state"}
        if set(action) != expected or any(not np.isfinite(value).all() for value in action.values()):
            raise ValueError("Policy returned missing, unexpected or nonfinite joint actions")
        self.checked_actions += 1

    def take_action(self, action):
        self._check_action(action)
        self.episode_step += 1

    def take_action_batch(self, actions, env_idx_list):
        if len(actions) != len(env_idx_list):
            raise ValueError("Action batch does not match active environments")
        for action in actions:
            self._check_action(action)
        self.episode_step += 1

    def is_episode_end(self):
        return self.episode_step >= self.episode_step_limit

    def get_running_env_idx_list(self):
        return [] if self.is_episode_end() else list(range(self.batch_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, default=POLICY_DIR / "deploy_robodojo.yml")
    parser.add_argument("--bench_name", default="RoboDojo")
    parser.add_argument("--task_name", default="sort_nesting_dolls_by_size")
    parser.add_argument("--env_cfg_type", default="arx_x5")
    parser.add_argument("--policy_name", default="WorldScape_Policy_2.5-Preview")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--instruction", default=os.environ.get(
        "WSP_DEBUG_INSTRUCTION",
        "Arrange the five nesting dolls in a row from left to right, from smallest to largest."))
    parser.add_argument("--encoded", action="store_true", default=(
        os.environ.get("DEBUG_OBS_ENCODED", "0").lower() in {"1", "true", "yes"}))
    args = parser.parse_args()
    if min(args.steps, args.episodes, args.batch_size) < 1:
        parser.error("steps, episodes and batch-size must be positive")
    config = yaml.safe_load(args.config_path.read_text())
    config.update(bench_name=args.bench_name, task_name=args.task_name,
                  env_cfg_type=args.env_cfg_type, policy_name=args.policy_name,
                  host=args.host, port=args.port, seed=args.seed,
                  obs_encoded=args.encoded, protocol="ws", policy_server_url=None,
                  evaluation_id="debug-" + uuid.uuid4().hex, trial_id="debug-trial",
                  action_case_id=None, repeat_index=0)
    env = InferenceDebugEnv(config, instruction=args.instruction, steps=args.steps,
                            batch_size=args.batch_size)
    try:
        runtime = env.model_client.call(func_name="get_runtime_info")
        if not runtime.get("history_memory") or not runtime.get("vlm_memory"):
            raise ValueError("Debug endpoint is not the expected fused memory policy")
        for episode in range(args.episodes):
            env.reset()
            if config.get("eval_batch", False) or args.batch_size > 1:
                env.eval_one_episode_batch()
            else:
                env.eval_one_episode()
            print(json.dumps({"event": "episode_complete", "episode": episode,
                              "steps": env.episode_step, "encoded": args.encoded}), flush=True)
        print(json.dumps({"event": "debug_passed", "episodes": args.episodes,
                          "actions_checked": env.checked_actions, "batch_size": args.batch_size,
                          "encoded": args.encoded}), flush=True)
        print("[MAIN] eval finished", flush=True)
    finally:
        env.model_client.close()


if __name__ == "__main__":
    main()
