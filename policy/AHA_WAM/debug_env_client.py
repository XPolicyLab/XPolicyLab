import argparse
import json
import os
from pathlib import Path

import numpy as np
from client_server.ws import WsModelClient


BATCH_SIZE = 10


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "1"):
        return True
    if lowered in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _load_simple_yaml(path: Path) -> dict:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        data = {}
        stack = [(-1, data)]
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.split("#", 1)[0].rstrip()
                if not raw.strip() or ":" not in raw:
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                key, value = raw.strip().split(":", 1)
                value = value.strip().strip("\"'")
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if value:
                    parent[key] = value
                else:
                    child = {}
                    parent[key] = child
                    stack.append((indent, child))
        return data


def _load_robot_action_dim_info(env_cfg_type: str, env_cfg_root: str) -> dict:
    root = Path(env_cfg_root)
    env_cfg = _load_simple_yaml(root / f"{env_cfg_type}.yml")
    robot_name = env_cfg["config"]["robot"]
    with (root / "robot" / "_robot_info.json").open("r", encoding="utf-8") as f:
        return json.load(f)[robot_name]


def _validate_robot_state_dict(state_dict: dict, robot_action_dim_info: dict) -> None:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(arm_dims) != len(ee_dims):
        raise ValueError("robot_action_dim_info arm/ee lengths do not match.")

    if len(arm_dims) == 1:
        expected = {
            "arm_joint_state": arm_dims[0],
            "ee_joint_state": ee_dims[0],
            "ee_pose": 7,
            "tcp_pose": 7,
            "delta_ee_pose": 7,
        }
    elif len(arm_dims) == 2:
        expected = {
            "left_arm_joint_state": arm_dims[0],
            "left_ee_joint_state": ee_dims[0],
            "left_ee_pose": 7,
            "left_tcp_pose": 7,
            "left_delta_ee_pose": 7,
            "right_arm_joint_state": arm_dims[1],
            "right_ee_joint_state": ee_dims[1],
            "right_ee_pose": 7,
            "right_tcp_pose": 7,
            "right_delta_ee_pose": 7,
        }
    else:
        raise ValueError(f"Unsupported arm count: {len(arm_dims)}")

    unexpected_keys = [key for key in state_dict if key not in expected]
    if unexpected_keys:
        raise ValueError(f"Unexpected state keys: {unexpected_keys}")

    for key, expected_dim in expected.items():
        if key not in state_dict:
            continue
        value = np.asarray(state_dict[key])
        if value.ndim != 1 or value.shape[0] != expected_dim:
            raise ValueError(
                f"state_dict['{key}'] dim mismatch: expected {expected_dim}, got {value.shape}."
            )


class TestEnv:
    def __init__(self, deploy_cfg):
        self.deploy_cfg = deploy_cfg
        self.episode_step_limit = 5
        self.robot_action_dim_info = _load_robot_action_dim_info(
            deploy_cfg["env_cfg_type"],
            deploy_cfg["env_cfg_root"],
        )
        self.model_client = WsModelClient(
            url=f"ws://{deploy_cfg['host']}:{deploy_cfg['port']}",
            evaluation_id="aha-wam-debug-eval",
            trial_id="aha-wam-debug-trial",
        )
        self.episode_step = 0

#region agent log
        try:
            with open("/personal/tianxing/RoboDojo/XPolicyLab/.cursor/debug-c13f7c.log", "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "c13f7c",
                            "runId": "post-fix",
                            "hypothesisId": "H3,H4",
                            "location": "policy/AHA_WAM/debug_env_client.py:TestEnv.__init__",
                            "message": "initialized AHA_WAM websocket debug client",
                            "data": {
                                "host": deploy_cfg["host"],
                                "port": deploy_cfg["port"],
                                "env_cfg_type": deploy_cfg["env_cfg_type"],
                                "env_cfg_root": deploy_cfg["env_cfg_root"],
                                "robot_action_dim_info": self.robot_action_dim_info,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except Exception:
            pass
#endregion

    def get_obs(self, env_idx=0):
        obs = {
            "vision": {
                "cam_head": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "shape": (480, 640),
                },
                "cam_left_wrist": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "shape": (480, 640),
                },
                "cam_right_wrist": {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                    "shape": (480, 640),
                },
            },
            "instruction": "language instruction",
            "state": {},
            "additional_info": {"frequency": 30},
            "data_format_version": "v1.0",
            "env_idx": env_idx,
        }

        arm_dims = self.robot_action_dim_info["arm_dim"]
        ee_dims = self.robot_action_dim_info["ee_dim"]
        prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
        for index, prefix in enumerate(prefixes):
            obs["state"][f"{prefix}arm_joint_state"] = np.zeros(arm_dims[index], dtype=np.float32)
            obs["state"][f"{prefix}ee_joint_state"] = np.zeros(ee_dims[index], dtype=np.float32)
            obs["state"][f"{prefix}ee_pose"] = np.ones(7, dtype=np.float32)
            obs["state"][f"{prefix}tcp_pose"] = np.zeros(7, dtype=np.float32)
            obs["state"][f"{prefix}delta_ee_pose"] = np.zeros(7, dtype=np.float32)

        return obs

    def get_obs_batch(self, env_idx_list):
        return [self.get_obs(env_idx) for env_idx in env_idx_list]

    def eval_one_episode(self):
        from XPolicyLab.policy.AHA_WAM.deploy import eval_one_episode

        eval_one_episode(TASK_ENV=self, model_client=self.model_client)

    def eval_one_episode_batch(self):
        from XPolicyLab.policy.AHA_WAM.deploy import eval_one_episode_batch

        eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

    def reset(self):
        self.model_client.call(func_name="reset")
        self.episode_step = 0

    def take_action(self, action):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)")
        self.episode_step += 1
        _validate_robot_state_dict(action, self.robot_action_dim_info)

    def take_action_batch(self, action_list, env_idx_list):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)")
        self.episode_step += 1
        if len(action_list) != len(env_idx_list):
            raise ValueError(f"action num != env num: {len(action_list)} != {len(env_idx_list)}")
        for action in action_list:
            _validate_robot_state_dict(action, self.robot_action_dim_info)

    def is_episode_end(self):
        done = self.episode_step >= self.episode_step_limit
        print("[TestEnv] Check Episode End:", done)
        return done

    def finish_episode(self):
        print("[TestEnv] Episode finished")

    def get_running_env_idx_list(self):
        return list(range(BATCH_SIZE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--env_cfg_type", required=True)
    parser.add_argument("--env_cfg_root", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--eval_episode_num", type=int, default=1)
    parser.add_argument("--eval_batch", type=_str2bool, default=False)
    args = parser.parse_args()

    deploy_cfg = vars(args)
    test_env = TestEnv(deploy_cfg)
    try:
        for idx in range(args.eval_episode_num):
            print(f"\033[94m🚀 Running Episode {idx}\033[0m")
            test_env.reset()
            if args.eval_batch:
                test_env.eval_one_episode_batch()
            else:
                test_env.eval_one_episode()
            test_env.finish_episode()
    finally:
        close = getattr(test_env.model_client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
