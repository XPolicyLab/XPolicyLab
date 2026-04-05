#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"

python3 -c '
import sys, os, json, yaml

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]

env_cfg = yaml.safe_load(
    open(os.path.join(root_dir, "env_cfg", f"{env_cfg_type}.yml"), "r", encoding="utf-8")
)
robot_name = env_cfg["config"]["robot"]
robot_action_dim_info = json.load(
    open(os.path.join(root_dir, "env_cfg", "robot", "_robot_info.json"), "r", encoding="utf-8")
)[robot_name]

print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}"