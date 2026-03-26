#!/bin/bash

get_free_port() {
python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
}

get_action_dim() {
    local env_cfg_name="$1"
    python3 -c '
import sys, os, json, yaml
repo_root = os.getcwd()
env_cfg = yaml.safe_load(open(os.path.join(repo_root, "env_cfg", f"{sys.argv[1]}.yml"), "r", encoding="utf-8"))
robot_name = env_cfg["config"]["robot"]
robot_action_dim_info = json.load(open(os.path.join(repo_root, "env_cfg", "robot", "_robot_info.json"), "r", encoding="utf-8"))[robot_name]
print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "$env_cfg_name"
}

activate_conda() {
    local env_name="$1"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
}