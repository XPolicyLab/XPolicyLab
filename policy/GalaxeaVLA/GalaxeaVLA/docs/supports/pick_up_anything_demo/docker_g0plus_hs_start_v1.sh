#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="g0_three_tasks"
CURL_RETRIES=5
CURL_SLEEP=2


usage() {
    echo "Usage: $0 <0 or1>"
    echo "  0 - indicates that VLM uses the default Gemini."
    echo "  1 - indicates that VLM uses Qwen."
}

if [ $# -ne 1 ]; then
    echo "Error: A parameter of 0 or 1 is required to select the model used by VLM."
    usage
    exit 1
fi

VLM_USE_QWEN=$1

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
	echo "Existed tmux session: ${SESSION_NAME}, eliminating..."
    tmux kill-session -t "${SESSION_NAME}"
    echo "Original tmux session: ${SESSION_NAME}has been killed！"
fi

ensure_tmux_installed() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not detected, installing..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y tmux
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y tmux
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y tmux
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -Sy --noconfirm tmux
    else
      echo "No known package manager found. Please install tmux manually and try again."
      exit 1
    fi
  fi
}

create_tmux_windows_and_run() {
  tmux new-session -d -s "${SESSION_NAME}" -n g0_VLA_node

  # ---------------- Window 0 ----------------
  CMD1="cd /home/ros/g0plus_ros2/EFMNode && bash scripts/run.sh"
  tmux send-keys -t "${SESSION_NAME}:0" "bash -lc '${CMD1}; exec \$SHELL'" C-m

  # ---------------- Window 1 ----------------
tmux new-window -t "${SESSION_NAME}" -n g0_VLM_node

CMD2=$(cat <<EOF

if [[ "${VLM_USE_QWEN}" == "1" ]]; then
  echo "[Code2] VLM_USE_QWEN=1, start directly using Qwen mode (skipping proxy and network detection)..."
  cd /home/ros/g0_ros2/Hierarchical_System || { echo "[Code2] cannot enter the work directory"; exec \$SHELL; }
  ros2 run g0_vlm_node vlm_main -- --use-qwen
  echo "[Code2] ros2 run exited."
  exec \$SHELL
fi

export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897

cd /home/ros/g0plus_ros2/Hierarchical_System || { echo "[Code2] 无法进入工作目录"; exec \$SHELL; }

echo "[Code2] VLM_USE_QWEN!=1, proxy has been set, start checking network connectivity (up to 5 times)...."
for i in {1..5}; do
  RESP=\$(curl -sI --max-time 8 www.google.com | head -n 1)
  echo "[Try #\$i] \$RESP"
  if echo "\$RESP" | grep -q "200 OK"; then
    echo "[Code2] Detected 200 OK, start to run ROS2 node..."
    ros2 run g0_vlm_node vlm_main -- --no-use-qwen
    echo "[Code2] ros2 run exited."
    exec \$SHELL
  fi
  sleep 2
done

echo "[Code2][Error] Unable to obtain 200 OK after multiple attempts, ROS2 run has stopped. Please check your proxy or network settings."
exec \$SHELL
EOF
)

  tmux send-keys -t "${SESSION_NAME}:1" "bash -lc '${CMD2}'" C-m

  # ---------------- Window 2 ----------------
  tmux new-window -t "${SESSION_NAME}" -n rosbridge_for_EHI
  CMD3="ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
  tmux send-keys -t "${SESSION_NAME}:2" "bash -lc '${CMD3}; exec \$SHELL'" C-m
  
  tmux select-window -t "${SESSION_NAME}:0"

  echo "tmux session: ${SESSION_NAME} created."
  echo "Enter command to check: tmux attach -t ${SESSION_NAME}"
}

main() {
  ensure_tmux_installed
  create_tmux_windows_and_run
}

main
