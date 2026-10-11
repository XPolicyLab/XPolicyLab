#!/usr/bin/env bash
# Usage: run_camera_model.sh GPU  -> writes agent/camera_model.json
set -euo pipefail
GPU=$1
source "$(dirname "$0")/sim_env.sh"
OUT=$ROBOSHELL_ROOT/agent/camera_model.json
inner() {
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ROBODOJO_REPO"
  exec timeout 900 "$ISAAC_PYTHON" -u "$ROBOSHELL_ROOT/tests/probes/sim/camera_model.py" --device_id "$GPU" --device cuda:0 --out "$OUT" --headless --enable_cameras --kit_args "$KIT_ARGS"
}
export -f inner; export GPU OUT KIT_ARGS
run_on_gpu "$GPU" "RoboShell-camera-model" "${MIN_FREE_MIB:-12000}" bash -c inner > "$ROBOSHELL_ROOT/runs/camera_model.log" 2>&1 || { tail -20 "$ROBOSHELL_ROOT/runs/camera_model.log"; exit 1; }
grep -E "CAMERA-MODEL-DONE|check_after" "$OUT" "$ROBOSHELL_ROOT/runs/camera_model.log" | tail -2
