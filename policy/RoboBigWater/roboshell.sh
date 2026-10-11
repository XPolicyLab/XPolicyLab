#!/usr/bin/env bash
# RoboBigWater entry point. Run from anywhere; settings come from config.env next to this file.
#   ./roboshell.sh setup                      build the agent image, install policy deps, link into RoboDojo
#   ./roboshell.sh check                      agent side only, no GPU: fake server + one Codex turn
#   ./roboshell.sh serve GPU TASK [SEED]      direct mode: start robo-server (foreground)
#   ./roboshell.sh episode TASK LAYOUT [OUT]  direct mode: run one episode against the running server
#   ./roboshell.sh regression [TASK]          contract regression against the running server
#   ./roboshell.sh check-task TASK            delivery check of tasks/TASK (files, tools load, manual has no task words)
# The official evaluation is eval.sh in this directory (XPolicyLab adapter entry).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cmd=${1:-help}; shift || true
case "$cmd" in
  setup)
    [ -f "$ROOT/config.env" ] || { echo "copy config.env.example to config.env and edit it first"; exit 1; }
    "$ROOT/agent/build_docs.sh"
    "$ROOT/agent/codex/build.sh"
    "$ROOT/scripts/install_policy_deps.sh"
    source "$ROOT/scripts/sim_env.sh"
    cp -n "$ROOT/env_cfg/arx_x5_rgbd.yml" "$ROBODOJO_REPO/env_cfg/" 2>/dev/null || true
    cp -n "$ROOT/env_cfg/camera_config_rgbd.yml" "$ROBODOJO_REPO/env_cfg/camera/" 2>/dev/null || true
    [ -f "$ROOT/agent/camera_model.json" ] || echo "note: agent/camera_model.json missing; run scripts/run_camera_model.sh GPU once"
    echo "setup done" ;;
  check)   LANE=check "$ROOT/tests/probes/codex/run_probe_container.sh" check "$ROOT/tests/probes/codex/prompt_min.txt" "${PORT:-28750}"
           cat "$ROOT/runs/m0/check/last_message.txt"; echo ;;
  serve)   exec "$ROOT/scripts/start_server.sh" "$@" ;;
  episode) exec "$ROOT/eval/run_episode.sh" "$@" ;;
  check-task) source "$ROOT/scripts/sim_env.sh"; exec "$ISAAC_PYTHON" "$ROOT/eval/check_task.py" "$@" ;;
  regression)
    export ROBO_ADMIN_SERVER=http://127.0.0.1:${ADMIN_PORT:-28701} ROBO_SERVER=http://$("$ROOT/agent/codex/run_agent.sh" --gateway):${PORT:-28700}
    exec python3 "$ROOT/tests/test_server_regression.py" "${1:-stack_bowls}" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
