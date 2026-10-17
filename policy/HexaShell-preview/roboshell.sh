#!/usr/bin/env bash
# HexaShell-preview entry point. Run from anywhere; settings come from config.env next to this file.
#   ./roboshell.sh setup                      build the agent image and install the policy-server deps
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
    echo "setup done" ;;
  check-task) source "$ROOT/scripts/sim_env.sh"; exec "$ISAAC_PYTHON" "$ROOT/eval/check_task.py" "$@" ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac
