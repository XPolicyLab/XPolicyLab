#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$POLICY_DIR/../../.." && pwd)"
METHOD=${1:?Usage: serve.sh jam|fastwam TASK VENV [PORT] [HOST] [GPU]}
TASK=${2:?}; VENV=${3:?}; PORT=${4:-9010}; HOST=${5:-127.0.0.1}; GPU=${6:-0}
case "$METHOD" in jam|fastwam) ;; *) echo 'Choose one fixed model: jam or fastwam' >&2; exit 2;; esac
exec bash "$POLICY_DIR/setup_eval_policy_server.sh" RoboDojo "$TASK" \
  "$WORKSPACE/models/$METHOD" piper_x joint 42 "$GPU" "$VENV" "$PORT" "$HOST"
