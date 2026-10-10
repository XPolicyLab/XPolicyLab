#!/usr/bin/env bash
# Detached single-node launch: <config> [overrides...]. Prints PID and log path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?usage: train_detached.sh <config> [overrides...]}
shift || true

RUN_ID=$(date +%Y%m%d-%H%M%S)
LOGDIR="$REPO/runs/detached"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${RUN_ID}.log"

# setsid detaches into a new session (immune to the caller's SIGHUP); nohup is a
# belt-and-braces second guard. Output goes to the log only, never the tty.
setsid nohup bash "$REPO/scripts/train_single_node.sh" "$CONFIG" "$@" \
  >"$LOG" 2>&1 < /dev/null &

PID=$!
disown "$PID" 2>/dev/null || true
sleep 1
echo "detached training launched"
echo "  launcher pid : $PID   (session leader; survives this shell exiting)"
echo "  log          : $LOG"
echo "watch with: tail -f $LOG"
