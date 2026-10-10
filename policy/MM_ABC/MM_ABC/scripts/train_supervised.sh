#!/usr/bin/env bash
# Detached, auto-resuming multi-node training: <config> [overrides...].
# Stop future attempts by creating STOP in the printed supervisor log directory.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?usage: train_supervised.sh <config> [overrides...]}
shift || true

MAX_ATTEMPTS=${MAX_ATTEMPTS:-1000}   # hard ceiling so a crash-loop cannot run forever
BACKOFF_SECONDS=${BACKOFF_SECONDS:-60}
# Stop retrying after repeated fast failures; healthy attempts reset the backoff.
MIN_HEALTHY_SECONDS=${MIN_HEALTHY_SECONDS:-300}

# When invoked from an interactive shell, re-exec detached so a disconnect of
# that shell cannot take the supervisor with it. MMABC_SUPERVISED_DETACHED marks
# the already-detached re-exec so it does not recurse.
if [[ "${MMABC_SUPERVISED_DETACHED:-0}" != "1" ]]; then
  RUN_ID=$(date +%Y%m%d-%H%M%S)
  LOGDIR="$REPO/runs/supervised/$RUN_ID"
  mkdir -p "$LOGDIR"
  export MMABC_SUPERVISED_RUN_ID="$RUN_ID"
  export MMABC_SUPERVISED_LOGDIR="$LOGDIR"
  export MMABC_SUPERVISED_DETACHED=1
  setsid nohup bash "$REPO/scripts/train_supervised.sh" "$CONFIG" "$@" \
    >"$LOGDIR/supervisor.log" 2>&1 </dev/null &
  PID=$!
  disown "$PID" 2>/dev/null || true
  sleep 1
  echo "supervised training launched (detached; survives this shell/agent exiting)"
  echo "  supervisor pid : $PID"
  echo "  run id         : $RUN_ID"
  echo "  supervisor log : $LOGDIR/supervisor.log"
  echo "  per-attempt/per-node logs land under runs/launch/<timestamp>/"
  echo "  stop cleanly   : touch $LOGDIR/STOP   (or kill $PID)"
  echo "watch with: tail -f $LOGDIR/supervisor.log"
  exit 0
fi

# ---- detached body -------------------------------------------------------
LOGDIR="$MMABC_SUPERVISED_LOGDIR"
STOP_FILE="$LOGDIR/STOP"

log() { echo "[$(date '+%F %T')] $*"; }

log "supervisor start: config=$CONFIG overrides=$* repo=$REPO"
log "max_attempts=$MAX_ATTEMPTS backoff=${BACKOFF_SECONDS}s min_healthy=${MIN_HEALTHY_SECONDS}s"
log "data root override: MMABC_DATA_ROOT=${MMABC_DATA_ROOT:-<unset, reading shared storage>}"

consecutive_failures=0
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  if [[ -f "$STOP_FILE" ]]; then
    log "STOP file present ($STOP_FILE); not launching. Remove it to resume."
    break
  fi

  # Clear leftover ranks/device memory from a previous crashed attempt, or the
  # next launch dies on a stale process holding the GPU.
  log "attempt $attempt: clearing cluster leftovers"
  bash "$REPO/scripts/cluster_cleanup.sh" --kill >>"$LOGDIR/cleanup.log" 2>&1 || true

  start=$(date +%s)
  log "attempt $attempt: launching train_multinode.sh"
  # SKIP_PREFLIGHT=1 because we just cleaned up ourselves above; the preflight
  # would otherwise refuse to launch on the memory it takes a moment to release.
  SKIP_PREFLIGHT=1 bash "$REPO/scripts/train_multinode.sh" "$CONFIG" "$@" \
    >>"$LOGDIR/attempt_${attempt}.log" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start ))
  log "attempt $attempt: exited status=$status after ${elapsed}s"

  if [[ $status -eq 0 ]]; then
    log "training completed cleanly (max_steps reached); supervisor exiting"
    break
  fi

  if [[ -f "$STOP_FILE" ]]; then
    log "STOP file present after exit; supervisor exiting"
    break
  fi

  if (( elapsed >= MIN_HEALTHY_SECONDS )); then
    consecutive_failures=0
    log "attempt ran ${elapsed}s (>= ${MIN_HEALTHY_SECONDS}s); treating as transient, resetting backoff"
  else
    consecutive_failures=$(( consecutive_failures + 1 ))
    log "attempt died young (${elapsed}s < ${MIN_HEALTHY_SECONDS}s); consecutive_failures=$consecutive_failures"
    if (( consecutive_failures >= 5 )); then
      log "5 consecutive fast failures -- this looks like a config/env error, not a node event."
      log "supervisor stopping so it does not crash-loop. Check runs/launch/*/node*.log."
      break
    fi
  fi

  wait_s=$(( BACKOFF_SECONDS * (consecutive_failures > 0 ? consecutive_failures : 1) ))
  log "backing off ${wait_s}s before relaunch (auto_resume will reload the newest checkpoint)"
  sleep "$wait_s"
done

log "supervisor done."
