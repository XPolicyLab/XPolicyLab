#!/usr/bin/env bash
# Launch nodes from HOSTFILE: <config> [overrides...]. First node is master.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${MMABC_PYTHON:-$(command -v python)}
HOSTFILE=${HOSTFILE:-$REPO/configs/cluster/hostfile}
CONFIG=${1:?usage: train_multinode.sh <config> [overrides...]}
shift || true

[[ -f "$HOSTFILE" ]] || { echo "missing hostfile: $HOSTFILE" >&2; exit 1; }
mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTFILE" | sed 's/#.*//')
NNODES=${#HOSTS[@]}
(( NNODES > 0 )) || { echo "hostfile is empty" >&2; exit 1; }

source "$REPO/scripts/cluster_common.sh"
load_wandb_env

read -r MASTER_SSH MASTER_ADDR MASTER_KEY <<<"$(host_fields "${HOSTS[0]}")"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$(pick_master_port "$MASTER_SSH" "$MASTER_ADDR" "$MASTER_KEY" || true)
fi
MASTER_PORT=${MASTER_PORT:?no free port in 29520-29599 on master $MASTER_ADDR}
NPROC=${NPROC:-8}
# torchrun restarts the worker group in place on a worker failure, up to this
# many times; coarser recovery (a node dropping out) is train_supervised.sh's job.
MAX_RESTARTS=${MAX_RESTARTS:-3}

# Refuse to launch on top of a previous run's leftovers: their device memory
# turns the next launch into an out-of-memory error that reads like a batch-size
# problem. Killing stays an explicit step since nodes may be shared.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  stale=$(bash "$REPO/scripts/cluster_cleanup.sh" 2>/dev/null | grep 'leftover pids' || true)
  if [[ -n "$stale" ]]; then
    echo "refusing to launch: earlier ranks are still alive" >&2
    echo "$stale" >&2
    echo "run 'bash scripts/cluster_cleanup.sh --kill' first, or set SKIP_PREFLIGHT=1" >&2
    exit 1
  fi
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)
LOGDIR="$REPO/runs/launch/$RUN_ID"
mkdir -p "$LOGDIR"
echo "launching $NNODES nodes x $NPROC gpus, master $MASTER_ADDR:$MASTER_PORT"
echo "logs: $LOGDIR"

PIDS=()
for rank in "${!HOSTS[@]}"; do
  read -r ssh_host ip key <<<"$(host_fields "${HOSTS[$rank]}")"

  # Variables named in FORWARD_ENV are set on every node *before* nccl.env, so
  # its `:-` defaults defer to them:
  #   FORWARD_ENV="NCCL_DEBUG" NCCL_DEBUG=INFO bash scripts/train_multinode.sh ...
  forwarded=""
  for name in ${FORWARD_ENV:-}; do
    forwarded+="export $name=${!name@Q}"$'\n'
  done

  remote_cmd=$(cat <<EOF
set -euo pipefail
$forwarded
source "$REPO/configs/cluster/nccl.env"
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
cd "$REPO"
rc=0
"$PY" -m torch.distributed.run \
  --nnodes=$NNODES --node_rank=$rank --nproc_per_node=$NPROC \
  --max-restarts=$MAX_RESTARTS \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  mmabc/train/entrypoint.py "$CONFIG" $* < /dev/null || rc=\$?
echo "[mmabc-node-exit] rank=$rank status=\$rc"
exit \$rc
EOF
)
  node_exec "$ssh_host" "$ip" "$key" <<<"$remote_cmd" >"$LOGDIR/node${rank}.log" 2>&1 &
  PIDS+=($!)
done

trap 'echo "terminating..."; kill ${PIDS[*]} 2>/dev/null || true' INT TERM

# Close lingering SSH sessions after the first node exits and the grace period elapses.
GRACE_SECONDS=${GRACE_SECONDS:-600}
node_status() {
  local line
  line=$(grep -a '\[mmabc-node-exit\]' "$LOGDIR/node$1.log" | tail -1)
  [[ -n "$line" ]] && echo "${line##*status=}" || echo ""
}
done_at=""
while :; do
  alive=0
  for rank in "${!PIDS[@]}"; do
    kill -0 "${PIDS[$rank]}" 2>/dev/null && alive=$((alive + 1))
  done
  (( alive == 0 )) && break
  if (( alive < ${#PIDS[@]} )); then
    [[ -z "$done_at" ]] && done_at=$(date +%s)
    if (( $(date +%s) - done_at > GRACE_SECONDS )); then
      echo "closing ${alive} lingering node session(s) ${GRACE_SECONDS}s after the first node finished"
      kill "${PIDS[@]}" 2>/dev/null || true
      break
    fi
  fi
  sleep 10
done

status=0
for rank in "${!PIDS[@]}"; do
  wait "${PIDS[$rank]}" 2>/dev/null
  s=$(node_status "$rank")
  echo "node$rank exit status: ${s:-unknown}"
  [[ "${s:-1}" == "0" ]] || status=1
done
echo "all nodes exited (status $status); logs in $LOGDIR"
exit "$status"
