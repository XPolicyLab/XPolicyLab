#!/usr/bin/env bash
# Run a Python entrypoint across HOSTFILE nodes: <script.py> [args...].
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${MMABC_PYTHON:-$(command -v python)}
HOSTFILE=${HOSTFILE:-$REPO/configs/cluster/hostfile}
SCRIPT=${1:?usage: run_multinode.sh <script.py> [args...]}
shift || true

source "$REPO/scripts/cluster_common.sh"

mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTFILE" | sed 's/#.*//')
NNODES=${#HOSTS[@]}
(( NNODES > 0 )) || { echo "hostfile is empty; configure HOSTFILE first" >&2; exit 1; }
read -r MASTER_SSH MASTER_ADDR MASTER_KEY <<<"$(host_fields "${HOSTS[0]}")"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$(pick_master_port "$MASTER_SSH" "$MASTER_ADDR" "$MASTER_KEY" || true)
fi
MASTER_PORT=${MASTER_PORT:?no free port in 29520-29599 on master $MASTER_ADDR}
NPROC=${NPROC:-8}
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOGDIR="$REPO/runs/launch/$RUN_ID"
mkdir -p "$LOGDIR"
echo "running $SCRIPT on $NNODES nodes x $NPROC gpus, master $MASTER_ADDR:$MASTER_PORT"
echo "logs: $LOGDIR"

PIDS=()
for rank in "${!HOSTS[@]}"; do
  read -r ssh_host ip key <<<"$(host_fields "${HOSTS[$rank]}")"
  forwarded=""
  for name in ${FORWARD_ENV:-}; do
    forwarded+="export $name=${!name@Q}"$'\n'
  done
  cmd=$(cat <<EOF
set -euo pipefail
$forwarded
source "$REPO/configs/cluster/nccl.env"
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
cd "$REPO"
exec "$PY" -m torch.distributed.run \
  --nnodes=$NNODES --node_rank=$rank --nproc_per_node=$NPROC \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT $*
EOF
)
  node_exec "$ssh_host" "$ip" "$key" <<<"$cmd" >"$LOGDIR/node${rank}.log" 2>&1 &
  PIDS+=($!)
done

trap 'kill ${PIDS[*]} 2>/dev/null || true' INT TERM
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=$?; done
echo "done (status $status); logs in $LOGDIR"
echo "LOGDIR=$LOGDIR"
exit "$status"
