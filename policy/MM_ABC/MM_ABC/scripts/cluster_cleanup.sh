#!/usr/bin/env bash
# Report stale training processes on HOSTFILE nodes; --kill terminates them.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE=${HOSTFILE:-$REPO/configs/cluster/hostfile}
source "$REPO/scripts/cluster_common.sh"

MODE=report
[[ "${1:-}" == "--kill" ]] && MODE=kill

# Matches this project's ranks only, so unrelated jobs on a shared node survive.
PATTERN='mmabc/train/entrypoint.py|torch.distributed.run|probe_batch.py|nccl_bench.py|nccl_verify.py|gpu_health.py'

# Find stale launchers and their GPU-worker descendants before optional cleanup.
remote=$(cat <<'EOS'
host=$(hostname)
by_cmd=$(ps -eo pid,args --no-headers 2>/dev/null | awk -v pat='PATTERN_PLACEHOLDER' '
  $2 ~ /python/ && $0 ~ pat { print $1 }')
by_gpu=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
  if [[ "$cwd" == "REPO_PLACEHOLDER" || "$cwd" == "REPO_PLACEHOLDER/"* ]]; then
    by_gpu="$by_gpu $p"
  fi
done
candidates=$(printf '%s\n%s\n' "$by_cmd" "$by_gpu" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
pids=""
for p in $candidates; do
  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
  if [[ "$cwd" == "REPO_PLACEHOLDER" || "$cwd" == "REPO_PLACEHOLDER/"* ]]; then
    pids="$pids $p"
  fi
done
if [[ -z "${pids// /}" ]]; then
  echo "$host: clean"
else
  echo "$host: leftover pids: $pids"
  if [[ "MODE_PLACEHOLDER" == "kill" ]]; then
    kill -9 $pids 2>/dev/null || true
    sleep 3
  fi
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
echo "$host: gpu MiB used: $used"
EOS
)
remote=${remote//PATTERN_PLACEHOLDER/$PATTERN}
remote=${remote//MODE_PLACEHOLDER/$MODE}
remote=${remote//REPO_PLACEHOLDER/$REPO}

mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTFILE" | sed 's/#.*//')
echo "mode: $MODE, ${#HOSTS[@]} nodes"

for line in "${HOSTS[@]}"; do
  read -r ssh_host ip key <<<"$(host_fields "$line")"
  node_exec "$ssh_host" "$ip" "$key" <<<"$remote"
done
