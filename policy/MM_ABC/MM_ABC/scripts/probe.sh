#!/usr/bin/env bash
# Sweep micro-batch sizes, one fresh process per size.
#
#   bash scripts/probe.sh configs/train/mobile.yaml "4 8 12 16"
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${MMABC_PYTHON:-$(command -v python)}
CONFIG=${1:?usage: probe.sh <config> ["sizes"]}
SIZES=${2:-"4 8 12 16"}

# shellcheck source=/dev/null
source "$REPO/configs/cluster/nccl.env"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
OUT="$REPO/runs/probe_$(date +%Y%m%d-%H%M%S).jsonl"
mkdir -p "$REPO/runs"

cd "$REPO"
echo "probing micro-batch sizes: $SIZES on $NPROC gpus -> $OUT"
for size in $SIZES; do
  echo "--- micro_batch=$size ---"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    scripts/probe_batch.py "$CONFIG" --micro-batch "$size" --out "$OUT" \
    2>&1 | grep -E '^\{|OutOfMemory|Error' || true
done

echo
echo "summary:"
"$PY" - "$OUT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'micro':>6s} {'status':>7s} {'peak GiB':>9s} {'s/step':>7s} {'smp/s/gpu':>10s} {'per node':>9s}")
best = None
for r in rows:
    print(f"{r['micro_batch']:6d} {r['status']:>7s} {r['peak_gib']:9.1f} "
          f"{r['s_per_step']:7.2f} {r['samples_per_s_per_gpu']:10.2f} {r['per_node']:9d}")
    if r['status'] == 'ok':
        best = r
if best:
    print(f"\nlargest fitting micro-batch: {best['micro_batch']}/gpu = {best['per_node']}/node")
    print("choose grad_accum_steps so micro_batch * accum * gpus_per_node hits your per-node target")
EOF
