#!/usr/bin/env bash
# Launch one PrismWAM training stage with accelerate.
#
#   bash scripts/train.sh <config.yaml> [--dry-run] [key=value ...]
#
# Uses all GPUs on this node; GPUS_PER_NODE overrides the count. Multi-node: run the same
# command on each node with NNODES, NODE_RANK, MASTER_ADDR (optionally MASTER_PORT) set.
# Gradient accumulation keeps the recipe's global_batch_size, so any GPU count that
# divides it works.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if (($# < 1)) || [[ "$1" == -h || "$1" == --help ]]; then
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi
CONFIG="$1"
shift
DRY_RUN=
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run || "$arg" == --check-assets ]]; then DRY_RUN="$arg"; else ARGS+=("$arg"); fi
done

PYTHON="${PYTHON:-python}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    GPUS_PER_NODE="$("$PYTHON" -c 'import torch; print(max(torch.cuda.device_count(), 1))')"
fi
for value in "$NNODES" "$GPUS_PER_NODE" "$NODE_RANK" "$MASTER_PORT"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "NNODES, GPUS_PER_NODE, NODE_RANK and MASTER_PORT must be integers." >&2; exit 2; }
done
if ((NNODES > 1)) && [[ "$MASTER_ADDR" == 127.0.0.1 ]]; then
    echo "Set MASTER_ADDR to the address of node 0 for multi-node training." >&2
    exit 2
fi
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# accelerate needs the accumulation, clipping and launcher config before it starts.
resolved="$("$PYTHON" scripts/train.py "$CONFIG" "${ARGS[@]}" --print-launch-options)"
mapfile -t LAUNCH <<< "$resolved"
((${#LAUNCH[@]} == 3)) || { echo "Could not resolve launch options for $CONFIG." >&2; exit 2; }

COMMAND=("$PYTHON" -m accelerate.commands.launch
    --config_file "${LAUNCH[2]}"
    --num_machines "$NNODES" --num_processes "$WORLD_SIZE" --machine_rank "$NODE_RANK"
    --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" --same_network
    --gradient_accumulation_steps "${LAUNCH[0]}" --gradient_clipping "${LAUNCH[1]}"
    scripts/train.py "$CONFIG" "${ARGS[@]}")

if [[ -n "$DRY_RUN" ]]; then
    "$PYTHON" scripts/train.py "$CONFIG" "${ARGS[@]}" "$DRY_RUN"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${COMMAND[@]}"
