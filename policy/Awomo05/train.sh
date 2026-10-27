#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo 'Usage: train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [--check-config | --resume PATH]' >&2
    exit 2
fi
bench=$1 label=$2 robot=$3 action=$4 seed=$5 gpu_ids=$6
shift 6
if [[ "$bench" != RoboDojo || "$robot" != arx_x5 || "$action" != joint || ! "$seed" =~ ^[0-9]+$ || ! "$label" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
    echo 'The reference recipe supports RoboDojo / arx_x5 / joint and a nonnegative seed.' >&2
    exit 2
fi
IFS=, read -r -a selected_gpus <<< "$gpu_ids"
if [[ ${#selected_gpus[@]} -ne 8 || $(printf '%s\n' "${selected_gpus[@]}" | sort -u | wc -l) -ne 8 ]]; then
    echo 'Provide eight distinct GPU IDs for each node, e.g. 0,1,2,3,4,5,6,7.' >&2
    exit 2
fi
for gpu in "${selected_gpus[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ && ! "$gpu" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
        echo "Invalid GPU ID: $gpu" >&2
        exit 2
    fi
done
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
export PYTHONPATH="$POLICY_DIR:$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
layout=$(mktemp -d)
cleanup() { unlink "$layout/XPolicyLab"; rmdir "$layout"; }
ln -s "$XPL_ROOT" "$layout/XPolicyLab"
trap cleanup EXIT
compact_dim=$(bash "$XPL_ROOT/utils/get_action_dim.sh" "$layout" "$robot")
if [[ "$compact_dim" != 14 ]]; then
    echo 'The reference action mapping requires the robot compact action width to be 14.' >&2
    exit 2
fi
cleanup
trap - EXIT
run_name=$(python - "$bench" "$label" "$robot" "$action" "$seed" <<'PY'
import sys
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
print(build_run_dir_name(dict(zip(('bench_name','ckpt_name','env_cfg_type','action_type','seed'), sys.argv[1:]))))
PY
)
export AWOMO_RUN_NAME="$run_name"
export AWOMO_RUN_DIR="${AWOMO05_TRAIN_OUTPUT:-$POLICY_DIR/checkpoints/$run_name}"
export AWOMO_DATA_INDEX="${AWOMO_DATA_INDEX:-$POLICY_DIR/data/$bench-$label-$robot-$action/index.pt}"
export AWOMO_DATA_STATS="${AWOMO_DATA_STATS:-$POLICY_DIR/training/dataset_stats.json}"
export TOKENIZERS_PARALLELISM=false
for arg in "$@"; do
    if [[ "$arg" == --check-config ]]; then
        exec python "$POLICY_DIR/training/train.py" --seed "$seed" "$@"
    fi
done
: "${NNODES:?Set NNODES=4}"
: "${NODE_RANK:?Set NODE_RANK=0,1,2,3}"
: "${MASTER_ADDR:?Set the training rendezvous host}"
: "${MASTER_PORT:?Set the rendezvous port}"
if [[ "$NNODES" != 4 || "${NPROC_PER_NODE:-8}" != 8 ]]; then
    echo 'The released recipe requires 4 nodes x 8 GPUs (global batch 256).' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_ids"
exec torchrun --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "$POLICY_DIR/training/train.py" --seed "$seed" "$@"
