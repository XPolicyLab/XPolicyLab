#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$PROJECT_ROOT"

export RUN_ENTRY=${RUN_ENTRY:-scripts/train.py}
export CONFIG=${CONFIG:-debug}
export EXP_NAME=${EXP_NAME:-"${CONFIG}_alaya"}
export EXTRA_ARGS=${EXTRA_ARGS:-}
export OPENPI_SHOW_TQDM=${OPENPI_SHOW_TQDM:-0}
export OPENPI_MIXED_READ_CHUNK_SIZE=${OPENPI_MIXED_READ_CHUNK_SIZE:-16}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"/root/nas/lfh/.cache/huggingface/datasets"}

ALAYA_NODE_COUNT=${PET_NNODES:-${NNODES:-${WORLD_SIZE:-1}}}
ALAYA_NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-${GROUP_RANK:-${RANK:-0}}}}
export MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-localhost}}
export MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-29500}}

LOCAL_GPU_COUNT=$(nvidia-smi -L | wc -l)
if ! [[ "$ALAYA_NODE_COUNT" =~ ^[1-9][0-9]*$ && "$ALAYA_NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "Invalid node layout: count=$ALAYA_NODE_COUNT rank=$ALAYA_NODE_RANK" >&2
    exit 1
fi
if (( ALAYA_NODE_RANK >= ALAYA_NODE_COUNT )); then
    echo "Node rank $ALAYA_NODE_RANK is outside node count $ALAYA_NODE_COUNT" >&2
    exit 1
fi
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT >= 65535 )); then
    echo "MASTER_PORT must be between 1 and 65534, got $MASTER_PORT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((LOCAL_GPU_COUNT - 1)))}
export JAX_NODE_COUNT=$ALAYA_NODE_COUNT
export JAX_COORDINATOR_ADDR=${JAX_COORDINATOR_ADDR:-$MASTER_ADDR}
export JAX_COORDINATOR_PORT=${JAX_COORDINATOR_PORT:-$((MASTER_PORT + 1))}
if [[ "$JAX_COORDINATOR_ADDR:$JAX_COORDINATOR_PORT" == "$MASTER_ADDR:$MASTER_PORT" ]]; then
    echo "JAX coordinator and torchrun rendezvous must use different ports" >&2
    exit 1
fi
unset JAX_PROCESS_COUNT JAX_PROCESS_INDEX JAX_LOCAL_PROCESS_COUNT JAX_LOCAL_PROCESS_INDEX JAX_LOCAL_DEVICE_IDS

unset XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

LOG_CONFIG=${CONFIG//\//_}
LOG_CONFIG=${LOG_CONFIG// /_}
TOPOLOGY="${ALAYA_NODE_COUNT}x${LOCAL_GPU_COUNT}x1"
LOG_TIME=${LOG_TIME:-$(date +%Y%m%d_%H%M%S)}
LOG_STEM=${LOG_STEM:-"${LOG_CONFIG}_${LOG_TIME}"}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-"$PROJECT_ROOT/.cache/jax/${LOG_CONFIG}/${TOPOLOGY}/process-0"}

mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR" slurm_logs
if (( ALAYA_NODE_RANK == 0 )); then
    exec > "slurm_logs/${LOG_STEM}.out" 2> "slurm_logs/${LOG_STEM}.err"
else
    exec > /dev/null 2>&1
fi
trap 'status=$?; echo "Launcher exited at $(date) with status=$status" >&2' EXIT

if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Virtualenv Python not found: $PROJECT_ROOT/.venv/bin/python" >&2
    exit 1
fi
if [ ! -f "$RUN_ENTRY" ]; then
    echo "RUN_ENTRY does not exist: $RUN_ENTRY" >&2
    exit 1
fi
echo "Job started at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "HOSTNAME=$(hostname)"
echo "NNODES=$ALAYA_NODE_COUNT"
echo "NODE_RANK=$ALAYA_NODE_RANK"
echo "LOCAL_GPU_COUNT=$LOCAL_GPU_COUNT"
echo "GLOBAL_DEVICE_COUNT=$((ALAYA_NODE_COUNT * LOCAL_GPU_COUNT))"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "JAX_COORDINATOR_ADDR=$JAX_COORDINATOR_ADDR"
echo "JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_ENTRY=$RUN_ENTRY"
echo "CONFIG=$CONFIG"
echo "EXP_NAME=$EXP_NAME"
echo "EXTRA_ARGS=$EXTRA_ARGS"
echo "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
echo "OPENPI_MIXED_READ_CHUNK_SIZE=$OPENPI_MIXED_READ_CHUNK_SIZE"
echo "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "NCCL_DEBUG=$NCCL_DEBUG"
nvidia-smi -L

read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
"$PROJECT_ROOT/.venv/bin/python" -m torch.distributed.run \
    --nnodes="$ALAYA_NODE_COUNT" \
    --nproc-per-node="$LOCAL_GPU_COUNT" \
    --node-rank="$ALAYA_NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --max-restarts=0 \
    "$RUN_ENTRY" "$CONFIG" --exp-name="$EXP_NAME" "${EXTRA_ARGV[@]}"
