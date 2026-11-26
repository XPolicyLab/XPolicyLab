#!/bin/bash
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [train.py args...]
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [train.py args...]" >&2
    exit 1
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${POLICY_DIR}/openpi/.venv/bin/python"

case "${bench_name}" in
    RoboDojo) train_config_name=train_robodojo_sim ;;
    RoboDojo_real) train_config_name=train_robodojo_real ;;
    *) echo "[SLA_Memory][ERROR] bench_name must be RoboDojo or RoboDojo_real, got ${bench_name}" >&2; exit 1 ;;
esac
if [[ "${action_type}" != "ee" ]]; then
    echo "[SLA_Memory][ERROR] SLA_Memory trains on end-effector actions; action_type must be ee, got ${action_type}" >&2
    exit 1
fi

# Checkpoints are written to checkpoints/<train config>/<run name>/<step>/.
run_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
# Parameters are sharded across the GPUs of one machine; machines run data parallel.
num_local_gpus=$(awk -F',' '{print NF}' <<<"${gpu_id}")

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# In a multi-node run JAX takes one GPU per process unless given the local device ids: use all GPUs given above.
if [[ "${JAX_PROCESS_COUNT:-1}" -gt 1 ]]; then
    export JAX_LOCAL_DEVICE_IDS="${JAX_LOCAL_DEVICE_IDS:-$(seq -s, 0 $((num_local_gpus - 1)))}"
fi
# JAX takes 75% of GPU memory by default, which is not enough for the history frames.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# Data/, assets/ and checkpoints/ are relative to the policy directory.
cd "${POLICY_DIR}"
if [[ ! -f "assets/${train_config_name}/norm_stats.json" ]]; then
    if [[ "${JAX_PROCESS_COUNT:-1}" -gt 1 ]]; then
        echo "[SLA_Memory][ERROR] assets/${train_config_name}/norm_stats.json is missing; compute it before multi-node training: ${PYTHON_BIN} openpi/scripts/compute_norm_stats.py --config-name ${train_config_name}" >&2
        exit 1
    fi
    "${PYTHON_BIN}" openpi/scripts/compute_norm_stats.py --config-name "${train_config_name}"
fi

mkdir -p logs
echo "[SLA_Memory] config=${train_config_name} run=${run_name} gpus=${gpu_id} process=${JAX_PROCESS_INDEX:-0}/${JAX_PROCESS_COUNT:-1}"
"${PYTHON_BIN}" openpi/scripts/train.py "${train_config_name}" \
    --exp-name "${run_name}" \
    --seed "${seed}" \
    --fsdp-devices "${num_local_gpus}" \
    "$@" 2>&1 | tee "logs/${run_name}.process${JAX_PROCESS_INDEX:-0}.log"
