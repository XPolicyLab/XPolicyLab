#!/bin/bash
set -euo pipefail

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}
num_gpus=${7:-}

if [[ "${bench_name}" != "RoboTwin" || "${env_cfg_type}" != "aloha_agilex" || "${action_type}" != "joint" ]]; then
    echo "[ERROR] OLA-SEM training supports only RoboTwin/aloha_agilex/joint." >&2
    exit 1
fi
if [[ -z "${num_gpus}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")
if [[ "${action_dim}" != "14" ]]; then
    echo "[ERROR] OLA-SEM requires 14 action dimensions; ${env_cfg_type} resolved to ${action_dim}." >&2
    exit 1
fi

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_root=${OLA_SEM_DATASET_ROOT:-${SCRIPT_DIR}/data/${data_setting}}
pretrained_root=${OLA_SEM_PRETRAINED_ROOT:-${SCRIPT_DIR}/ola_sem/pretrained_models}
wan_path=${OLA_SEM_WAN_PATH:-${pretrained_root}/Wan2.2-TI2V-5B}
vlm_path=${OLA_SEM_VLM_PATH:-${pretrained_root}/Qwen3-VL-2B-Instruct}
finetune_path=${OLA_SEM_FINETUNE_CHECKPOINT:-${pretrained_root}/d0_v}
output_root=${OLA_SEM_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}
run_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
run_dir="${output_root}/${run_name}"
runtime_config="${run_dir}/ola_sem_runtime.yml"

for path in "${dataset_root}" "${wan_path}" "${vlm_path}" "${finetune_path}"; do
    [[ -e "${path}" ]] || { echo "[ERROR] Required path not found: ${path}" >&2; exit 1; }
done

config_args=(
    --base "${SCRIPT_DIR}/ola_sem/configs/robotwin_lap_history_flow_clean.yaml"
    --output "${runtime_config}"
    --dataset "${dataset_root}"
    --wan "${wan_path}"
    --vlm "${vlm_path}"
    --finetune "${finetune_path}"
    --checkpoint-dir "${run_dir}"
    --run-name "${run_name}"
    --max-steps "${OLA_SEM_MAX_STEPS:-40000}"
    --report-to "${OLA_SEM_REPORT_TO:-tensorboard}"
)
[[ -z "${OLA_SEM_MAX_EPISODES:-}" ]] || config_args+=(--max-episodes "${OLA_SEM_MAX_EPISODES}")
[[ -z "${OLA_SEM_BATCH_SIZE:-}" ]] || config_args+=(--batch-size "${OLA_SEM_BATCH_SIZE}")
[[ -z "${OLA_SEM_NUM_WORKERS:-}" ]] || config_args+=(--num-workers "${OLA_SEM_NUM_WORKERS}")

python "${SCRIPT_DIR}/prepare_train_config.py" "${config_args[@]}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${SCRIPT_DIR}/ola_sem:${WORKSPACE_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}

cd "${SCRIPT_DIR}/ola_sem"
torchrun \
    --nnodes=1 \
    --nproc_per_node="${num_gpus}" \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train/train.py \
    --deepspeed configs/zero2.json \
    --config "${runtime_config}" \
    --run_name "${run_name}" \
    --report_to "${OLA_SEM_REPORT_TO:-tensorboard}"
