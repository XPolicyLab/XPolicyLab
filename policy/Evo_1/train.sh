#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh RoboTwin <ckpt_name> <env_cfg_type> joint <seed> <gpu_ids> [upstream training args...]" >&2
    exit 2
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_ids=$6
shift 6
if [[ "${bench_name}" != RoboTwin || "${action_type}" != joint || ! "${env_cfg_type}" =~ ^(arx_x5|aloha_agilex)$ ]]; then
    echo "[ERROR] Supported: RoboTwin, arx_x5|aloha_agilex, joint." >&2
    exit 2
fi
if [[ ! "${ckpt_name}" =~ ^[A-Za-z0-9_-]+$ || ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Use a simple ckpt_name and a nonnegative seed." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVO1_SOURCE_DIR="${EVO1_SOURCE_DIR:-${SCRIPT_DIR}/upstream}"
DATA_NAME="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
RUN_NAME="${DATA_NAME}-${seed}"
DATA_CONFIG="${EVO1_DATA_CONFIG:-${SCRIPT_DIR}/processed_data/${DATA_NAME}/config.yaml}"
RUN_DIR="${SCRIPT_DIR}/checkpoints/${RUN_NAME}"
if [[ ! -f "${DATA_CONFIG}" ]]; then
    echo "[ERROR] Missing ${DATA_CONFIG}; run process_data.sh first." >&2
    exit 1
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "[ERROR] Run directory exists; choose a new ckpt_name. No checkpoints were overwritten." >&2
    exit 1
fi
action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${XPL_ROOT}/.." "${env_cfg_type}")
if [[ "${action_dim}" != 14 ]]; then
    echo "[ERROR] Training metadata must describe the released 14-D bimanual embodiment." >&2
    exit 1
fi
IFS=',' read -r -a gpu_list <<< "${gpu_ids}"
launch_args=(--config_file "${SCRIPT_DIR}/accelerate.yml" --num_processes "${#gpu_list[@]}" --num_machines 1 --mixed_precision bf16)
if (( ${#gpu_list[@]} > 1 )); then
    launch_args+=(--multi_gpu)
fi
cd "${EVO1_SOURCE_DIR}/Evo_1"
export PYTHONPATH="${PWD}:${PWD}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
# Upstream initializes SwanLab even with --disable_wandb. Keep the standard
# training entry usable without a cloud account; users may opt in explicitly.
export SWANLAB_MODE="${SWANLAB_MODE:-disabled}"
# The launcher sets all RNGs before executing the unchanged upstream trainer.
exec python -m accelerate.commands.launch "${launch_args[@]}" "${SCRIPT_DIR}/train_entry.py" \
    --evo1-source "${EVO1_SOURCE_DIR}" --evo1-seed "${seed}" -- \
    --action_head flowmatching --use_augmentation --lr 1e-5 --batch_size 16 \
    --max_steps 20000 --warmup_steps 1000 --ckpt_interval 1000 --dropout 0.2 \
    --weight_decay 1e-3 --finetune_vlm --finetune_action_head --disable_wandb "$@" \
    --dataset_config_path "${DATA_CONFIG}" --run_name "${RUN_NAME}" \
    --save_dir "${RUN_DIR}" --cache_dir "${SCRIPT_DIR}/training_cache/${DATA_NAME}" \
    --image_size 448 --horizon 50 --per_action_dim 24 --state_dim 24
