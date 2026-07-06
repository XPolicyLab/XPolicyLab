#!/bin/bash
set -euo pipefail

# XPolicyLab-standard training wrapper for the X_WAM (X-WAM) policy.
#
# It launches scripts/train_sft.py inside the X-WAM subdir and, via Hydra/
# OmegaConf CLI overrides, redirects the experiment output so training artefacts
# land exactly where the eval side resolves them by default:
#
#   exp_root=<POLICY_DIR>/checkpoints, exp_name=<bench>-<ckpt>-<env>-<action>-<seed>
#     -> <POLICY_DIR>/checkpoints/<ckpt_setting>/config.yaml
#     -> <POLICY_DIR>/checkpoints/<ckpt_setting>/checkpoints/{last|<step>}.ckpt/checkpoint/mp_rank_00_model_states.pt
#
# setup_eval_policy_server.sh defaults exp_path to
#   ${POLICY_DIR}/checkpoints/${ckpt_name}
# and deploy_policy.py loads ${exp_path}/checkpoints/${steps}.ckpt/...  So after
# training, evaluate with ckpt_name = <bench>-<ckpt>-<env>-<action>-<seed> and it
# hits the checkpoint with no symlinks or XWAM_EXP_PATH tweaks required.
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}

if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

if [[ "${action_type}" != "ee" ]]; then
    echo "[X_WAM] ERROR: X-WAM is an EE-space policy; action_type must be 'ee', got '${action_type}'." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
POLICY_DIR="${SCRIPT_DIR}"
XWAM_DIR="${POLICY_DIR}/X-WAM"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

# eval ckpt_name == this ckpt_setting; must equal exp_name so eval's default
# exp_path (<POLICY_DIR>/checkpoints/<ckpt_name>) points at these artefacts.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

# Converted dataset location; kept in sync with process_data.sh via the
# XWAM_DATASET_PATH env resolver referenced by configs/data/robodojo.yaml.
dataset_path="${XWAM_DATASET_PATH:-${POLICY_DIR}/data/${data_key}}"
exp_root="${POLICY_DIR}/checkpoints"
raw_task_dirs="${XWAM_RAW_TASK_DIRS:-${ckpt_name}}"
data_limit="${XWAM_DATA_LIMIT:-}"

if [[ ! -d "${dataset_path}/data" ]]; then
    echo "[X_WAM] Converted dataset not found at ${dataset_path}; running process_data.sh first."
    echo "[X_WAM] Auto-converting raw_task_dirs=${raw_task_dirs}, limit=${data_limit:-all}."
    XWAM_DATASET_PATH="${dataset_path}" bash "${POLICY_DIR}/process_data.sh" \
        "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${data_limit}" "${raw_task_dirs}"
fi

# Base / pretrained weights: honor the config.yaml defaults unless overridden.
wan_checkpoint_dir="${XWAM_WAN_CHECKPOINT_DIR:-}"
pretrained_checkpoint="${XWAM_PRETRAINED_CHECKPOINT:-}"

# Training hyperparameters (all overridable; sensible defaults from the yaml).
num_training_steps="${XWAM_NUM_TRAINING_STEPS:-}"
save_interval="${XWAM_SAVE_INTERVAL:-}"
batch_size_per_gpu="${XWAM_BATCH_SIZE_PER_GPU:-}"
num_workers_per_gpu="${XWAM_NUM_WORKERS_PER_GPU:-}"

master_addr="${XWAM_MASTER_ADDR:-localhost}"
if [[ -n "${XWAM_MASTER_PORT:-}" ]]; then
    master_port="${XWAM_MASTER_PORT}"
elif [[ -x "${UTILS_DIR}/get_free_port.sh" ]]; then
    master_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
else
    master_port=29500
fi

mkdir -p "${exp_root}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# Exported so configs/data/robodojo.yaml's ${oc.env:XWAM_DATASET_PATH,...}
# resolver picks up the same dataset dir that process_data.sh wrote.
export XWAM_DATASET_PATH="${dataset_path}"
export PYTHONPATH="${ROOT_DIR}:${XWAM_DIR}:${PYTHONPATH:-}"

echo "[X_WAM train] bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed}"
echo "[X_WAM train] exp_root=${exp_root} exp_name=${ckpt_setting}"
echo "[X_WAM train] dataset_path=${dataset_path}"
echo "[X_WAM train] gpus=${gpu_id} nproc_per_node=${num_gpus} master=${master_addr}:${master_port}"

train_overrides=(
    "dataset=robodojo"
    "exp_root=${exp_root}"
    "exp_name=${ckpt_setting}"
    "seed=${seed}"
)
[[ -n "${wan_checkpoint_dir}" ]]    && train_overrides+=("wan_checkpoint_dir=${wan_checkpoint_dir}")
[[ -n "${pretrained_checkpoint}" ]] && train_overrides+=("pretrained_checkpoint=${pretrained_checkpoint}")
[[ -n "${num_training_steps}" ]]    && train_overrides+=("num_training_steps=${num_training_steps}")
[[ -n "${save_interval}" ]]         && train_overrides+=("save_interval=${save_interval}")
[[ -n "${batch_size_per_gpu}" ]]    && train_overrides+=("batch_size_per_gpu=${batch_size_per_gpu}")
[[ -n "${num_workers_per_gpu}" ]]   && train_overrides+=("num_workers_per_gpu=${num_workers_per_gpu}")

cd "${XWAM_DIR}"
exec torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${num_gpus}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    scripts/train_sft.py "${train_overrides[@]}"
