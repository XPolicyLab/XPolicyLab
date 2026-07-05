#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_DIR="${POLICY_DIR}/giga_world_policy"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

resolve_lerobot_repo_id() {
  if [[ -n "${LEROBOT_DATASET_REPO_ID:-}" ]]; then
    echo "${LEROBOT_DATASET_REPO_ID}"
    return
  fi
  case "${env_cfg_type}" in
    arx_x5) echo "XPolicyLab_sim_arx-x5_v30" ;;
    *) echo "XPolicyLab_sim_${env_cfg_type}" ;;
  esac
}

export XPOLICYLAB_LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT:-${LEROBOT_DATA_ROOT:-${XPOLICYLAB_ROOT}/data}}"
export LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT}"
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-$(resolve_lerobot_repo_id)}"

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
data_dir="${GIGAWORLD_DATA_DIR:-${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}}"
output_root="${GIGAWORLD_OUTPUT_ROOT:-${POLICY_DIR}/experiments}"
ckpt_dir="${GIGAWORLD_CKPT_DIR:-${output_root}/checkpoints/${ckpt_setting}}"
record_config="${ckpt_dir}/xpolicylab_train_config.json"
base_config="${GIGAWORLD_CONFIG:-configs.xpolicylab_gigaworld.config}"
accel_config="${GIGAWORLD_ACCEL_CONFIG:-${INNER_DIR}/scripts/accelerate_configs/config_deepspeed_zero2.json}"
norm_path="${GIGAWORLD_NORM_PATH:-}"
pretrained_path="${GIGAWORLD_PRETRAINED_PATH:-${WAN22_DIFFUSERS_PATH:?Set GIGAWORLD_PRETRAINED_PATH (or WAN22_DIFFUSERS_PATH) to the local Wan2.2-TI2V-5B-Diffusers directory}}"
wandb_project="${GIGAWORLD_WANDB_PROJECT:-gwp-xpolicylab}"
wandb_name="${GIGAWORLD_WANDB_NAME:-GigaWorldPolicy_${ckpt_setting}}"
wandb_mode="${WANDB_MODE:-offline}"
model_action_dim="${GIGAWORLD_MODEL_ACTION_DIM:-14}"
model_state_dim="${GIGAWORLD_MODEL_STATE_DIM:-${model_action_dim}}"
num_frames="${GIGAWORLD_NUM_FRAMES:-24}"
action_chunk="${GIGAWORLD_ACTION_CHUNK:-${num_frames}}"
python_bin="${GIGAWORLD_PYTHON:-python}"
accelerate_bin="${GIGAWORLD_ACCELERATE:-accelerate}"
torch_dist_timeout_sec="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-${GIGAWORLD_DISTRIBUTED_TIMEOUT:-3600}}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${torch_dist_timeout_sec}"
export GIGAWORLD_DISTRIBUTED_TIMEOUT="${GIGAWORLD_DISTRIBUTED_TIMEOUT:-${torch_dist_timeout_sec}}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-60}"

# giga-train requires seed > 0; keep XPolicyLab seed=0 valid and distinct from seed=1.
TRAIN_SEED=$((seed + 1))
export PYTHONHASHSEED="${TRAIN_SEED}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export GIGAWORLD_DATA_DIR="${data_dir}"
export GIGAWORLD_MODEL_ACTION_DIM="${model_action_dim}"
export GIGAWORLD_MODEL_STATE_DIM="${model_state_dim}"
export GIGAWORLD_NUM_FRAMES="${num_frames}"

IFS="," read -r -a gpu_array <<< "${gpu_id}"
default_nproc=0
for g in "${gpu_array[@]}"; do
  [[ -n "${g// /}" ]] && default_nproc=$((default_nproc + 1))
done
[[ ${default_nproc} -gt 0 ]] || default_nproc=1

num_nodes="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
node_rank="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
nproc_per_node="${NPROC_PER_NODE:-${default_nproc}}"
total_procs=$((num_nodes * nproc_per_node))
master_addr="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
master_port="${MASTER_PORT:-29531}"

cd "${INNER_DIR}"

common_args=(
  --config "${base_config}"
  --project_dir "${ckpt_dir}"
  --record_config "${record_config}"
  --data_dir "${data_dir}"
  --pretrained_path "${pretrained_path}"
  --seed "${TRAIN_SEED}"
  --gpu_ids "${gpu_id}"
  --wandb_project "${wandb_project}"
  --wandb_name "${wandb_name}"
  --wandb_mode "${wandb_mode}"
  --model_action_dim "${model_action_dim}"
  --model_state_dim "${model_state_dim}"
  --num_frames "${num_frames}"
  --action_chunk "${action_chunk}"
)

if [[ -n "${norm_path}" ]]; then
  common_args+=(--norm_path "${norm_path}")
elif [[ -f "${data_dir}/norm_stats_delta.json" ]]; then
  common_args+=(--norm_path "${data_dir}/norm_stats_delta.json")
fi

if [[ "${GIGAWORLD_FORCE_DATA_DIR:-0}" == "1" ]]; then
  common_args+=(--force_data_dir)
fi

echo "[GigaWorldPolicy] LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT}"
echo "[GigaWorldPolicy] LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID}"
cat <<EOF
[GigaWorldPolicy] training
  config:      ${base_config}
  data_dir:    ${data_dir}
  ckpt_dir:    ${ckpt_dir}
  gpu_id:      ${gpu_id}
  train_seed:  ${TRAIN_SEED} (XPolicyLab seed=${seed})
  nproc:       ${total_procs}
  action_dim:  ${model_action_dim}
  state_dim:   ${model_state_dim}
  wandb_mode:  ${wandb_mode}
  python:      ${python_bin}
  accelerate:  ${accelerate_bin}
  TORCH_DIST_TIMEOUT: ${TORCH_DISTRIBUTED_TIMEOUT_SEC}s
  DEEPSPEED_TIMEOUT:  ${DEEPSPEED_TIMEOUT}min
  ACCEL_CONFIG:       ${accel_config}
EOF

if [[ "${GIGAWORLD_DRY_RUN:-0}" == "1" ]]; then
  "${python_bin}" scripts/train_xpolicylab.py "${common_args[@]}" --dry-run
  exit 0
fi

"${accelerate_bin}" launch \
  --config_file "${accel_config}" \
  --gpu_ids "${gpu_id}" \
  --num_processes "${total_procs}" \
  --num_machines "${num_nodes}" \
  --machine_rank "${node_rank}" \
  --main_process_ip "${master_addr}" \
  --main_process_port "${master_port}" \
  scripts/train_xpolicylab.py "${common_args[@]}"
