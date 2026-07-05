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
ROBODOJO_TEST_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

resolve_lerobot_repo_id() {
  if [[ -n "${LEROBOT_DATASET_REPO_ID:-}" ]]; then
    echo "${LEROBOT_DATASET_REPO_ID}"
    return
  fi
  case "${env_cfg_type}" in
    arx_x5) echo "RoboDojo_sim_arx-x5_v30" ;;
    *) echo "RoboDojo_sim_${env_cfg_type}" ;;
  esac
}

export XPOLICYLAB_LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT:-${LEROBOT_DATA_ROOT:-${ROBODOJO_TEST_ROOT}/data}}"
export LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT}"
export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-$(resolve_lerobot_repo_id)}"

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NGPU
NGPU="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | xargs)"
export CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-robotwin30_train}"
export LINGBOT_VA_DATASET_PATH="${LINGBOT_VA_DATASET_PATH:-${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}}"
export LINGBOT_VA_BASE_MODEL_PATH="${LINGBOT_VA_BASE_MODEL_PATH:-}"
export PYTHONHASHSEED="${seed}"

echo "[LingBot_VA] LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT}"
echo "[LingBot_VA] LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID}"
echo "[LingBot_VA] config=${CONFIG_NAME}"
echo "[LingBot_VA] dataset=${LINGBOT_VA_DATASET_PATH}"
echo "[LingBot_VA] base_model=${LINGBOT_VA_BASE_MODEL_PATH:-<unset>}"
echo "[LingBot_VA] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/lingbot_va/script/run_va_posttrain.sh" \
  --save-root "${ckpt_dir}" \
  --seed "${seed}"
