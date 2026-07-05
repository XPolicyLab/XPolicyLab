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
config_path="${LINGBOT_VLA_CONFIG_PATH:-configs/vla/robodojo_sim_arx_x5.yaml}"
data_path="${LINGBOT_VLA_DATA_PATH:-${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}}"
export LINGBOT_VLA_DATA_PATH="${data_path}"
export PYTHONHASHSEED="${seed}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[LingBot_VLA] LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT}"
echo "[LingBot_VLA] LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID}"
echo "[LingBot_VLA] config=${config_path}"
echo "[LingBot_VLA] data_path=${data_path}"
echo "[LingBot_VLA] checkpoint_dir=${ckpt_dir}"

cd "${POLICY_DIR}/lingbot_vla"
bash train_origin.sh tasks/vla/train_lingbotvla.py \
  "${config_path}" \
  --data.train_path "${data_path}" \
  --train.output_dir "${ckpt_dir}" \
  --train.seed "${seed}"