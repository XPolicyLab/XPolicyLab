#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

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

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
config_path="${LINGBOT_VLA_CONFIG_PATH:-configs/vla/robotwin_load20000h.yaml}"
data_path="${LINGBOT_VLA_DATA_PATH:-${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}}"
export LINGBOT_VLA_DATA_PATH="${data_path}"
export PYTHONHASHSEED="${seed}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

if [[ ! -f "${POLICY_DIR}/lingbot_vla/${config_path}" && ! -f "${config_path}" ]]; then
  echo "[LingBot_VLA] ERROR: config not found: ${config_path}" >&2
  echo "[LingBot_VLA] Set LINGBOT_VLA_CONFIG_PATH to an existing config under lingbot_vla/." >&2
  exit 1
fi

extra_train_args=()
if [[ -n "${LINGBOT_VLA_MODEL_PATH:-}" ]]; then
  extra_train_args+=(--model.model_path "${LINGBOT_VLA_MODEL_PATH}")
fi
if [[ -n "${LINGBOT_VLA_TOKENIZER_PATH:-${QWEN25_PATH:-}}" ]]; then
  extra_train_args+=(--model.tokenizer_path "${LINGBOT_VLA_TOKENIZER_PATH:-${QWEN25_PATH:-}}")
fi
if [[ -n "${LINGBOT_VLA_NORM_STATS_FILE:-}" ]]; then
  extra_train_args+=(--data.norm_stats_file "${LINGBOT_VLA_NORM_STATS_FILE}")
fi

echo "[LingBot_VLA] LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT}"
echo "[LingBot_VLA] LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID}"
echo "[LingBot_VLA] config=${config_path}"
echo "[LingBot_VLA] data_path=${data_path}"
echo "[LingBot_VLA] checkpoint_dir=${ckpt_dir}"

cd "${POLICY_DIR}/lingbot_vla"
bash train.sh tasks/vla/train_lingbotvla.py \
  "${config_path}" \
  --data.train_path "${data_path}" \
  --train.output_dir "${ckpt_dir}" \
  --train.seed "${seed}" \
  "${extra_train_args[@]}"
