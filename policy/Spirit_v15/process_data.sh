#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
converted_data_root="${SPIRIT_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
raw_data_root="${SPIRIT_RAW_DATA_ROOT:-/vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo}"

resolve_patterns_csv() {
  if [[ -n "${SPIRIT_PATTERNS_CSV:-}" ]]; then
    echo "${SPIRIT_PATTERNS_CSV}"
    return
  fi
  if [[ "${ckpt_name}" == "cotrain" ]]; then
    if [[ "${dataset_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud" ]]; then
      echo "sim_cloud.*.${env_cfg_type}"
      return
    fi
    echo "${dataset_name}.*.${env_cfg_type}"
    return
  fi
  if [[ "${dataset_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud" ]]; then
    echo "sim_cloud.${ckpt_name}.${env_cfg_type}"
    return
  fi
  echo "${dataset_name}.${ckpt_name}.${env_cfg_type}"
}

patterns_csv="$(resolve_patterns_csv)"

echo "[Spirit_v15] raw_data_root=${raw_data_root}"
echo "[Spirit_v15] patterns_csv=${patterns_csv}"
echo "[Spirit_v15] converted_data_root=${converted_data_root}"

bash "${POLICY_DIR}/spirit_v15/scripts/train_xpolicylab_from_raw.sh" \
  "${raw_data_root}" \
  "${patterns_csv}" \
  "${converted_data_root}" \
  "${SPIRIT_PRETRAINED_PATH:-/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Spirit-v1.5}" \
  "${POLICY_DIR}/checkpoints/_process_data_placeholder" \
  "1" \
  "32" \
  "40000" \
  "25" \
  "2500" \
  "4" \
  "8" \
  "disabled" \
  "${ckpt_name}" \
  "${SPIRIT_TASK_PROMPT:-Perform the instructed bimanual manipulation task.}" \
  "${SPIRIT_FPS:-auto}" \
  "${SPIRIT_OVERWRITE_DATASET:-0}" \
  "${SPIRIT_MAX_EPISODES_PER_TARGET:-${expert_data_num}}" \
  "${SPIRIT_ROBOT_TYPE:-aloha}" \
  "${SPIRIT_DATA_TYPE:-xspark}" \
  "${SPIRIT_DATA_VERSION:-v1.0}" \
  "0" \
  "1"

echo "[Spirit_v15] process_data done."
echo "[Spirit_v15] converted_data_root=${converted_data_root}"
