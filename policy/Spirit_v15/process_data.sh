#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  echo "  expert_data_num: optional; empty = use all episodes" >&2
  echo "  raw_task_dirs: optional source task name or comma-separated task list" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
converted_data_root="${SPIRIT_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
raw_data_root="${SPIRIT_RAW_DATA_ROOT:?set SPIRIT_RAW_DATA_ROOT to your RoboDojo raw dataset root}"

resolve_patterns_csv() {
  if [[ -n "${SPIRIT_PATTERNS_CSV:-}" ]]; then
    echo "${SPIRIT_PATTERNS_CSV}"
    return
  fi
  if [[ -n "${raw_task_dirs}" ]]; then
    local source_bench="${bench_name}"
    if [[ "${bench_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud" ]]; then
      source_bench="sim_cloud"
    fi
    local patterns=()
    IFS=',' read -r -a task_names <<< "${raw_task_dirs}"
    for task_name in "${task_names[@]}"; do
      task_name="${task_name//[[:space:]]/}"
      if [[ -n "${task_name}" ]]; then
        patterns+=("${source_bench}.${task_name}.${env_cfg_type}")
      fi
    done
    if [[ ${#patterns[@]} -eq 0 ]]; then
      echo "raw_task_dirs did not contain any task names" >&2
      exit 1
    fi
    local IFS=,
    echo "${patterns[*]}"
    return
  fi
  if [[ "${ckpt_name}" == "cotrain" ]]; then
    if [[ "${bench_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud" ]]; then
      echo "sim_cloud.*.${env_cfg_type}"
      return
    fi
    echo "${bench_name}.*.${env_cfg_type}"
    return
  fi
  if [[ "${bench_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud" ]]; then
    echo "sim_cloud.${ckpt_name}.${env_cfg_type}"
    return
  fi
  echo "${bench_name}.${ckpt_name}.${env_cfg_type}"
}

patterns_csv="$(resolve_patterns_csv)"

echo "[Spirit_v15] raw_data_root=${raw_data_root}"
echo "[Spirit_v15] patterns_csv=${patterns_csv}"
echo "[Spirit_v15] converted_data_root=${converted_data_root}"
echo "[Spirit_v15] expert_data_num=${expert_data_num:-<all>}"
echo "[Spirit_v15] raw_task_dirs=${raw_task_dirs:-<from ckpt_name>}"

bash "${POLICY_DIR}/spirit_v15/scripts/train_xpolicylab_from_raw.sh" \
  "${raw_data_root}" \
  "${patterns_csv}" \
  "${converted_data_root}" \
  "${SPIRIT_PRETRAINED_PATH:?set SPIRIT_PRETRAINED_PATH to your Spirit-v1.5 weights dir}" \
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
