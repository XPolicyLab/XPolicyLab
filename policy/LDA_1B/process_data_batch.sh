#!/bin/bash
# Usage: bash process_data_batch.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
set -euo pipefail

dataset_name=${1:?dataset_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:?action_type required}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_DIR="${ROOT_DIR}/data/${dataset_name}"

shopt -s nullglob
task_names=()
for task_dir in "${DATASET_DIR}"/*/; do
  if compgen -G "${task_dir}${env_cfg_type}/data/episode_*.hdf5" > /dev/null; then
    task_names+=("$(basename "${task_dir}")")
  fi
done
shopt -u nullglob

if [[ ${#task_names[@]} -eq 0 ]]; then
  echo "[process_data_batch] no tasks with ${env_cfg_type}/data/episode_*.hdf5 under ${DATASET_DIR}" >&2
  exit 1
fi

IFS=$'\n' read -r -d '' -a sorted < <(printf '%s\n' "${task_names[@]}" | sort && printf '\0')
joined="$(IFS=,; printf '%s' "${sorted[*]}")"
echo "[process_data_batch] merging ${#sorted[@]} tasks -> ckpt_name=${ckpt_name}: ${joined}"

bash "${SCRIPT_DIR}/process_data.sh" \
  "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" \
  "${joined}" "${ckpt_name}"
