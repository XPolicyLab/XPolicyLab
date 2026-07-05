#!/bin/bash
set -euo pipefail

# Discover every task under data/<bench_name>/ that has episodes for the given
# env_cfg_type, then merge them all into one LeRobot dataset via process_data.sh.
#   bash process_data_batch.sh RoboDojo arx_x5 3 joint [dataset_id]
bench_name=${1}
env_cfg_type=${2}
expert_data_num=${3}    # episodes kept PER task
action_type=${4}
dataset_id=${5:-}       # optional output folder name; default cotrain_dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_DIR="${ROOT_DIR}/final_data/${bench_name}"

# Collect task dirs that actually contain <task>/<env_cfg_type>/data/episode_*.hdf5.
shopt -s nullglob
task_names=()
for task_dir in "${DATASET_DIR}"/*/; do
  if compgen -G "${task_dir}${env_cfg_type}/data/episode_*.hdf5" > /dev/null; then
    task_names+=("$(basename "${task_dir}")")
  fi
done
shopt -u nullglob

if [[ ${#task_names[@]} -eq 0 ]]; then
  echo "[process_data_batch] no tasks with ${env_cfg_type}/final_data/episode_*.hdf5 under ${DATASET_DIR}" >&2
  exit 1
fi

# Sort for deterministic episode ordering, then comma-join for process_data.sh.
IFS=$'\n' read -r -d '' -a sorted < <(printf '%s\n' "${task_names[@]}" | sort && printf '\0')
joined="$(IFS=,; printf '%s' "${sorted[*]}")"
echo "[process_data_batch] merging ${#sorted[@]} tasks: ${joined}"

bash "${SCRIPT_DIR}/process_data.sh" \
  "${bench_name}" "${joined}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" "${dataset_id}"
