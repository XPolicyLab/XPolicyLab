#!/bin/bash
set -euo pipefail

# Discover every task under final_data/<bench_name>/ that has episodes for the
# given env_cfg_type, then merge them all into one LeRobot dataset via
# process_data.sh.
#   bash process_data_batch.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
#       [expert_data_num] [dataset_id]
# expert_data_num: optional; empty = all episodes (kept PER task).
# dataset_id: optional output folder name; default <bench>-<ckpt>-<env>-<action>.
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}
dataset_id=${6:-}

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
echo "[process_data_batch] merging ${#sorted[@]} tasks -> ckpt_name=${ckpt_name}: ${joined}"

bash "${SCRIPT_DIR}/process_data.sh" \
  "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
  "${expert_data_num}" "${joined}" "${dataset_id}"
