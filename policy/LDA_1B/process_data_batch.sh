#!/bin/bash
# Usage: bash process_data_batch.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Discovers every task under data/<bench_name>/ with episodes for env_cfg_type and
# merges them into one dataset named <bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>.
# expert_data_num: optional; empty = all episodes (kept PER task).
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_DIR="${ROOT_DIR}/data/${bench_name}"

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
  "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
  "${expert_data_num}" "${joined}"
