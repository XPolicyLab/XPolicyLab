#!/bin/bash


dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"

ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
out_dir="${POLICY_DIR}/data/${ckpt_setting}"

echo "[TinyVLA process_data] output: ${out_dir}"

#If the 5-tuple output directory already exists, let the user decide:
#   - y  : skip processing entirely, reuse the existing dataset as-is
#   - N  : abort, the user must remove the directory manually before rerunning
if [[ -d "${out_dir}" ]]; then
  echo "[TinyVLA process_data] dataset already exists: ${out_dir}"
  read -r -p "Skip processing and reuse the existing dataset? [y/N]: " ans
  case "${ans}" in
    [yY]|[yY][eE][sS])
      echo "[TinyVLA process_data] skipping; reusing existing dataset."
      exit 0
      ;;
    *)
      echo "[TinyVLA process_data] aborting. Remove ${out_dir} manually and rerun." >&2
      exit 1
      ;;
  esac
fi
mkdir -p "${out_dir}"

merged_idx=0
for task_dir in "${ROOT_DIR}/data/${dataset_name}"/*/; do
  task_name="$(basename "${task_dir}")"
  src_dir="${task_dir}${env_cfg_type}/data"
  for ((i=0; i<expert_data_num; i++)); do
    src_file="${src_dir}/$(printf 'episode_%07d.hdf5' "${i}")"
    dst_file="${out_dir}/$(printf 'episode_%07d.hdf5' "${merged_idx}")"
    ln -s -- "${src_file}" "${dst_file}"
    merged_idx=$((merged_idx + 1))
  done
  echo "[TinyVLA process_data] task='${task_name}': linked ${expert_data_num} episodes"
done

echo "[TinyVLA process_data] total: ${merged_idx} episodes -> ${out_dir}"
