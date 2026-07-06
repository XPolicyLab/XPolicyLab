#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [source_ckpt_name]" >&2
  echo "  expert_data_num: optional; empty = use all episodes" >&2
  echo "  source_ckpt_name: optional raw-data source name; defaults to ckpt_name" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
source_ckpt_name=${6:-${ckpt_name}}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXBOTIC_ROOT="${POLICY_DIR}/dexbotic"
DATA_SOURCE_DIR="${DEXBOTIC_ROOT}/dexbotic/data/data_source"
TRANSFORM_SCRIPT="${POLICY_DIR}/scripts/transform_dm0_dexdata_format.py"
GENERATE_SOURCE_SCRIPT="${POLICY_DIR}/scripts/generate_data_source.py"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
converted_data_root="${DM0_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
raw_data_root="${DM0_RAW_DATA_ROOT:?set DM0_RAW_DATA_ROOT to your RoboDojo raw dataset root}"
data_source_path="${DATA_SOURCE_DIR}/robodojo_${data_setting}.py"

resolve_single_input_dir() {
  if [[ "${bench_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud/${source_ckpt_name}/${env_cfg_type}" ]]; then
    echo "${raw_data_root}/sim_cloud/${source_ckpt_name}/${env_cfg_type}"
    return
  fi
  if [[ -d "${raw_data_root}/${bench_name}/${source_ckpt_name}/${env_cfg_type}" ]]; then
    echo "${raw_data_root}/${bench_name}/${source_ckpt_name}/${env_cfg_type}"
    return
  fi
  echo "Input directory not found for ${bench_name}/${source_ckpt_name}/${env_cfg_type}" >&2
  echo "For 35-task co-train, use source_ckpt_name=cotrain." >&2
  echo "For a single task, use ckpt_name=<task_name>, e.g. sweep_blocks." >&2
  echo "Example: bash process_data.sh RoboDojo cotrain arx_x5 ee" >&2
  echo "For an ablation run, keep ckpt_name unique and pass the source as the sixth argument." >&2
  echo "Set DM0_RAW_DATA_ROOT or check source_ckpt_name." >&2
  exit 1
}

build_cotrain_staging_dir() {
  local staging_dir="${converted_data_root}/.raw_staging"
  rm -rf "${staging_dir}"
  mkdir -p "${staging_dir}"

  local sim_root="${raw_data_root}/sim_cloud"
  if [[ ! -d "${sim_root}" ]]; then
    echo "Co-train expects ${sim_root}" >&2
    exit 1
  fi

  local task_dir env_dir task_name hdf5_path count
  for task_dir in "${sim_root}"/*; do
    [[ -d "${task_dir}" ]] || continue
    env_dir="${task_dir}/${env_cfg_type}"
    [[ -d "${env_dir}" ]] || continue
    task_name="$(basename "${task_dir}")"
    count=0
    while IFS= read -r hdf5_path; do
      [[ -n "${hdf5_path}" ]] || continue
      ln -sf "$(readlink -f "${hdf5_path}")" "${staging_dir}/${task_name}_$(basename "${hdf5_path}")"
      count=$((count + 1))
      if [[ -n "${expert_data_num}" && "${count}" -ge "${expert_data_num}" ]]; then
        break
      fi
    done < <(find "${env_dir}" -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort)
  done

  if [[ -z "$(find "${staging_dir}" -maxdepth 1 -type l 2>/dev/null | head -n 1)" ]]; then
    echo "No HDF5 files found under ${sim_root}/*/${env_cfg_type}" >&2
    exit 1
  fi

  echo "${staging_dir}"
}

resolve_input_dir() {
  if [[ "${source_ckpt_name}" == "cotrain" ]]; then
    build_cotrain_staging_dir
    return
  fi
  resolve_single_input_dir
}

echo "[Dexbotic_DM0] bench_name=${bench_name}"
echo "[Dexbotic_DM0] ckpt_name=${ckpt_name}"
echo "[Dexbotic_DM0] source_ckpt_name=${source_ckpt_name}"
echo "[Dexbotic_DM0] env_cfg_type=${env_cfg_type}"
echo "[Dexbotic_DM0] expert_data_num=${expert_data_num:-<all>}"
echo "[Dexbotic_DM0] action_type=${action_type}"
echo "[Dexbotic_DM0] raw_data_root=${raw_data_root}"
echo "[Dexbotic_DM0] converted_data_root=${converted_data_root}"
echo "[Dexbotic_DM0] dexbotic_root=${DEXBOTIC_ROOT}"

input_dir="$(resolve_input_dir)"
echo "[Dexbotic_DM0] input_dir=${input_dir}"

mkdir -p "${converted_data_root}" "${DATA_SOURCE_DIR}"

echo "[Dexbotic_DM0] num_workers=${DM0_CONVERT_WORKERS:-8}"

python "${TRANSFORM_SCRIPT}" \
  "${input_dir}" \
  "${converted_data_root}" \
  --data_type xspark \
  --data_version v1.0 \
  --num_workers "${DM0_CONVERT_WORKERS:-8}"

python "${GENERATE_SOURCE_SCRIPT}" \
  "${converted_data_root}" \
  "${data_setting}" \
  "${data_source_path}"

if [[ "${source_ckpt_name}" == "cotrain" && -d "${converted_data_root}/.raw_staging" ]]; then
  rm -rf "${converted_data_root}/.raw_staging"
fi

echo "[Dexbotic_DM0] process_data done."
echo "[Dexbotic_DM0] dexdata_root=${converted_data_root}"
echo "[Dexbotic_DM0] bench_name=robodojo_${data_setting}"
echo "[Dexbotic_DM0] data_source=${data_source_path}"
