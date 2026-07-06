#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  echo "  expert_data_num: optional; empty = use all episodes" >&2
  echo "  raw_task_dirs: optional source task dir or comma-separated task dirs; defaults to ckpt_name" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-}
if [[ -n "${expert_data_num}" && ! "${expert_data_num}" =~ ^[0-9]+$ ]]; then
  raw_task_dirs="${expert_data_num}"
  expert_data_num=""
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR0_ROOT="${POLICY_DIR}/xiaomi_robotics_0/xr0"
TRANSFORM_SCRIPT="${POLICY_DIR}/scripts/transform_xr0_json_format.py"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
converted_data_root="${XR0_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
raw_data_root="${XR0_RAW_DATA_ROOT:-}"
if [[ -z "${raw_data_root}" ]]; then
  echo "Set XR0_RAW_DATA_ROOT to the RoboDojo HDF5 root (contains sim_cloud/)." >&2
  exit 1
fi
data_config_name="${XR0_DATA_CONFIG_NAME:-${data_setting}}"

resolve_task_input_dir() {
  local task_dir_name=$1
  if [[ "${bench_name}" == "RoboDojo" && -d "${raw_data_root}/sim_cloud/${task_dir_name}/${env_cfg_type}" ]]; then
    echo "${raw_data_root}/sim_cloud/${task_dir_name}/${env_cfg_type}"
    return
  fi
  if [[ -d "${raw_data_root}/${bench_name}/${task_dir_name}/${env_cfg_type}" ]]; then
    echo "${raw_data_root}/${bench_name}/${task_dir_name}/${env_cfg_type}"
    return
  fi
  echo "Input directory not found for ${bench_name}/${task_dir_name}/${env_cfg_type}" >&2
  echo "Set XR0_RAW_DATA_ROOT or check raw_task_dirs/ckpt_name." >&2
  exit 1
}

link_task_episodes() {
  local source_dir=$1
  local staging_dir=$2
  local output_prefix=$3
  local hdf5_path count=0

  shopt -s nullglob
  for hdf5_path in "${source_dir}"/*.hdf5 "${source_dir}"/*.h5; do
    ln -sf "$(readlink -f "${hdf5_path}")" "${staging_dir}/${output_prefix}$(basename "${hdf5_path}")"
    count=$((count + 1))
    if [[ -n "${expert_data_num}" && "${count}" -ge "${expert_data_num}" ]]; then
      break
    fi
  done
  shopt -u nullglob

  echo "${count}"
}

build_staging_dir_from_task_list() {
  local task_list=$1
  local staging_dir="${converted_data_root}/.raw_staging"
  rm -rf "${staging_dir}"
  mkdir -p "${staging_dir}"

  local task_name source_dir linked_count total_count=0
  IFS=',' read -r -a task_names <<< "${task_list}"
  for task_name in "${task_names[@]}"; do
    task_name="$(echo "${task_name}" | xargs)"
    [[ -n "${task_name}" ]] || continue
    source_dir="$(resolve_task_input_dir "${task_name}")"
    linked_count="$(link_task_episodes "${source_dir}" "${staging_dir}" "${task_name}_")"
    total_count=$((total_count + linked_count))
  done

  if [[ "${total_count}" -eq 0 ]]; then
    echo "No .hdf5 or .h5 files found for raw_task_dirs=${task_list}" >&2
    exit 1
  fi
  echo "${staging_dir}"
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

  local task_dir env_dir task_name total_count=0 linked_count
  for task_dir in "${sim_root}"/*; do
    [[ -d "${task_dir}" ]] || continue
    env_dir="${task_dir}/${env_cfg_type}"
    [[ -d "${env_dir}" ]] || continue
    task_name="$(basename "${task_dir}")"
    linked_count="$(link_task_episodes "${env_dir}" "${staging_dir}" "${task_name}_")"
    total_count=$((total_count + linked_count))
  done

  if [[ "${total_count}" -eq 0 ]]; then
    echo "No .hdf5 or .h5 files found under ${sim_root}/*/${env_cfg_type}" >&2
    exit 1
  fi
  echo "${staging_dir}"
}

resolve_input_dir() {
  if [[ -n "${raw_task_dirs}" ]]; then
    build_staging_dir_from_task_list "${raw_task_dirs}"
    return
  fi
  if [[ "${ckpt_name}" == "cotrain" ]]; then
    build_cotrain_staging_dir
    return
  fi
  local single_dir
  single_dir="$(resolve_task_input_dir "${ckpt_name}")"
  if [[ -n "${expert_data_num}" ]]; then
    build_staging_dir_from_task_list "${ckpt_name}"
    return
  fi
  echo "${single_dir}"
}

echo "[Xiaomi_Robotics_0] bench_name=${bench_name}"
echo "[Xiaomi_Robotics_0] ckpt_name=${ckpt_name}"
echo "[Xiaomi_Robotics_0] env_cfg_type=${env_cfg_type}"
echo "[Xiaomi_Robotics_0] expert_data_num=${expert_data_num:-<all>}"
echo "[Xiaomi_Robotics_0] raw_task_dirs=${raw_task_dirs:-<ckpt_name>}"
echo "[Xiaomi_Robotics_0] action_type=${action_type}"
echo "[Xiaomi_Robotics_0] raw_data_root=${raw_data_root}"
echo "[Xiaomi_Robotics_0] converted_data_root=${converted_data_root}"

input_dir="$(resolve_input_dir)"
echo "[Xiaomi_Robotics_0] input_dir=${input_dir}"

mkdir -p "${converted_data_root}"

python "${TRANSFORM_SCRIPT}" \
  "${input_dir}" \
  "${converted_data_root}" \
  --data_type xspark \
  --data_version v1.0 \
  --num_workers "${XR0_CONVERT_WORKERS:-8}" \
  --compute_stats

python "${POLICY_DIR}/scripts/generate_data_config.py" \
  "${converted_data_root}/action_stats.json" \
  "${XR0_ROOT}/configs/data/${data_config_name}.yaml" \
  --json_dir "${converted_data_root}/json" \
  --batch_size "${XR0_BATCH_SIZE:-16}"

if [[ -d "${converted_data_root}/.raw_staging" ]]; then
  rm -rf "${converted_data_root}/.raw_staging"
fi

echo "[Xiaomi_Robotics_0] process_data done."
echo "[Xiaomi_Robotics_0] json_dir=${converted_data_root}/json"
echo "[Xiaomi_Robotics_0] videos_dir=${converted_data_root}/videos"
echo "[Xiaomi_Robotics_0] data_config=${XR0_ROOT}/configs/data/${data_config_name}.yaml"
