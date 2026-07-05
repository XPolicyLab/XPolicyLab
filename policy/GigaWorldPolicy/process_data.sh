#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="${XPOLICYLAB_ROOT:-$(cd "${POLICY_DIR}/../.." && pwd)}"
INNER_DIR="${POLICY_DIR}/giga_world_policy"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
out_dir="${POLICY_DIR}/data/${data_setting}"
source_dir="${GIGAWORLD_SOURCE_DATA_DIR:-}"
task_names="${GIGAWORLD_TASK_NAMES:-${ckpt_name}}"
python_bin="${GIGAWORLD_PYTHON:-python}"
wan_model_path="${GIGAWORLD_PRETRAINED_PATH:-${WAN22_DIFFUSERS_PATH:-}}"
action_dim="${GIGAWORLD_MODEL_ACTION_DIM:-14}"
state_dim="${GIGAWORLD_MODEL_STATE_DIM:-${action_dim}}"
image_width="${GIGAWORLD_IMAGE_WIDTH:-640}"
image_height="${GIGAWORLD_IMAGE_HEIGHT:-480}"

mkdir -p "${POLICY_DIR}/data"

if [[ -n "${source_dir}" ]]; then
  if [[ ! -d "${source_dir}" ]]; then
    echo "[GigaWorldPolicy] source data does not exist: ${source_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${source_dir}/meta" || ! -d "${source_dir}/data" ]]; then
    echo "[GigaWorldPolicy] source data is not a LeRobot v2.1 dataset: ${source_dir}" >&2
    exit 1
  fi
  rm -rf "${out_dir}"
  ln -s "$(realpath "${source_dir}")" "${out_dir}"
  echo "[GigaWorldPolicy] linked ${out_dir} -> ${source_dir}"
else
  echo "[GigaWorldPolicy] converting XPolicyLab HDF5 -> LeRobot v2.1"
  echo "  source root: ${XPL_ROOT}/data/${bench_name}/{${task_names}}/${env_cfg_type}"
  echo "  output:      ${out_dir}"
  "${python_bin}" "${INNER_DIR}/scripts/convert_xpolicylab_hdf5_to_lerobot.py" \
    --xpolicylab-root "${XPL_ROOT}" \
    --bench-name "${bench_name}" \
    --task-names "${task_names}" \
    --env-cfg-type "${env_cfg_type}" \
    --expert-data-num "${expert_data_num}" \
    --action-type "${action_type}" \
    --output-dir "${out_dir}" \
    --image-width "${image_width}" \
    --image-height "${image_height}" \
    --input-color-space "${GIGAWORLD_INPUT_COLOR_SPACE:-rgb}" \
    --video-codec "${GIGAWORLD_VIDEO_CODEC:-mp4v}" \
    --overwrite
fi

cd "${INNER_DIR}"

if [[ "${GIGAWORLD_COMPUTE_NORM:-1}" == "1" ]]; then
  "${python_bin}" scripts/fast_norm_stats.py \
    --data_root "${out_dir}" \
    --output "${out_dir}/norm_stats_delta.json" \
    --action_dim "${action_dim}" \
    --state_dim "${state_dim}"
fi

if [[ "${GIGAWORLD_GENERATE_T5:-0}" == "1" ]]; then
  if [[ -z "${wan_model_path}" ]]; then
    echo "[GigaWorldPolicy] set GIGAWORLD_PRETRAINED_PATH (or WAN22_DIFFUSERS_PATH) to the local Wan2.2-TI2V-5B-Diffusers directory to generate T5 embeddings" >&2
    exit 1
  fi
  "${python_bin}" scripts/generate_t5_embeddings.py \
    --data_root "${out_dir}" \
    --wan_model_path "${wan_model_path}"
fi

echo "[GigaWorldPolicy] data ready: ${out_dir}"
