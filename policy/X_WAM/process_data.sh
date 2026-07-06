#!/bin/bash
set -euo pipefail

# XPolicyLab-standard data conversion wrapper for the X_WAM policy.
#
# Converts RoboDojo HDF5 episodes into the X-WAM dataset format
# (metadata + data/ + video/) by calling transform_robodojo_to_xwam.py, and
# writes the output where train.sh expects it: <policy>/data/<data_key>, where
#   data_key = <bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
# This matches the XWAM_DATASET_PATH default used by train.sh, so the two stay
# in sync without any manual path juggling.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
#       [limit] [raw_task_dirs]
#
# limit         : optional; empty = convert all episodes. Maps to the transform's
#                 global --limit (applied across the staged tasks).
# raw_task_dirs : comma-separated raw task dir name(s) under the raw bench root;
#                 defaults to <ckpt_name>. Multiple tasks merge into one dataset.
#
# Raw data location (no personal paths are hard-coded):
#   XWAM_RAW_INPUT_DIR : if set, used directly as the transform --input-dir
#                        (it must contain <task>/<env_cfg_type>/data/*.hdf5 subdirs).
#   XWAM_RAW_DATA_ROOT : raw root that holds per-bench dirs; the bench root is
#                        ${XWAM_RAW_DATA_ROOT}/<bench_name>. Default:
#                        ${ROOT_DIR}/final_data.
#   XWAM_DATASET_PATH  : override the converted-dataset output dir.
bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [limit] [raw_task_dirs]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
limit=${5:-${XWAM_DATA_LIMIT:-}}
raw_task_dirs=${6:-${XWAM_RAW_TASK_DIRS:-${ckpt_name}}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
POLICY_DIR="${SCRIPT_DIR}"

data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
output_dir="${XWAM_DATASET_PATH:-${POLICY_DIR}/data/${data_key}}"
workers="${XWAM_TRANSFORM_WORKERS:-16}"

raw_bench_root="${XWAM_RAW_DATA_ROOT:-${ROOT_DIR}/final_data}/${bench_name}"

# Assemble the transform input dir. When XWAM_RAW_INPUT_DIR is given we trust it
# verbatim; otherwise we stage symlinks for exactly the requested task dirs so a
# single-task ckpt_name does not pull in every task under the bench root.
staging_dir=""
cleanup() {
    if [[ -n "${staging_dir}" && -d "${staging_dir}" ]]; then
        rm -rf "${staging_dir}"
    fi
}
trap cleanup EXIT

if [[ -n "${XWAM_RAW_INPUT_DIR:-}" ]]; then
    input_dir="${XWAM_RAW_INPUT_DIR}"
else
    if [[ ! -d "${raw_bench_root}" ]]; then
        echo "[X_WAM] ERROR: raw bench root not found: ${raw_bench_root}" >&2
        echo "[X_WAM] Set XWAM_RAW_DATA_ROOT (default \${ROOT_DIR}/final_data) or XWAM_RAW_INPUT_DIR." >&2
        exit 1
    fi
    staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/xwam_raw_XXXXXX")"
    input_dir="${staging_dir}"
    IFS=',' read -r -a task_arr <<< "${raw_task_dirs}"
    for task in "${task_arr[@]}"; do
        src="${raw_bench_root}/${task}"
        if [[ ! -d "${src}/${env_cfg_type}/data" ]]; then
            echo "[X_WAM] ERROR: missing episodes: ${src}/${env_cfg_type}/data" >&2
            exit 1
        fi
        ln -s "${src}" "${staging_dir}/${task}"
    done
fi

echo "[X_WAM] data_key=${data_key} (raw_task_dirs=${raw_task_dirs})"
echo "[X_WAM] input_dir:  ${input_dir}"
echo "[X_WAM] output_dir: ${output_dir}"

transform_args=(
    --input-dir "${input_dir}"
    --output-dir "${output_dir}"
    --workers "${workers}"
    --env-cfg-type "${env_cfg_type}"
)
if [[ -n "${limit}" ]]; then
    transform_args+=(--limit "${limit}")
fi

python "${POLICY_DIR}/transform_robodojo_to_xwam.py" "${transform_args[@]}"

echo "[X_WAM] done. Train with the matching data_key by running:"
echo "  bash ${POLICY_DIR}/train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_id>"
