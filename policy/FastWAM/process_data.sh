#!/bin/bash
set -euo pipefail

# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
#       [expert_data_num] [raw_task_dirs] [dataset_id]
# expert_data_num : optional; empty = use all episodes (kept PER task).
# raw_task_dirs   : raw HDF5 task dir(s) under final_data/<bench_name>/;
#                   comma-separated list merges into one dataset, e.g.
#                   "stack_bowls,press_by_number". Defaults to ${ckpt_name}.
# dataset_id      : controls the output folder name under <policy>/data/<dataset_id>/;
#                   defaults to "<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>".
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-${ckpt_name}}
dataset_id=${7:-}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"

# Resolve the effective dataset_id the same way process_data.py does, so the
# text-embed cache path and the lerobot output path stay in sync without a
# second round-trip into python.
if [[ -z "${dataset_id}" ]]; then
    dataset_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
fi
dataset_dir="${POLICY_DIR}/data/${dataset_id}/lerobot"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"
export PYTHONPATH="${ROOT_DIR}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"

echo "[FastWAM] dataset_id=${dataset_id} (raw_task_dirs=${raw_task_dirs})"
echo "[FastWAM] output lerobot dir: ${dataset_dir}"
echo "[FastWAM] text embed cache:   ${text_cache_dir}"

py_args=(
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"
    --raw-task-dirs "${raw_task_dirs}"
    --project-root "${ROOT_DIR}"
    --dataset-id "${dataset_id}"
)
if [[ -n "${expert_data_num}" ]]; then
    py_args+=(--expert-data-num "${expert_data_num}")
fi
python "${FASTWAM_DIR}/process_data.py" "${py_args[@]}"

if [[ "${FASTWAM_PRECOMPUTE_TEXT_EMBEDS:-true}" == "true" ]]; then
    if [[ ! -d "${text_cache_dir}" || -z "$(find "${text_cache_dir}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
        cd "${FASTWAM_DIR}"
        python scripts/precompute_text_embeds.py \
            "task=robotwin_uncond_3cam_384_1e-4" \
            "data.train.dataset_dirs=[${dataset_dir}]" \
            "data.val.dataset_dirs=[${dataset_dir}]" \
            "data.train.text_embedding_cache_dir=${text_cache_dir}" \
            "data.val.text_embedding_cache_dir=${text_cache_dir}"
    else
        echo "[FastWAM] Reusing text embedding cache: ${text_cache_dir}"
    fi
fi
