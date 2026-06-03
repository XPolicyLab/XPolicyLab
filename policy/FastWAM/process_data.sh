#!/bin/bash
set -euo pipefail

# Usage:
#   single task : bash process_data.sh <dataset> <task> <env_cfg> <num> <action_type>
#   merged co.  : bash process_data.sh <dataset> "task_a,task_b,..." <env_cfg> <num> <action_type> [dataset_id]
# `dataset_id` controls the output folder name under <policy>/data/<dataset_id>/;
# for a single task it defaults to "<dataset>-<task>-<env_cfg>-<num>-<action_type>",
# for a comma-list it defaults to "cotrain_dataset" (matches process_data.py).
dataset_name=${1}
task_name=${2}          # single task, or comma-separated list to merge, e.g. "stack_bowls,press_by_number"
env_cfg_type=${3}
expert_data_num=${4}    # episodes kept PER task
action_type=${5}
dataset_id=${6:-}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"

# Resolve the effective dataset_id the same way process_data.py does, so the
# text-embed cache path and the lerobot output path stay in sync without a
# second round-trip into python.
n_tasks=$(awk -F',' '{n=0; for(i=1;i<=NF;i++){gsub(/^ +| +$/,"",$i); if($i!="")n++} print n}' <<< "${task_name}")
if [[ -z "${dataset_id}" ]]; then
    if [[ "${n_tasks}" == "1" ]]; then
        dataset_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
    else
        dataset_id="cotrain_dataset"
    fi
fi
dataset_dir="${POLICY_DIR}/data/${dataset_id}/lerobot"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"
export PYTHONPATH="${ROOT_DIR}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"

echo "[FastWAM] dataset_id=${dataset_id} (tasks=${n_tasks})"
echo "[FastWAM] output lerobot dir: ${dataset_dir}"
echo "[FastWAM] text embed cache:   ${text_cache_dir}"

py_args=(
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
    --project-root "${ROOT_DIR}"
)
if [[ -n "${dataset_id}" ]]; then
    py_args+=(--dataset-id "${dataset_id}")
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
