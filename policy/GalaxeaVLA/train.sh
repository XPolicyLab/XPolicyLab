#!/bin/bash
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> \
#                      <action_type> <seed> <gpu_id> [hydra overrides...]
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:-joint}
seed=${6:-0}
gpu_id=${7:-0}
shift 7 2>/dev/null || shift $#
extra_overrides=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

ADAPTER_DIR="${SCRIPT_DIR}/GalaxeaVLA/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

if [[ "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[train] GalaxeaVLA only supports action_type=joint, got '${action_type}'\033[0m" >&2
    exit 1
fi
task_config="${GALAXEA_TASK_CONFIG:-real/g0plus_xpolicylab_finetune}"

default_dataset_dir="$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${expert_data_num}")"
dataset_dir="${GALAXEA_DATASET_DIR:-${default_dataset_dir}}"
if [[ ! -d "${dataset_dir}" ]]; then
    echo -e "\033[31m[train] dataset dir not found: ${dataset_dir}\033[0m" >&2
    echo "  Run process_data.sh first, or set GALAXEA_DATASET_DIR." >&2
    exit 1
fi
dataset_dir="$(cd "${dataset_dir}" && pwd)"

pretrained_ckpt="${GALAXEA_PRETRAINED_CKPT:-${SCRIPT_DIR}/checkpoints/G0Plus_3B_base/checkpoints}"
if [[ ! -d "${pretrained_ckpt}" ]]; then
    echo -e "\033[31m[train] pretrained ckpt dir not found: ${pretrained_ckpt}\033[0m" >&2
    echo "  Set GALAXEA_PRETRAINED_CKPT (see INSTALLATION.md)." >&2
    exit 1
fi
pretrained_ckpt="$(cd "${pretrained_ckpt}" && pwd)"

paligemma_path="${GALAXEA_PALIGEMMA_PATH:-${SCRIPT_DIR}/weights/paligemma-3b-pt-224}"
if ! ls "${paligemma_path}"/*.safetensors >/dev/null 2>&1; then
    echo -e "\033[31m[train] PaliGemma weights not found under: ${paligemma_path}\033[0m" >&2
    echo "  Set GALAXEA_PALIGEMMA_PATH (see INSTALLATION.md)." >&2
    exit 1
fi
paligemma_path="$(cd "${paligemma_path}" && pwd)"

# Upstream rejects seed=0; shift so XPolicyLab seed N maps to upstream N+1.
VENV_PYTHON="${UPSTREAM_DIR}/.venv/bin/python3"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="$(command -v python3)"
fi
tasks_jsonl="${dataset_dir}/meta/tasks.jsonl"
tasks_parquet="${dataset_dir}/meta/tasks.parquet"
if [[ "${ALLOW_PLACEHOLDER_LANG:-false}" != "true" ]]; then
    if [[ -f "${tasks_jsonl}" ]]; then
        read -r n_idx n_uniq < <("${VENV_PYTHON}" - "${tasks_jsonl}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(len(rows), len({r.get("task", "") for r in rows}))
PY
)
        tasks_meta="${tasks_jsonl}"
    elif [[ -f "${tasks_parquet}" ]]; then
        lang_check_out="$("${VENV_PYTHON}" - "${tasks_parquet}" <<'PY' 2>/dev/null || true
import sys
import pyarrow.parquet as pq
table = pq.read_table(sys.argv[1])
df = table.to_pandas()
if "task" in df.columns:
    tasks = df["task"].tolist()
else:
    tasks = list(df.index)
print(len(tasks), len({t for t in tasks if t is not None and str(t).strip()}))
PY
)"
        if [[ -z "${lang_check_out}" || ! "${lang_check_out}" =~ ^[0-9]+\ [0-9]+$ ]]; then
            echo -e "\033[33m[train] language check skipped: could not parse ${tasks_parquet}\033[0m" >&2
            n_idx=0
            n_uniq=0
            tasks_meta=""
        else
            read -r n_idx n_uniq <<< "${lang_check_out}"
            tasks_meta="${tasks_parquet}"
        fi
    else
        n_idx=0
        n_uniq=0
        tasks_meta=""
    fi
    if [[ -n "${tasks_meta}" ]]; then
        if [[ "${n_idx}" -gt 1 && "${n_uniq}" -le 1 ]]; then
            echo -e "\033[31m[train] placeholder language detected: ${n_idx} task_index entries but only ${n_uniq} unique instruction(s) in ${tasks_meta}.\033[0m" >&2
            echo "  Fix tasks metadata or set ALLOW_PLACEHOLDER_LANG=true." >&2
            exit 1
        fi
        echo -e "\033[33m[train] language check OK: ${n_uniq} unique instruction(s) over ${n_idx} task_index entries (${tasks_meta})\033[0m"
    fi
fi

if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo -e "\033[31m[train] invalid seed '${seed}' (expected non-negative integer)\033[0m" >&2; exit 1
fi
effective_seed=$((seed + 1))

export CUDA_VISIBLE_DEVICES="${gpu_id}"
if [[ "${gpu_id}" == *","* ]]; then
    num_gpu="$(awk -F, '{print NF}' <<< "${gpu_id}")"
else
    num_gpu="1"
fi

ckpt_run_id="$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
export GALAXEA_FM_OUTPUT_DIR="${GALAXEA_FM_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints}"
export GALAXEA_CKPT_RUN_ID="${ckpt_run_id}"
export GALAXEA_FM_DATASET_STATS_CACHE_DIR="${GALAXEA_FM_DATASET_STATS_CACHE_DIR:-${SCRIPT_DIR}/.cache/galaxea_stats}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRIPT_DIR}/.cache/hf_datasets}"
mkdir -p "${GALAXEA_FM_OUTPUT_DIR}/${GALAXEA_CKPT_RUN_ID}" "${GALAXEA_FM_DATASET_STATS_CACHE_DIR}" "${HF_DATASETS_CACHE}"
logger_mode="${GALAXEA_LOGGER_MODE:-disabled}"

action_dim="$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}" 2>/dev/null || echo "?")"

echo -e "\033[33m[train] bench_name=${bench_name} ckpt_name=${ckpt_name} env_cfg_type=${env_cfg_type} expert_data_num=${expert_data_num} action_type=${action_type}\033[0m"
echo -e "\033[33m[train] task_config=${task_config} | gpus=${gpu_id} (n=${num_gpu}) | seed=${seed} (upstream seed=${effective_seed}) | action_dim(info)=${action_dim}\033[0m"
echo -e "\033[33m[train] dataset_dir=${dataset_dir}\033[0m"
echo -e "\033[33m[train] pretrained_ckpt=${pretrained_ckpt}\033[0m"
echo -e "\033[33m[train] paligemma_path=${paligemma_path}\033[0m"
echo -e "\033[33m[train] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[train] output_dir=${GALAXEA_FM_OUTPUT_DIR}/${GALAXEA_CKPT_RUN_ID}/<timestamp>\033[0m"

source "${UPSTREAM_DIR}/.venv/bin/activate"
cd "${UPSTREAM_DIR}"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
bash scripts/run/finetune.sh "${num_gpu}" "${task_config}" \
    "model.pretrained_ckpt=${pretrained_ckpt}" \
    "model.model_arch.pretrained_model_path=${paligemma_path}" \
    "model.tokenizer.tokenizer_params.pretrained_model_name_or_path=${paligemma_path}" \
    "model.tokenizer.tokenizer_params.local_files_only=True" \
    "data.dataset.dataset_dirs=[${dataset_dir}]" \
    "seed=${effective_seed}" \
    "logger.mode=${logger_mode}" \
    "${extra_overrides[@]}"
