#!/bin/bash
set -euo pipefail

# GalaxeaVLA (G0Plus_3B) fine-tune launcher — XPolicyLab standard arg format.
#
# Usage (mirrors process_data.sh / eval.sh family):
#   bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> \
#                 <action_type> <gpu_id> <seed> [extra hydra overrides...]
#
# Examples:
#   # ee (end-effector) fine-tune on the pre-converted RoboDojo arx-x5 dataset
#   # (read-only), 4 GPUs, seed 0. The shared dataset path is wired into the ee
#   # data config; override with GALAXEA_DATASET_DIR if it lives elsewhere.
#   bash train.sh RoboDojo cotrain arx_x5 100 ee 0,1,2,3 0
#
#   # joint fine-tune on a process_data.sh output under ./data/
#   bash train.sh RoboDojo robodojo_joint arx_x5 100 joint 0 0
#
# Checkpoint identity is keyed by the XPolicyLab 6-tuple (same join used by
# setup_eval_policy_server.sh / eval.sh):
#   <dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
# Override knobs (env vars):
#   GALAXEA_DATASET_DIR        - LeRobot dataset dir (default ./data/<tuple>-lerobot)
#   GALAXEA_PRETRAINED_CKPT    - base checkpoint dir (default ./checkpoints/G0Plus_3B_base/checkpoints)
#   GALAXEA_FM_OUTPUT_DIR      - training output root (default ./checkpoints)
#   GALAXEA_LOGGER_MODE        - swanlab/wandb mode (default disabled)
#   ALLOW_PLACEHOLDER_LANG     - set true to bypass the placeholder-language guard

dataset_name=${1:?dataset_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:-ee}
gpu_id=${6:-0}
seed=${7:-0}
shift 7 2>/dev/null || shift $# 
extra_overrides=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

# ---- action_type -> task config ----
case "${action_type}" in
    ee)    task_config="real/g0plus_xpolicylab_ee_finetune" ;;
    joint) task_config="real/g0plus_xpolicylab_finetune" ;;
    *) echo -e "\033[31m[train] unknown action_type '${action_type}' (expected ee|joint)\033[0m" >&2; exit 1 ;;
esac

# ---- resolve dataset dir (env override, else process_data.sh output tuple) ----
default_dataset_dir="${SCRIPT_DIR}/data/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-lerobot"
dataset_dir="${GALAXEA_DATASET_DIR:-${default_dataset_dir}}"
if [[ ! -d "${dataset_dir}" ]]; then
    echo -e "\033[31m[train] dataset dir not found: ${dataset_dir}\033[0m" >&2
    echo "  Run process_data.sh first, or set GALAXEA_DATASET_DIR to an existing LeRobot dataset." >&2
    exit 1
fi
dataset_dir="$(cd "${dataset_dir}" && pwd)"

# ---- resolve pretrained checkpoint ----
pretrained_ckpt="${GALAXEA_PRETRAINED_CKPT:-${SCRIPT_DIR}/checkpoints/G0Plus_3B_base/checkpoints}"
if [[ ! -d "${pretrained_ckpt}" ]]; then
    echo -e "\033[31m[train] pretrained ckpt dir not found: ${pretrained_ckpt}\033[0m" >&2
    echo "  Set GALAXEA_PRETRAINED_CKPT to the G0Plus_3B_base checkpoints dir (see INSTALLATION.md)." >&2
    exit 1
fi
pretrained_ckpt="$(cd "${pretrained_ckpt}" && pwd)"

# ---- resolve PaliGemma backbone weights (model config ships a placeholder path) ----
paligemma_path="${GALAXEA_PALIGEMMA_PATH:-${SCRIPT_DIR}/weights/paligemma-3b-pt-224}"
if ! ls "${paligemma_path}"/*.safetensors >/dev/null 2>&1; then
    echo -e "\033[31m[train] PaliGemma weights not found (no *.safetensors) under: ${paligemma_path}\033[0m" >&2
    echo "  Set GALAXEA_PALIGEMMA_PATH to the google/paligemma-3b-pt-224 dir (see INSTALLATION.md)." >&2
    exit 1
fi
paligemma_path="$(cd "${paligemma_path}" && pwd)"

# ---- language placeholder guard (contract requirement) ----
# A language-conditioned VLA needs distinct per-task instructions. Abort if the
# dataset's task strings have collapsed to a single placeholder across multiple
# task_index entries (e.g. all "stack the bowls"). Bypass with ALLOW_PLACEHOLDER_LANG=true.
tasks_meta="${dataset_dir}/meta/tasks.jsonl"
if [[ "${ALLOW_PLACEHOLDER_LANG:-false}" != "true" && -f "${tasks_meta}" ]]; then
    read -r n_idx n_uniq < <(python3 - "${tasks_meta}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(len(rows), len({r.get("task", "") for r in rows}))
PY
)
    if [[ "${n_idx}" -gt 1 && "${n_uniq}" -le 1 ]]; then
        echo -e "\033[31m[train] placeholder language detected: ${n_idx} task_index entries but only ${n_uniq} unique instruction(s) in ${tasks_meta}.\033[0m" >&2
        echo "  This dataset's per-task language is collapsed; a language-conditioned VLA would train on a single instruction." >&2
        echo "  Fix the dataset's tasks.jsonl, or set ALLOW_PLACEHOLDER_LANG=true to train anyway (e.g. pure visuomotor)." >&2
        exit 1
    fi
    echo -e "\033[33m[train] language check OK: ${n_uniq} unique instruction(s) over ${n_idx} task_index entries\033[0m"
fi

# ---- effective seed ----
# Upstream set_global_seed() asserts 0 < seed < uint32_max (rejects seed=0), but
# the XPolicyLab convention commonly uses seed=0. Shift by +1 so the standard
# seed values map to valid upstream seeds (0->1, 1->2, ...).
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo -e "\033[31m[train] invalid seed '${seed}' (expected non-negative integer)\033[0m" >&2; exit 1
fi
effective_seed=$((seed + 1))

# ---- gpu count from gpu_id (single id or comma list) ----
export CUDA_VISIBLE_DEVICES="${gpu_id}"
if [[ "${gpu_id}" == *","* ]]; then
    num_gpu="$(awk -F, '{print NF}' <<< "${gpu_id}")"
else
    num_gpu="1"
fi

# ---- output / cache dirs (sensible defaults, overridable) ----
ckpt_run_id="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
export GALAXEA_FM_OUTPUT_DIR="${GALAXEA_FM_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints}"
export GALAXEA_CKPT_RUN_ID="${ckpt_run_id}"
export GALAXEA_FM_DATASET_STATS_CACHE_DIR="${GALAXEA_FM_DATASET_STATS_CACHE_DIR:-${SCRIPT_DIR}/.cache/galaxea_stats}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRIPT_DIR}/.cache/hf_datasets}"
mkdir -p "${GALAXEA_FM_OUTPUT_DIR}/${GALAXEA_CKPT_RUN_ID}" "${GALAXEA_FM_DATASET_STATS_CACHE_DIR}" "${HF_DATASETS_CACHE}"
logger_mode="${GALAXEA_LOGGER_MODE:-disabled}"

# ---- informational action dim (dims actually come from the data config) ----
action_dim="$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}" 2>/dev/null || echo "?")"

echo -e "\033[33m[train] dataset_name=${dataset_name} ckpt_name=${ckpt_name} env_cfg_type=${env_cfg_type} expert_data_num=${expert_data_num} action_type=${action_type}\033[0m"
echo -e "\033[33m[train] task_config=${task_config} | gpus=${gpu_id} (n=${num_gpu}) | seed=${seed} (upstream seed=${effective_seed}) | action_dim(info)=${action_dim}\033[0m"
echo -e "\033[33m[train] dataset_dir=${dataset_dir}\033[0m"
echo -e "\033[33m[train] pretrained_ckpt=${pretrained_ckpt}\033[0m"
echo -e "\033[33m[train] paligemma_path=${paligemma_path}\033[0m"
echo -e "\033[33m[train] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[train] output_dir=${GALAXEA_FM_OUTPUT_DIR}/${GALAXEA_CKPT_RUN_ID}/<timestamp>\033[0m"

# ---- launch ----
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
