#!/bin/bash
set -euo pipefail

# XPolicyLab-standard data preparation for PICA_WAM (RoboTwin 2.0, clean demos).
#
# 1. Downloads the official RoboTwin 2.0 clean demonstrations
#    (aloha-agilex_clean_50) in LeRobot v3.0 form, from the clean split of
#    OpenMOSS-Team/robotwin2.0-lerobot-v3.0 (revision 4beb760, every file
#    checksum-verified), and rewrites it as the LeRobot v2.0 layout the
#    training loader reads, without re-encoding video:
#      <POLICY_DIR>/data/<bench>-<ckpt>-<env>-<action>/{meta,data,videos}
# 2. Builds the UMT5 text-embedding cache for that dataset's instructions:
#      <POLICY_DIR>/.cache/text_embed/<bench>-<ckpt>-<env>-<action>/
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [gpu_ids]
#
# Optional environment:
#   PICA_WAM_SOURCE_DIR    v3.0 download directory (default: <POLICY_DIR>/data/robotwin2.0-lerobot-v3.0)
#   PICA_WAM_DATASET_DIR   v2.0 export (default: <POLICY_DIR>/data/<bench>-<ckpt>-<env>-<action>)
#   PICA_WAM_TEXT_CACHE    text-embedding cache directory
#   PICA_WAM_SKIP_DOWNLOAD=1  reuse an already verified download
#   HF_ENDPOINT            Hugging Face mirror, e.g. https://hf-mirror.com

bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [gpu_ids]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
gpu_ids=${5:-${CUDA_VISIBLE_DEVICES:-0}}

if [[ "${bench_name}" != "RoboTwin" || "${env_cfg_type}" != "aloha_agilex" || "${action_type}" != "joint" ]]; then
    echo "[PICA_WAM] ERROR: the recipe is RoboTwin / aloha_agilex / joint (14D), got ${bench_name} / ${env_cfg_type} / ${action_type}." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${POLICY_DIR}/internw0_pica"
ASSET_ROOT="${INTERNW0_ASSET_ROOT:-${POLICY_DIR}/assets}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
source_dir="${PICA_WAM_SOURCE_DIR:-${POLICY_DIR}/data/robotwin2.0-lerobot-v3.0}"
dataset_dir="${PICA_WAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"
text_cache="${PICA_WAM_TEXT_CACHE:-${POLICY_DIR}/.cache/text_embed/${data_setting}}"
wan_dir="${WAM_WAN22_PATH:-${ASSET_ROOT}/Wan-AI/Wan2.2-TI2V-5B}"

dataset_ready() {
    [[ -f "$1/meta/info.json" && -f "$1/meta/tasks.jsonl" && -f "$1/meta/episodes.jsonl" && -d "$1/data" && -d "$1/videos" ]]
}

if dataset_ready "${dataset_dir}"; then
    echo "[PICA_WAM] existing training export found: ${dataset_dir}"
else
    prepare_args=(--source "${source_dir}" --output "${dataset_dir}")
    [[ "${PICA_WAM_SKIP_DOWNLOAD:-0}" == "1" ]] && prepare_args+=(--skip-download)
    echo "[PICA_WAM] preparing ${dataset_dir} from ${source_dir}"
    python "${RUNTIME_ROOT}/tools/prepare_robotwin_data.py" "${prepare_args[@]}"
fi

# The dataset looks up caches named *.wan22ti2v5b.pt, which text_cache.py
# derives from the basename of model_id.
if [[ "$(basename "${wan_dir}")" != "Wan2.2-TI2V-5B" ]]; then
    echo "[PICA_WAM] ERROR: the Wan2.2 directory must be named Wan2.2-TI2V-5B, got ${wan_dir}." >&2
    exit 1
fi
for required in "${wan_dir}/models_t5_umt5-xxl-enc-bf16.pth" "${wan_dir}/google/umt5-xxl"; do
    if [[ ! -e "${required}" ]]; then
        echo "[PICA_WAM] ERROR: missing ${required}; run download_assets.sh first." >&2
        exit 1
    fi
done

IFS=',' read -r -a gpu_arr <<< "${gpu_ids}"
echo "[PICA_WAM] building the text-embedding cache at ${text_cache} on GPUs ${gpu_ids}"
(
    cd "${RUNTIME_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    export ROBOTWIN_DATA_ROOT="${dataset_dir}"
    export ROBOTWIN_TEXT_CACHE="${text_cache}"
    export WAM_CACHE_ROOT="${WAM_CACHE_ROOT:-${POLICY_DIR}/.cache}"
    export DIFFSYNTH_SKIP_DOWNLOAD=true
    export TOKENIZERS_PARALLELISM=false
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#gpu_arr[@]}" \
        tools/text_cache.py task=robotwin_pica \
        "model.model_id=${wan_dir}" \
        "model.tokenizer_model_id=${wan_dir}" \
        model.redirect_common_files=false \
        "+text_cache_batch_size=${TEXT_CACHE_BATCH_SIZE:-64}"
)

echo "[PICA_WAM] done. Train with:"
echo "  bash ${POLICY_DIR}/train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_ids>"
