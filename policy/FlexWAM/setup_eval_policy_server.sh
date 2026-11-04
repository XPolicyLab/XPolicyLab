#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${XPL_ROOT}/policy/${policy_name}"
yaml_file="${POLICY_DIR}/deploy.yml"

# Vendored FlexWAM checkout (weights are not vendored).
FLEXWAM_ROOT="${FLEXWAM_ROOT:-${POLICY_DIR}/flexwam}"
CKPT_ROOT="${POLICY_DIR}/checkpoints"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

# Checkpoint dir resolution precedence (highest first):
#   1. FLEXWAM_CKPT_DIR        explicit full path override
#   2. ckpt_name as an absolute path
#   3. ckpt_name as a relative path (POLICY_DIR-relative)
#   4. checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/
#   5. checkpoints/<ckpt_name>/
if [[ -n "${FLEXWAM_CKPT_DIR:-}" ]]; then
    ckpt_dir="${FLEXWAM_CKPT_DIR}"
elif [[ "${ckpt_name}" == /* ]]; then
    ckpt_dir="${ckpt_name}"
elif [[ "${ckpt_name}" == */* ]]; then
    ckpt_dir="${POLICY_DIR}/${ckpt_name}"
elif [[ -d "${CKPT_ROOT}/${run_dir_name}" ]]; then
    ckpt_dir="${CKPT_ROOT}/${run_dir_name}"
else
    ckpt_dir="${CKPT_ROOT}/${ckpt_name}"
fi

allow_dummy_policy="${FLEXWAM_ALLOW_DUMMY_POLICY:-false}"
cuda_graph="${FLEXWAM_CUDA_GRAPH:-true}"
dataset_stats_path="${FLEXWAM_DATASET_STATS:-}"
checkpoint_path="${FLEXWAM_CHECKPOINT_PATH:-}"
task_name_override="${FLEXWAM_TASK:-}"
# Frozen VAE, UMT5 encoder, and UMT5 tokenizer. Shared by arx_x5, piper, and piper_x.
model_base="${FLEXWAM_MODEL_BASE_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-${CKPT_ROOT}/FlexWAM-base}}"

if [[ "${allow_dummy_policy}" != "true" ]]; then
    vae_path="${model_base}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    umt5_path="${model_base}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    tokenizer_dir="${model_base}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    if [[ ! -f "${vae_path}" || ! -f "${umt5_path}" || ! -d "${tokenizer_dir}" ]]; then
        echo -e "\033[31m[SERVER] FlexWAM-base is missing under ${model_base}\033[0m" >&2
        echo -e "\033[31m[SERVER] Download the frozen VAE, UMT5 encoder, and tokenizer first:\033[0m" >&2
        echo -e "\033[31m[SERVER]   huggingface-cli download Zzming/FlexWAM-base --local-dir ${CKPT_ROOT}/FlexWAM-base\033[0m" >&2
        exit 1
    fi
fi

if [[ ! -d "${FLEXWAM_ROOT}/src/flexwam" ]]; then
    echo -e "\033[31m[SERVER] flexwam package not found under ${FLEXWAM_ROOT}\033[0m" >&2
    exit 1
fi

if [[ "${allow_dummy_policy}" != "true" && ! -e "${ckpt_dir}" ]]; then
    echo -e "\033[31m[SERVER] checkpoint path does not exist: ${ckpt_dir}\033[0m" >&2
    exit 1
fi

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] ckpt_dir: ${ckpt_dir}\033[0m"
echo -e "\033[33m[SERVER] model_base: ${model_base}\033[0m"
echo -e "\033[33m[SERVER] flexwam_root: ${FLEXWAM_ROOT}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port} cuda_graph=${cuda_graph}\033[0m"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="python"
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${FLEXWAM_ROOT}:${FLEXWAM_ROOT}/src:${BENCH_ROOT}:${PYTHONPATH:-}" \
    FLEXWAM_MODEL_BASE_PATH="${model_base}" \
    DIFFSYNTH_MODEL_BASE_PATH="${model_base}" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            flexwam_root="${FLEXWAM_ROOT}" \
            ckpt_dir="${ckpt_dir}" \
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            task="${task_name_override}" \
            cuda_graph="${cuda_graph}" \
            allow_dummy_policy="${allow_dummy_policy}"
