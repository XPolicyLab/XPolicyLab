#!/usr/bin/env bash
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
WSA_DIR="${WSA_ROOT:-${SCRIPT_DIR}/WSA}"
yaml_file="${SCRIPT_DIR}/deploy.yml"
policy_name="$(basename "${SCRIPT_DIR}")"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN=python
fi

overrides=(
    "port=${policy_server_port}"
    "host=${policy_server_host}"
    "bench_name=${bench_name}"
    "task_name=${task_name}"
    "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}"
    "seed=${seed}"
    "policy_name=${policy_name}"
    "action_type=${action_type}"
    "wsa_root=${WSA_DIR}"
    "allow_dummy_policy=${WSA_ALLOW_DUMMY_POLICY:-false}"
)

append_env_override() {
    local env_name=$1
    local config_key=$2
    if [[ -n "${!env_name:-}" ]]; then
        overrides+=("${config_key}=${!env_name}")
    fi
}

append_env_override WSA_CHECKPOINT_PATH checkpoint_path
append_env_override WSA_ACTION_MODE action_mode
append_env_override WSA_DTYPE dtype
append_env_override WSA_LOAD_CHECKPOINT_ON_CPU load_checkpoint_on_cpu
append_env_override WSA_INFER_HORIZON infer_horizon
append_env_override WSA_BINARIZE_GRIPPER binarize_gripper
append_env_override WSA_LARGE_STATS_PATH wsa_large_stats_path
append_env_override QWEN3_VL_PRETRAINED_PATH qwen3_vl_pretrained_path
append_env_override QWEN3_VL_PROCESSOR_PATH qwen3_vl_processor_path
append_env_override COSMOS_TOKENIZER_PATH_OR_NAME cosmos_tokenizer_path_or_name
append_env_override DA3_MODEL_PATH_OR_NAME da3_model_path_or_name
append_env_override DA3_CODE_ROOT da3_code_root
append_env_override WAN_MODEL_ID wsa_large_model_id
append_env_override WAN_TOKENIZER_MODEL_ID wsa_large_tokenizer_model_id
append_env_override ACTION_DIT_PRETRAINED_PATH wsa_large_action_dit_pretrained_path
append_env_override FUTURE_3D_PRETRAINED_PATH wsa_large_future_3d_pretrained_path
append_env_override WSA_LARGE_LOAD_TEXT_ENCODER wsa_large_load_text_encoder

echo "[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}"
echo "[SERVER] ${policy_server_host}:${policy_server_port}, WSA_ROOT=${WSA_DIR}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${WSA_DIR}/src:${WSA_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
