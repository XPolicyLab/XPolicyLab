#!/bin/bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

policy_python="${policy_conda_env}/bin/python"
[[ -x "$policy_python" ]] || { echo "Pass an existing Python environment prefix as argument 8." >&2; exit 1; }
export PATH="${policy_conda_env}/bin:$PATH"
export PYTHONPATH="${XPL_ROOT}:$(dirname "$XPL_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
: "${LIBER0_MODEL_PATH:?Set the base model assets directory}"

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
exec "$policy_python" -u "${XPL_ROOT}/setup_policy_server.py" \
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
            model_assets_path="${LIBER0_MODEL_PATH}" \
            weights_file="${LIBER0_WEIGHTS_FILE:-model.pt}"
