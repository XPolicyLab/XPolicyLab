#!/bin/bash
set -euo pipefail

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

if [[ "${policy_conda_env}" != "system" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi

weights_dir=${AWOMO05_WEIGHTS:-$SCRIPT_DIR/weights}
export AWOMO05_QWEN_VL=${AWOMO05_QWEN_VL:-${AWOMO_QWEN3_VL:-$weights_dir/Qwen3-VL-4B-Instruct}}
export AWOMO05_FLUX2_VAE=${AWOMO05_FLUX2_VAE:-${AWOMO_FLUX2_VAE:-$weights_dir/FLUX.2-klein-base-4B/vae}}

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${AWOMO05_PYTHON:-python}" "${XPL_ROOT}/setup_policy_server.py" \
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
            input_color_order="${AWOMO05_INPUT_COLOR_ORDER:-bgr}"
