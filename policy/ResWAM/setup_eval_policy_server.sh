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
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UPSTREAM_DIR="${RESWAM_REPO:-${SCRIPT_DIR}/ResWAM}"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

# <policy_conda_env> is either the venv created by install.sh (a path with bin/python) or a conda env.
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN=python
fi

overrides=()
[[ -n "${RESWAM_CHECKPOINT_PATH:-}" ]] && overrides+=(checkpoint_path="${RESWAM_CHECKPOINT_PATH}")
[[ -n "${RESWAM_CHECKPOINT_SHA256:-}" ]] && overrides+=(checkpoint_sha256="${RESWAM_CHECKPOINT_SHA256}")

exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${BENCH_ROOT}:${UPSTREAM_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
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
            "${overrides[@]}"
