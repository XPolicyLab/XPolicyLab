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

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Optional env-var overrides; unset ones keep the deploy.yml values.
extra_overrides=()
[[ -n "${EWAM_CKPT_PATH:-}" ]] && extra_overrides+=("checkpoint_path=${EWAM_CKPT_PATH}")
[[ -n "${EWAM_WAN_PATH:-}" ]] && extra_overrides+=("wan_path=${EWAM_WAN_PATH}")
[[ -n "${EWAM_VLM_PATH:-}" ]] && extra_overrides+=("vlm_path=${EWAM_VLM_PATH}")
[[ -n "${EWAM_ROOT:-}" ]] && extra_overrides+=("ewam_root=${EWAM_ROOT}")
[[ -n "${EWAM_ALLOW_DUMMY_POLICY:-}" ]] && extra_overrides+=("allow_dummy_policy=${EWAM_ALLOW_DUMMY_POLICY}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}, policy_server_port=${policy_server_port}"

# Accept either a conda env prefix path or an env name for `conda activate`.
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="python"
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${PYTHONPATH:-}" \
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
            "${extra_overrides[@]}"
