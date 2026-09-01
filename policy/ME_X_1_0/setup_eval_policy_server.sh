#!/bin/bash
set -euo pipefail

bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}
action_type=${5}; seed=${6}; policy_gpu_id=${7}; policy_conda_env=${8}
policy_server_port=${9}; policy_server_host=${10:-localhost}
MEX_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${MEX_POLICY_DIR}/../.." && pwd)"
policy_name="$(basename "${MEX_POLICY_DIR}")"
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    export CONDA_PREFIX="${policy_conda_env}"
    export PATH="${policy_conda_env}/bin:${PATH}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi
if [[ -f "${MEX_POLICY_DIR}/runtime_env.sh" ]]; then
    source "${MEX_POLICY_DIR}/runtime_env.sh"
fi
exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${MEX_POLICY_DIR}/deploy.yml" --overrides \
    port="${policy_server_port}" host="${policy_server_host}" \
    bench_name="${bench_name}" task_name="${task_name}" ckpt_name="${ckpt_name}" \
    env_cfg_type="${env_cfg_type}" seed="${seed}" policy_name="${policy_name}" \
    action_type="${action_type}"
