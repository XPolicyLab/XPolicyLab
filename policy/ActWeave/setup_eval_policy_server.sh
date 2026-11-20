#!/bin/bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
policy_env=${8:?}
server_port=${9:?}
server_host=${10:-localhost}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "${script_dir}/../.." && pwd -P)"
: "${ACTWEAVE_ROOT:?Set ACTWEAVE_ROOT to the ActWeave runtime checkout}"
: "${OPENAI_BASE_URL:?}"
: "${OPENAI_API_KEY:?}"

if [[ -f "${policy_env}/bin/activate" ]]; then
    source "${policy_env}/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
fi

export EVO_CONFIG="$(realpath "${ckpt_name}")"
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
exec python "${script_dir}/serve.py" \
    --config_path "${script_dir}/deploy.yml" \
    --overrides "port=${server_port}" "host=${server_host}" \
    "bench_name=${bench_name}" "task_name=${task_name}" "ckpt_name=${EVO_CONFIG}" \
    "env_cfg_type=${env_cfg_type}" "seed=${seed}" "policy_name=ActWeave" "action_type=${action_type}" "gpu_id=${policy_gpu_id}"
