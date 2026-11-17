#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_env=${8}
policy_server_port=${9}
policy_server_host=${10:-127.0.0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"

if [[ -x "${policy_env}/bin/python" ]]; then
    policy_python="$(cd "${policy_env}" && pwd)/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
    policy_python="$(command -v python)"
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"
server_entry=("${XPL_ROOT}/setup_policy_server.py")
case "${HWM_DYNAMIC_BATCH:-1}" in
    0) ;;
    1) server_entry=(-m XPolicyLab.policy.HWM_0_preview.hwm.runtime.server) ;;
    *) echo 'HWM_DYNAMIC_BATCH must be 0 or 1' >&2; exit 2 ;;
esac
batch_overrides=()
if [[ -n "${HWM_INFERENCE_BATCH_SIZE:-}" ]]; then
    batch_overrides+=("inference_batch_size=${HWM_INFERENCE_BATCH_SIZE}")
fi
exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${policy_python}" "${server_entry[@]}" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides \
        port="${policy_server_port}" host="${policy_server_host}" \
        bench_name="${bench_name}" task_name="${task_name}" \
        ckpt_name="${ckpt_name}" env_cfg_type="${env_cfg_type}" \
        seed="${seed}" policy_name="${policy_name}" action_type="${action_type}" "${batch_overrides[@]}"
