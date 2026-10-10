#!/usr/bin/env bash
set -euo pipefail
bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_conda_env=${8:?policy_conda_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-127.0.0.1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
extra=()
[[ -z "${VPP2_BUNDLE:-}" ]] || extra+=(--bundle "${VPP2_BUNDLE}")
[[ -z "${VPP2_WAN_ROOT:-}" ]] || extra+=(--wan "${VPP2_WAN_ROOT}")
exec "${CONDA_PREFIX}/bin/python" -u "${SCRIPT_DIR}/launch_policy.py" \
  --bench "${bench_name}" --task "${task_name}" --ckpt "${ckpt_name}" \
  --env-cfg "${env_cfg_type}" --action-type "${action_type}" --seed "${seed}" \
  --gpu "${policy_gpu_id}" --port "${policy_server_port}" --host "${policy_server_host}" \
  --steps "${VPP2_NUM_INFERENCE_STEPS:-10}" --shift "${VPP2_SIGMA_SHIFT:-1}" \
  --replan "${VPP2_REPLAN_STEPS:-24}" "${extra[@]}"
