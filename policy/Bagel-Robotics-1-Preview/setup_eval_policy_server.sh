#!/usr/bin/env bash
set -euo pipefail
bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_env=${8:?policy_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-127.0.0.1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${policy_env}/bin/python" ]]; then
  python_bin="${policy_env}/bin/python"
elif [[ -x "${SCRIPT_DIR}/${policy_env}/bin/python" ]]; then
  python_bin="${SCRIPT_DIR}/${policy_env}/bin/python"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${policy_env}"
  python_bin="${CONDA_PREFIX}/bin/python"
fi
extra=()
[[ -z "${BAGEL_ROBOTICS_1_BUNDLE:-}" ]] || extra+=(--bundle "${BAGEL_ROBOTICS_1_BUNDLE}")
[[ -z "${BAGEL_ROBOTICS_1_WAN_ROOT:-}" ]] || extra+=(--wan "${BAGEL_ROBOTICS_1_WAN_ROOT}")
exec "${python_bin}" -u "${SCRIPT_DIR}/launch_policy.py" \
  --bench "${bench_name}" --task "${task_name}" --ckpt "${ckpt_name}" \
  --env-cfg "${env_cfg_type}" --action-type "${action_type}" --seed "${seed}" \
  --gpu "${policy_gpu_id}" --port "${policy_server_port}" --host "${policy_server_host}" \
  "${extra[@]}"
