#!/usr/bin/env bash
set -euo pipefail
bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?evaluation seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_conda_env=${8:?policy_conda_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-127.0.0.1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
exec "${CONDA_PREFIX}/bin/python" -u "${SCRIPT_DIR}/launch_policy.py" \
  --bench "${bench_name}" --task "${task_name}" --ckpt "${ckpt_name}" \
  --env-cfg "${env_cfg_type}" --action-type "${action_type}" --seed "${seed}" \
  --gpu "${policy_gpu_id}" --port "${policy_server_port}" --host "${policy_server_host}" \
  --steps "${LOGIC_PREVIEW_NUM_INFERENCE_STEPS:-10}" --shift "${LOGIC_PREVIEW_SIGMA_SHIFT:-1}" \
  --replan "${LOGIC_PREVIEW_REPLAN_STEPS:-24}"
