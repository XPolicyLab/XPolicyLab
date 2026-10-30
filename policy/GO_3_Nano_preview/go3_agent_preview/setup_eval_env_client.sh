#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
  eval_python="${eval_env_conda_env}/bin/python"
else
  eval_python="${BENCH_ROOT}/.conda/envs/${eval_env_conda_env}/bin/python"
fi
[[ -x "${eval_python}" ]] || {
  echo "[error] simulator Python is not executable: ${eval_python}" >&2
  exit 1
}

export PATH="$(dirname "${eval_python}"):${PATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
eval_batch="$("${eval_python}" -c 'import sys, yaml; print(str(bool(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["eval_batch"])).lower())' "${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}")"
exec bash "${BENCH_ROOT}/scripts/eval_policy.sh" \
  --bench_name "${bench_name}" \
  --task_name "${task_name}" \
  --env_cfg_type "${env_cfg_type}" \
  --policy_name go3_agent_preview \
  --host "${policy_server_ip}" \
  --port "${policy_server_port}" \
  --protocol ws \
  --eval_batch "${eval_batch}" \
  --root_dir "${BENCH_ROOT}" \
  --device_id "${env_gpu_id}" \
  --additional_info "${additional_info}" \
  --seed "${seed}"
