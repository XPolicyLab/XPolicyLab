#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name is required}
task_name=${2:?task_name is required}
ckpt_name=${3:?finalized joint checkpoint path is required}
env_cfg_type=${4:?env_cfg_type is required}
action_type=${5:?action_type is required}
seed=${6:?seed is required}
policy_gpu_id=${7:?policy_gpu_id is required}
policy_conda_env=${8:-${EGOVLA_CONDA_ENV:-}}
policy_server_port=${9:?port is required}
policy_server_host=${10:-localhost}

[[ "${bench_name}" == "SparkArena" ]] || {
  echo "this server entry point requires bench_name=SparkArena" >&2; exit 2;
}
[[ "${env_cfg_type}" == "tianji_marvin_wuji" && "${action_type}" == "joint" ]] || {
  echo "SparkArena EgoVLA evaluation requires tianji_marvin_wuji/joint" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${EGOVLA_UPSTREAM_ROOT:-${SCRIPT_DIR}/EgoVLA_Release}"
if [[ -n "${EGOVLA_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${EGOVLA_PYTHON_BIN}"
elif [[ -n "${policy_conda_env}" && -x "${policy_conda_env}/bin/python" ]]; then
  PYTHON_BIN="${policy_conda_env}/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "missing policy Python: ${PYTHON_BIN}" >&2; exit 2; }

unset LD_LIBRARY_PATH
# .deps vendors a numpy source tree. Keep the conda install ahead of it so
# `import numpy` does not load that checkout.
conda_site="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${conda_site}:${SCRIPT_DIR}/.deps:${SCRIPT_DIR}/EgoVLA_Release/VILA:${SCRIPT_DIR}/EgoVLA_Release:${SCRIPT_DIR}:${XPL_ROOT}:${UPSTREAM_ROOT}/VILA:${UPSTREAM_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
export EGOVLA_STRICT_ACTION_CONTRACT=1

# Verify every model/data/contract hash before importing the multi-GB model.
"${PYTHON_BIN}" -c \
  'import sys; from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.utils.sparkarena_provenance import verify_sparkarena_checkpoint; verify_sparkarena_checkpoint(sys.argv[1]); print("[egovla-sparkarena] checkpoint provenance verified")' \
  "${ckpt_name}"

exec env PYTHONWARNINGS=ignore::UserWarning \
  "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides \
      port="${policy_server_port}" \
      host="${policy_server_host}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      pretrained_path="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name=EgoVLA \
      action_type=joint
