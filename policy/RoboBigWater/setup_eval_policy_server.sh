#!/bin/bash
# Policy server of RoboBigWater. The Python of the Isaac environment is used (it has cuRobo); the
# policy env argument is ignored. Settings come from RoboBigWater's config.env.
set -euo pipefail
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}; seed=${6}
policy_gpu_id=${7}; policy_conda_env=${8}; policy_server_port=${9}; policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROBOSHELL_ROOT="${SCRIPT_DIR}"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
if [ -f "${ROBOSHELL_ROOT}/config.env" ]; then set -a; source "${ROBOSHELL_ROOT}/config.env"; set +a; fi
# Defaults when config.env does not set them: the RoboDojo checkout XPolicyLab lives in, and the Python passed as
# <policy_env> (a python binary, or a directory holding .venv/bin/python or bin/python). The server needs cuRobo, so
# this must be the Isaac Sim environment of the RoboDojo checkout.
ROBODOJO_REPO="${ROBODOJO_REPO:-${BENCH_ROOT}}"
if [ -z "${ISAAC_PYTHON:-}" ]; then
  if [ -x "${policy_conda_env}" ]; then ISAAC_PYTHON="${policy_conda_env}"
  elif [ -x "${policy_conda_env}/.venv/bin/python" ]; then ISAAC_PYTHON="${policy_conda_env}/.venv/bin/python"
  elif [ -x "${policy_conda_env}/bin/python" ]; then ISAAC_PYTHON="${policy_conda_env}/bin/python"
  elif [ -x "${BENCH_ROOT}/../isaacsim-5.1.0/.venv/bin/python" ]; then ISAAC_PYTHON="$(cd "${BENCH_ROOT}/../isaacsim-5.1.0/.venv/bin" && pwd)/python"
  else echo "[SERVER] set ISAAC_PYTHON in config.env or pass the Isaac Sim python as <policy_env>" >&2; exit 1; fi
fi
export ROBODOJO_REPO ISAAC_PYTHON

echo "[SERVER] policy=RoboBigWater task=${task_name} port=${policy_server_port} gpu=${policy_gpu_id}"
# The Isaac environment bundles an old `websockets`; the policy server needs a newer one (see scripts/install_policy_deps.sh)
DEPS="${ROBOSHELL_ROOT}/.cache/policy-deps"
export PYTHONPATH="${DEPS}:${ROBOSHELL_ROOT}:${ROBODOJO_REPO}:${XPL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 ROBOSHELL_RUN_ROOT="${ROBOSHELL_RUN_ROOT:-${ROBOSHELL_ROOT}/runs/bridge/${task_name}}"
cd "${XPL_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" "${ISAAC_PYTHON}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides port="${policy_server_port}" host="${policy_server_host}" bench_name="${bench_name}" \
        task_name="${task_name}" ckpt_name="${ckpt_name}" env_cfg_type="${env_cfg_type}" seed="${seed}" \
        policy_name="RoboBigWater" action_type="${action_type}"
