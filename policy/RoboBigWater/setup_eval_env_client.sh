#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
# Simulator-side environment of this adapter (library paths, caches, TMPDIR; from config.env). The eval-env argument
# may name the Isaac Sim environment directory or its python when config.env does not set ISAAC_PYTHON.
if [ -z "${ISAAC_PYTHON:-}" ] && [ ! -f "${SCRIPT_DIR}/config.env" ]; then
  if [ -x "${eval_env_conda_env}/bin/python" ]; then export ISAAC_PYTHON="${eval_env_conda_env}/bin/python"
  elif [ -x "${eval_env_conda_env}" ]; then export ISAAC_PYTHON="${eval_env_conda_env}"; fi
fi
if [ -f "${SCRIPT_DIR}/config.env" ] || [ -n "${ISAAC_PYTHON:-}" ]; then
  # shellcheck source=scripts/sim_env.sh
  source "${SCRIPT_DIR}/scripts/sim_env.sh"
fi
# RoboBigWater observes depth: its env configs are added next to the stock ones (nothing upstream is modified)
cp -n "${SCRIPT_DIR}/env_cfg/arx_x5_rgbd.yml" "${BENCH_ROOT}/env_cfg/" 2>/dev/null || true
cp -n "${SCRIPT_DIR}/env_cfg/camera_config_rgbd.yml" "${BENCH_ROOT}/env_cfg/camera/" 2>/dev/null || true

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
