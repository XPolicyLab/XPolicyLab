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

if [[ "${bench_name}" != "RoboDojo" || "${action_type}" != "ee" ]]; then
    echo "[CLIENT][ERROR] OEH_Review runs RoboDojo with action_type ee; got ${bench_name}/${action_type}" >&2
    exit 1
fi

# deploy.py runs inside the evaluation client and imports the vendored runtime (numpy, Pillow and pyyaml, which the
# RoboDojo environment has). When the eval env is given as a directory, check the runtime there too.
export PYTHONPATH="${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    "${eval_env_conda_env}/bin/python" "${SCRIPT_DIR}/tools/verify_runtime.py"
    "${eval_env_conda_env}/bin/python" "${SCRIPT_DIR}/tools/check_deps.py" eval
fi
# Per-episode evidence (observations, decisions, receipts) of the OEH gateway: README, Notes.
export OEH_REVIEW_RUNS_ROOT="${OEH_REVIEW_RUNS_ROOT:-${SCRIPT_DIR}/runs}"

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
