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

if [[ "${action_type}" != "joint" ]]; then
    echo "[CLIENT][ERROR] Mibot-RSI sends joint actions; use action_type joint, not ${action_type}" >&2
    exit 1
fi

# The agent runs in the evaluation client: check its Python, the harness runtime and, for a simulator run,
# the model endpoint and key (names only; values are never printed).
if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    client_python="${eval_env_conda_env}/bin/python"
else
    client_python="$(
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
        conda activate "${eval_env_conda_env}" || exit 1
        command -v python
    )" || client_python=""
fi
if [[ -z "${client_python}" || ! -x "${client_python}" ]]; then
    echo "[CLIENT][ERROR] could not resolve the Python of eval env ${eval_env_conda_env} (a conda env name or path)" >&2
    exit 1
fi
check_args=(check)
if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" ]]; then
    check_args+=(--model-env)
fi
"${client_python}" "${SCRIPT_DIR}/artifacts.py" "${check_args[@]}"

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
