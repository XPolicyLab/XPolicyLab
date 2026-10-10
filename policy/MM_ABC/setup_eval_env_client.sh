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
additional_info=${9:-}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}, EVAL_ENV_TYPE=${EVAL_ENV_TYPE:-sim}"

if [[ "${EVAL_ENV_TYPE:-}" != "debug" ]]; then
    exec bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${policy_server_port}" \
        "${bench_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" \
        "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
fi

# shellcheck source=resolve_python.sh
source "${SCRIPT_DIR}/resolve_python.sh"
PY="${MMABC_PYTHON:-$(resolve_python "${eval_env_conda_env}")}"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

replay_args=()
if [[ -n "${REPLAY_HDF5:-}" ]]; then
    # shellcheck disable=SC2206
    replay_args=(--hdf5 ${REPLAY_HDF5} --start "${REPLAY_START:-0}")
fi
send_args=()
[[ -n "${SEND:-}" ]] && send_args=(--send "${SEND}")

exec env \
    CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    "${PY}" "${SCRIPT_DIR}/debug_mobile_client.py" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --seed "${seed}" \
        --eval_episode_num "${EVAL_EPISODE_NUM:-2}" \
        --episode_step_limit "${EPISODE_STEP_LIMIT:-48}" \
        --eval_batch "${EVAL_BATCH:-false}" \
        "${replay_args[@]}" "${send_args[@]}"
