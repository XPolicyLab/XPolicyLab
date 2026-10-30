#!/bin/bash
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
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

# Resolve action representation before starting either process. "auto" selects
# the registered task contract; an incompatible explicit action is rejected.
ROUTE_PYTHON="${ME_BRAIN_CONFIG_PYTHON:-python3}"
backend="${ME_BRAIN_BACKEND:-auto}"
action_type="$("${ROUTE_PYTHON}" "${SCRIPT_DIR}/routing.py" --task "${task_name}" --backend "${backend}" --action "${action_type}" --field action_type)"


# The official simulator reads its request timeout from this environment key.
if [[ "${backend}" == "auto" && ( "${task_name}" == "deposit_coin" || "${task_name}" == "fasten_screws" ) ]]; then
    export ROBODOJO_WS_REQUEST_TIMEOUT_S="${ROBODOJO_WS_REQUEST_TIMEOUT_S:-1800}"
fi

policy_name="$(basename "${SCRIPT_DIR}")"

yaml_file="$(mktemp "${TMPDIR:-/tmp}/me-brain-config.XXXXXX")"
trap 'rm -f "${yaml_file}"' EXIT
"${ROUTE_PYTHON}" "${SCRIPT_DIR}/routing.py" --task "${task_name}" --backend "${backend}" --action "${action_type}" --write-config "${yaml_file}" >/dev/null


additional_info="${additional_info},action_type=${action_type}"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${EVAL_ENV_TYPE:-}" == "debug" && -n "${ME_BRAIN_EVAL_PYTHON:-}" ]]; then
    eval_batch="$("${ROUTE_PYTHON}" "${SCRIPT_DIR}/routing.py" --task "${task_name}" --field eval_batch)"
    export PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    EVAL_PYTHON="${ME_BRAIN_EVAL_PYTHON}"
    [[ "${EVAL_PYTHON}" = /* ]] || EVAL_PYTHON="${SCRIPT_DIR}/${EVAL_PYTHON}"
    exec "${EVAL_PYTHON}" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name "${policy_name}" \
        --protocol ws --host "${policy_server_ip}" --port "${policy_server_port}" \
        --eval_batch "${eval_batch}"
fi

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
