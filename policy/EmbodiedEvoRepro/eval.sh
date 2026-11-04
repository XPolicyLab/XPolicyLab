#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 10 ]] || { echo 'Usage: eval.sh bench task ckpt robot action seed policy_gpu env_gpu policy_env eval_env' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/shell_common.sh"
validate_mode
[[ $1 == RoboDojo ]] || { echo "[ERROR] This adapter supports RoboDojo only" >&2; exit 2; }
port=$(bash "${XPL_ROOT}/utils/get_free_port.sh")
cleanup() {
    local status=$?
    local server_status=0
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID:-}" ]]; then
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            # A server may exit between this check and kill; wait still owns
            # and reports its final status in either case.
            kill -TERM "${SERVER_PID}" 2>/dev/null || true
        fi
        if wait "${SERVER_PID}"; then
            server_status=0
        else
            server_status=$?
            echo "[ERROR] Policy server cleanup/exit failed with status ${server_status}" >&2
        fi
        if [[ ${status} -eq 0 && ${server_status} -ne 0 ]]; then
            status=${server_status}
        fi
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "${port}" &
SERVER_PID=$!
bash "${XPL_ROOT}/utils/wait_for_policy_server.sh" localhost "${port}" "${SERVER_PID}" 'EmbodiedEvoRepro server' 120
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
    "ckpt_name=$3,action_type=$5" "${port}" localhost
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    echo '[INTERFACE PROBE COMPLETE] No simulator/model/task score was evaluated.'
else
    echo '[CLIENT EXITED] Inspect native reports and worker receipts; exit status alone is not success.'
fi
