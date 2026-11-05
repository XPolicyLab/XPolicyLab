#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 10 ]]; then
    echo "Usage: eval.sh bench task ckpt robot action seed policy_gpu env_gpu policy_env eval_env" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
port="$(bash "${XPL_ROOT}/utils/get_free_port.sh")"
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
echo "[MAIN] start server, policy_server_port=${port}"
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "${port}" &
SERVER_PID=$!
bash "${XPL_ROOT}/utils/wait_for_policy_server.sh" localhost "${port}" "${SERVER_PID}" "MR0 policy server" 1200
echo "[MAIN] start client, server=localhost:${port}"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
    "ckpt_name=$3,action_type=$5" "${port}" localhost
echo "[MAIN] eval finished"
