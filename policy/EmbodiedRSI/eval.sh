#!/usr/bin/env bash
set -euo pipefail
[[ $# == 10 ]] || { echo 'Usage: eval.sh bench task checkpoint robot action seed policy_gpu env_gpu policy_env eval_env' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime/env.sh"
activate_policy_env "$9"
policy_server_port="$(bash "$XPL_ROOT/utils/get_free_port.sh")"
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "[MAIN] start server, policy_server_port=$policy_server_port"
bash "$SCRIPT_DIR/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$policy_server_port" \
    </dev/null &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" 127.0.0.1 "$policy_server_port" "$SERVER_PID" EmbodiedRSI 120
echo "[MAIN] start client, server=127.0.0.1:$policy_server_port"
bash "$SCRIPT_DIR/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
    "ckpt_name=$3,action_type=$5" "$policy_server_port"
echo '[MAIN] eval finished'
